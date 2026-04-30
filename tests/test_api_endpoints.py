"""Integration tests for REST API endpoints."""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

import cadence.api as api_module
from cadence.api import app
from cadence.core.config import Config
from cadence.core.trace import TraceLogger
from cadence.storage.chat_store import ChatStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def chat_store(tmp_path):
    """Provide a ChatStore backed by a temp database."""
    return ChatStore(db_path=str(tmp_path / "test_chats.db"))


@pytest.fixture()
def client(tmp_path, monkeypatch, chat_store):
    """Create a TestClient with isolated chat store (no CadenceApp needed for most endpoints)."""
    monkeypatch.setattr(api_module, "_chat_store", None)
    monkeypatch.setattr(api_module, "get_chat_store", lambda: chat_store)
    return TestClient(app, raise_server_exceptions=False)


def _make_api_config() -> Config:
    cfg = Config()
    cfg.prompt_evolution.enabled = False
    cfg.message_bus.enabled = False
    cfg.checkpoints.enabled = False
    cfg.learning.enabled = False
    cfg.knowledge_graph.enabled = False
    cfg.mcp.enabled = False
    return cfg


class FakeAgentApp:
    def __init__(self):
        self.config = _make_api_config()
        self.trace = TraceLogger(console=False)
        self.reconfigured_with: Config | None = None

    async def run(
        self,
        message,
        conversation_history=None,
        images=None,
        session_id="",
        trace=None,
        on_task_update=None,
    ):
        trace.thought("fake-agent", f"request trace for {session_id}: {message}")
        await asyncio.sleep(0)
        return f"reply for {session_id}"

    async def run_streaming(
        self,
        message,
        collector,
        conversation_history=None,
        images=None,
        session_id="",
        trace=None,
        on_task_update=None,
    ):
        trace.thought("fake-agent", f"stream trace for {session_id}: {message}")
        await collector.emit_token("streamed", agent_id="fake-agent")
        await asyncio.sleep(0)
        return "streamed response"

    async def reconfigure(self, new_config: Config) -> None:
        self.reconfigured_with = new_config
        self.config = new_config
        self.trace = TraceLogger(console=False)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


class TestChatRuntimeIsolation:
    def test_chat_response_uses_request_local_trace(self, client, monkeypatch):
        fake_app = FakeAgentApp()
        fake_app.trace.thought("global", "unrelated global trace")
        monkeypatch.setattr(api_module, "get_app", lambda: fake_app)

        resp = client.post("/api/chat", json={"message": "hello", "session_id": "session-a"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "session-a"
        assert data["response"] == "reply for session-a"
        assert len(data["trace_steps"]) == 1
        assert data["trace_steps"][0]["content"] == "request trace for session-a: hello"

    def test_streaming_chat_emits_token_before_done(self, client, monkeypatch):
        fake_app = FakeAgentApp()
        monkeypatch.setattr(api_module, "get_app", lambda: fake_app)

        resp = client.post(
            "/api/chat/stream",
            json={"message": "hello", "session_id": "stream-session"},
        )

        assert resp.status_code == 200
        body = resp.text
        assert "event: token" in body
        assert "event: done" in body
        assert body.index("event: token") < body.index("event: done")
        assert '"session_id": "stream-session"' in body
        assert '"trace_steps": [' in body

    def test_dag_endpoint_returns_session_scoped_snapshot(self, client):
        from cadence.agents.orchestrator import TaskDAG
        from cadence.core.types import Task

        api_module._dag_snapshots.clear()
        dag = TaskDAG()
        dag.add(Task(description="Session scoped task"))

        handler = api_module._make_task_update_handler("dag-session")
        handler(dag)

        resp = client.get("/api/dag?session_id=dag-session")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "dag-session"
        assert data["nodes"][0]["description"] == "Session scoped task"

        empty_resp = client.get("/api/dag?session_id=other-session")
        assert empty_resp.status_code == 200
        assert empty_resp.json() == {
            "nodes": [],
            "edges": [],
            "session_id": "other-session",
        }

    def test_put_config_reconfigures_live_app(self, client, monkeypatch):
        fake_app = FakeAgentApp()
        new_config = _make_api_config()
        new_config.agents.max_parallel = 2

        monkeypatch.setattr(api_module, "get_app", lambda: fake_app)
        monkeypatch.setattr(api_module, "_update_config", lambda _updates: new_config)

        resp = client.put("/api/config", json={"updates": {"agents": {"max_parallel": 2}}})

        assert resp.status_code == 200
        assert fake_app.reconfigured_with is new_config
        assert resp.json()["agents"]["max_parallel"] == 2


# ---------------------------------------------------------------------------
# Chats CRUD
# ---------------------------------------------------------------------------

class TestChatsCRUD:
    def test_create_chat(self, client):
        resp = client.post("/api/chats", json={"title": "My Chat"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "My Chat"
        assert "id" in data
        assert data["created_at"] > 0

    def test_create_chat_with_custom_id(self, client):
        resp = client.post("/api/chats", json={"id": "custom-123", "title": "Custom"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "custom-123"
        assert data["title"] == "Custom"

    def test_list_chats_empty(self, client):
        resp = client.get("/api/chats")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_chats_returns_created(self, client):
        client.post("/api/chats", json={"title": "Chat A"})
        client.post("/api/chats", json={"title": "Chat B"})

        resp = client.get("/api/chats")
        assert resp.status_code == 200
        chats = resp.json()
        assert len(chats) == 2
        titles = {c["title"] for c in chats}
        assert "Chat A" in titles
        assert "Chat B" in titles

    def test_get_single_chat(self, client):
        create_resp = client.post("/api/chats", json={"id": "get-test", "title": "Get Test"})
        assert create_resp.status_code == 200

        resp = client.get("/api/chats/get-test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "get-test"
        assert data["title"] == "Get Test"
        assert "messages" in data

    def test_get_nonexistent_chat(self, client):
        resp = client.get("/api/chats/does-not-exist")
        assert resp.status_code == 404
        assert resp.json()["error"] == "Chat not found"

    def test_update_chat_title(self, client):
        client.post("/api/chats", json={"id": "upd-1", "title": "Original"})

        resp = client.put("/api/chats/upd-1", json={"title": "Updated Title"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Updated Title"

    def test_update_chat_session_id(self, client):
        client.post("/api/chats", json={"id": "upd-2", "title": "Chat"})

        resp = client.put("/api/chats/upd-2", json={"session_id": "sess-abc"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-abc"

    def test_delete_chat(self, client):
        client.post("/api/chats", json={"id": "del-1", "title": "To Delete"})

        resp = client.delete("/api/chats/del-1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's gone from the list
        list_resp = client.get("/api/chats")
        assert len(list_resp.json()) == 0

    def test_delete_nonexistent_chat(self, client):
        resp = client.delete("/api/chats/ghost")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_found"

    def test_add_message_to_chat(self, client):
        client.post("/api/chats", json={"id": "msg-1", "title": "Message Chat"})

        resp = client.post("/api/chats/msg-1/messages", json={
            "id": "m1",
            "role": "user",
            "content": "Hello agent",
            "timestamp": time.time(),
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify message is in the chat
        chat_resp = client.get("/api/chats/msg-1")
        data = chat_resp.json()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Hello agent"
        assert data["messages"][0]["role"] == "user"

    def test_add_multiple_messages(self, client):
        client.post("/api/chats", json={"id": "msg-2", "title": "Multi Msg"})
        now = time.time()

        client.post("/api/chats/msg-2/messages", json={
            "id": "m1", "role": "user", "content": "First", "timestamp": now,
        })
        client.post("/api/chats/msg-2/messages", json={
            "id": "m2", "role": "agent", "content": "Response", "timestamp": now + 1,
        })
        client.post("/api/chats/msg-2/messages", json={
            "id": "m3", "role": "user", "content": "Follow-up", "timestamp": now + 2,
        })

        chat_resp = client.get("/api/chats/msg-2")
        messages = chat_resp.json()["messages"]
        assert len(messages) == 3
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Response"
        assert messages[2]["content"] == "Follow-up"

    def test_add_message_with_trace_steps(self, client):
        client.post("/api/chats", json={"id": "msg-3", "title": "Trace Chat"})

        trace = [{"step_type": "thought", "content": "thinking..."}]
        resp = client.post("/api/chats/msg-3/messages", json={
            "id": "m1",
            "role": "agent",
            "content": "Here's my answer",
            "timestamp": time.time(),
            "duration_ms": 1234.5,
            "trace_steps": trace,
        })
        assert resp.status_code == 200

        chat_resp = client.get("/api/chats/msg-3")
        msg = chat_resp.json()["messages"][0]
        assert msg["duration_ms"] == 1234.5
        assert msg["trace_steps"] == trace


# ---------------------------------------------------------------------------
# Chat Store (unit-level)
# ---------------------------------------------------------------------------

class TestChatStore:
    def test_session_history_roundtrip(self, chat_store):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        chat_store.save_session_history("sess-1", history)
        loaded = chat_store.get_session_history("sess-1")
        assert loaded == history

    def test_session_history_empty(self, chat_store):
        loaded = chat_store.get_session_history("nonexistent")
        assert loaded == []

    def test_session_summary_roundtrip(self, chat_store):
        chat_store.save_session_summary("sess-2", "User asked about Python.")
        summary = chat_store.get_session_summary("sess-2")
        assert summary == "User asked about Python."

    def test_session_summary_empty(self, chat_store):
        summary = chat_store.get_session_summary("nonexistent")
        assert summary == ""

    def test_session_history_update(self, chat_store):
        chat_store.save_session_history("sess-3", [{"role": "user", "content": "v1"}])
        chat_store.save_session_history("sess-3", [{"role": "user", "content": "v2"}])
        loaded = chat_store.get_session_history("sess-3")
        assert loaded[0]["content"] == "v2"

    def test_chat_crud_full_lifecycle(self, chat_store):
        # Create
        chat = chat_store.create_chat(title="Lifecycle Test")
        assert chat.title == "Lifecycle Test"

        # List
        chats = chat_store.list_chats()
        assert len(chats) == 1

        # Update
        updated = chat_store.update_chat(chat.id, title="Updated")
        assert updated.title == "Updated"

        # Get
        fetched = chat_store.get_chat(chat.id)
        assert fetched.title == "Updated"

        # Delete
        assert chat_store.delete_chat(chat.id) is True
        assert chat_store.get_chat(chat.id) is None
        assert chat_store.list_chats() == []
