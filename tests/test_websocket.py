"""Integration tests for WebSocket connection and trace broadcasting."""

import json

import pytest
from fastapi.testclient import TestClient

import cadence.api as api_module
from cadence.api import app, _broadcast_trace
from cadence.core.types import TraceStep
from cadence.storage.chat_store import ChatStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Create a TestClient for WebSocket testing."""
    store = ChatStore(db_path=str(tmp_path / "ws_chats.db"))
    monkeypatch.setattr(api_module, "_chat_store", None)
    monkeypatch.setattr(api_module, "get_chat_store", lambda: store)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# WebSocket connection tests
# ---------------------------------------------------------------------------

class TestWebSocketConnection:
    def test_websocket_connect_and_disconnect(self, client):
        """Client can connect and disconnect from /ws."""
        with client.websocket_connect("/ws") as ws:
            # Connection should be established
            ws.send_text("ping")
            resp = ws.receive_text()
            data = json.loads(resp)
            assert data["type"] == "pong"

    def test_websocket_ping_pong(self, client):
        """Server responds to ping with pong."""
        with client.websocket_connect("/ws") as ws:
            ws.send_text("ping")
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "pong"

            # Multiple pings
            ws.send_text("ping")
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "pong"

    def test_websocket_multiple_clients(self, client):
        """Multiple clients can connect simultaneously."""
        with client.websocket_connect("/ws") as ws1:
            with client.websocket_connect("/ws") as ws2:
                ws1.send_text("ping")
                resp1 = json.loads(ws1.receive_text())
                assert resp1["type"] == "pong"

                ws2.send_text("ping")
                resp2 = json.loads(ws2.receive_text())
                assert resp2["type"] == "pong"


# ---------------------------------------------------------------------------
# Trace broadcast tests
# ---------------------------------------------------------------------------

class TestTraceBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_trace_no_clients(self):
        """Broadcasting with no connected clients should not error."""
        step = TraceStep(
            agent_id="test-agent",
            step_type="thought",
            content="thinking...",
        )
        # Should not raise
        api_module._ws_clients = set()
        await _broadcast_trace(step)

    def test_trace_step_serialization(self):
        """TraceStep can be serialized to JSON for broadcasting."""
        step = TraceStep(
            agent_id="agent-1",
            task_id="task-1",
            step_type="action",
            content="Running tool read_file",
            metadata={"tool": "read_file"},
        )
        data = step.model_dump()
        json_str = json.dumps({"type": "trace", "data": data})

        parsed = json.loads(json_str)
        assert parsed["type"] == "trace"
        assert parsed["data"]["agent_id"] == "agent-1"
        assert parsed["data"]["step_type"] == "action"
        assert parsed["data"]["content"] == "Running tool read_file"
        assert parsed["data"]["metadata"]["tool"] == "read_file"

    def test_trace_step_types(self):
        """All trace step types can be created."""
        for step_type in ["observation", "thought", "action", "result", "error"]:
            step = TraceStep(
                agent_id="agent-1",
                step_type=step_type,
                content=f"A {step_type} step",
            )
            assert step.step_type == step_type
            assert step.timestamp > 0
