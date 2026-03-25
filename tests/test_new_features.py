"""Tests for the 6 new features:
1. Agent-to-Agent Message Bus
2. Streaming Response Support
3. Human-in-the-Loop Checkpoints
4. Cross-Session Learning
5. Structured Knowledge Graphs
6. Multi-Modal Input Support
"""

import asyncio
import json
import time
import pytest

# ===========================================================================
# 1. Message Bus Tests
# ===========================================================================


class TestMessageBus:
    """Tests for cadence.core.message_bus."""

    def _make_bus(self):
        from cadence.core.message_bus import MessageBus
        return MessageBus(history_limit=50)

    @pytest.mark.asyncio
    async def test_publish_and_peek(self):
        bus = self._make_bus()
        msg = await bus.publish(
            topic="discovery",
            sender_id="agent-1",
            content="Found important data",
        )
        assert msg.id
        assert msg.topic == "discovery"
        assert msg.sender_id == "agent-1"

        messages = bus.peek("discovery", limit=10)
        assert len(messages) == 1
        assert messages[0].content == "Found important data"

    @pytest.mark.asyncio
    async def test_subscribe_receives_messages(self):
        bus = self._make_bus()
        received = []

        async def callback(msg):
            received.append(msg)

        bus.subscribe("status", "agent-2", callback)
        await bus.publish(topic="status", sender_id="agent-1", content="Done")

        # Give the callback task a moment to run
        await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0].content == "Done"

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = self._make_bus()
        received = []

        async def callback(msg):
            received.append(msg)

        bus.subscribe("test", "agent-1", callback)
        bus.unsubscribe("test", "agent-1")
        await bus.publish(topic="test", sender_id="agent-2", content="Hello")

        await asyncio.sleep(0.05)
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self):
        bus = self._make_bus()
        received = []

        async def callback(msg):
            received.append(msg)

        bus.subscribe("topic-a", "agent-1", callback)
        bus.subscribe("topic-b", "agent-1", callback)
        bus.unsubscribe_all("agent-1")

        await bus.publish(topic="topic-a", sender_id="agent-2", content="A")
        await bus.publish(topic="topic-b", sender_id="agent-2", content="B")

        await asyncio.sleep(0.05)
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_request_reply(self):
        bus = self._make_bus()

        async def responder(msg):
            await bus.publish(
                topic=msg.topic,
                sender_id="responder",
                content="Here is the answer",
                reply_to=msg.id,
            )

        bus.subscribe("questions", "responder", responder)

        reply = await bus.request(
            topic="questions",
            sender_id="asker",
            content="What is 2+2?",
            timeout=2.0,
        )
        assert reply is not None
        assert reply.content == "Here is the answer"

    @pytest.mark.asyncio
    async def test_request_timeout(self):
        bus = self._make_bus()
        reply = await bus.request(
            topic="no-listeners",
            sender_id="asker",
            content="Hello?",
            timeout=0.1,
        )
        assert reply is None

    @pytest.mark.asyncio
    async def test_history_limit(self):
        bus = self._make_bus()
        bus._history_limit = 5
        for i in range(10):
            await bus.publish(topic="flood", sender_id="agent", content=f"msg-{i}")

        messages = bus.peek("flood", limit=100)
        assert len(messages) == 5
        assert messages[0].content == "msg-5"

    @pytest.mark.asyncio
    async def test_peek_with_since(self):
        bus = self._make_bus()
        await bus.publish(topic="t", sender_id="a", content="old")
        cutoff = time.time()
        await asyncio.sleep(0.01)
        await bus.publish(topic="t", sender_id="a", content="new")

        messages = bus.peek("t", since=cutoff)
        assert len(messages) == 1
        assert messages[0].content == "new"

    @pytest.mark.asyncio
    async def test_topics_and_stats(self):
        bus = self._make_bus()
        await bus.publish(topic="alpha", sender_id="a", content="x")
        await bus.publish(topic="beta", sender_id="b", content="y")

        topics = bus.topics()
        assert "alpha" in topics
        assert "beta" in topics

        stats = bus.stats()
        assert stats["topics"] >= 2
        assert stats["total_messages"] >= 2

    @pytest.mark.asyncio
    async def test_priority_preserved(self):
        from cadence.core.message_bus import MessagePriority
        bus = self._make_bus()
        msg = await bus.publish(
            topic="urgent-topic",
            sender_id="agent",
            content="URGENT",
            priority=MessagePriority.URGENT,
        )
        assert msg.priority == MessagePriority.URGENT

    @pytest.mark.asyncio
    async def test_no_duplicate_subscriptions(self):
        bus = self._make_bus()
        received = []

        async def callback(msg):
            received.append(msg)

        bus.subscribe("t", "agent-1", callback)
        bus.subscribe("t", "agent-1", callback)  # duplicate
        await bus.publish(topic="t", sender_id="agent-2", content="hi")

        await asyncio.sleep(0.05)
        assert len(received) == 1  # Should only receive once


# ===========================================================================
# 2. Streaming Tests
# ===========================================================================


class TestStreaming:
    """Tests for cadence.core.streaming."""

    @pytest.mark.asyncio
    async def test_stream_collector_basic(self):
        from cadence.core.streaming import StreamCollector
        collector = StreamCollector()

        await collector.emit_token("Hello")
        await collector.emit_token(" World")
        await collector.emit_done(full_response="Hello World", session_id="s1")

        events = []
        async for event in collector:
            events.append(event)

        assert len(events) == 3
        assert events[0].event == "token"
        assert events[0].data["token"] == "Hello"
        assert events[1].event == "token"
        assert events[2].event == "done"
        assert events[2].data["response"] == "Hello World"

    @pytest.mark.asyncio
    async def test_stream_event_sse_format(self):
        from cadence.core.streaming import StreamEvent
        event = StreamEvent(event="token", data={"token": "hi"})
        sse = event.to_sse()
        assert sse.startswith("event: token\n")
        assert "data:" in sse
        assert sse.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_stream_collector_error(self):
        from cadence.core.streaming import StreamCollector
        collector = StreamCollector()

        await collector.emit_error("Something went wrong")

        events = []
        async for event in collector:
            events.append(event)

        assert len(events) == 1
        assert events[0].event == "error"
        assert "Something went wrong" in events[0].data["error"]

    @pytest.mark.asyncio
    async def test_stream_collector_all_event_types(self):
        from cadence.core.streaming import StreamCollector
        collector = StreamCollector()

        await collector.emit_thinking("Analyzing...", agent_id="a1")
        await collector.emit_tool_start("read_file", {"path": "/tmp/x"}, agent_id="a1")
        await collector.emit_tool_result("read_file", "file contents", True, agent_id="a1")
        await collector.emit_status("Phase 2: Execution", agent_id="a1")
        await collector.emit_done("Final answer", session_id="s1")

        events = []
        async for event in collector:
            events.append(event)

        event_types = [e.event for e in events]
        assert "thinking" in event_types
        assert "tool_start" in event_types
        assert "tool_result" in event_types
        assert "status" in event_types
        assert "done" in event_types


# ===========================================================================
# 3. Checkpoint Tests
# ===========================================================================


class TestCheckpoints:
    """Tests for cadence.core.checkpoint."""

    @pytest.mark.asyncio
    async def test_request_and_approve(self):
        from cadence.core.checkpoint import CheckpointManager, CheckpointStatus
        mgr = CheckpointManager()

        async def _approve_after_delay():
            await asyncio.sleep(0.05)
            pending = mgr.get_pending()
            assert len(pending) == 1
            mgr.resolve(pending[0].id, approved=True, response="Go ahead")

        asyncio.create_task(_approve_after_delay())

        approved, response = await mgr.request_approval(
            agent_id="coder-1",
            title="Delete file",
            description="About to delete important.txt",
            timeout=2.0,
        )
        assert approved is True
        assert response == "Go ahead"

    @pytest.mark.asyncio
    async def test_request_and_reject(self):
        from cadence.core.checkpoint import CheckpointManager
        mgr = CheckpointManager()

        async def _reject_after_delay():
            await asyncio.sleep(0.05)
            pending = mgr.get_pending()
            mgr.resolve(pending[0].id, approved=False, response="No way")

        asyncio.create_task(_reject_after_delay())

        approved, response = await mgr.request_approval(
            agent_id="coder-1",
            title="Drop table",
            description="DROP TABLE users",
            timeout=2.0,
        )
        assert approved is False
        assert response == "No way"

    @pytest.mark.asyncio
    async def test_checkpoint_timeout(self):
        from cadence.core.checkpoint import CheckpointManager
        mgr = CheckpointManager()

        approved, response = await mgr.request_approval(
            agent_id="coder-1",
            title="Timeout test",
            description="This should timeout",
            timeout=0.1,
        )
        assert approved is False
        assert "expired" in response.lower()

    def test_get_all_checkpoints(self):
        from cadence.core.checkpoint import CheckpointManager, Checkpoint, CheckpointStatus
        mgr = CheckpointManager()
        # Manually add some checkpoints
        cp1 = Checkpoint(
            agent_id="a1", title="Test 1", description="Desc 1",
            status=CheckpointStatus.APPROVED,
        )
        cp2 = Checkpoint(
            agent_id="a2", title="Test 2", description="Desc 2",
            status=CheckpointStatus.PENDING,
        )
        mgr._checkpoints[cp1.id] = cp1
        mgr._checkpoints[cp2.id] = cp2

        all_cps = mgr.get_all()
        assert len(all_cps) == 2

        pending = mgr.get_pending()
        assert len(pending) == 1
        assert pending[0].title == "Test 2"

    def test_resolve_nonexistent(self):
        from cadence.core.checkpoint import CheckpointManager
        mgr = CheckpointManager()
        result = mgr.resolve("nonexistent", approved=True)
        assert result is None


# ===========================================================================
# 4. Cross-Session Learning Tests
# ===========================================================================


class TestLearning:
    """Tests for cadence.learning.store."""

    def _make_store(self, tmp_path):
        from cadence.learning.store import LearningStore
        return LearningStore(db_path=str(tmp_path / "test_learning.db"))

    def test_record_and_get_stats(self, tmp_path):
        from cadence.learning.store import StrategyRecord, OutcomeRating
        store = self._make_store(tmp_path)

        store.record(StrategyRecord(
            session_id="s1",
            task_type="code_generation",
            task_description="Write a function",
            strategy="Direct implementation",
            tools_used=["write_file", "execute_code"],
            model_used="claude-sonnet",
            outcome=OutcomeRating.SUCCESS,
        ))
        store.record(StrategyRecord(
            session_id="s1",
            task_type="code_generation",
            task_description="Write a test",
            strategy="Test-driven approach",
            tools_used=["read_file", "write_file"],
            model_used="claude-sonnet",
            outcome=OutcomeRating.SUCCESS,
        ))

        stats = store.get_stats()
        assert stats["total_strategies"] == 2
        assert "code_generation" in stats["by_task_type"]
        assert stats["by_task_type"]["code_generation"]["successes"] == 2

    def test_get_insights(self, tmp_path):
        from cadence.learning.store import StrategyRecord, OutcomeRating
        store = self._make_store(tmp_path)

        for i in range(5):
            store.record(StrategyRecord(
                session_id=f"s{i}",
                task_type="debugging",
                task_description=f"Debug issue {i}",
                strategy="Read logs then fix",
                tools_used=["read_file", "search_files"],
                model_used="claude-sonnet",
                outcome=OutcomeRating.SUCCESS if i < 4 else OutcomeRating.FAILURE,
            ))

        insights = store.get_insights("debugging")
        assert len(insights) > 0
        assert insights[0].avg_success_rate > 0.5

    def test_get_insights_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        insights = store.get_insights("nonexistent")
        assert insights == []

    def test_classify_task(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.classify_task("Debug the authentication bug") == "debugging"
        assert store.classify_task("Write a new API endpoint") == "code_generation"
        assert store.classify_task("Refactor the database module") == "refactoring"
        assert store.classify_task("Research best practices") == "research"
        assert store.classify_task("Something random") == "general"

    def test_get_best_tools(self, tmp_path):
        from cadence.learning.store import StrategyRecord, OutcomeRating
        store = self._make_store(tmp_path)

        store.record(StrategyRecord(
            session_id="s1",
            task_type="testing",
            task_description="Write tests",
            strategy="Direct",
            tools_used=["execute_code", "write_file"],
            outcome=OutcomeRating.SUCCESS,
        ))
        store.record(StrategyRecord(
            session_id="s2",
            task_type="testing",
            task_description="Run tests",
            strategy="Direct",
            tools_used=["execute_code", "shell"],
            outcome=OutcomeRating.FAILURE,
        ))

        ranked = store.get_best_tools("testing")
        assert len(ranked) > 0
        # execute_code appears in both (1 success, 1 failure)
        tool_names = [t[0] for t in ranked]
        assert "execute_code" in tool_names


# ===========================================================================
# 5. Knowledge Graph Tests
# ===========================================================================


class TestKnowledgeGraph:
    """Tests for cadence.knowledge.graph."""

    def _make_graph(self, tmp_path):
        from cadence.knowledge.graph import KnowledgeGraph
        return KnowledgeGraph(persist_path=str(tmp_path / "test_graph.json"))

    def test_add_entity(self, tmp_path):
        graph = self._make_graph(tmp_path)
        entity = graph.add_entity("UserService", "class", {"module": "auth"})
        assert entity.name == "UserService"
        assert entity.entity_type == "class"
        assert entity.properties["module"] == "auth"

    def test_add_duplicate_entity_updates(self, tmp_path):
        graph = self._make_graph(tmp_path)
        e1 = graph.add_entity("Foo", "class")
        e2 = graph.add_entity("Foo", "class", {"new_prop": True})
        assert e1.id == e2.id  # Same entity
        assert e2.properties.get("new_prop") is True

    def test_add_relationship(self, tmp_path):
        graph = self._make_graph(tmp_path)
        e1 = graph.add_entity("A", "module")
        e2 = graph.add_entity("B", "module")
        rel = graph.add_relationship(e1.id, e2.id, "imports")
        assert rel is not None
        assert rel.relation_type == "imports"

    def test_relationship_invalid_entities(self, tmp_path):
        graph = self._make_graph(tmp_path)
        rel = graph.add_relationship("fake1", "fake2", "calls")
        assert rel is None

    def test_find_entities(self, tmp_path):
        graph = self._make_graph(tmp_path)
        graph.add_entity("UserService", "class")
        graph.add_entity("UserModel", "class")
        graph.add_entity("auth_handler", "function")

        results = graph.find_entities(name="User")
        assert len(results) == 2

        results = graph.find_entities(entity_type="function")
        assert len(results) == 1
        assert results[0].name == "auth_handler"

    def test_get_neighbors(self, tmp_path):
        graph = self._make_graph(tmp_path)
        a = graph.add_entity("A", "module")
        b = graph.add_entity("B", "module")
        c = graph.add_entity("C", "module")
        graph.add_relationship(a.id, b.id, "imports")
        graph.add_relationship(a.id, c.id, "imports")

        result = graph.get_neighbors(a.id, direction="outgoing")
        assert len(result.entities) == 2

        result = graph.get_neighbors(b.id, direction="incoming")
        assert len(result.entities) == 1
        assert result.entities[0].name == "A"

    def test_find_path(self, tmp_path):
        graph = self._make_graph(tmp_path)
        a = graph.add_entity("A", "module")
        b = graph.add_entity("B", "module")
        c = graph.add_entity("C", "module")
        graph.add_relationship(a.id, b.id, "calls")
        graph.add_relationship(b.id, c.id, "calls")

        path = graph.find_path(a.id, c.id)
        assert path is not None
        assert len(path) == 3
        assert path[0] == a.id
        assert path[2] == c.id

    def test_find_path_no_connection(self, tmp_path):
        graph = self._make_graph(tmp_path)
        a = graph.add_entity("A", "module")
        b = graph.add_entity("B", "module")
        # No relationship
        path = graph.find_path(a.id, b.id)
        assert path is None

    def test_get_subgraph(self, tmp_path):
        graph = self._make_graph(tmp_path)
        a = graph.add_entity("A", "module")
        b = graph.add_entity("B", "module")
        c = graph.add_entity("C", "module")
        d = graph.add_entity("D", "module")
        graph.add_relationship(a.id, b.id, "calls")
        graph.add_relationship(b.id, c.id, "calls")
        graph.add_relationship(c.id, d.id, "calls")

        result = graph.get_subgraph(a.id, depth=1)
        entity_names = {e.name for e in result.entities}
        assert "A" in entity_names
        assert "B" in entity_names

    def test_delete_entity(self, tmp_path):
        graph = self._make_graph(tmp_path)
        e = graph.add_entity("ToDelete", "class")
        assert graph.delete_entity(e.id) is True
        assert graph.get_entity(e.id) is None
        assert graph.delete_entity("nonexistent") is False

    def test_delete_relationship(self, tmp_path):
        graph = self._make_graph(tmp_path)
        a = graph.add_entity("A", "module")
        b = graph.add_entity("B", "module")
        rel = graph.add_relationship(a.id, b.id, "calls")
        assert graph.delete_relationship(rel.id) is True
        assert graph.delete_relationship("nonexistent") is False

    def test_stats(self, tmp_path):
        graph = self._make_graph(tmp_path)
        graph.add_entity("A", "class")
        graph.add_entity("B", "function")
        a = graph.find_entities(name="A")[0]
        b = graph.find_entities(name="B")[0]
        graph.add_relationship(a.id, b.id, "contains")

        stats = graph.stats()
        assert stats["total_entities"] == 2
        assert stats["total_relationships"] == 1
        assert "class" in stats["entity_types"]

    def test_persistence(self, tmp_path):
        from cadence.knowledge.graph import KnowledgeGraph
        path = str(tmp_path / "persist_test.json")

        g1 = KnowledgeGraph(persist_path=path)
        e = g1.add_entity("Persistent", "class")
        eid = e.id

        # Create new instance, should load from disk
        g2 = KnowledgeGraph(persist_path=path)
        loaded = g2.get_entity(eid)
        assert loaded is not None
        assert loaded.name == "Persistent"


# ===========================================================================
# 6. Multi-Modal Input Tests
# ===========================================================================


class TestMultiModal:
    """Tests for cadence.core.multimodal."""

    def test_supports_vision(self):
        from cadence.core.multimodal import supports_vision
        assert supports_vision("claude-3-opus-20240229") is True
        assert supports_vision("claude-sonnet-4-5-20250514") is True
        assert supports_vision("gpt-4o") is True
        assert supports_vision("gemini-pro") is True
        assert supports_vision("bedrock/converse/us.anthropic.claude-sonnet-4-5-20250514-v1:0") is True
        assert supports_vision("text-davinci-003") is False

    def test_image_input_from_base64(self):
        import base64
        from cadence.core.multimodal import ImageInput
        # Create a tiny test image (1x1 PNG)
        png_data = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100).decode()
        img = ImageInput.from_base64(png_data, media_type="image/png")
        assert img.media_type == "image/png"
        assert img.source == "base64"
        assert len(img.data) > 0

    def test_image_input_from_data_url(self):
        import base64
        from cadence.core.multimodal import ImageInput
        raw = base64.b64encode(b"fake-image-data").decode()
        data_url = f"data:image/jpeg;base64,{raw}"
        img = ImageInput.from_base64(data_url)
        assert img.media_type == "image/jpeg"

    def test_image_input_from_url(self):
        from cadence.core.multimodal import ImageInput
        img = ImageInput.from_url("https://example.com/photo.jpg")
        assert img.media_type == "image/url"
        assert img.source == "https://example.com/photo.jpg"

    def test_content_block_base64(self):
        import base64
        from cadence.core.multimodal import ImageInput
        img = ImageInput(
            data=b"fake-data",
            media_type="image/png",
        )
        block = img.to_content_block()
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"

    def test_content_block_url(self):
        from cadence.core.multimodal import ImageInput
        img = ImageInput.from_url("https://example.com/img.png")
        block = img.to_content_block()
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/img.png"

    def test_build_multimodal_content_text_only(self):
        from cadence.core.multimodal import build_multimodal_content
        result = build_multimodal_content("Hello world")
        assert result == "Hello world"  # Plain string, backward compatible

    def test_build_multimodal_content_with_images(self):
        from cadence.core.multimodal import build_multimodal_content, ImageInput
        img = ImageInput(data=b"data", media_type="image/png")
        result = build_multimodal_content("Describe this image", images=[img])
        assert isinstance(result, list)
        assert len(result) == 2  # image block + text block
        assert result[0]["type"] == "image"
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "Describe this image"

    def test_image_input_file_not_found(self):
        from cadence.core.multimodal import ImageInput
        with pytest.raises(FileNotFoundError):
            ImageInput.from_file("/nonexistent/image.png")


# ===========================================================================
# Integration: Config Tests
# ===========================================================================


class TestNewConfig:
    """Tests for the new configuration sections."""

    def test_config_has_new_sections(self):
        from cadence.core.config import Config
        config = Config()
        assert config.message_bus.enabled is True
        assert config.message_bus.history_limit == 100
        assert config.checkpoints.enabled is True
        assert config.checkpoints.default_timeout == 300.0
        assert config.learning.enabled is True
        assert config.knowledge_graph.enabled is True
        assert config.multimodal.enabled is True
        assert config.multimodal.max_images_per_message == 5

    def test_config_defaults_can_be_overridden(self):
        from cadence.core.config import Config
        config = Config(
            message_bus={"enabled": False},
            checkpoints={"default_timeout": 60.0},
        )
        assert config.message_bus.enabled is False
        assert config.checkpoints.default_timeout == 60.0


# ===========================================================================
# Integration: Message type with content_blocks
# ===========================================================================


class TestMessageContentBlocks:
    """Tests for multi-modal Message content_blocks."""

    def test_message_with_content_blocks(self):
        from cadence.core.types import Message, Role
        msg = Message(
            role=Role.USER,
            content="Describe this image",
            content_blocks=[
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                {"type": "text", "text": "Describe this image"},
            ],
        )
        assert msg.content_blocks is not None
        assert len(msg.content_blocks) == 2

    def test_message_without_content_blocks(self):
        from cadence.core.types import Message, Role
        msg = Message(role=Role.USER, content="Plain text")
        assert msg.content_blocks is None
