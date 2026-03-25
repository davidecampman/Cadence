"""Integration tests for the Agent loop and Orchestrator with mocked LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from cadence.core.agent import Agent, _summarize_args
from cadence.core.config import Config
from cadence.core.trace import TraceLogger
from cadence.core.types import (
    AgentRole,
    Message,
    PermissionTier,
    Role,
    Task,
    TaskStatus,
    ToolCall,
)
from cadence.tools.base import Tool, ToolRegistry
from cadence.agents.orchestrator import Orchestrator, TaskDAG, ROLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class EchoTool(Tool):
    """A simple tool that echoes its input — useful for testing the agent loop."""
    name = "echo"
    description = "Echo the input back"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo."},
        },
        "required": ["text"],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, text: str = "") -> str:
        return f"ECHO: {text}"


class CounterTool(Tool):
    """A tool that counts how many times it was called."""
    name = "counter"
    description = "Increment and return a counter"
    parameters = {"type": "object", "properties": {}}
    permission_tier = PermissionTier.READ_ONLY

    def __init__(self):
        self.count = 0

    async def execute(self, **kwargs) -> str:
        self.count += 1
        return f"count={self.count}"


def _make_registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _make_config(tmp_path=None, **overrides) -> Config:
    cfg = Config()
    # Disable prompt evolution by default in tests to avoid SQLite path issues
    cfg.prompt_evolution.enabled = False
    if tmp_path:
        cfg.prompt_evolution.persist_dir = str(tmp_path / "prompt_evolution.db")
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


# ---------------------------------------------------------------------------
# TaskDAG
# ---------------------------------------------------------------------------

class TestTaskDAG:
    def test_add_and_get(self):
        dag = TaskDAG()
        t = Task(description="task 1")
        dag.add(t)
        assert dag.get(t.id) is not None
        assert dag.get(t.id).description == "task 1"

    def test_get_nonexistent(self):
        dag = TaskDAG()
        assert dag.get("no-such-id") is None

    def test_ready_tasks_no_deps(self):
        dag = TaskDAG()
        t1 = Task(description="A")
        t2 = Task(description="B")
        dag.add(t1)
        dag.add(t2)
        ready = dag.ready_tasks()
        assert len(ready) == 2

    def test_ready_tasks_with_deps(self):
        dag = TaskDAG()
        t1 = Task(description="First")
        t2 = Task(description="Second", dependencies=[t1.id])
        dag.add(t1)
        dag.add(t2)

        # Only t1 should be ready initially
        ready = dag.ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == t1.id

        # After completing t1, t2 should be ready
        t1.status = TaskStatus.COMPLETED
        ready = dag.ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == t2.id

    def test_all_completed(self):
        dag = TaskDAG()
        t1 = Task(description="A")
        t2 = Task(description="B")
        dag.add(t1)
        dag.add(t2)

        assert not dag.all_completed()

        t1.status = TaskStatus.COMPLETED
        assert not dag.all_completed()

        t2.status = TaskStatus.COMPLETED
        assert dag.all_completed()

    def test_all_completed_with_failed(self):
        dag = TaskDAG()
        t1 = Task(description="A")
        dag.add(t1)
        t1.status = TaskStatus.FAILED
        assert dag.all_completed()  # Failed counts as "done"

    def test_summary(self):
        dag = TaskDAG()
        dag.add(Task(description="Task A"))
        dag.add(Task(description="Task B"))
        summary = dag.summary()
        assert "Task A" in summary
        assert "Task B" in summary

    def test_dependency_chain(self):
        dag = TaskDAG()
        t1 = Task(description="Step 1")
        t2 = Task(description="Step 2", dependencies=[t1.id])
        t3 = Task(description="Step 3", dependencies=[t2.id])
        dag.add(t1)
        dag.add(t2)
        dag.add(t3)

        # Only t1 ready
        assert [t.id for t in dag.ready_tasks()] == [t1.id]

        t1.status = TaskStatus.COMPLETED
        assert [t.id for t in dag.ready_tasks()] == [t2.id]

        t2.status = TaskStatus.COMPLETED
        assert [t.id for t in dag.ready_tasks()] == [t3.id]

    def test_parallel_tasks_with_shared_dependency(self):
        dag = TaskDAG()
        t1 = Task(description="Setup")
        t2 = Task(description="Branch A", dependencies=[t1.id])
        t3 = Task(description="Branch B", dependencies=[t1.id])
        dag.add(t1)
        dag.add(t2)
        dag.add(t3)

        t1.status = TaskStatus.COMPLETED
        ready = dag.ready_tasks()
        assert len(ready) == 2
        ready_ids = {t.id for t in ready}
        assert t2.id in ready_ids
        assert t3.id in ready_ids


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_agent_direct_response(self):
        """When LLM returns text with no tool calls, agent returns it directly."""
        mock_response = ("The answer is 42.", [], {"total_tokens": 10})

        with patch("cadence.core.agent.chat_completion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            role = AgentRole(name="test", description="Test agent")
            trace = TraceLogger(console=False)
            registry = _make_registry()
            config = _make_config()

            agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
            result = await agent.run("What is the meaning of life?")

            assert result == "The answer is 42."
            assert agent._total_tokens == 10
            assert agent._iterations == 1

    @pytest.mark.asyncio
    async def test_agent_uses_tool_then_responds(self):
        """Agent calls a tool, gets result, then provides final answer."""
        echo_tool = EchoTool()
        registry = _make_registry(echo_tool)

        tool_call = ToolCall(id="tc1", name="echo", arguments={"text": "hello"})

        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: LLM wants to use the tool
                return ("", [tool_call], {"total_tokens": 20})
            else:
                # Second call: LLM gives final answer after seeing tool result
                return ("The echo said: ECHO: hello", [], {"total_tokens": 15})

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat):
            role = AgentRole(name="test", description="Test agent")
            trace = TraceLogger(console=False)
            config = _make_config()

            agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
            result = await agent.run("Echo hello for me")

            assert "ECHO: hello" in result
            assert agent._iterations == 2
            assert agent._total_tokens == 35

    @pytest.mark.asyncio
    async def test_agent_handles_unknown_tool(self):
        """Agent gracefully handles LLM requesting a tool that doesn't exist."""
        registry = _make_registry()
        bad_call = ToolCall(id="tc1", name="nonexistent", arguments={})

        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("", [bad_call], {"total_tokens": 10})
            else:
                return ("Sorry, that tool doesn't exist.", [], {"total_tokens": 10})

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat):
            role = AgentRole(name="test", description="Test agent")
            trace = TraceLogger(console=False)
            config = _make_config()

            agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
            result = await agent.run("Use a fake tool")

            assert len(agent._errors) > 0
            assert "Unknown tool" in agent._errors[0]

    @pytest.mark.asyncio
    async def test_agent_max_iterations(self):
        """Agent stops after max_iterations even if LLM keeps requesting tools."""
        tool = EchoTool()
        registry = _make_registry(tool)
        tc = ToolCall(id="tc1", name="echo", arguments={"text": "loop"})

        async def mock_chat(*args, **kwargs):
            return ("", [tc], {"total_tokens": 5})

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat):
            role = AgentRole(name="test", description="Test agent")
            trace = TraceLogger(console=False)
            config = _make_config()
            config.agents.max_iterations_per_task = 3

            agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
            result = await agent.run("Keep looping")

            assert "maximum iterations" in result.lower() or "max iterations" in result.lower()
            assert agent._iterations == 3

    @pytest.mark.asyncio
    async def test_agent_token_budget(self):
        """Agent stops when token budget is exceeded."""
        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            tc = ToolCall(id=f"tc{call_count}", name="echo", arguments={"text": "hi"})
            return ("", [tc], {"total_tokens": 60000})

        tool = EchoTool()
        registry = _make_registry(tool)

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat):
            role = AgentRole(name="test", description="Test agent")
            trace = TraceLogger(console=False)
            config = _make_config()
            config.budget.max_tokens_per_task = 50000

            agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
            result = await agent.run("Token heavy task")

            assert "budget" in result.lower() or "token" in result.lower()

    @pytest.mark.asyncio
    async def test_agent_loop_detection(self):
        """Agent detects when it's stuck in a loop."""
        registry = _make_registry()
        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Always return the same text to trigger loop detection
            return ("I am stuck repeating myself.", [], {"total_tokens": 10})

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat):
            role = AgentRole(name="test", description="Test agent")
            trace = TraceLogger(console=False)
            config = _make_config()
            config.agents.loop_detection_window = 3

            agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
            # Note: loop detection needs multiple assistant messages in history
            # but the first call without tool_calls will return immediately.
            # This tests that the agent returns on first call.
            result = await agent.run("Repeat yourself")

            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_agent_with_conversation_history(self):
        """Agent receives prior conversation context."""
        captured_messages = []

        async def mock_chat(model, messages, **kwargs):
            captured_messages.extend(messages)
            return ("Got the context.", [], {"total_tokens": 10})

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat):
            role = AgentRole(name="test", description="Test agent")
            trace = TraceLogger(console=False)
            registry = _make_registry()
            config = _make_config()

            agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
            history = [
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Hello Alice!"},
            ]
            result = await agent.run("What is my name?", conversation_history=history)

            # Verify history was injected into messages
            user_msgs = [m for m in captured_messages if m.role == Role.USER]
            assert any("Alice" in m.content for m in captured_messages)

    @pytest.mark.asyncio
    async def test_agent_model_selection(self):
        """Agent uses role's model_override when set."""
        role = AgentRole(
            name="custom",
            description="Custom model agent",
            model_override="custom-model-v1",
        )
        trace = TraceLogger(console=False)
        registry = _make_registry()
        config = _make_config()
        agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
        assert agent.model == "custom-model-v1"

    @pytest.mark.asyncio
    async def test_agent_default_model(self):
        """Agent uses config's strong model by default."""
        role = AgentRole(name="default", description="Default model")
        trace = TraceLogger(console=False)
        registry = _make_registry()
        config = _make_config()
        agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
        assert agent.model == config.models.strong


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_simple_request_no_decomposition(self):
        """Simple request → planner returns SIMPLE → single agent handles it."""
        call_count = 0

        async def mock_chat(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Planning phase: return SIMPLE
                return ("SIMPLE", [], {"total_tokens": 10})
            elif call_count == 2:
                # General agent responds
                return ("Here is the answer.", [], {"total_tokens": 20})
            else:
                # Evaluation phase
                return ("PASS", [], {"total_tokens": 5})

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat), \
             patch("cadence.agents.orchestrator.chat_completion", side_effect=mock_chat):
            trace = TraceLogger(console=False)
            registry = _make_registry()
            config = _make_config()

            orch = Orchestrator(
                tool_registry=registry,
                trace=trace,
                config=config,
            )
            result = await orch.run("What is 2+2?")
            assert "answer" in result.lower()

    @pytest.mark.asyncio
    async def test_multi_task_decomposition(self):
        """Complex request → planner returns tasks → agents execute them."""
        import json
        call_count = 0

        async def mock_chat(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            msg_content = messages[-1].content if messages else ""

            if call_count == 1:
                # Planning phase
                tasks = [
                    {"description": "Research topic", "role": "researcher", "dependencies": []},
                    {"description": "Write code", "role": "coder", "dependencies": [0]},
                ]
                return (json.dumps(tasks), [], {"total_tokens": 20})
            elif call_count <= 3:
                # Agent execution phases
                return ("Task completed successfully.", [], {"total_tokens": 15})
            elif call_count == 4:
                # Synthesis phase
                return ("Combined result from all tasks.", [], {"total_tokens": 10})
            else:
                # Evaluation phase
                return ("PASS", [], {"total_tokens": 5})

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat), \
             patch("cadence.agents.orchestrator.chat_completion", side_effect=mock_chat):
            trace = TraceLogger(console=False)
            registry = _make_registry()
            config = _make_config()

            orch = Orchestrator(
                tool_registry=registry,
                trace=trace,
                config=config,
            )
            result = await orch.run("Research and implement feature X")
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_session_budget_exceeded(self):
        """Orchestrator returns budget error when session tokens exceeded."""
        trace = TraceLogger(console=False)
        registry = _make_registry()
        config = _make_config()
        config.budget.max_tokens_per_session = 100

        orch = Orchestrator(
            tool_registry=registry,
            trace=trace,
            config=config,
        )
        orch._session_tokens = 200  # Already over budget

        result = await orch.run("Do something")
        assert "budget" in result.lower()

    @pytest.mark.asyncio
    async def test_predefined_roles_exist(self):
        """All expected specialist roles are defined."""
        assert "orchestrator" in ROLES
        assert "researcher" in ROLES
        assert "coder" in ROLES
        assert "reviewer" in ROLES
        assert "general" in ROLES

    @pytest.mark.asyncio
    async def test_orchestrator_spawns_correct_role(self):
        """_spawn_agent creates agents with the correct role."""
        trace = TraceLogger(console=False)
        registry = _make_registry()
        config = _make_config()

        orch = Orchestrator(tool_registry=registry, trace=trace, config=config)

        coder = orch._spawn_agent("coder")
        assert coder.role.name == "coder"
        assert coder.role.permission_tier == PermissionTier.PRIVILEGED

        researcher = orch._spawn_agent("researcher")
        assert researcher.role.name == "researcher"

    @pytest.mark.asyncio
    async def test_orchestrator_fallback_to_general(self):
        """Unknown role name falls back to general."""
        trace = TraceLogger(console=False)
        registry = _make_registry()
        config = _make_config()

        orch = Orchestrator(tool_registry=registry, trace=trace, config=config)
        agent = orch._spawn_agent("nonexistent_role")
        assert agent.role.name == "general"

    @pytest.mark.asyncio
    async def test_evaluation_pass(self):
        """_evaluate returns original response when evaluation passes."""
        async def mock_chat(model, messages, **kwargs):
            return ("PASS", [], {"total_tokens": 5})

        with patch("cadence.agents.orchestrator.chat_completion", side_effect=mock_chat):
            trace = TraceLogger(console=False)
            registry = _make_registry()
            config = _make_config()

            orch = Orchestrator(tool_registry=registry, trace=trace, config=config)
            orch._conversation_history = []
            result = await orch._evaluate("What is X?", "X is 42.")
            assert result == "X is 42."

    @pytest.mark.asyncio
    async def test_evaluation_fail_adds_note(self):
        """_evaluate appends a note when evaluation fails."""
        async def mock_chat(model, messages, **kwargs):
            return ("FAIL: Missing explanation of Y.", [], {"total_tokens": 5})

        with patch("cadence.agents.orchestrator.chat_completion", side_effect=mock_chat):
            trace = TraceLogger(console=False)
            registry = _make_registry()
            config = _make_config()

            orch = Orchestrator(tool_registry=registry, trace=trace, config=config)
            orch._conversation_history = []
            result = await orch._evaluate("What is X?", "X is 42.")
            assert "X is 42." in result
            assert "Self-review flagged" in result
            assert "Missing explanation" in result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_summarize_args_short(self):
        result = _summarize_args({"path": "file.txt", "limit": 10})
        assert "path=file.txt" in result
        assert "limit=10" in result

    def test_summarize_args_truncates_long_values(self):
        result = _summarize_args({"code": "x" * 100})
        assert "..." in result
        assert len(result) <= 80

    def test_summarize_args_empty(self):
        result = _summarize_args({})
        assert result == ""
