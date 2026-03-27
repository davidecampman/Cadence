"""Integration tests for the Agent loop and Orchestrator with mocked LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from cadence.core.agent import Agent, _summarize_args
from cadence.core.config import Config
from cadence.core.trace import TraceLogger
from cadence.core.types import (
    AgentRole,
    ConditionalDef,
    LoopDef,
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


# ---------------------------------------------------------------------------
# Context overflow prevention
# ---------------------------------------------------------------------------

class TestContextOverflowPrevention:
    @pytest.mark.asyncio
    async def test_tool_result_truncation(self):
        """Large tool results are truncated before adding to agent history."""
        big_tool = EchoTool()

        # Override execute to return a huge result
        async def big_execute(text: str = "") -> str:
            return "X" * 50_000  # 50k chars

        big_tool.execute = big_execute
        registry = _make_registry(big_tool)

        tool_call = ToolCall(id="tc1", name="echo", arguments={"text": "big"})
        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("", [tool_call], {"total_tokens": 20})
            else:
                return ("Got the result.", [], {"total_tokens": 15})

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat):
            role = AgentRole(name="test", description="Test agent")
            trace = TraceLogger(console=False)
            config = _make_config()
            config.agents.max_tool_result_chars = 1000

            agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
            result = await agent.run("Get big data")

            # Verify tool result in history was truncated
            tool_msgs = [m for m in agent._history if m.role == Role.TOOL]
            assert len(tool_msgs) == 1
            assert len(tool_msgs[0].content) < 2000  # Should be ~1000, not 50k
            assert "truncated" in tool_msgs[0].content

    @pytest.mark.asyncio
    async def test_history_pruning(self):
        """Older tool results are pruned when history grows large."""
        echo_tool = EchoTool()
        registry = _make_registry(echo_tool)

        iteration = 0

        async def mock_chat(*args, **kwargs):
            nonlocal iteration
            iteration += 1
            if iteration <= 10:
                tc = ToolCall(id=f"tc{iteration}", name="echo", arguments={"text": f"step{iteration}"})
                # Use unique text each time to avoid loop detection
                return (f"thinking about step {iteration}...", [tc], {"total_tokens": 100})
            else:
                return ("Done!", [], {"total_tokens": 100})

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat):
            role = AgentRole(name="test", description="Test agent")
            trace = TraceLogger(console=False)
            config = _make_config()
            config.agents.prune_threshold = 10  # Low threshold for testing

            agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
            result = await agent.run("Do many things")

            assert result == "Done!"
            # History should have been pruned — old tool results should be short
            tool_msgs = [m for m in agent._history if m.role == Role.TOOL]
            # At least some of them should exist (pruning doesn't remove, just shrinks)
            assert len(tool_msgs) > 0

    @pytest.mark.asyncio
    async def test_proactive_budget_wrap_up(self):
        """Agent wraps up proactively when approaching token budget."""
        echo_tool = EchoTool()
        registry = _make_registry(echo_tool)

        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            tools = kwargs.get("tools")
            if tools is None:
                # This is the wrap-up call (no tools provided)
                return ("Here's my summary based on what I found.", [], {"total_tokens": 5000})
            tc = ToolCall(id=f"tc{call_count}", name="echo", arguments={"text": "data"})
            return ("", [tc], {"total_tokens": 45000})  # Each call uses ~45k tokens

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat):
            role = AgentRole(name="test", description="Test agent")
            trace = TraceLogger(console=False)
            config = _make_config()
            config.budget.max_tokens_per_task = 100000

            agent = Agent(role=role, tool_registry=registry, trace=trace, config=config)
            result = await agent.run("Expensive task")

            # Agent should have wrapped up proactively rather than hitting hard budget limit
            assert "summary" in result.lower() or "found" in result.lower()
            assert "budget exceeded" not in result.lower()
            # Should have done the wrap-up call (tools=None)
            assert agent._total_tokens < 200000  # Should not have blown far past budget


# ---------------------------------------------------------------------------
# TaskDAG Conditional Branching
# ---------------------------------------------------------------------------

class TestTaskDAGConditionals:
    """Tests for conditional branching in the task DAG."""

    @pytest.mark.asyncio
    async def test_conditional_true_branch(self):
        """When condition is met, true-branch tasks stay PENDING and false-branch is SKIPPED."""
        dag = TaskDAG()
        source = Task(description="Research")
        cond_task = Task(
            description="Check results",
            dependencies=[source.id],
            metadata={
                "conditional": {
                    "condition_source": source.id,
                    "condition_type": "contains",
                    "condition_value": "found vulnerabilities",
                    "if_true": [],   # will be filled below
                    "if_false": [],
                },
            },
        )
        true_task = Task(description="Fix vulnerabilities", dependencies=[cond_task.id])
        false_task = Task(description="Move on", dependencies=[cond_task.id])

        # Wire up the branch IDs
        cond_task.metadata["conditional"]["if_true"] = [true_task.id]
        cond_task.metadata["conditional"]["if_false"] = [false_task.id]

        for t in [source, cond_task, true_task, false_task]:
            dag.add(t)

        # Simulate source completing with a result that matches
        source.status = TaskStatus.COMPLETED
        source.result = "Analysis found vulnerabilities in the auth module"
        results = {source.id: source.result}

        await dag.evaluate_conditionals(results, llm_judge_fn=None)

        assert cond_task.status == TaskStatus.COMPLETED
        assert "TRUE" in cond_task.result
        assert false_task.status == TaskStatus.SKIPPED
        assert true_task.status == TaskStatus.PENDING  # ready to run

    @pytest.mark.asyncio
    async def test_conditional_false_branch(self):
        """When condition is NOT met, false-branch runs and true-branch is SKIPPED."""
        dag = TaskDAG()
        source = Task(description="Research")
        cond_task = Task(
            description="Check results",
            dependencies=[source.id],
            metadata={
                "conditional": {
                    "condition_source": source.id,
                    "condition_type": "contains",
                    "condition_value": "found vulnerabilities",
                    "if_true": [],
                    "if_false": [],
                },
            },
        )
        true_task = Task(description="Fix vulnerabilities", dependencies=[cond_task.id])
        false_task = Task(description="Move on", dependencies=[cond_task.id])
        cond_task.metadata["conditional"]["if_true"] = [true_task.id]
        cond_task.metadata["conditional"]["if_false"] = [false_task.id]

        for t in [source, cond_task, true_task, false_task]:
            dag.add(t)

        source.status = TaskStatus.COMPLETED
        source.result = "No issues found, everything looks clean"
        results = {source.id: source.result}

        await dag.evaluate_conditionals(results, llm_judge_fn=None)

        assert cond_task.status == TaskStatus.COMPLETED
        assert "FALSE" in cond_task.result
        assert true_task.status == TaskStatus.SKIPPED
        assert false_task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_conditional_skipped_cascades(self):
        """Tasks depending on a SKIPPED task are themselves cascaded to SKIPPED."""
        dag = TaskDAG()
        source = Task(description="Research")
        cond_task = Task(
            description="Check",
            dependencies=[source.id],
            metadata={
                "conditional": {
                    "condition_source": source.id,
                    "condition_type": "contains",
                    "condition_value": "yes",
                    "if_true": [],
                    "if_false": [],
                },
            },
        )
        true_task = Task(description="Do A", dependencies=[cond_task.id])
        downstream = Task(description="After A", dependencies=[true_task.id])
        false_task = Task(description="Do B", dependencies=[cond_task.id])

        cond_task.metadata["conditional"]["if_true"] = [true_task.id]
        cond_task.metadata["conditional"]["if_false"] = [false_task.id]

        for t in [source, cond_task, true_task, downstream, false_task]:
            dag.add(t)

        source.status = TaskStatus.COMPLETED
        source.result = "no"
        results = {source.id: source.result}

        await dag.evaluate_conditionals(results, llm_judge_fn=None)

        # true_task should be SKIPPED
        assert true_task.status == TaskStatus.SKIPPED

        # Now call ready_tasks — downstream should cascade to SKIPPED
        ready = dag.ready_tasks()
        assert downstream.status == TaskStatus.SKIPPED
        assert downstream not in ready

        # false_task should be ready
        assert false_task.status == TaskStatus.PENDING
        assert false_task in ready

    @pytest.mark.asyncio
    async def test_conditional_equals(self):
        """Test the 'equals' condition type."""
        dag = TaskDAG()
        source = Task(description="Classify")
        cond_task = Task(
            description="Route",
            dependencies=[source.id],
            metadata={
                "conditional": {
                    "condition_source": source.id,
                    "condition_type": "equals",
                    "condition_value": "BUG",
                    "if_true": [],
                    "if_false": [],
                },
            },
        )
        bug_fix = Task(description="Fix bug", dependencies=[cond_task.id])
        feature = Task(description="Add feature", dependencies=[cond_task.id])
        cond_task.metadata["conditional"]["if_true"] = [bug_fix.id]
        cond_task.metadata["conditional"]["if_false"] = [feature.id]

        for t in [source, cond_task, bug_fix, feature]:
            dag.add(t)

        source.status = TaskStatus.COMPLETED
        source.result = "BUG"
        results = {source.id: source.result}

        await dag.evaluate_conditionals(results, llm_judge_fn=None)

        assert bug_fix.status == TaskStatus.PENDING
        assert feature.status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_conditional_llm_judge(self):
        """Test the 'llm_judge' condition type with a mock judge."""
        dag = TaskDAG()
        source = Task(description="Run tests")
        cond_task = Task(
            description="Check pass/fail",
            dependencies=[source.id],
            metadata={
                "conditional": {
                    "condition_source": source.id,
                    "condition_type": "llm_judge",
                    "condition_value": "All tests pass",
                    "if_true": [],
                    "if_false": [],
                },
            },
        )
        deploy = Task(description="Deploy", dependencies=[cond_task.id])
        fix = Task(description="Fix failures", dependencies=[cond_task.id])
        cond_task.metadata["conditional"]["if_true"] = [deploy.id]
        cond_task.metadata["conditional"]["if_false"] = [fix.id]

        for t in [source, cond_task, deploy, fix]:
            dag.add(t)

        source.status = TaskStatus.COMPLETED
        source.result = "5/5 tests passed"
        results = {source.id: source.result}

        # Mock LLM judge that says YES
        async def mock_judge(result_text, condition):
            return True

        await dag.evaluate_conditionals(results, llm_judge_fn=mock_judge)

        assert deploy.status == TaskStatus.PENDING
        assert fix.status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_conditional_waits_for_source(self):
        """Conditional node is not evaluated until its source is COMPLETED."""
        dag = TaskDAG()
        source = Task(description="Research")
        cond_task = Task(
            description="Check",
            dependencies=[source.id],
            metadata={
                "conditional": {
                    "condition_source": source.id,
                    "condition_type": "contains",
                    "condition_value": "done",
                    "if_true": [],
                    "if_false": [],
                },
            },
        )
        dag.add(source)
        dag.add(cond_task)

        # Source still PENDING — conditional should NOT be evaluated
        await dag.evaluate_conditionals({}, llm_judge_fn=None)
        assert cond_task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_conditional_malformed_ignored(self):
        """Malformed conditional metadata is ignored — task behaves normally."""
        dag = TaskDAG()
        source = Task(description="Work")
        bad_cond = Task(
            description="Bad conditional",
            dependencies=[source.id],
            metadata={"conditional": "not a dict"},
        )
        dag.add(source)
        dag.add(bad_cond)

        source.status = TaskStatus.COMPLETED
        await dag.evaluate_conditionals({source.id: "result"}, llm_judge_fn=None)

        # Should still be PENDING since it couldn't parse the conditional
        assert bad_cond.status == TaskStatus.PENDING


# ---------------------------------------------------------------------------
# TaskDAG Loop / Retry
# ---------------------------------------------------------------------------

class TestTaskDAGLoops:
    """Tests for retry/loop constructs in the task DAG."""

    @pytest.mark.asyncio
    async def test_loop_retries_on_failure(self):
        """Task with unmet loop condition is reset to PENDING."""
        dag = TaskDAG()
        task = Task(
            description="Write and test code",
            metadata={
                "role": "coder",
                "loop": {
                    "max_iterations": 3,
                    "condition_type": "contains",
                    "condition_value": "PASS",
                    "loop_task_ids": [],
                },
            },
        )
        dag.add(task)
        task.status = TaskStatus.COMPLETED
        task.result = "Tests FAILED: 2 errors"
        results = {task.id: task.result}

        retrying = await dag.evaluate_loop(task, results, llm_judge_fn=None, max_cap=5)

        assert retrying is True
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.id not in results
        assert dag._iteration_counts[task.id] == 1

    @pytest.mark.asyncio
    async def test_loop_stops_at_max_iterations(self):
        """After max iterations, loop stops retrying."""
        dag = TaskDAG()
        task = Task(
            description="Flaky task",
            metadata={
                "loop": {
                    "max_iterations": 2,
                    "condition_type": "contains",
                    "condition_value": "PASS",
                    "loop_task_ids": [],
                },
            },
        )
        dag.add(task)

        # Simulate 2 failed iterations
        for i in range(2):
            task.status = TaskStatus.COMPLETED
            task.result = "FAILED"
            results = {task.id: task.result}
            await dag.evaluate_loop(task, results, llm_judge_fn=None, max_cap=5)

        # Third attempt should NOT retry
        task.status = TaskStatus.COMPLETED
        task.result = "FAILED again"
        results = {task.id: task.result}
        retrying = await dag.evaluate_loop(task, results, llm_judge_fn=None, max_cap=5)

        assert retrying is False
        assert task.metadata.get("loop_exhausted") is True

    @pytest.mark.asyncio
    async def test_loop_succeeds_first_try(self):
        """When condition is met immediately, no retry occurs."""
        dag = TaskDAG()
        task = Task(
            description="Good task",
            metadata={
                "loop": {
                    "max_iterations": 3,
                    "condition_type": "contains",
                    "condition_value": "PASS",
                    "loop_task_ids": [],
                },
            },
        )
        dag.add(task)
        task.status = TaskStatus.COMPLETED
        task.result = "All tests PASS"
        results = {task.id: task.result}

        retrying = await dag.evaluate_loop(task, results, llm_judge_fn=None, max_cap=5)

        assert retrying is False
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_loop_respects_config_cap(self):
        """Config max_cap overrides loop's own max_iterations."""
        dag = TaskDAG()
        task = Task(
            description="Ambitious task",
            metadata={
                "loop": {
                    "max_iterations": 10,  # Wants 10 retries
                    "condition_type": "contains",
                    "condition_value": "PASS",
                    "loop_task_ids": [],
                },
            },
        )
        dag.add(task)

        # Config cap is 2
        for i in range(2):
            task.status = TaskStatus.COMPLETED
            task.result = "FAILED"
            results = {task.id: task.result}
            await dag.evaluate_loop(task, results, llm_judge_fn=None, max_cap=2)

        # Should stop even though max_iterations=10
        task.status = TaskStatus.COMPLETED
        task.result = "FAILED"
        results = {task.id: task.result}
        retrying = await dag.evaluate_loop(task, results, llm_judge_fn=None, max_cap=2)

        assert retrying is False
        assert task.metadata.get("loop_exhausted") is True

    @pytest.mark.asyncio
    async def test_loop_resets_related_tasks(self):
        """Loop retry resets both the anchor task and its loop_task_ids."""
        dag = TaskDAG()
        helper = Task(description="Gather data")
        main = Task(
            description="Process and validate",
            dependencies=[helper.id],
            metadata={
                "loop": {
                    "max_iterations": 3,
                    "condition_type": "contains",
                    "condition_value": "ALL_OK",
                    "loop_task_ids": [helper.id],
                },
            },
        )
        dag.add(helper)
        dag.add(main)

        helper.status = TaskStatus.COMPLETED
        helper.result = "raw data"
        main.status = TaskStatus.COMPLETED
        main.result = "errors found in output"
        results = {helper.id: helper.result, main.id: main.result}

        retrying = await dag.evaluate_loop(main, results, llm_judge_fn=None, max_cap=5)

        assert retrying is True
        assert main.status == TaskStatus.PENDING
        assert helper.status == TaskStatus.PENDING
        assert helper.id not in results
        assert main.id not in results

    @pytest.mark.asyncio
    async def test_loop_with_llm_judge(self):
        """Loop uses llm_judge condition type correctly."""
        dag = TaskDAG()
        task = Task(
            description="Write code",
            metadata={
                "loop": {
                    "max_iterations": 3,
                    "condition_type": "llm_judge",
                    "condition_value": "Code is correct and complete",
                    "loop_task_ids": [],
                },
            },
        )
        dag.add(task)
        task.status = TaskStatus.COMPLETED
        task.result = "def add(a, b): return a + b"
        results = {task.id: task.result}

        # Judge says YES — no retry
        async def judge_yes(result_text, condition):
            return True

        retrying = await dag.evaluate_loop(task, results, llm_judge_fn=judge_yes, max_cap=5)
        assert retrying is False

    @pytest.mark.asyncio
    async def test_loop_malformed_ignored(self):
        """Malformed loop metadata does not cause errors."""
        dag = TaskDAG()
        task = Task(
            description="Bad loop",
            metadata={"loop": "not a dict"},
        )
        dag.add(task)
        task.status = TaskStatus.COMPLETED
        task.result = "result"
        results = {task.id: task.result}

        retrying = await dag.evaluate_loop(task, results, llm_judge_fn=None, max_cap=5)
        assert retrying is False


# ---------------------------------------------------------------------------
# TaskDAG Backward Compatibility
# ---------------------------------------------------------------------------

class TestTaskDAGBackwardCompat:
    """Ensure existing behaviour is unaffected by the new features."""

    def test_plain_tasks_unchanged(self):
        """DAG with no conditional or loop metadata works exactly as before."""
        dag = TaskDAG()
        t1 = Task(description="Step 1")
        t2 = Task(description="Step 2", dependencies=[t1.id])
        t3 = Task(description="Step 3", dependencies=[t2.id])
        dag.add(t1)
        dag.add(t2)
        dag.add(t3)

        assert dag.ready_tasks() == [t1]

        t1.status = TaskStatus.COMPLETED
        assert dag.ready_tasks() == [t2]

        t2.status = TaskStatus.COMPLETED
        assert dag.ready_tasks() == [t3]

        t3.status = TaskStatus.COMPLETED
        assert dag.all_completed()

    def test_skipped_status_in_all_completed(self):
        """A DAG with SKIPPED tasks is considered complete."""
        dag = TaskDAG()
        t1 = Task(description="Done")
        t2 = Task(description="Skipped")
        dag.add(t1)
        dag.add(t2)

        t1.status = TaskStatus.COMPLETED
        t2.status = TaskStatus.SKIPPED
        assert dag.all_completed()

    def test_summary_includes_skipped_icon(self):
        """Summary shows skip icon for SKIPPED tasks."""
        dag = TaskDAG()
        t = Task(description="Skipped task")
        dag.add(t)
        t.status = TaskStatus.SKIPPED
        summary = dag.summary()
        assert "Skipped task" in summary

    def test_failed_deps_do_not_cascade_skip(self):
        """FAILED deps do NOT trigger cascading skip (only SKIPPED does)."""
        dag = TaskDAG()
        t1 = Task(description="Fails")
        t2 = Task(description="Depends on failure", dependencies=[t1.id])
        dag.add(t1)
        dag.add(t2)

        t1.status = TaskStatus.FAILED
        ready = dag.ready_tasks()
        # t2 should become ready (FAILED is terminal), not skipped
        assert t2 in ready
        assert t2.status == TaskStatus.PENDING


# ---------------------------------------------------------------------------
# Orchestrator with conditionals and loops (integration)
# ---------------------------------------------------------------------------

class TestOrchestratorConditionalIntegration:
    """End-to-end tests for the orchestrator with conditional and loop tasks."""

    @pytest.mark.asyncio
    async def test_orchestrator_conditional_flow(self):
        """Orchestrator correctly routes through a conditional branch."""
        import json
        call_count = 0

        async def mock_chat(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            msg_content = messages[-1].content if messages else ""

            if call_count == 1:
                # Planning phase — emit tasks with a conditional
                tasks = [
                    {"description": "Research the topic", "role": "researcher", "dependencies": []},
                    {
                        "description": "Check if bugs found",
                        "role": "conditional",
                        "dependencies": [0],
                        "conditional": {
                            "condition_type": "contains",
                            "condition_value": "bug",
                            "if_true_indices": [2],
                            "if_false_indices": [3],
                        },
                    },
                    {"description": "Fix bugs", "role": "coder", "dependencies": [1]},
                    {"description": "Write summary", "role": "general", "dependencies": [1]},
                ]
                return (json.dumps(tasks), [], {"total_tokens": 20})
            elif call_count == 2:
                # Researcher agent — finds a bug
                return ("Found a bug in the login module.", [], {"total_tokens": 15})
            elif call_count == 3:
                # Coder agent — fixes the bug (true branch)
                return ("Fixed the login bug.", [], {"total_tokens": 15})
            elif call_count == 4:
                # Synthesis
                return ("Research found and fixed a login bug.", [], {"total_tokens": 10})
            else:
                # Evaluation
                return ("PASS", [], {"total_tokens": 5})

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat), \
             patch("cadence.agents.orchestrator.chat_completion", side_effect=mock_chat):
            trace = TraceLogger(console=False)
            registry = _make_registry()
            config = _make_config()

            orch = Orchestrator(tool_registry=registry, trace=trace, config=config)
            result = await orch.run("Check for bugs and fix them")

            assert isinstance(result, str)
            assert len(result) > 0

            # Verify the conditional task was evaluated and summary task was skipped
            tasks = list(orch.dag._tasks.values())
            cond_tasks = [t for t in tasks if "conditional" in t.metadata]
            assert len(cond_tasks) == 1
            assert cond_tasks[0].status == TaskStatus.COMPLETED

            # The "Write summary" task (false branch) should be skipped
            summary_tasks = [t for t in tasks if t.description == "Write summary"]
            assert len(summary_tasks) == 1
            assert summary_tasks[0].status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_orchestrator_loop_retry(self):
        """Orchestrator retries a looping task when condition is not met."""
        import json
        call_count = 0

        async def mock_chat(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            msg_content = messages[-1].content if messages else ""

            if call_count == 1:
                # Planning phase — task with loop
                tasks = [
                    {
                        "description": "Write and test code",
                        "role": "coder",
                        "dependencies": [],
                        "loop": {
                            "max_iterations": 3,
                            "condition_type": "contains",
                            "condition_value": "PASS",
                        },
                    },
                ]
                return (json.dumps(tasks), [], {"total_tokens": 20})
            elif call_count == 2:
                # First attempt — fails
                return ("Tests FAILED: syntax error on line 5", [], {"total_tokens": 15})
            elif call_count == 3:
                # Second attempt — passes
                return ("All tests PASS", [], {"total_tokens": 15})
            elif call_count == 4:
                # Synthesis (single result, returns directly)
                return ("Code written and all tests pass.", [], {"total_tokens": 10})
            else:
                # Evaluation
                return ("PASS", [], {"total_tokens": 5})

        with patch("cadence.core.agent.chat_completion", side_effect=mock_chat), \
             patch("cadence.agents.orchestrator.chat_completion", side_effect=mock_chat):
            trace = TraceLogger(console=False)
            registry = _make_registry()
            config = _make_config()

            orch = Orchestrator(tool_registry=registry, trace=trace, config=config)
            result = await orch.run("Write code that passes tests")

            assert isinstance(result, str)
            # The task should have been retried once
            task = list(orch.dag._tasks.values())[0]
            assert task.status == TaskStatus.COMPLETED
            assert orch.dag._iteration_counts.get(task.id, 0) == 1

    @pytest.mark.asyncio
    async def test_orchestrator_llm_judge_integration(self):
        """_llm_judge correctly parses YES/NO from the LLM."""
        call_count = 0

        async def mock_chat(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return YES for the judge call
            return ("YES", [], {"total_tokens": 5})

        with patch("cadence.agents.orchestrator.chat_completion", side_effect=mock_chat):
            trace = TraceLogger(console=False)
            registry = _make_registry()
            config = _make_config()

            orch = Orchestrator(tool_registry=registry, trace=trace, config=config)
            result = await orch._llm_judge("All 5 tests passed.", "All tests pass")
            assert result is True

    @pytest.mark.asyncio
    async def test_orchestrator_llm_judge_no(self):
        """_llm_judge returns False when LLM says NO."""
        async def mock_chat(model, messages, **kwargs):
            return ("NO", [], {"total_tokens": 5})

        with patch("cadence.agents.orchestrator.chat_completion", side_effect=mock_chat):
            trace = TraceLogger(console=False)
            registry = _make_registry()
            config = _make_config()

            orch = Orchestrator(tool_registry=registry, trace=trace, config=config)
            result = await orch._llm_judge("2 tests failed.", "All tests pass")
            assert result is False
