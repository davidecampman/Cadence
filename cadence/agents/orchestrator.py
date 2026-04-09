"""Multi-agent orchestrator with task DAG execution."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from cadence.core.agent import Agent
from cadence.core.checkpoint import CheckpointManager
from cadence.core.config import Config, get_config
from cadence.core.llm import chat_completion
from cadence.core.message_bus import MessageBus
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
)
from cadence.learning.store import LearningStore, OutcomeRating, StrategyRecord
from cadence.prompts.evolution import PromptEvolver
from cadence.prompts.store import PromptEvolutionStore
from cadence.skills.loader import SkillLoader
from cadence.tools.base import ToolRegistry


# --- Predefined specialist roles ---

ROLES = {
    "orchestrator": AgentRole(
        name="orchestrator",
        description=(
            "You coordinate complex tasks by breaking them into subtasks and delegating "
            "to specialist agents. You synthesize results into a coherent final answer. "
            "You do NOT do the work yourself — you plan and delegate."
        ),
    ),
    "researcher": AgentRole(
        name="researcher",
        description=(
            "You gather information by reading files, searching codebases, fetching web pages, "
            "and querying memory. You report findings clearly and factually."
        ),
        allowed_tools=["read_file", "list_files", "search_files", "web_fetch",
                        "browse_web", "browser_extract", "memory_query"],
    ),
    "coder": AgentRole(
        name="coder",
        description=(
            "You write, modify, and debug code. You can execute code to test it. "
            "You write clean, correct, minimal code."
        ),
        allowed_tools=["read_file", "write_file", "list_files", "search_files",
                        "execute_code", "shell", "memory_query", "memory_save"],
        permission_tier=PermissionTier.PRIVILEGED,
    ),
    "reviewer": AgentRole(
        name="reviewer",
        description=(
            "You review code and task outputs for correctness, security issues, and quality. "
            "You provide specific, actionable feedback."
        ),
        allowed_tools=["read_file", "list_files", "search_files", "execute_code", "memory_query"],
    ),
    "general": AgentRole(
        name="general",
        description="A general-purpose agent that can use all available tools.",
        permission_tier=PermissionTier.PRIVILEGED,
    ),
}


class TaskDAG:
    """Manages a directed acyclic graph of tasks with dependency resolution.

    Supports conditional branching and retry/loop constructs via metadata
    on individual tasks.
    """

    _TERMINAL_STATES = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._iteration_counts: dict[str, int] = {}  # loop-anchor task ID -> count

    def add(self, task: Task) -> Task:
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def ready_tasks(self) -> list[Task]:
        """Return tasks whose dependencies are all completed/skipped.

        If a dependency was SKIPPED, the dependent task is also cascaded to
        SKIPPED (it can never run because its prerequisite was never executed).
        """
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            dep_statuses = [
                self._tasks[dep_id].status
                for dep_id in task.dependencies
                if dep_id in self._tasks
            ]

            all_terminal = all(s in self._TERMINAL_STATES for s in dep_statuses)
            if not all_terminal:
                continue

            # Cascade skip: if any dependency was skipped, skip this task too
            if any(s == TaskStatus.SKIPPED for s in dep_statuses):
                task.status = TaskStatus.SKIPPED
                task.result = "Skipped — prerequisite task was skipped"
                continue

            ready.append(task)
        return ready

    def all_completed(self) -> bool:
        return all(
            t.status in self._TERMINAL_STATES
            for t in self._tasks.values()
        )

    def summary(self) -> str:
        lines = []
        for t in self._tasks.values():
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.RUNNING: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.BLOCKED: "🚫",
                TaskStatus.SKIPPED: "⏭️",
            }.get(t.status, "?")
            lines.append(f"  {status_icon} [{t.id[:6]}] {t.description}")
        return "\n".join(lines)

    # --- Conditional branching ---

    async def evaluate_conditionals(
        self,
        results: dict[str, str],
        llm_judge_fn,
    ) -> None:
        """Evaluate conditional nodes whose source dependency has completed.

        For each conditional task that is still PENDING and whose
        ``condition_source`` is COMPLETED, evaluate the condition and
        activate one branch while skipping the other.
        """
        for task in list(self._tasks.values()):
            if task.status != TaskStatus.PENDING:
                continue
            cond_data = task.metadata.get("conditional")
            if not cond_data or not isinstance(cond_data, dict):
                continue

            try:
                cond = ConditionalDef(**cond_data)
            except Exception:
                continue  # Malformed — treat as normal task

            source = self._tasks.get(cond.condition_source)
            if not source or source.status != TaskStatus.COMPLETED:
                continue

            source_result = results.get(cond.condition_source, "")

            # Evaluate the condition
            condition_met = await self._evaluate_condition(
                cond.condition_type,
                cond.condition_value,
                source_result,
                llm_judge_fn,
            )

            # Mark the conditional node itself as completed
            task.status = TaskStatus.COMPLETED
            task.result = f"Condition evaluated: {'TRUE' if condition_met else 'FALSE'}"
            results[task.id] = task.result

            # Activate chosen branch, skip the other
            active_ids = cond.if_true if condition_met else cond.if_false
            skipped_ids = cond.if_false if condition_met else cond.if_true

            for tid in skipped_ids:
                t = self._tasks.get(tid)
                if t and t.status == TaskStatus.PENDING:
                    t.status = TaskStatus.SKIPPED
                    t.result = "Skipped — conditional branch not taken"

    async def evaluate_loop(
        self,
        task: Task,
        results: dict[str, str],
        llm_judge_fn,
        max_cap: int = 5,
    ) -> bool:
        """Check loop condition on a just-completed task. Returns True if retrying."""
        loop_data = task.metadata.get("loop")
        if not loop_data or not isinstance(loop_data, dict):
            return False

        try:
            loop = LoopDef(**loop_data)
        except Exception:
            return False

        effective_max = min(loop.max_iterations, max_cap)
        current = self._iteration_counts.get(task.id, 0)

        if current >= effective_max:
            task.metadata["loop_exhausted"] = True
            return False

        task_result = results.get(task.id, "")
        condition_met = await self._evaluate_condition(
            loop.condition_type,
            loop.condition_value,
            task_result,
            llm_judge_fn,
        )

        if condition_met:
            # Success — no retry needed
            return False

        # Retry: reset this task and any extra loop tasks
        self._iteration_counts[task.id] = current + 1
        tasks_to_reset = [task.id] + list(loop.loop_task_ids)
        for tid in tasks_to_reset:
            t = self._tasks.get(tid)
            if t:
                t.status = TaskStatus.PENDING
                t.result = None
                results.pop(tid, None)

        return True

    @staticmethod
    async def _evaluate_condition(
        condition_type: str,
        condition_value: str,
        source_result: str,
        llm_judge_fn,
    ) -> bool:
        """Evaluate a condition against a task result."""
        if condition_type == "contains":
            return condition_value.lower() in source_result.lower()
        elif condition_type == "equals":
            return source_result.strip() == condition_value
        elif condition_type == "llm_judge":
            if llm_judge_fn:
                return await llm_judge_fn(source_result, condition_value)
            return False
        return False


class Orchestrator:
    """Breaks a user request into a task DAG and executes it with specialist agents.

    Flow:
      1. Planning: LLM decomposes the request into tasks with dependencies
      2. Execution: Run ready tasks in parallel (up to max_parallel)
      3. Synthesis: Combine results into a final response
      4. Evaluation: Optional self-check of the final output
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        trace: TraceLogger,
        config: Config | None = None,
        skill_loader: SkillLoader | None = None,
        prompt_evolver: PromptEvolver | None = None,
        message_bus: MessageBus | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        learning_store: LearningStore | None = None,
    ):
        self.tools = tool_registry
        self.trace = trace
        self.config = config or get_config()
        self.skill_loader = skill_loader
        self.dag = TaskDAG()
        self._session_tokens: int = 0
        self._session_id: str = ""

        # Inter-agent communication
        self.message_bus = message_bus

        # Human-in-the-loop
        self.checkpoint_manager = checkpoint_manager

        # Cross-session learning
        self.learning_store = learning_store

        # Initialize prompt evolver if evolution is enabled
        if prompt_evolver:
            self.prompt_evolver = prompt_evolver
        elif self.config.prompt_evolution.enabled:
            store = PromptEvolutionStore(db_path=self.config.prompt_evolution.persist_dir)
            self.prompt_evolver = PromptEvolver(store=store, config=self.config)
        else:
            self.prompt_evolver = None

    @property
    def session_tokens(self) -> int:
        """Total tokens consumed across all agents in this session."""
        return self._session_tokens

    def _check_session_budget(self) -> str | None:
        """Return an error message if session budget is exceeded, else None."""
        limit = self.config.budget.max_tokens_per_session
        if self._session_tokens >= limit:
            return (
                f"Session token budget exhausted ({self._session_tokens:,} / {limit:,} tokens). "
                "Please start a new session."
            )
        warn_pct = self.config.budget.warn_at_percentage
        if warn_pct and self._session_tokens >= limit * warn_pct / 100:
            self.trace.thought(
                "orchestrator",
                f"Session budget warning: {self._session_tokens:,} / {limit:,} tokens "
                f"({self._session_tokens * 100 // limit}% used)",
            )
        return None

    async def run(
        self,
        user_request: str,
        conversation_history: list[dict[str, str]] | None = None,
        session_id: str = "",
        images: list[dict] | None = None,
    ) -> str:
        """Process a user request end-to-end."""
        self._session_id = session_id
        self._images = images
        _start_time = time.time()

        # Check session budget before starting
        budget_error = self._check_session_budget()
        if budget_error:
            return budget_error

        self.trace.observation("orchestrator", f"Request: {user_request}")

        # Reset DAG for each request
        self.dag = TaskDAG()

        self._conversation_history = conversation_history or []

        # Query learning insights if available
        learning_context = ""
        if self.learning_store:
            task_type = self.learning_store.classify_task(user_request)
            insights = self.learning_store.get_insights(task_type, limit=3)
            if insights:
                tips = "; ".join(i.recommendation for i in insights[:2])
                learning_context = f"\n[Learning from past tasks ({task_type})]: {tips}"

        # Phase 1: Plan — decompose into tasks
        tasks = await self._plan(user_request + learning_context)

        if not tasks:
            # Simple request, no decomposition needed — just run directly
            agent = self._spawn_agent("general")
            result = await agent.run(
                user_request,
                conversation_history=self._conversation_history,
                images=self._images,
            )
            self._session_tokens += agent._total_tokens
            return result

        for task in tasks:
            self.dag.add(task)

        self.trace.thought("orchestrator", f"Plan:\n{self.dag.summary()}")

        # Phase 2: Execute the DAG
        results: dict[str, str] = {}
        max_parallel = self.config.agents.max_parallel

        while not self.dag.all_completed():
            # Check budget between batches
            budget_error = self._check_session_budget()
            if budget_error:
                self.trace.error("orchestrator", "Session token budget exceeded during execution")
                # Mark remaining pending tasks as failed
                for t in self.dag._tasks.values():
                    if t.status == TaskStatus.PENDING:
                        t.status = TaskStatus.FAILED
                        t.result = "Skipped — session budget exceeded"
                break

            # Evaluate any conditional nodes whose sources are done
            await self.dag.evaluate_conditionals(results, self._llm_judge)

            ready = self.dag.ready_tasks()
            if not ready:
                # Check for deadlock
                pending = [t for t in self.dag._tasks.values() if t.status == TaskStatus.PENDING]
                if pending:
                    self.trace.error("orchestrator", "Deadlock detected — unresolvable dependencies")
                    for t in pending:
                        t.status = TaskStatus.FAILED
                        t.result = "Blocked by unresolved dependencies"
                break

            # Run ready tasks in parallel, up to limit
            batch = ready[:max_parallel]
            coros = []
            for task in batch:
                task.status = TaskStatus.RUNNING
                coros.append(self._execute_task(task, results))

            await asyncio.gather(*coros)

            # Evaluate loop conditions on just-completed tasks
            for task in batch:
                if task.status == TaskStatus.COMPLETED and "loop" in task.metadata:
                    retrying = await self.dag.evaluate_loop(
                        task, results, self._llm_judge,
                        max_cap=self.config.agents.max_loop_iterations,
                    )
                    if retrying:
                        iteration = self.dag._iteration_counts.get(task.id, 0)
                        self.trace.thought(
                            "orchestrator",
                            f"Task [{task.id[:6]}] loop retry {iteration} — "
                            f"condition not met, retrying",
                        )

        # Phase 3: Synthesize results
        final = await self._synthesize(user_request, results)

        # Phase 4: Self-evaluate
        final = await self._evaluate(user_request, final)

        # Record learning outcome
        if self.learning_store:
            task_type = self.learning_store.classify_task(user_request)
            all_failed = all(
                t.status == TaskStatus.FAILED for t in self.dag._tasks.values()
            )
            has_failures = any(
                t.status == TaskStatus.FAILED for t in self.dag._tasks.values()
            )
            outcome = (
                OutcomeRating.FAILURE if all_failed
                else OutcomeRating.PARTIAL if has_failures
                else OutcomeRating.SUCCESS
            )
            self.learning_store.record(StrategyRecord(
                session_id=self._session_id,
                task_type=task_type,
                task_description=user_request[:500],
                strategy=f"Decomposed into {len(tasks)} tasks" if tasks else "Direct execution",
                tools_used=[],
                model_used=self.config.models.strong,
                role_used="orchestrator",
                outcome=outcome,
                tokens_used=self._session_tokens,
                duration_ms=(time.time() - _start_time) * 1000,
            ))

        return final

    def _format_history_block(self) -> str:
        """Format prior conversation turns into a context block for prompts."""
        if not self._conversation_history:
            return ""
        lines = []
        for entry in self._conversation_history:
            role = "User" if entry["role"] == "user" else "Assistant"
            lines.append(f"{role}: {entry['content']}")
        return (
            "Here is the prior conversation for context:\n"
            + "\n".join(lines)
            + "\n\n"
        )

    async def _plan(self, request: str) -> list[Task]:
        """Use the fast model to decompose a request into tasks."""
        history_block = self._format_history_block()
        planning_prompt = (
            "You are a task planner. Given a user request, decide if it needs to be broken "
            "into subtasks. If it's simple enough for one agent, return SIMPLE.\n\n"
            "If it needs decomposition, return a JSON array of tasks:\n"
            '[\n  {"description": "...", "role": "researcher|coder|reviewer|general", '
            '"dependencies": []},\n  ...\n]\n\n'
            "Dependencies are indices (0-based) of tasks that must complete first.\n"
            "Keep it minimal — don't over-decompose.\n\n"
            "ADVANCED FEATURES (use only when truly needed):\n\n"
            "1. CONDITIONAL BRANCHING — add a task with role \"conditional\" to make decisions:\n"
            '   {"description": "Check if tests pass", "role": "conditional", '
            '"dependencies": [1],\n'
            '    "conditional": {"condition_type": "llm_judge", '
            '"condition_value": "All tests pass", "if_true_indices": [3], '
            '"if_false_indices": [4]}}\n'
            "   condition_type can be: \"contains\", \"equals\", or \"llm_judge\"\n"
            "   The condition is evaluated against the result of the dependency task.\n\n"
            "2. RETRY LOOP — add a \"loop\" key to retry a task until a condition is met:\n"
            '   {"description": "Fix and test code", "role": "coder", '
            '"dependencies": [0],\n'
            '    "loop": {"max_iterations": 3, "condition_type": "llm_judge", '
            '"condition_value": "Code passes all tests"}}\n'
            "   The task will be re-run up to max_iterations times if the condition is NOT met.\n\n"
            f"{history_block}"
            f"User request: {request}"
        )

        text, _, usage = await chat_completion(
            model=self.config.models.fast,
            messages=[Message(role=Role.USER, content=planning_prompt)],
            temperature=0.3,
            max_tokens=1024,
            bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
            local_config=self.config.models.local if self.config.models.local.enabled else None,
        )
        self._session_tokens += usage.get("total_tokens", 0)

        if "SIMPLE" in text.upper():
            return []

        # Parse task list from LLM response
        import json
        import re

        # Extract JSON from response (may be wrapped in markdown)
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            return []

        try:
            task_defs = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        tasks: list[Task] = []
        for i, td in enumerate(task_defs):
            task = Task(
                description=td.get("description", f"Task {i}"),
                metadata={"role": td.get("role", "general")},
            )
            tasks.append(task)

        # Resolve dependency indices to task IDs
        for i, td in enumerate(task_defs):
            dep_indices = td.get("dependencies", [])
            for idx in dep_indices:
                if 0 <= idx < len(tasks) and idx != i:
                    tasks[i].dependencies.append(tasks[idx].id)

        # Resolve conditional branching metadata
        for i, td in enumerate(task_defs):
            cond = td.get("conditional")
            if cond and isinstance(cond, dict):
                # Resolve branch indices to task IDs
                if_true_ids = []
                for idx in cond.get("if_true_indices", []):
                    if 0 <= idx < len(tasks):
                        if_true_ids.append(tasks[idx].id)
                if_false_ids = []
                for idx in cond.get("if_false_indices", []):
                    if 0 <= idx < len(tasks):
                        if_false_ids.append(tasks[idx].id)

                # The condition source is the first dependency
                condition_source = tasks[i].dependencies[0] if tasks[i].dependencies else ""

                tasks[i].metadata["conditional"] = {
                    "condition_source": condition_source,
                    "condition_type": cond.get("condition_type", "llm_judge"),
                    "condition_value": cond.get("condition_value", ""),
                    "if_true": if_true_ids,
                    "if_false": if_false_ids,
                }

            # Resolve loop metadata
            loop = td.get("loop")
            if loop and isinstance(loop, dict):
                loop_task_ids = []
                for idx in loop.get("loop_task_indices", []):
                    if 0 <= idx < len(tasks):
                        loop_task_ids.append(tasks[idx].id)

                tasks[i].metadata["loop"] = {
                    "max_iterations": loop.get("max_iterations", 3),
                    "condition_type": loop.get("condition_type", "llm_judge"),
                    "condition_value": loop.get("condition_value", ""),
                    "loop_task_ids": loop_task_ids,
                }

        return tasks

    async def _llm_judge(self, result_text: str, condition: str) -> bool:
        """Ask the fast model whether *result_text* satisfies *condition*."""
        prompt = (
            "You are a strict evaluator. Does the following result satisfy the condition?\n\n"
            f"CONDITION: {condition}\n\n"
            f"RESULT:\n{result_text[:2000]}\n\n"
            "Answer with exactly YES or NO."
        )
        text, _, usage = await chat_completion(
            model=self.config.models.fast,
            messages=[Message(role=Role.USER, content=prompt)],
            temperature=0.0,
            max_tokens=8,
            bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
            local_config=self.config.models.local if self.config.models.local.enabled else None,
        )
        self._session_tokens += usage.get("total_tokens", 0)
        return text.strip().upper().startswith("YES")

    async def _execute_task(self, task: Task, results: dict[str, str]) -> None:
        """Execute a single task with the appropriate specialist agent."""
        # Conditional nodes are resolved by evaluate_conditionals, not agent execution
        if "conditional" in task.metadata:
            task.status = TaskStatus.COMPLETED
            task.result = "Conditional node — awaiting evaluation"
            results[task.id] = task.result
            return

        role_name = task.metadata.get("role", "general")
        agent = self._spawn_agent(role_name)
        task.assigned_agent = agent.id

        # Build context: include results of dependency tasks
        context_parts = [task.description]
        for dep_id in task.dependencies:
            if dep_id in results:
                context_parts.append(f"\n[Result from prerequisite task {dep_id[:6]}]:\n{results[dep_id]}")

        # Include recent bus messages so the agent sees discoveries from other agents
        if self.message_bus:
            discovery_msgs = self.message_bus.peek("discovery", limit=5)
            if discovery_msgs:
                bus_context = "\n".join(
                    f"[{m.sender_id}]: {m.content[:200]}" for m in discovery_msgs
                )
                context_parts.append(f"\n[Recent discoveries from other agents]:\n{bus_context}")

        task_prompt = "\n".join(context_parts)

        self.trace.action("orchestrator", f"Assigning [{task.id[:6]}] to {agent.id}")
        try:
            result = await agent.run(task_prompt)
            task.status = TaskStatus.COMPLETED
            task.result = result
            results[task.id] = result
            self.trace.result("orchestrator", f"Task [{task.id[:6]}] completed")

            # Publish task completion on message bus
            if self.message_bus:
                await self.message_bus.publish(
                    topic="status",
                    sender_id=agent.id,
                    content=f"Task [{task.id[:6]}] completed: {task.description[:100]}",
                )
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = f"Error: {e}"
            results[task.id] = task.result
            self.trace.error("orchestrator", f"Task [{task.id[:6]}] failed: {e}")

            # Publish failure on message bus
            if self.message_bus:
                await self.message_bus.publish(
                    topic="error",
                    sender_id=agent.id,
                    content=f"Task [{task.id[:6]}] failed: {e}",
                )
        finally:
            self._session_tokens += agent._total_tokens

    async def _synthesize(self, request: str, results: dict[str, str]) -> str:
        """Combine task results into a coherent final response."""
        if len(results) == 1:
            return next(iter(results.values()))

        result_block = "\n\n".join(
            f"### Task Result\n{text}" for text in results.values()
        )
        history_block = self._format_history_block()
        synthesis_prompt = (
            f"{history_block}"
            f"The user asked: {request}\n\n"
            f"Multiple agents worked on subtasks. Here are their results:\n\n{result_block}\n\n"
            "Synthesize these into a single, coherent response to the user's original request. "
            "Be concise. Don't mention the subtask structure."
        )

        text, _, usage = await chat_completion(
            model=self.config.models.strong,
            messages=[Message(role=Role.USER, content=synthesis_prompt)],
            temperature=0.5,
            bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
            local_config=self.config.models.local if self.config.models.local.enabled else None,
        )
        self._session_tokens += usage.get("total_tokens", 0)
        return text

    async def _evaluate(self, request: str, response: str) -> str:
        """Self-evaluate: does the response actually answer the request?"""
        eval_prompt = (
            f"User request: {request}\n\n"
            f"Proposed response:\n{response}\n\n"
            "Does this response fully and correctly answer the user's request?\n"
            "If yes, reply: PASS\n"
            "If no, reply: FAIL: <brief explanation of what's missing or wrong>"
        )

        text, _, usage = await chat_completion(
            model=self.config.models.fast,
            messages=[Message(role=Role.USER, content=eval_prompt)],
            temperature=0.2,
            max_tokens=256,
            bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
            local_config=self.config.models.local if self.config.models.local.enabled else None,
        )
        self._session_tokens += usage.get("total_tokens", 0)

        if text.strip().upper().startswith("PASS"):
            self.trace.result("orchestrator", "Self-evaluation: PASS")
            return response

        # If evaluation fails, note it but return the response anyway
        self.trace.thought("orchestrator", f"Self-evaluation flagged issues: {text[:200]}")
        return response + f"\n\n---\n*Note: Self-review flagged: {text.strip()}*"

    def _spawn_agent(self, role_name: str) -> Agent:
        """Create a specialist agent."""
        role = ROLES.get(role_name, ROLES["general"])
        return Agent(
            role=role,
            tool_registry=self.tools,
            trace=self.trace,
            config=self.config,
            skill_loader=self.skill_loader,
            prompt_evolver=self.prompt_evolver,
        )
