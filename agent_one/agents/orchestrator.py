"""Multi-agent orchestrator with task DAG execution."""

from __future__ import annotations

import asyncio
from typing import Any

from agent_one.core.agent import Agent
from agent_one.core.config import Config, get_config
from agent_one.core.llm import chat_completion
from agent_one.core.trace import TraceLogger
from agent_one.core.types import (
    AgentRole,
    Message,
    Role,
    Task,
    TaskStatus,
)
from agent_one.tools.base import ToolRegistry


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
        allowed_tools=["read_file", "list_files", "search_files", "web_fetch", "memory_query"],
    ),
    "coder": AgentRole(
        name="coder",
        description=(
            "You write, modify, and debug code. You can execute code to test it. "
            "You write clean, correct, minimal code."
        ),
        allowed_tools=["read_file", "write_file", "list_files", "search_files",
                        "execute_code", "shell", "memory_query", "memory_save"],
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
    ),
}


class TaskDAG:
    """Manages a directed acyclic graph of tasks with dependency resolution."""

    def __init__(self):
        self._tasks: dict[str, Task] = {}

    def add(self, task: Task) -> Task:
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def ready_tasks(self) -> list[Task]:
        """Return tasks whose dependencies are all completed."""
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self._tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self._tasks
            )
            if deps_met:
                ready.append(task)
        return ready

    def all_completed(self) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
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
            }.get(t.status, "?")
            lines.append(f"  {status_icon} [{t.id[:6]}] {t.description}")
        return "\n".join(lines)


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
    ):
        self.tools = tool_registry
        self.trace = trace
        self.config = config or get_config()
        self.dag = TaskDAG()

    async def run(self, user_request: str) -> str:
        """Process a user request end-to-end."""
        self.trace.observation("orchestrator", f"Request: {user_request}")

        # Phase 1: Plan — decompose into tasks
        tasks = await self._plan(user_request)

        if not tasks:
            # Simple request, no decomposition needed — just run directly
            agent = self._spawn_agent("general")
            return await agent.run(user_request)

        for task in tasks:
            self.dag.add(task)

        self.trace.thought("orchestrator", f"Plan:\n{self.dag.summary()}")

        # Phase 2: Execute the DAG
        results: dict[str, str] = {}
        max_parallel = self.config.agents.max_parallel

        while not self.dag.all_completed():
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

        # Phase 3: Synthesize results
        final = await self._synthesize(user_request, results)

        # Phase 4: Self-evaluate
        final = await self._evaluate(user_request, final)

        return final

    async def _plan(self, request: str) -> list[Task]:
        """Use the fast model to decompose a request into tasks."""
        planning_prompt = (
            "You are a task planner. Given a user request, decide if it needs to be broken "
            "into subtasks. If it's simple enough for one agent, return SIMPLE.\n\n"
            "If it needs decomposition, return a JSON array of tasks:\n"
            '[\n  {"description": "...", "role": "researcher|coder|reviewer|general", '
            '"dependencies": []},\n  ...\n]\n\n'
            "Dependencies are indices (0-based) of tasks that must complete first.\n"
            "Keep it minimal — don't over-decompose.\n\n"
            f"User request: {request}"
        )

        text, _, _ = await chat_completion(
            model=self.config.models.fast,
            messages=[Message(role=Role.USER, content=planning_prompt)],
            temperature=0.3,
            max_tokens=1024,
            bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
        )

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

        return tasks

    async def _execute_task(self, task: Task, results: dict[str, str]) -> None:
        """Execute a single task with the appropriate specialist agent."""
        role_name = task.metadata.get("role", "general")
        agent = self._spawn_agent(role_name)
        task.assigned_agent = agent.id

        # Build context: include results of dependency tasks
        context_parts = [task.description]
        for dep_id in task.dependencies:
            if dep_id in results:
                context_parts.append(f"\n[Result from prerequisite task {dep_id[:6]}]:\n{results[dep_id]}")

        task_prompt = "\n".join(context_parts)

        self.trace.action("orchestrator", f"Assigning [{task.id[:6]}] to {agent.id}")
        try:
            result = await agent.run(task_prompt)
            task.status = TaskStatus.COMPLETED
            task.result = result
            results[task.id] = result
            self.trace.result("orchestrator", f"Task [{task.id[:6]}] completed")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = f"Error: {e}"
            results[task.id] = task.result
            self.trace.error("orchestrator", f"Task [{task.id[:6]}] failed: {e}")

    async def _synthesize(self, request: str, results: dict[str, str]) -> str:
        """Combine task results into a coherent final response."""
        if len(results) == 1:
            return next(iter(results.values()))

        result_block = "\n\n".join(
            f"### Task Result\n{text}" for text in results.values()
        )
        synthesis_prompt = (
            f"The user asked: {request}\n\n"
            f"Multiple agents worked on subtasks. Here are their results:\n\n{result_block}\n\n"
            "Synthesize these into a single, coherent response to the user's original request. "
            "Be concise. Don't mention the subtask structure."
        )

        text, _, _ = await chat_completion(
            model=self.config.models.strong,
            messages=[Message(role=Role.USER, content=synthesis_prompt)],
            temperature=0.5,
            bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
        )
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

        text, _, _ = await chat_completion(
            model=self.config.models.fast,
            messages=[Message(role=Role.USER, content=eval_prompt)],
            temperature=0.2,
            max_tokens=256,
            bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
        )

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
        )
