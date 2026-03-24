"""Core agent loop — the think→act→observe cycle."""

from __future__ import annotations

import uuid
from typing import Any

from sentinel.core.config import Config, get_config
from sentinel.core.llm import chat_completion
from sentinel.core.trace import TraceLogger
from sentinel.core.types import (
    AgentRole,
    Message,
    PermissionTier,
    Role,
    Task,
    TaskStatus,
    ToolCall,
)
from sentinel.skills.loader import SkillLoader
from sentinel.tools.base import Tool, ToolRegistry


class Agent:
    """A single agent instance with its own identity, role, history, and tools.

    The agent loop:
      1. Build context (system prompt + history + task)
      2. Call LLM → get text + tool_calls
      3. If tool_calls: execute them, append results, goto 2
      4. If no tool_calls: return final text response
      5. Circuit breaker: stop after max_iterations
    """

    def __init__(
        self,
        role: AgentRole,
        tool_registry: ToolRegistry,
        trace: TraceLogger,
        config: Config | None = None,
        agent_id: str | None = None,
        parent_id: str | None = None,
        depth: int = 0,
        skill_loader: SkillLoader | None = None,
    ):
        self.id = agent_id or f"{role.name}-{str(uuid.uuid4())[:6]}"
        self.role = role
        self.tools = tool_registry
        self.trace = trace
        self.config = config or get_config()
        self.parent_id = parent_id
        self.depth = depth
        self.skill_loader = skill_loader

        self._history: list[Message] = []
        self._total_tokens: int = 0
        self._iterations: int = 0

    @property
    def model(self) -> str:
        """Which model this agent uses."""
        return self.role.model_override or self.config.models.strong

    def _system_prompt(self) -> str:
        """Build the system prompt for this agent, including any loaded skills."""
        tool_names = ", ".join(self.tools.names())
        prompt = (
            f"You are '{self.role.name}', an AI agent.\n"
            f"Role: {self.role.description}\n\n"
            f"Available tools: [{tool_names}]\n\n"
            "## Instructions\n"
            "- Think step by step. Use tools to gather information before answering.\n"
            "- When you have enough information, provide a clear, direct response.\n"
            "- If a task requires multiple steps, break it down and work through each one.\n"
            "- If you're stuck, explain what's blocking you.\n"
            "- Do NOT guess when you can look things up.\n"
            "- Be concise. Lead with the answer.\n"
        )

        # Append skill instructions if a skill loader is available
        if self.skill_loader:
            skill_prompts = self._build_skill_prompt()
            if skill_prompts:
                prompt += f"\n## Skills\n{skill_prompts}\n"

        return prompt

    def _build_skill_prompt(self) -> str:
        """Collect instructions from all loaded skills (with dependency resolution)."""
        if not self.skill_loader:
            return ""
        parts: list[str] = []
        for skill in self.skill_loader.all_skills.values():
            prompt = self.skill_loader.get_skill_prompt(skill.name)
            if prompt:
                parts.append(prompt)
        return "\n".join(parts)

    def _check_loop_detection(self) -> bool:
        """Detect if the agent is stuck in a loop by checking recent outputs."""
        window = self.config.agents.loop_detection_window
        if len(self._history) < window * 2:
            return False

        recent_assistant = [
            m.content for m in self._history[-window * 2:]
            if m.role == Role.ASSISTANT and m.content
        ][-window:]

        if len(recent_assistant) >= window:
            # If all recent outputs are identical, we're looping
            if len(set(recent_assistant)) == 1:
                return True
        return False

    async def run(self, task: str) -> str:
        """Execute the agent loop on a task. Returns the final response."""
        self.trace.observation(self.id, f"Task received: {task}")

        # Scope memory tools to this agent's namespace
        scoped_tools = self.tools.scoped_copy(self.id)

        # Initialize with system prompt
        self._history = [
            Message(role=Role.SYSTEM, content=self._system_prompt()),
            Message(role=Role.USER, content=task),
        ]

        max_iter = self.config.agents.max_iterations_per_task

        while self._iterations < max_iter:
            self._iterations += 1

            # Loop detection
            if self._check_loop_detection():
                self.trace.error(self.id, "Loop detected — breaking out")
                return "I appear to be stuck in a loop. Here's what I have so far:\n" + (
                    self._last_assistant_text() or "(no output yet)"
                )

            # Budget check
            if self._total_tokens >= self.config.budget.max_tokens_per_task:
                self.trace.error(self.id, f"Token budget exceeded: {self._total_tokens}")
                return "Token budget exceeded. Partial result:\n" + (
                    self._last_assistant_text() or "(no output yet)"
                )

            # Get available tool definitions for this agent's permission tier
            tool_defs = scoped_tools.definitions(
                max_tier=self.role.permission_tier,
                allowed_names=self.role.allowed_tools,
            )

            # Call LLM
            self.trace.thought(self.id, f"Iteration {self._iterations} — calling {self.model}")
            text, tool_calls, usage = await chat_completion(
                model=self.model,
                messages=self._history,
                tools=tool_defs if tool_defs else None,
                bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
            )
            self._total_tokens += usage.get("total_tokens", 0)

            # No tool calls → we have our final answer
            if not tool_calls:
                self._history.append(Message(role=Role.ASSISTANT, content=text))
                self.trace.result(self.id, f"Final response ({self._iterations} iterations)")
                return text

            # Record assistant message with tool calls
            self._history.append(Message(
                role=Role.ASSISTANT,
                content=text,
                tool_calls=tool_calls,
            ))

            # Execute tool calls
            for tc in tool_calls:
                self.trace.action(self.id, f"Tool: {tc.name}({_summarize_args(tc.arguments)})")
                tool = scoped_tools.get(tc.name)
                if tool is None:
                    result_text = f"Unknown tool: {tc.name}"
                    self.trace.error(self.id, result_text)
                else:
                    result = await tool.run(tc.id, tc.arguments)
                    result_text = result.output
                    if not result.success:
                        self.trace.error(self.id, f"Tool {tc.name} failed: {result_text[:200]}")

                self._history.append(Message(
                    role=Role.TOOL,
                    content=result_text,
                    tool_call_id=tc.id,
                    name=tc.name,
                ))

        # Hit max iterations
        self.trace.error(self.id, f"Max iterations ({max_iter}) reached")
        return f"Reached maximum iterations ({max_iter}). Best result:\n" + (
            self._last_assistant_text() or "(no output)"
        )

    def _last_assistant_text(self) -> str | None:
        for msg in reversed(self._history):
            if msg.role == Role.ASSISTANT and msg.content:
                return msg.content
        return None


def _summarize_args(args: dict[str, Any], max_len: int = 80) -> str:
    """Short summary of tool arguments for trace logging."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 30:
            s = s[:27] + "..."
        parts.append(f"{k}={s}")
    summary = ", ".join(parts)
    return summary[:max_len]
