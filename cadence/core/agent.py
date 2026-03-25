"""Core agent loop — the think→act→observe cycle."""

from __future__ import annotations

import uuid
from typing import Any

from cadence.core.config import Config, get_config
from cadence.core.llm import chat_completion
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
from cadence.prompts.evolution import PromptEvolver
from cadence.skills.loader import SkillLoader
from cadence.tools.base import Tool, ToolRegistry


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
        prompt_evolver: PromptEvolver | None = None,
    ):
        self.id = agent_id or f"{role.name}-{str(uuid.uuid4())[:6]}"
        self.role = role
        self.tools = tool_registry
        self.trace = trace
        self.config = config or get_config()
        self.parent_id = parent_id
        self.depth = depth
        self.skill_loader = skill_loader
        self.prompt_evolver = prompt_evolver

        self._history: list[Message] = []
        self._total_tokens: int = 0
        self._iterations: int = 0
        self._errors: list[str] = []

    @property
    def model(self) -> str:
        """Which model this agent uses."""
        return self.role.model_override or self.config.models.strong

    def _system_prompt(self) -> str:
        """Build the system prompt for this agent, including skills and evolved modifications."""
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

        # Apply evolved prompt modifications
        if self.prompt_evolver and self.config.prompt_evolution.enabled:
            prompt = self.prompt_evolver.build_evolved_prompt(self.role.name, prompt)

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

    async def run(
        self,
        task: str,
        conversation_history: list[dict[str, str]] | None = None,
        images: list[dict] | None = None,
    ) -> str:
        """Execute the agent loop on a task. Returns the final response."""
        self.trace.observation(self.id, f"Task received: {task}")

        # Scope memory tools to this agent's namespace
        scoped_tools = self.tools.scoped_copy(self.id)

        # Initialize with system prompt, prior conversation context, and current task
        self._history = [
            Message(role=Role.SYSTEM, content=self._system_prompt()),
        ]

        # Inject prior conversation turns so the agent knows what was discussed
        for entry in (conversation_history or []):
            role = Role.USER if entry["role"] == "user" else Role.ASSISTANT
            self._history.append(Message(role=role, content=entry["content"]))

        # Add the current user message, with images if provided
        if images:
            content_blocks: list[dict] = [{"type": "text", "text": task}]
            for img in images:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['media_type']};base64,{img['data']}",
                    },
                })
            self._history.append(Message(
                role=Role.USER,
                content=task,
                content_blocks=content_blocks,
            ))
        else:
            self._history.append(Message(role=Role.USER, content=task))

        max_iter = self.config.agents.max_iterations_per_task
        self._errors = []

        while self._iterations < max_iter:
            self._iterations += 1

            # Loop detection
            if self._check_loop_detection():
                self.trace.error(self.id, "Loop detected — breaking out")
                self._errors.append("Loop detected")
                result = "I appear to be stuck in a loop. Here's what I have so far:\n" + (
                    self._last_assistant_text() or "(no output yet)"
                )
                await self._maybe_reflect(task, result, max_iter)
                return result

            # Budget check
            if self._total_tokens >= self.config.budget.max_tokens_per_task:
                self.trace.error(self.id, f"Token budget exceeded: {self._total_tokens}")
                self._errors.append("Token budget exceeded")
                result = "Token budget exceeded. Partial result:\n" + (
                    self._last_assistant_text() or "(no output yet)"
                )
                await self._maybe_reflect(task, result, max_iter)
                return result

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
                await self._maybe_reflect(task, text, max_iter)
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
                    self._errors.append(result_text)
                else:
                    result = await tool.run(tc.id, tc.arguments)
                    result_text = result.output
                    if not result.success:
                        self.trace.error(self.id, f"Tool {tc.name} failed: {result_text[:200]}")
                        self._errors.append(f"Tool {tc.name} failed: {result_text[:200]}")

                self._history.append(Message(
                    role=Role.TOOL,
                    content=result_text,
                    tool_call_id=tc.id,
                    name=tc.name,
                ))

        # Hit max iterations
        self.trace.error(self.id, f"Max iterations ({max_iter}) reached")
        self._errors.append(f"Max iterations ({max_iter}) reached")
        result = f"Reached maximum iterations ({max_iter}). Best result:\n" + (
            self._last_assistant_text() or "(no output)"
        )
        await self._maybe_reflect(task, result, max_iter)
        return result

    async def _maybe_reflect(self, task: str, result: str, max_iter: int) -> None:
        """Trigger prompt self-reflection if evolution is enabled."""
        if not self.prompt_evolver:
            return
        if not self.config.prompt_evolution.enabled:
            return
        if not self.config.prompt_evolution.reflect_after_task:
            return

        try:
            modifications = await self.prompt_evolver.reflect_and_evolve(
                role_name=self.role.name,
                task_description=task[:500],
                task_result=result,
                iterations_used=self._iterations,
                max_iterations=max_iter,
                errors=self._errors if self._errors else None,
            )
            if modifications:
                mod_summary = ", ".join(
                    f"v{m.version}({m.modification_type.value})" for m in modifications
                )
                self.trace.thought(
                    self.id,
                    f"Prompt self-reflection added {len(modifications)} modification(s): {mod_summary}",
                )
        except Exception as e:
            # Reflection failures should never break task execution
            self.trace.error(self.id, f"Prompt reflection failed (non-fatal): {e}")

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
