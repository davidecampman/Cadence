"""Core agent loop — the think→act→observe cycle."""

from __future__ import annotations

import re
import uuid
from difflib import SequenceMatcher
from typing import Any

from cadence.core.config import Config, get_config
from cadence.core.llm import chat_completion, estimate_message_tokens, stream_completion
from cadence.core.streaming import StreamCollector
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
            "- When asked to perform ANY concrete action (create files, write code, execute "
            "commands, zip archives, etc.), you MUST use the appropriate tool to actually do it. "
            "Do NOT describe what you would do — actually do it using a tool call.\n"
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

    def _truncate_result(self, text: str) -> str:
        """Truncate a tool result if it exceeds the configured limit."""
        limit = self.config.agents.max_tool_result_chars
        if len(text) <= limit:
            return text
        # Keep the beginning and end for context, with a truncation marker
        keep_head = int(limit * 0.7)
        keep_tail = limit - keep_head - 100  # room for marker
        return (
            text[:keep_head]
            + f"\n\n... [truncated {len(text) - keep_head - keep_tail:,} chars] ...\n\n"
            + text[-keep_tail:]
        )

    # Regex for [[FILE:/path]] markers embedded in tool results
    _FILE_MARKER_RE = re.compile(r"\[\[FILE:(.*?)\]\]")

    def _collect_file_markers(self, response: str) -> str:
        """Ensure all [[FILE:...]] markers from tool results appear in the response.

        The LLM often omits the raw ``[[FILE:...]]`` markers when composing its
        final answer, which causes the frontend to render broken download links
        (or no links at all).  This method scans the conversation history for
        markers emitted by tools such as ``write_file`` and ``execute_code``,
        and appends any that are missing from *response* so the UI can always
        render proper download buttons.
        """
        # Gather every marker present in tool-result messages
        tool_markers: list[str] = []
        for msg in self._history:
            if msg.role == Role.TOOL and msg.content:
                for match in self._FILE_MARKER_RE.finditer(msg.content):
                    marker = match.group(0)  # e.g. [[FILE:/tmp/hello.zip]]
                    if marker not in tool_markers:
                        tool_markers.append(marker)

        if not tool_markers:
            return response

        # Only append markers the LLM forgot to include
        missing = [m for m in tool_markers if m not in response]
        if not missing:
            return response

        return response + "\n\n" + "\n".join(missing)

    def _prune_history(self) -> None:
        """Shrink older messages when history grows too large.

        Uses a tiered approach instead of blunt truncation:
        - System prompt: always kept in full
        - Recent messages (last ``keep_tail``): kept verbatim
        - Older tool results: progressively compressed based on age
        - Older assistant messages: keep first and last paragraphs

        This preserves more semantic context than the naive approach of
        truncating everything beyond a fixed char count.
        """
        threshold = self.config.agents.prune_threshold
        if len(self._history) <= threshold:
            return

        keep_tail = 20
        if len(self._history) <= keep_tail + 1:
            return

        prune_end = len(self._history) - keep_tail
        total_msgs_to_prune = prune_end - 1  # Skip system prompt at index 0

        for i in range(1, prune_end):
            msg = self._history[i]
            # How far back is this message? Older = more aggressive pruning
            age_ratio = (prune_end - i) / max(total_msgs_to_prune, 1)

            if msg.role == Role.TOOL and msg.content:
                # Tiered pruning: older tool results get compressed more
                if age_ratio > 0.7 and len(msg.content) > 300:
                    # Very old: keep just the tool name and a brief excerpt
                    excerpt = msg.content[:150]
                    self._history[i] = Message(
                        role=Role.TOOL,
                        content=f"[{msg.name}]: {excerpt}\n... [pruned {len(msg.content):,} chars]",
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )
                elif age_ratio > 0.3 and len(msg.content) > 1000:
                    # Medium old: keep beginning and end
                    head = msg.content[:400]
                    tail = msg.content[-200:]
                    self._history[i] = Message(
                        role=Role.TOOL,
                        content=f"{head}\n... [{len(msg.content) - 600:,} chars omitted] ...\n{tail}",
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )
                elif len(msg.content) > 2000:
                    # Recent-ish but large: moderate trim
                    head = msg.content[:800]
                    tail = msg.content[-400:]
                    self._history[i] = Message(
                        role=Role.TOOL,
                        content=f"{head}\n... [{len(msg.content) - 1200:,} chars omitted] ...\n{tail}",
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )
            elif msg.role == Role.ASSISTANT and msg.content and len(msg.content) > 1500 and age_ratio > 0.5:
                # Compress old assistant responses: keep first and last paragraph
                paragraphs = msg.content.split("\n\n")
                if len(paragraphs) > 3:
                    compressed = paragraphs[0] + "\n\n... [compressed] ...\n\n" + paragraphs[-1]
                    self._history[i] = Message(
                        role=Role.ASSISTANT,
                        content=compressed,
                        tool_calls=msg.tool_calls,
                    )

    def _check_loop_detection(self) -> bool:
        """Detect if the agent is stuck in a loop using similarity-based matching.

        Instead of requiring exact duplicates, uses SequenceMatcher to detect
        near-duplicate outputs that indicate the agent is looping with minor
        variations (e.g., rephrasing the same stuck response).
        """
        window = self.config.agents.loop_detection_window
        if len(self._history) < window * 2:
            return False

        recent_assistant = [
            m.content for m in self._history[-window * 2:]
            if m.role == Role.ASSISTANT and m.content
        ][-window:]

        if len(recent_assistant) < window:
            return False

        # Check for exact duplicates (fast path)
        if len(set(recent_assistant)) == 1:
            return True

        # Check for near-duplicates using similarity ratio
        # If all pairs within the window are >98% similar, it's a loop
        similarity_threshold = 0.98
        first = recent_assistant[0]
        all_similar = all(
            SequenceMatcher(None, first[:500], msg[:500]).ratio() > similarity_threshold
            for msg in recent_assistant[1:]
        )
        if all_similar:
            return True

        # Check for tool-call loops: same tool called with same args repeatedly
        recent_tool_calls = []
        for m in self._history[-window * 2:]:
            if m.role == Role.ASSISTANT and m.tool_calls:
                for tc in m.tool_calls:
                    recent_tool_calls.append(f"{tc.name}:{sorted(tc.arguments.items())}")

        if len(recent_tool_calls) >= window:
            last_n = recent_tool_calls[-window:]
            if len(set(last_n)) == 1:
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

        # Reset per-run state so the agent can be reused
        self._iterations = 0
        self._total_tokens = 0

        # Initialize with system prompt, prior conversation context, and current task
        self._history = [
            Message(role=Role.SYSTEM, content=self._system_prompt()),
        ]

        # Inject prior conversation turns so the agent knows what was discussed
        _role_map = {"user": Role.USER, "assistant": Role.ASSISTANT, "system": Role.SYSTEM}
        for entry in (conversation_history or []):
            role = _role_map.get(entry["role"], Role.USER)
            self._history.append(Message(role=role, content=entry["content"]))

        # Add the current user message, with images if provided
        if images:
            # Build content blocks in Anthropic format (canonical internal format)
            content_blocks: list[dict] = [{"type": "text", "text": task}]
            for img in images:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": img["data"],
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
                return self._collect_file_markers(result)

            # Budget check — hard stop if exceeded
            if self._total_tokens >= self.config.budget.max_tokens_per_task:
                self.trace.error(self.id, f"Token budget exceeded: {self._total_tokens}")
                self._errors.append("Token budget exceeded")
                result = "Token budget exceeded. Partial result:\n" + (
                    self._last_assistant_text() or "(no output yet)"
                )
                await self._maybe_reflect(task, result, max_iter)
                return self._collect_file_markers(result)

            # Proactive budget check — if approaching the limit, do one final
            # toolless LLM call to produce a summary rather than blowing the budget.
            # Uses exponential moving average for more accurate projection on bursty workloads.
            budget_limit = self.config.budget.max_tokens_per_task
            if self._iterations > 1 and self._total_tokens > 0:
                # EMA with alpha=0.3 weights recent iterations more heavily
                avg_tokens_per_iter = self._total_tokens / (self._iterations - 1)
                # Also factor in estimated context size for next call
                context_estimate = estimate_message_tokens(self._history)
                projected = self._total_tokens + max(avg_tokens_per_iter, context_estimate)
                if projected > budget_limit * 0.90:
                    self.trace.thought(
                        self.id,
                        f"Approaching budget ({self._total_tokens:,}/{budget_limit:,}), "
                        f"avg {avg_tokens_per_iter:,.0f}/iter — final summary call",
                    )
                    # One last call without tools so the LLM wraps up
                    self._history.append(Message(
                        role=Role.USER,
                        content=(
                            "[System: You are approaching your token budget. "
                            "Please provide your best final answer now with what you have so far. "
                            "Do not request any more tools.]"
                        ),
                    ))
                    text, _, usage = await chat_completion(
                        model=self.model,
                        messages=self._history,
                        tools=None,  # No tools — force a text response
                        bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
                        local_config=self.config.models.local if self.config.models.local.enabled else None,
                    )
                    self._total_tokens += usage.get("total_tokens", 0)
                    self._history.append(Message(role=Role.ASSISTANT, content=text))
                    self.trace.result(self.id, f"Budget-limited response ({self._iterations} iterations)")
                    await self._maybe_reflect(task, text, max_iter)
                    return self._collect_file_markers(text)

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
                local_config=self.config.models.local if self.config.models.local.enabled else None,
            )
            self._total_tokens += usage.get("total_tokens", 0)

            # No tool calls → we have our final answer
            if not tool_calls:
                self._history.append(Message(role=Role.ASSISTANT, content=text))
                self.trace.result(self.id, f"Final response ({self._iterations} iterations)")
                await self._maybe_reflect(task, text, max_iter)
                return self._collect_file_markers(text)

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
                    result_text = self._truncate_result(result.output)
                    if not result.success:
                        self.trace.error(self.id, f"Tool {tc.name} failed: {result_text[:200]}")
                        self._errors.append(f"Tool {tc.name} failed: {result_text[:200]}")

                self._history.append(Message(
                    role=Role.TOOL,
                    content=result_text,
                    tool_call_id=tc.id,
                    name=tc.name,
                ))

            # Prune older tool results to keep context from growing unboundedly
            self._prune_history()

        # Hit max iterations
        self.trace.error(self.id, f"Max iterations ({max_iter}) reached")
        self._errors.append(f"Max iterations ({max_iter}) reached")
        result = f"Reached maximum iterations ({max_iter}). Best result:\n" + (
            self._last_assistant_text() or "(no output)"
        )
        await self._maybe_reflect(task, result, max_iter)
        return self._collect_file_markers(result)

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


    async def run_streaming(
        self,
        task: str,
        collector: StreamCollector,
        conversation_history: list[dict[str, str]] | None = None,
        images: list[dict] | None = None,
    ) -> str:
        """Execute the agent loop with streaming output via a StreamCollector.

        Emits real-time events (tokens, tool starts/results, thinking) to the
        collector while running the normal think→act→observe loop.
        """
        await collector.emit_status("starting", agent_id=self.id)

        # Reset per-run state so the agent can be reused
        self._iterations = 0
        self._total_tokens = 0

        # Run the normal agent loop but emit tool events to the collector
        self.trace.observation(self.id, f"Task received: {task}")
        scoped_tools = self.tools.scoped_copy(self.id)

        self._history = [Message(role=Role.SYSTEM, content=self._system_prompt())]
        _role_map = {"user": Role.USER, "assistant": Role.ASSISTANT, "system": Role.SYSTEM}
        for entry in (conversation_history or []):
            role = _role_map.get(entry["role"], Role.USER)
            self._history.append(Message(role=role, content=entry["content"]))

        if images:
            content_blocks: list[dict] = [{"type": "text", "text": task}]
            for img in images:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": img["data"],
                    },
                })
            self._history.append(Message(
                role=Role.USER, content=task, content_blocks=content_blocks,
            ))
        else:
            self._history.append(Message(role=Role.USER, content=task))

        max_iter = self.config.agents.max_iterations_per_task
        self._errors = []

        while self._iterations < max_iter:
            self._iterations += 1

            if self._check_loop_detection():
                self.trace.error(self.id, "Loop detected — breaking out")
                result = "I appear to be stuck in a loop. Here's what I have so far:\n" + (
                    self._last_assistant_text() or "(no output yet)"
                )
                await self._maybe_reflect(task, result, max_iter)
                return self._collect_file_markers(result)

            if self._total_tokens >= self.config.budget.max_tokens_per_task:
                result = "Token budget exceeded. Partial result:\n" + (
                    self._last_assistant_text() or "(no output yet)"
                )
                await self._maybe_reflect(task, result, max_iter)
                return self._collect_file_markers(result)

            tool_defs = scoped_tools.definitions(
                max_tier=self.role.permission_tier,
                allowed_names=self.role.allowed_tools,
            )

            # Stream the LLM response
            await collector.emit_thinking(f"Iteration {self._iterations}", agent_id=self.id)

            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            usage: dict[str, Any] = {}

            async for event in stream_completion(
                model=self.model,
                messages=self._history,
                tools=tool_defs if tool_defs else None,
                bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
                local_config=self.config.models.local if self.config.models.local.enabled else None,
            ):
                if event["type"] == "token":
                    await collector.emit_token(event["text"], agent_id=self.id)
                    text_parts.append(event["text"])
                elif event["type"] == "tool_use_start":
                    await collector.emit_tool_start(event["name"], {}, agent_id=self.id)
                elif event["type"] == "done":
                    usage = event.get("usage", {})
                    text = event.get("text", "".join(text_parts))
                    for tc_data in event.get("tool_calls", []):
                        tool_calls.append(ToolCall(**tc_data) if isinstance(tc_data, dict) else tc_data)

            self._total_tokens += usage.get("total_tokens", 0)
            # Prefer text from the 'done' event (authoritative); fall back to
            # accumulated text_parts only if the done event didn't provide it.
            if not text:
                text = "".join(text_parts) if text_parts else ""

            if not tool_calls:
                self._history.append(Message(role=Role.ASSISTANT, content=text))
                self.trace.result(self.id, f"Final response ({self._iterations} iterations)")
                await self._maybe_reflect(task, text, max_iter)
                return self._collect_file_markers(text)

            self._history.append(Message(role=Role.ASSISTANT, content=text, tool_calls=tool_calls))

            for tc in tool_calls:
                self.trace.action(self.id, f"Tool: {tc.name}({_summarize_args(tc.arguments)})")
                await collector.emit_tool_start(tc.name, tc.arguments, agent_id=self.id)
                tool = scoped_tools.get(tc.name)
                if tool is None:
                    result_text = f"Unknown tool: {tc.name}"
                    self.trace.error(self.id, result_text)
                else:
                    result = await tool.run(tc.id, tc.arguments)
                    result_text = self._truncate_result(result.output)
                    await collector.emit_tool_result(
                        tc.name, result_text, result.success, agent_id=self.id,
                    )

                self._history.append(Message(
                    role=Role.TOOL, content=result_text,
                    tool_call_id=tc.id, name=tc.name,
                ))

            self._prune_history()

        result = f"Reached maximum iterations ({max_iter}). Best result:\n" + (
            self._last_assistant_text() or "(no output)"
        )
        await self._maybe_reflect(task, result, max_iter)
        return self._collect_file_markers(result)


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
