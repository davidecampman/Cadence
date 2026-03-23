"""Smart model routing — pick the right model for the job."""

from __future__ import annotations

import time
from typing import Any

from agent_one.core.config import Config, get_config
from agent_one.core.llm import chat_completion
from agent_one.core.types import Message, Role


class ModelStats:
    """Track per-model success rates and latency."""

    def __init__(self):
        self._stats: dict[str, dict[str, Any]] = {}

    def record(self, model: str, success: bool, latency_ms: float, tokens: int):
        if model not in self._stats:
            self._stats[model] = {
                "total": 0, "successes": 0,
                "total_latency_ms": 0.0, "total_tokens": 0,
            }
        s = self._stats[model]
        s["total"] += 1
        if success:
            s["successes"] += 1
        s["total_latency_ms"] += latency_ms
        s["total_tokens"] += tokens

    def success_rate(self, model: str) -> float:
        s = self._stats.get(model)
        if not s or s["total"] == 0:
            return 1.0  # Assume good until proven otherwise
        return s["successes"] / s["total"]

    def avg_latency(self, model: str) -> float:
        s = self._stats.get(model)
        if not s or s["total"] == 0:
            return 0.0
        return s["total_latency_ms"] / s["total"]


class SmartRouter:
    """Routes requests to the appropriate model tier.

    Tiers:
      - fast:   Planning, memory queries, classification, simple Q&A
      - strong: Complex reasoning, code generation, multi-step analysis

    Also handles fallback: if the primary model fails, try the fallback chain.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.stats = ModelStats()

    def select_model(self, task_type: str) -> str:
        """Select the best model for a given task type."""
        fast_tasks = {"plan", "classify", "memory_query", "evaluate", "summarize", "simple"}
        strong_tasks = {"code", "reason", "analyze", "create", "debug", "complex"}

        if task_type in fast_tasks:
            model = self.config.models.fast
        elif task_type in strong_tasks:
            model = self.config.models.strong
        else:
            # Default: use strong if success rate of fast is below threshold
            fast_rate = self.stats.success_rate(self.config.models.fast)
            model = self.config.models.fast if fast_rate > 0.8 else self.config.models.strong

        return model

    async def completion_with_fallback(
        self,
        messages: list[Message],
        task_type: str = "default",
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, list, dict[str, Any]]:
        """Call a model with automatic fallback on failure."""
        primary = self.select_model(task_type)
        models_to_try = [primary] + [
            m for m in self.config.models.fallback_chain if m != primary
        ]

        last_error = None
        for model in models_to_try:
            start = time.time()
            try:
                text, tool_calls, usage = await chat_completion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
                )
                latency = (time.time() - start) * 1000
                self.stats.record(model, True, latency, usage.get("total_tokens", 0))
                return text, tool_calls, usage
            except Exception as e:
                latency = (time.time() - start) * 1000
                self.stats.record(model, False, latency, 0)
                last_error = e
                continue

        raise RuntimeError(f"All models failed. Last error: {last_error}")
