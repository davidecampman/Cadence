"""Smart model routing — pick the right model for the job.

Uses a cost-quality tradeoff function instead of hard-coded task classification.
Tracks per-model cost, latency percentiles, and success rates for adaptive routing.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

from cadence.core.config import Config, get_config
from cadence.core.llm import LLMError, chat_completion, _classify_error, estimate_message_tokens
from cadence.core.types import Message, Role


# Per-model pricing (USD per 1M tokens, input/output)
# These are approximate; users can override via config in the future.
_MODEL_COSTS: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
    # OpenAI
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    # Defaults for unknown models
    "_default": {"input": 3.0, "output": 15.0},
}


def _get_cost(model: str) -> dict[str, float]:
    """Look up pricing for a model, using prefix matching."""
    model_lower = model.lower()
    for prefix, cost in _MODEL_COSTS.items():
        if prefix != "_default" and model_lower.startswith(prefix):
            return cost
    return _MODEL_COSTS["_default"]


class ModelStats:
    """Track per-model success rates, latency percentiles, and token usage."""

    def __init__(self):
        self._stats: dict[str, dict[str, Any]] = {}
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._max_latency_samples = 100  # Rolling window

    def record(self, model: str, success: bool, latency_ms: float, tokens: int):
        if model not in self._stats:
            self._stats[model] = {
                "total": 0, "successes": 0,
                "total_latency_ms": 0.0, "total_tokens": 0,
                "total_cost_usd": 0.0,
            }
        s = self._stats[model]
        s["total"] += 1
        if success:
            s["successes"] += 1
        s["total_latency_ms"] += latency_ms
        s["total_tokens"] += tokens

        # Track cost
        cost = _get_cost(model)
        # Rough split: assume 60% input, 40% output
        estimated_cost = (tokens * 0.6 * cost["input"] + tokens * 0.4 * cost["output"]) / 1_000_000
        s["total_cost_usd"] += estimated_cost

        # Track latency distribution
        lat_list = self._latencies[model]
        lat_list.append(latency_ms)
        if len(lat_list) > self._max_latency_samples:
            self._latencies[model] = lat_list[-self._max_latency_samples:]

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

    def p95_latency(self, model: str) -> float:
        """Return the 95th percentile latency for a model."""
        lat_list = self._latencies.get(model, [])
        if not lat_list:
            return 0.0
        sorted_lat = sorted(lat_list)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def total_cost(self, model: str) -> float:
        s = self._stats.get(model)
        return s.get("total_cost_usd", 0.0) if s else 0.0

    def total_tokens(self, model: str) -> int:
        s = self._stats.get(model)
        return s.get("total_tokens", 0) if s else 0

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return a summary of all model stats for diagnostics."""
        result = {}
        for model, s in self._stats.items():
            result[model] = {
                "total_calls": s["total"],
                "success_rate": self.success_rate(model),
                "avg_latency_ms": self.avg_latency(model),
                "p95_latency_ms": self.p95_latency(model),
                "total_tokens": s["total_tokens"],
                "total_cost_usd": round(s.get("total_cost_usd", 0), 4),
            }
        return result


class SmartRouter:
    """Routes requests to the appropriate model tier.

    Uses a cost-quality tradeoff function instead of hard-coded task lists:
    - Scores each candidate model: score = (success_rate * quality) / (cost + latency_weight * latency)
    - Considers context length: small contexts can use fast models even for complex tasks
    - Error-aware fallback: transient errors → retry with backoff, permanent → skip to next
    - Tracks per-model statistics for adaptive routing over time
    """

    # Task types that are inherently simple (can use fast model)
    _FAST_HINTS = {"plan", "classify", "memory_query", "evaluate", "summarize", "simple"}
    # Task types that need strong reasoning
    _STRONG_HINTS = {"code", "reason", "analyze", "create", "debug", "complex"}

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.stats = ModelStats()

    def _quality_weight(self, task_type: str) -> float:
        """Return a quality weight (0-1) indicating how much quality matters for this task.

        Higher = favor strong model. Lower = fast model is fine.
        """
        if task_type in self._FAST_HINTS:
            return 0.3
        if task_type in self._STRONG_HINTS:
            return 0.9
        return 0.6

    def _score_model(
        self,
        model: str,
        task_type: str,
        context_tokens: int = 0,
        latency_weight: float = 0.001,
    ) -> float:
        """Score a model for a given task using cost-quality tradeoff.

        score = (success_rate * quality_weight) / (normalized_cost + latency_factor)

        Higher score = better candidate.
        """
        success = self.stats.success_rate(model)
        quality = self._quality_weight(task_type)

        # Cost per 1K tokens
        cost_info = _get_cost(model)
        cost_per_1k = (cost_info["input"] + cost_info["output"]) / 2 / 1000

        # Latency factor (use p95 to avoid tail latency issues)
        p95 = self.stats.p95_latency(model) or 1000  # default 1s
        latency_factor = latency_weight * p95

        # Context bonus: for small contexts, fast models are more efficient
        context_bonus = 0.0
        if context_tokens < 2000:
            context_bonus = 0.2  # Bonus for using a fast model on small context

        # For strong models, quality matters more
        is_strong = model == self.config.models.strong
        quality_multiplier = quality if is_strong else (1.0 - quality * 0.5)

        # Avoid division by zero
        denominator = max(cost_per_1k + latency_factor, 0.001)
        score = (success * quality_multiplier + context_bonus) / denominator

        return score

    def select_model(self, task_type: str, context_tokens: int = 0) -> str:
        """Select the best model for a given task type and context size."""
        fast = self.config.models.fast
        strong = self.config.models.strong

        # Hard rules: if fast model has very low success rate, always use strong
        fast_rate = self.stats.success_rate(fast)
        if fast_rate < 0.5 and self.stats._stats.get(fast, {}).get("total", 0) >= 5:
            return strong

        # Score both models
        score_fast = self._score_model(fast, task_type, context_tokens)
        score_strong = self._score_model(strong, task_type, context_tokens)

        return fast if score_fast >= score_strong else strong

    async def completion_with_fallback(
        self,
        messages: list[Message],
        task_type: str = "default",
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, list, dict[str, Any]]:
        """Call a model with error-aware automatic fallback.

        Error handling strategy:
        - Transient errors (rate limit, timeout): retry same model with backoff
        - Permanent errors (auth, invalid): skip immediately to fallback chain
        """
        context_tokens = estimate_message_tokens(messages, tools)
        primary = self.select_model(task_type, context_tokens)
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
                    local_config=self.config.models.local if self.config.models.local.enabled else None,
                )
                latency = (time.time() - start) * 1000
                self.stats.record(model, True, latency, usage.get("total_tokens", 0))
                return text, tool_calls, usage
            except LLMError as e:
                latency = (time.time() - start) * 1000
                self.stats.record(model, False, latency, 0)
                last_error = e
                if e.transient:
                    # Already retried internally by chat_completion, move to fallback
                    continue
                else:
                    # Permanent error — skip to next model immediately
                    continue
            except Exception as e:
                latency = (time.time() - start) * 1000
                self.stats.record(model, False, latency, 0)
                last_error = e
                continue

        raise RuntimeError(f"All models failed. Last error: {last_error}")
