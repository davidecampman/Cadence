"""Tools for cross-session learning insights."""

from __future__ import annotations

from typing import Any

from cadence.core.types import PermissionTier
from cadence.learning.store import LearningStore
from cadence.tools.base import Tool


class LearningInsightsTool(Tool):
    """Query cross-session learning insights for a task type."""

    name = "learning_insights"
    description = (
        "Get insights from past task executions. Shows which strategies, tools, "
        "and models have worked best for similar tasks. Use this before planning "
        "complex tasks to leverage past experience."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "description": (
                    "Task type to query: 'code_generation', 'debugging', 'research', "
                    "'refactoring', 'testing', 'review', 'documentation', 'general'"
                ),
            },
        },
        "required": ["task_type"],
    }
    permission_tier = PermissionTier.READ_ONLY

    def __init__(self, store: LearningStore):
        self._store = store

    async def execute(self, task_type: str) -> str:
        insights = self._store.get_insights(task_type)
        if not insights:
            return f"No learning data yet for task type '{task_type}'."

        lines = [f"Insights for '{task_type}':"]
        for i, insight in enumerate(insights, 1):
            lines.append(
                f"  {i}. {insight.recommendation}\n"
                f"     Success rate: {insight.avg_success_rate:.0%} "
                f"(based on {insight.based_on_count} tasks, "
                f"confidence: {insight.confidence:.0%})"
            )
            if insight.preferred_tools:
                lines.append(f"     Best tools: {', '.join(insight.preferred_tools[:3])}")
            if insight.preferred_model:
                lines.append(f"     Best model: {insight.preferred_model}")

        return "\n".join(lines)


class LearningStatsTool(Tool):
    """Get aggregate statistics from cross-session learning."""

    name = "learning_stats"
    description = "Get aggregate statistics about past task strategies and outcomes."
    parameters = {
        "type": "object",
        "properties": {},
    }
    permission_tier = PermissionTier.READ_ONLY

    def __init__(self, store: LearningStore):
        self._store = store

    async def execute(self) -> str:
        stats = self._store.get_stats()
        lines = [f"Learning Stats — {stats['total_strategies']} total strategies recorded"]
        for task_type, data in stats.get("by_task_type", {}).items():
            lines.append(
                f"  {task_type}: {data['total']} tasks, "
                f"{data['successes']} successes ({data['success_rate']:.0%})"
            )
        return "\n".join(lines) if lines else "No learning data recorded yet."
