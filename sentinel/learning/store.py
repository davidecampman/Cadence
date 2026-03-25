"""Cross-session learning — tracks strategy effectiveness across sessions."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class OutcomeRating(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class StrategyRecord(BaseModel):
    """Records a strategy used to accomplish a task and its outcome."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    task_type: str          # e.g., "code_generation", "research", "debugging"
    task_description: str
    strategy: str           # Description of the approach taken
    tools_used: list[str] = Field(default_factory=list)
    model_used: str = ""
    role_used: str = ""
    outcome: OutcomeRating = OutcomeRating.SUCCESS
    iterations_used: int = 0
    tokens_used: int = 0
    duration_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LearningInsight(BaseModel):
    """An insight derived from analyzing strategy outcomes."""
    task_type: str
    recommendation: str
    confidence: float       # 0.0 to 1.0
    based_on_count: int     # Number of records this insight is based on
    avg_success_rate: float
    preferred_tools: list[str] = Field(default_factory=list)
    preferred_model: str = ""


class LearningStore:
    """SQLite-backed store for cross-session learning data.

    Tracks which strategies, tools, and models work best for different
    task types, enabling the system to improve planning over time.
    """

    def __init__(self, db_path: str = "./data/learning.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                task_description TEXT NOT NULL,
                strategy TEXT NOT NULL,
                tools_used TEXT NOT NULL DEFAULT '[]',
                model_used TEXT NOT NULL DEFAULT '',
                role_used TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT 'success',
                iterations_used INTEGER NOT NULL DEFAULT 0,
                tokens_used INTEGER NOT NULL DEFAULT 0,
                duration_ms REAL NOT NULL DEFAULT 0.0,
                errors TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_strategies_task_type ON strategies(task_type);
            CREATE INDEX IF NOT EXISTS idx_strategies_outcome ON strategies(outcome);
            CREATE INDEX IF NOT EXISTS idx_strategies_created ON strategies(created_at);
        """)
        conn.commit()

    def record(self, strategy: StrategyRecord) -> StrategyRecord:
        """Record a strategy and its outcome."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO strategies
               (id, session_id, task_type, task_description, strategy,
                tools_used, model_used, role_used, outcome, iterations_used,
                tokens_used, duration_ms, errors, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                strategy.id,
                strategy.session_id,
                strategy.task_type,
                strategy.task_description,
                strategy.strategy,
                json.dumps(strategy.tools_used),
                strategy.model_used,
                strategy.role_used,
                strategy.outcome.value,
                strategy.iterations_used,
                strategy.tokens_used,
                strategy.duration_ms,
                json.dumps(strategy.errors),
                strategy.created_at,
                json.dumps(strategy.metadata),
            ),
        )
        conn.commit()
        return strategy

    def get_insights(self, task_type: str, limit: int = 5) -> list[LearningInsight]:
        """Analyze past strategies for a task type and return recommendations."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT task_type, strategy, tools_used, model_used, outcome,
                      iterations_used, tokens_used
               FROM strategies
               WHERE task_type = ?
               ORDER BY created_at DESC
               LIMIT 100""",
            (task_type,),
        ).fetchall()

        if not rows:
            return []

        # Aggregate statistics
        total = len(rows)
        successes = sum(1 for r in rows if r["outcome"] == "success")
        success_rate = successes / total if total > 0 else 0.0

        # Count tool usage in successful strategies
        tool_counts: dict[str, int] = {}
        model_counts: dict[str, int] = {}
        strategy_success: dict[str, tuple[int, int]] = {}  # strategy -> (successes, total)

        for row in rows:
            tools = json.loads(row["tools_used"])
            is_success = row["outcome"] == "success"

            for tool in tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + (1 if is_success else 0)

            model = row["model_used"]
            if model:
                model_counts[model] = model_counts.get(model, 0) + (1 if is_success else 0)

            strat = row["strategy"][:100]  # Truncate for grouping
            s, t = strategy_success.get(strat, (0, 0))
            strategy_success[strat] = (s + (1 if is_success else 0), t + 1)

        # Build insights
        insights: list[LearningInsight] = []

        # Best tools
        preferred_tools = sorted(
            tool_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Best model
        preferred_model = max(model_counts.items(), key=lambda x: x[1])[0] if model_counts else ""

        # Top strategies
        sorted_strategies = sorted(
            strategy_success.items(),
            key=lambda x: x[1][0] / max(x[1][1], 1),
            reverse=True,
        )

        for strat, (s, t) in sorted_strategies[:limit]:
            strat_rate = s / t if t > 0 else 0.0
            insights.append(LearningInsight(
                task_type=task_type,
                recommendation=f"Strategy: {strat}",
                confidence=min(1.0, t / 10),  # More data = higher confidence
                based_on_count=t,
                avg_success_rate=strat_rate,
                preferred_tools=[t[0] for t in preferred_tools],
                preferred_model=preferred_model,
            ))

        return insights

    def get_best_tools(self, task_type: str) -> list[tuple[str, float]]:
        """Return tools ranked by success rate for a given task type."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT tools_used, outcome FROM strategies
               WHERE task_type = ?
               ORDER BY created_at DESC LIMIT 100""",
            (task_type,),
        ).fetchall()

        tool_stats: dict[str, tuple[int, int]] = {}  # tool -> (successes, total)
        for row in rows:
            tools = json.loads(row["tools_used"])
            is_success = row["outcome"] == "success"
            for tool in tools:
                s, t = tool_stats.get(tool, (0, 0))
                tool_stats[tool] = (s + (1 if is_success else 0), t + 1)

        ranked = [
            (tool, s / t if t > 0 else 0.0)
            for tool, (s, t) in tool_stats.items()
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def classify_task(self, description: str) -> str:
        """Simple heuristic task type classification.

        Returns a task_type string based on keywords in the description.
        """
        desc_lower = description.lower()
        if any(kw in desc_lower for kw in ("debug", "fix", "bug", "error", "issue")):
            return "debugging"
        if any(kw in desc_lower for kw in ("write", "create", "implement", "add", "build")):
            return "code_generation"
        if any(kw in desc_lower for kw in ("refactor", "clean", "optimize", "improve")):
            return "refactoring"
        if any(kw in desc_lower for kw in ("test", "verify", "check", "validate")):
            return "testing"
        if any(kw in desc_lower for kw in ("research", "find", "search", "look up", "investigate")):
            return "research"
        if any(kw in desc_lower for kw in ("review", "audit", "analyze", "assess")):
            return "review"
        if any(kw in desc_lower for kw in ("explain", "document", "describe")):
            return "documentation"
        return "general"

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate learning statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) as c FROM strategies").fetchone()["c"]
        by_type = conn.execute(
            """SELECT task_type, COUNT(*) as c,
                      SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes
               FROM strategies GROUP BY task_type"""
        ).fetchall()

        return {
            "total_strategies": total,
            "by_task_type": {
                row["task_type"]: {
                    "total": row["c"],
                    "successes": row["successes"],
                    "success_rate": round(row["successes"] / row["c"], 3) if row["c"] > 0 else 0,
                }
                for row in by_type
            },
        }
