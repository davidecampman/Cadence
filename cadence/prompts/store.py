"""SQLite-backed persistence for prompt modifications with versioning and rollback."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from enum import Enum
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, Field


class ModificationType(str, Enum):
    """What kind of prompt modification this is."""
    APPEND = "append"          # Add new instruction/context
    REPLACE = "replace"        # Replace an existing section
    STRATEGY = "strategy"      # High-level behavioral adjustment
    CONSTRAINT = "constraint"  # Add a new constraint or guardrail
    REMOVE = "remove"          # Remove a previously added modification


class PromptModification(BaseModel):
    """A single tracked change to an agent's prompt."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    role_name: str                          # Which agent role this applies to
    modification_type: ModificationType
    content: str                            # The actual modification text
    reasoning: str = ""                     # Why this modification was made
    source_task: str = ""                   # Task that triggered the reflection
    performance_score: float = 0.0          # 0.0-1.0 how well the task went
    version: int = 1                        # Auto-incremented per role
    active: bool = True                     # Whether this modification is currently applied
    created_at: float = Field(default_factory=time.time)
    metadata: dict = Field(default_factory=dict)


class PromptEvolutionStore:
    """SQLite-backed store for prompt modifications with versioning.

    Each agent role has its own modification history. Modifications are
    versioned and can be activated/deactivated or rolled back.
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            data_dir = Path("./data")
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "prompt_evolution.db")
        self._db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS prompt_modifications (
                        id TEXT PRIMARY KEY,
                        role_name TEXT NOT NULL,
                        modification_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        reasoning TEXT DEFAULT '',
                        source_task TEXT DEFAULT '',
                        performance_score REAL DEFAULT 0.0,
                        version INTEGER NOT NULL,
                        active INTEGER DEFAULT 1,
                        created_at REAL NOT NULL,
                        metadata TEXT DEFAULT '{}'
                    );

                    CREATE INDEX IF NOT EXISTS idx_modifications_role
                        ON prompt_modifications(role_name, active);

                    CREATE INDEX IF NOT EXISTS idx_modifications_version
                        ON prompt_modifications(role_name, version);
                """)
                conn.commit()
            finally:
                conn.close()

    def save(self, modification: PromptModification) -> PromptModification:
        """Save a new prompt modification. Auto-assigns the next version number."""
        with self._lock:
            conn = self._get_conn()
            try:
                # Get next version number for this role
                row = conn.execute(
                    "SELECT MAX(version) as max_v FROM prompt_modifications WHERE role_name = ?",
                    (modification.role_name,),
                ).fetchone()
                next_version = (row["max_v"] or 0) + 1
                modification.version = next_version

                conn.execute(
                    """INSERT INTO prompt_modifications
                       (id, role_name, modification_type, content, reasoning,
                        source_task, performance_score, version, active, created_at, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        modification.id,
                        modification.role_name,
                        modification.modification_type.value,
                        modification.content,
                        modification.reasoning,
                        modification.source_task,
                        modification.performance_score,
                        modification.version,
                        1 if modification.active else 0,
                        modification.created_at,
                        json.dumps(modification.metadata),
                    ),
                )
                conn.commit()
                return modification
            finally:
                conn.close()

    def get_active(self, role_name: str) -> list[PromptModification]:
        """Get all active modifications for a role, ordered by version."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM prompt_modifications
                       WHERE role_name = ? AND active = 1
                       ORDER BY version ASC""",
                    (role_name,),
                ).fetchall()
                return [self._row_to_modification(r) for r in rows]
            finally:
                conn.close()

    def get_history(self, role_name: str, limit: int = 50) -> list[PromptModification]:
        """Get full modification history for a role (active and inactive)."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM prompt_modifications
                       WHERE role_name = ?
                       ORDER BY version DESC
                       LIMIT ?""",
                    (role_name, limit),
                ).fetchall()
                return [self._row_to_modification(r) for r in rows]
            finally:
                conn.close()

    def deactivate(self, modification_id: str) -> bool:
        """Deactivate a specific modification (soft delete)."""
        with self._lock:
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "UPDATE prompt_modifications SET active = 0 WHERE id = ?",
                    (modification_id,),
                )
                conn.commit()
                return result.rowcount > 0
            finally:
                conn.close()

    def rollback_to_version(self, role_name: str, version: int) -> int:
        """Deactivate all modifications after the given version. Returns count deactivated."""
        with self._lock:
            conn = self._get_conn()
            try:
                result = conn.execute(
                    """UPDATE prompt_modifications
                       SET active = 0
                       WHERE role_name = ? AND version > ? AND active = 1""",
                    (role_name, version),
                )
                conn.commit()
                return result.rowcount
            finally:
                conn.close()

    def clear_role(self, role_name: str) -> int:
        """Deactivate all modifications for a role. Returns count deactivated."""
        with self._lock:
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "UPDATE prompt_modifications SET active = 0 WHERE role_name = ? AND active = 1",
                    (role_name,),
                )
                conn.commit()
                return result.rowcount
            finally:
                conn.close()

    def get_by_id(self, modification_id: str) -> PromptModification | None:
        """Fetch a single modification by ID."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM prompt_modifications WHERE id = ?",
                    (modification_id,),
                ).fetchone()
                return self._row_to_modification(row) if row else None
            finally:
                conn.close()

    def reactivate(self, modification_id: str) -> bool:
        """Re-enable a previously deactivated modification."""
        with self._lock:
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "UPDATE prompt_modifications SET active = 1 WHERE id = ?",
                    (modification_id,),
                )
                conn.commit()
                return result.rowcount > 0
            finally:
                conn.close()

    @staticmethod
    def _row_to_modification(row: sqlite3.Row) -> PromptModification:
        return PromptModification(
            id=row["id"],
            role_name=row["role_name"],
            modification_type=ModificationType(row["modification_type"]),
            content=row["content"],
            reasoning=row["reasoning"],
            source_task=row["source_task"],
            performance_score=row["performance_score"],
            version=row["version"],
            active=bool(row["active"]),
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
