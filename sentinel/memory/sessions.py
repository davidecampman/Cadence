"""Session persistence — conversation history survives restarts.

Stores conversation sessions in SQLite so users can resume where they left
off.  Each session tracks its message history, active task context, and
metadata.  Works identically on macOS, Windows, and Linux.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sentinel.memory.paths import get_sessions_dir


class SessionMessage(BaseModel):
    """A single message within a persisted session."""

    role: str
    content: str
    name: str | None = None
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """A complete persisted conversation session."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    messages: list[SessionMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    active_task_ids: list[str] = Field(default_factory=list)


class SessionStore:
    """SQLite-backed session persistence.

    Schema:
        sessions: id, title, created_at, updated_at, metadata_json
        messages: session_id, role, content, name, timestamp, metadata_json

    One shared SQLite file for all sessions, kept in the platform-appropriate
    data directory.
    """

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = get_sessions_dir() / "sessions.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                active_task_ids_json TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                name        TEXT,
                timestamp   REAL NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, timestamp);
        """)

    async def create_session(
        self,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session."""
        session = Session(title=title, metadata=metadata or {})
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO sessions (id, title, created_at, updated_at,
                                     metadata_json, active_task_ids_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session.id,
                session.title,
                session.created_at,
                session.updated_at,
                json.dumps(session.metadata),
                json.dumps(session.active_task_ids),
            ),
        )
        conn.commit()
        return session

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        """Append a message to a session."""
        msg = SessionMessage(
            role=role,
            content=content,
            name=name,
            metadata=metadata or {},
        )
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO messages (session_id, role, content, name, timestamp, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, msg.role, msg.content, msg.name, msg.timestamp, json.dumps(msg.metadata)),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        conn.commit()
        return msg

    async def get_session(self, session_id: str) -> Session | None:
        """Load a session with all its messages."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, title, created_at, updated_at, metadata_json, active_task_ids_json FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None

        msg_rows = conn.execute(
            "SELECT role, content, name, timestamp, metadata_json FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()

        messages = [
            SessionMessage(
                role=r[0],
                content=r[1],
                name=r[2],
                timestamp=r[3],
                metadata=json.loads(r[4]) if r[4] else {},
            )
            for r in msg_rows
        ]

        return Session(
            id=row[0],
            title=row[1],
            created_at=row[2],
            updated_at=row[3],
            messages=messages,
            metadata=json.loads(row[4]) if row[4] else {},
            active_task_ids=json.loads(row[5]) if row[5] else [],
        )

    async def list_sessions(self, limit: int = 20) -> list[Session]:
        """List recent sessions (without messages, for performance)."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, title, created_at, updated_at, metadata_json, active_task_ids_json
               FROM sessions ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()

        return [
            Session(
                id=r[0],
                title=r[1],
                created_at=r[2],
                updated_at=r[3],
                metadata=json.loads(r[4]) if r[4] else {},
                active_task_ids=json.loads(r[5]) if r[5] else [],
            )
            for r in rows
        ]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0

    async def update_task_ids(self, session_id: str, task_ids: list[str]) -> None:
        """Update the active task IDs for a session."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET active_task_ids_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(task_ids), time.time(), session_id),
        )
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
