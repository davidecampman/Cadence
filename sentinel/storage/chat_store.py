"""SQLite-backed persistent storage for chats and session history."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from threading import Lock

from pydantic import BaseModel


class ChatMessageRecord(BaseModel):
    id: str
    chat_id: str
    role: str  # "user" | "agent"
    content: str
    timestamp: float
    duration_ms: float | None = None
    trace_steps: list[dict] | None = None


class ChatRecord(BaseModel):
    id: str
    title: str
    session_id: str | None = None
    created_at: float
    updated_at: float
    messages: list[ChatMessageRecord] = []


class ChatStore:
    """Persistent chat storage using SQLite.

    Stores chats, messages, and server-side session history so they survive
    server restarts and are accessible across devices.
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            data_dir = Path("./data")
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "chats.db")
        self._db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New Chat',
                    session_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    duration_ms REAL,
                    trace_steps TEXT,
                    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_chat_id
                    ON chat_messages(chat_id);
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                    ON chat_messages(timestamp);

                CREATE TABLE IF NOT EXISTS session_history (
                    session_id TEXT PRIMARY KEY,
                    history TEXT NOT NULL DEFAULT '[]',
                    summary TEXT DEFAULT '',
                    updated_at REAL NOT NULL
                );
            """)

    # ---- Chat CRUD ----

    def list_chats(self) -> list[ChatRecord]:
        """Return all chats (without messages) ordered by most recent first."""
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chats ORDER BY updated_at DESC"
            ).fetchall()
            return [
                ChatRecord(
                    id=r["id"],
                    title=r["title"],
                    session_id=r["session_id"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]

    def get_chat(self, chat_id: str) -> ChatRecord | None:
        """Return a chat with all its messages."""
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM chats WHERE id = ?", (chat_id,)
            ).fetchone()
            if not row:
                return None

            msg_rows = conn.execute(
                "SELECT * FROM chat_messages WHERE chat_id = ? ORDER BY timestamp ASC",
                (chat_id,),
            ).fetchall()

            messages = [
                ChatMessageRecord(
                    id=m["id"],
                    chat_id=m["chat_id"],
                    role=m["role"],
                    content=m["content"],
                    timestamp=m["timestamp"],
                    duration_ms=m["duration_ms"],
                    trace_steps=json.loads(m["trace_steps"]) if m["trace_steps"] else None,
                )
                for m in msg_rows
            ]

            return ChatRecord(
                id=row["id"],
                title=row["title"],
                session_id=row["session_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                messages=messages,
            )

    def create_chat(
        self,
        chat_id: str | None = None,
        title: str = "New Chat",
        created_at: float | None = None,
    ) -> ChatRecord:
        """Create a new empty chat."""
        now = created_at or time.time()
        cid = chat_id or str(uuid.uuid4())
        with self._lock, self._get_conn() as conn:
            conn.execute(
                "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (cid, title, now, now),
            )
        return ChatRecord(id=cid, title=title, created_at=now, updated_at=now)

    def update_chat(
        self,
        chat_id: str,
        title: str | None = None,
        session_id: str | None = None,
    ) -> ChatRecord | None:
        """Update chat metadata (title, session_id)."""
        with self._lock, self._get_conn() as conn:
            row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if not row:
                return None

            new_title = title if title is not None else row["title"]
            new_session = session_id if session_id is not None else row["session_id"]
            now = time.time()

            conn.execute(
                "UPDATE chats SET title = ?, session_id = ?, updated_at = ? WHERE id = ?",
                (new_title, new_session, now, chat_id),
            )

            return ChatRecord(
                id=chat_id,
                title=new_title,
                session_id=new_session,
                created_at=row["created_at"],
                updated_at=now,
            )

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and all its messages. Returns True if found."""
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            return cursor.rowcount > 0

    # ---- Messages ----

    def add_message(self, msg: ChatMessageRecord) -> None:
        """Append a message to a chat."""
        trace_json = json.dumps(msg.trace_steps) if msg.trace_steps else None
        now = time.time()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """INSERT INTO chat_messages
                   (id, chat_id, role, content, timestamp, duration_ms, trace_steps)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (msg.id, msg.chat_id, msg.role, msg.content,
                 msg.timestamp, msg.duration_ms, trace_json),
            )
            conn.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?",
                (now, msg.chat_id),
            )

    # ---- Session history (server-side context for LLM) ----

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        """Load conversation history for a session."""
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT history FROM session_history WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                return json.loads(row["history"])
            return []

    def save_session_history(
        self,
        session_id: str,
        history: list[dict[str, str]],
        summary: str = "",
    ) -> None:
        """Persist conversation history for a session."""
        now = time.time()
        history_json = json.dumps(history)
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """INSERT INTO session_history (session_id, history, summary, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     history = excluded.history,
                     summary = excluded.summary,
                     updated_at = excluded.updated_at""",
                (session_id, history_json, summary, now),
            )

    def get_session_summary(self, session_id: str) -> str:
        """Load compressed summary for a session."""
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT summary FROM session_history WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["summary"] if row and row["summary"] else ""

    def save_session_summary(self, session_id: str, summary: str) -> None:
        """Update just the summary for a session."""
        now = time.time()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """INSERT INTO session_history (session_id, history, summary, updated_at)
                   VALUES (?, '[]', ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     summary = excluded.summary,
                     updated_at = excluded.updated_at""",
                (session_id, summary, now),
            )
