"""SQLite-backed memory store — zero external dependencies, cross-platform.

Uses Python's built-in sqlite3 module with a simple TF-IDF-like keyword
matching strategy for retrieval.  This trades the vector-similarity precision
of ChromaDB for universal portability: it works identically on macOS, Windows,
and Linux with no native extensions to compile.

The schema stores full MemoryEntry metadata and uses SQLite FTS5 (full-text
search) for fast keyword retrieval, combined with the same time-decay and
importance scoring used by the ChromaDB backend.
"""

from __future__ import annotations

import math
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_one.core.config import get_config
from agent_one.memory.paths import get_memory_dir


class MemoryEntry(BaseModel):
    """A single memory stored in the database."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    namespace: str = "shared"
    importance: float = 0.5
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_agent: str | None = None


class MemoryResult(BaseModel):
    """A memory retrieval result with computed relevance score."""

    entry: MemoryEntry
    similarity: float
    relevance: float


class SQLiteMemoryStore:
    """Cross-platform memory store backed by SQLite + FTS5.

    Design decisions:
        - One SQLite file per namespace keeps data isolated and easy to
          export/backup individually.
        - FTS5 provides fast keyword search with BM25 ranking which acts as
          our "similarity" proxy (no embedding model needed).
        - The same decay formula (exp(-rate * days)) is applied on top of BM25
          rank so behaviour is consistent with the ChromaDB backend.
        - Thread-safe: uses ``check_same_thread=False`` with WAL journal mode.
    """

    def __init__(self, db_dir: str | Path | None = None):
        self._config = get_config().memory
        self._db_dir = Path(db_dir) if db_dir else get_memory_dir()
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._connections: dict[str, sqlite3.Connection] = {}

    # -- connection management ------------------------------------------------

    def _db_path(self, namespace: str) -> Path:
        safe_name = namespace.replace(":", "_").replace("/", "_")
        return self._db_dir / f"{safe_name}.db"

    def _get_conn(self, namespace: str) -> sqlite3.Connection:
        if namespace not in self._connections:
            db_path = self._db_path(namespace)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema(conn)
            self._connections[namespace] = conn
        return self._connections[namespace]

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id            TEXT PRIMARY KEY,
                content       TEXT NOT NULL,
                importance    REAL NOT NULL DEFAULT 0.5,
                created_at    REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0,
                source_agent  TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, content=memories, content_rowid=rowid);

            -- Triggers to keep FTS index in sync
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO memories_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;
        """)

    # -- public API (mirrors MemoryStore interface) ---------------------------

    async def save(
        self,
        content: str,
        namespace: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        source_agent: str | None = None,
    ) -> MemoryEntry:
        """Store a memory entry."""
        import json

        ns = namespace or self._config.default_namespace
        entry = MemoryEntry(
            content=content,
            namespace=ns,
            importance=max(0.0, min(1.0, importance)),
            metadata=metadata or {},
            source_agent=source_agent,
        )

        conn = self._get_conn(ns)
        conn.execute(
            """INSERT INTO memories
               (id, content, importance, created_at, last_accessed, access_count,
                source_agent, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.content,
                entry.importance,
                entry.created_at,
                entry.last_accessed,
                entry.access_count,
                entry.source_agent or "",
                json.dumps(entry.metadata),
            ),
        )
        conn.commit()
        return entry

    async def query(
        self,
        query: str,
        namespace: str | None = None,
        max_results: int | None = None,
        min_relevance: float | None = None,
    ) -> list[MemoryResult]:
        """Search memory using FTS5 BM25 ranking + time-decay scoring."""
        import json

        ns = namespace or self._config.default_namespace
        n = max_results or self._config.max_results
        # SQLite uses keyword matching, not vector similarity, so a lower
        # threshold is appropriate (vector cosine 0.7 ≈ keyword 0.3).
        threshold = min_relevance or min(self._config.similarity_threshold, 0.4)

        conn = self._get_conn(ns)

        # FTS5 match query — bm25() returns negative scores (lower = better)
        rows = conn.execute(
            """SELECT m.id, m.content, m.importance, m.created_at, m.last_accessed,
                      m.access_count, m.source_agent, m.metadata_json,
                      bm25(memories_fts) AS rank
               FROM memories_fts f
               JOIN memories m ON f.rowid = m.rowid
               WHERE memories_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (self._fts_query(query), n * 3),
        ).fetchall()

        if not rows:
            # Fallback: case-insensitive substring search when FTS match fails
            rows = conn.execute(
                """SELECT id, content, importance, created_at, last_accessed,
                          access_count, source_agent, metadata_json,
                          -1.0 AS rank
                   FROM memories
                   WHERE content LIKE ? COLLATE NOCASE
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (f"%{query}%", n * 3),
            ).fetchall()
            is_fallback = True
        else:
            is_fallback = False

        if not rows:
            return []

        now = time.time()
        decay_rate = self._config.decay_rate

        # BM25 scores from FTS5 are negative (more negative = better match).
        # We convert to a 0..1 similarity where 1.0 = best match.
        # For keyword search, any FTS match is a strong signal, so we use a
        # high baseline (0.7) and spread the range above it.
        ranks = [abs(r[8]) for r in rows]
        max_rank = max(ranks) if ranks else 1.0
        min_rank = min(ranks) if ranks else 0.0
        rank_range = max_rank - min_rank if max_rank > min_rank else 1.0

        results: list[MemoryResult] = []
        for row in rows:
            if is_fallback:
                # Substring matches get a fixed similarity of 0.85
                similarity = 0.85
            elif len(rows) == 1:
                similarity = 1.0
            else:
                raw_rank = abs(row[8])
                # Lower rank = better match → higher similarity
                normalized = (max_rank - raw_rank) / rank_range
                similarity = 0.7 + 0.3 * normalized

            importance = row[2]
            created_at = row[3]
            age_days = (now - created_at) / 86400
            decay_factor = math.exp(-decay_rate * age_days)

            relevance = similarity * importance * decay_factor
            if relevance < threshold:
                continue

            meta = json.loads(row[7]) if row[7] else {}
            entry = MemoryEntry(
                id=row[0],
                content=row[1],
                namespace=ns,
                importance=importance,
                created_at=created_at,
                last_accessed=row[4],
                access_count=row[5],
                source_agent=row[6] or None,
                metadata=meta,
            )
            results.append(MemoryResult(
                entry=entry,
                similarity=similarity,
                relevance=relevance,
            ))

        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:n]

    async def delete(self, memory_id: str, namespace: str | None = None) -> bool:
        """Delete a specific memory by ID."""
        ns = namespace or self._config.default_namespace
        conn = self._get_conn(ns)
        try:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            return False

    async def list_namespaces(self) -> list[str]:
        """List all memory namespaces (one per .db file)."""
        dbs = self._db_dir.glob("*.db")
        return [p.stem.replace("_", ":") for p in dbs]

    async def count(self, namespace: str | None = None) -> int:
        """Return the number of memories in a namespace."""
        ns = namespace or self._config.default_namespace
        conn = self._get_conn(ns)
        row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    async def get_all(self, namespace: str | None = None) -> list[MemoryEntry]:
        """Return every memory in a namespace (for export)."""
        import json

        ns = namespace or self._config.default_namespace
        conn = self._get_conn(ns)
        rows = conn.execute(
            """SELECT id, content, importance, created_at, last_accessed,
                      access_count, source_agent, metadata_json
               FROM memories ORDER BY created_at"""
        ).fetchall()

        entries = []
        for row in rows:
            entries.append(MemoryEntry(
                id=row[0],
                content=row[1],
                namespace=ns,
                importance=row[2],
                created_at=row[3],
                last_accessed=row[4],
                access_count=row[5],
                source_agent=row[6] or None,
                metadata=json.loads(row[7]) if row[7] else {},
            ))
        return entries

    def close(self) -> None:
        """Close all database connections."""
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _fts_query(query: str) -> str:
        """Convert a natural-language query to an FTS5 query string.

        Splits on whitespace and joins with OR so partial matches still work.
        Tokens are quoted to escape FTS5 special characters.
        """
        tokens = query.split()
        if not tokens:
            return '""'
        escaped = [f'"{t}"' for t in tokens]
        return " OR ".join(escaped)
