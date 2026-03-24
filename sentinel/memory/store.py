"""Memory system with namespaces, importance scoring, and time decay."""

from __future__ import annotations

import math
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from sentinel.core.config import get_config


class MemoryEntry(BaseModel):
    """A single memory stored in the vector database."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    namespace: str = "shared"
    importance: float = 0.5         # 0.0 (trivial) to 1.0 (critical)
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_agent: str | None = None


class MemoryResult(BaseModel):
    """A memory retrieval result with computed relevance score."""
    entry: MemoryEntry
    similarity: float           # Raw cosine similarity
    relevance: float            # Final score after decay + importance


class MemoryStore:
    """ChromaDB-backed memory with time decay and importance scoring.

    Relevance formula:
        relevance = similarity * importance * decay_factor

    Where:
        decay_factor = exp(-decay_rate * days_since_creation)

    Namespaces isolate memories:
        - "shared"    → all agents can access
        - "project:X" → project-specific
        - "agent:X"   → private to a specific agent
    """

    def __init__(self):
        self._config = get_config().memory
        self._client = None
        self._collections: dict[str, Any] = {}

    def _get_client(self):
        if self._client is None:
            import chromadb
            from pathlib import Path

            persist_dir = Path(self._config.persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        return self._client

    def _get_collection(self, namespace: str):
        if namespace not in self._collections:
            client = self._get_client()
            # Sanitize namespace for ChromaDB collection name
            safe_name = namespace.replace(":", "_").replace("/", "_")
            self._collections[namespace] = client.get_or_create_collection(
                name=safe_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[namespace]

    async def save(
        self,
        content: str,
        namespace: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        source_agent: str | None = None,
    ) -> MemoryEntry:
        """Store a memory entry."""
        ns = namespace or self._config.default_namespace
        entry = MemoryEntry(
            content=content,
            namespace=ns,
            importance=max(0.0, min(1.0, importance)),
            metadata=metadata or {},
            source_agent=source_agent,
        )

        collection = self._get_collection(ns)
        collection.add(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[{
                "importance": entry.importance,
                "created_at": entry.created_at,
                "source_agent": entry.source_agent or "",
                **entry.metadata,
            }],
        )
        return entry

    async def query(
        self,
        query: str,
        namespace: str | None = None,
        max_results: int | None = None,
        min_relevance: float | None = None,
    ) -> list[MemoryResult]:
        """Search memory with time-decay-weighted relevance scoring."""
        ns = namespace or self._config.default_namespace
        n = max_results or self._config.max_results
        threshold = min_relevance or self._config.similarity_threshold

        collection = self._get_collection(ns)

        # ChromaDB returns distances (lower = more similar for cosine)
        results = collection.query(
            query_texts=[query],
            n_results=min(n * 2, collection.count() or 1),  # Over-fetch for post-filtering
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        now = time.time()
        decay_rate = self._config.decay_rate
        memory_results = []

        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            similarity = max(0.0, 1.0 - distance)  # Convert distance to similarity

            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            content = results["documents"][0][i] if results["documents"] else ""

            importance = meta.get("importance", 0.5)
            created_at = meta.get("created_at", now)

            # Time decay: exponential decay based on age in days
            age_days = (now - created_at) / 86400
            decay_factor = math.exp(-decay_rate * age_days)

            relevance = similarity * importance * decay_factor

            if relevance < threshold:
                continue

            entry = MemoryEntry(
                id=doc_id,
                content=content,
                namespace=ns,
                importance=importance,
                created_at=created_at,
                metadata={k: v for k, v in meta.items()
                          if k not in ("importance", "created_at", "source_agent")},
                source_agent=meta.get("source_agent") or None,
            )
            memory_results.append(MemoryResult(
                entry=entry,
                similarity=similarity,
                relevance=relevance,
            ))

        # Sort by relevance descending, take top N
        memory_results.sort(key=lambda r: r.relevance, reverse=True)
        return memory_results[:n]

    async def delete(self, memory_id: str, namespace: str | None = None) -> bool:
        """Delete a specific memory by ID."""
        ns = namespace or self._config.default_namespace
        collection = self._get_collection(ns)
        try:
            collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    async def list_namespaces(self) -> list[str]:
        """List all memory namespaces."""
        client = self._get_client()
        collections = client.list_collections()
        return [c.name for c in collections]

    async def get_all(self, namespace: str | None = None) -> list[MemoryEntry]:
        """Return every memory in a namespace (for export/migration)."""
        ns = namespace or self._config.default_namespace
        collection = self._get_collection(ns)
        count = collection.count()
        if count == 0:
            return []

        results = collection.get(limit=count)
        entries = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            content = results["documents"][i] if results["documents"] else ""
            entries.append(MemoryEntry(
                id=doc_id,
                content=content,
                namespace=ns,
                importance=meta.get("importance", 0.5),
                created_at=meta.get("created_at", 0.0),
                metadata={k: v for k, v in meta.items()
                          if k not in ("importance", "created_at", "source_agent")},
                source_agent=meta.get("source_agent") or None,
            ))
        return entries
