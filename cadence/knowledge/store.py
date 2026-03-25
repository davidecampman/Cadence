"""Knowledge base store — indexes document chunks in ChromaDB for semantic search."""

from __future__ import annotations

import math
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from cadence.core.config import get_config


class DocumentRecord(BaseModel):
    """Metadata for an ingested document."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    source: str  # "pdf", "docx", "email", "web", "text"
    origin: str = ""  # URL, file path, or email subject
    chunk_count: int = 0
    ingested_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    """A single chunk stored in the vector DB."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A search hit from the knowledge base."""
    chunk: ChunkRecord
    document: DocumentRecord | None = None
    similarity: float
    relevance: float


class KnowledgeStore:
    """ChromaDB-backed knowledge base with document chunking and semantic search.

    Uses a dedicated ``knowledge`` ChromaDB collection, separate from the
    agent memory system, so knowledge persists independently of agent sessions.
    """

    COLLECTION_NAME = "cadence_knowledge"
    DOCUMENTS_COLLECTION = "cadence_kb_documents"

    def __init__(self):
        self._config = get_config().memory
        self._client = None
        self._collection = None
        self._doc_collection = None

    def _get_client(self):
        if self._client is None:
            import chromadb
            from pathlib import Path

            persist_dir = Path(self._config.persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_doc_collection(self):
        if self._doc_collection is None:
            client = self._get_client()
            self._doc_collection = client.get_or_create_collection(
                name=self.DOCUMENTS_COLLECTION,
            )
        return self._doc_collection

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Split text into overlapping chunks by character count.

        Tries to break on paragraph/sentence boundaries when possible.
        """
        if not text or not text.strip():
            return []

        chunks: list[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + chunk_size, length)

            # If we can fit the rest in one chunk, take it all
            if end >= length:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Try to break at paragraph boundary
            para_break = text.rfind("\n\n", start + chunk_size // 2, end)
            if para_break != -1:
                end = para_break + 2
            else:
                # Try sentence boundary (period followed by space or newline)
                for sep in (". ", ".\n", "? ", "!\n", "! ", "?\n"):
                    sent_break = text.rfind(sep, start + chunk_size // 2, end)
                    if sent_break != -1:
                        end = sent_break + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move forward, accounting for overlap
            start = max(start + 1, end - overlap)

        return chunks

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest(
        self,
        title: str,
        content: str,
        source: str,
        origin: str = "",
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> DocumentRecord:
        """Ingest a document: chunk it and store all chunks in the vector DB."""
        doc = DocumentRecord(
            title=title,
            source=source,
            origin=origin,
            metadata=metadata or {},
        )

        chunks = self.chunk_text(content, chunk_size=chunk_size, overlap=chunk_overlap)
        doc.chunk_count = len(chunks)

        if not chunks:
            return doc

        collection = self._get_collection()

        chunk_ids = []
        chunk_docs = []
        chunk_metas = []

        for i, chunk_text in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            chunk_docs.append(chunk_text)
            chunk_metas.append({
                "document_id": doc.id,
                "document_title": title,
                "source": source,
                "origin": origin,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingested_at": doc.ingested_at,
                **(metadata or {}),
            })

        collection.add(ids=chunk_ids, documents=chunk_docs, metadatas=chunk_metas)

        # Store document record in the documents collection
        doc_collection = self._get_doc_collection()
        doc_collection.add(
            ids=[doc.id],
            documents=[title],
            metadatas=[{
                "title": title,
                "source": source,
                "origin": origin,
                "chunk_count": doc.chunk_count,
                "ingested_at": doc.ingested_at,
                **(metadata or {}),
            }],
        )

        return doc

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        max_results: int = 5,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """Semantic search across all ingested documents."""
        collection = self._get_collection()

        count = collection.count()
        if count == 0:
            return []

        where_filter = None
        if source_filter:
            where_filter = {"source": source_filter}

        results = collection.query(
            query_texts=[query],
            n_results=min(max_results * 2, count),
            where=where_filter,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        now = time.time()
        decay_rate = self._config.decay_rate
        search_results: list[SearchResult] = []

        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            similarity = max(0.0, 1.0 - distance)

            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            content = results["documents"][0][i] if results["documents"] else ""

            ingested_at = meta.get("ingested_at", now)
            age_days = (now - ingested_at) / 86400
            decay_factor = math.exp(-decay_rate * age_days)
            relevance = similarity * decay_factor

            chunk = ChunkRecord(
                id=chunk_id,
                document_id=meta.get("document_id", ""),
                content=content,
                chunk_index=meta.get("chunk_index", 0),
                metadata={k: v for k, v in meta.items()
                          if k not in ("document_id", "chunk_index", "ingested_at")},
            )

            doc = DocumentRecord(
                id=meta.get("document_id", ""),
                title=meta.get("document_title", ""),
                source=meta.get("source", ""),
                origin=meta.get("origin", ""),
                chunk_count=meta.get("total_chunks", 0),
                ingested_at=ingested_at,
            )

            search_results.append(SearchResult(
                chunk=chunk,
                document=doc,
                similarity=similarity,
                relevance=relevance,
            ))

        search_results.sort(key=lambda r: r.relevance, reverse=True)
        return search_results[:max_results]

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def list_documents(self) -> list[DocumentRecord]:
        """List all ingested documents."""
        doc_collection = self._get_doc_collection()
        count = doc_collection.count()
        if count == 0:
            return []

        results = doc_collection.get(limit=count)
        docs = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            docs.append(DocumentRecord(
                id=doc_id,
                title=meta.get("title", ""),
                source=meta.get("source", ""),
                origin=meta.get("origin", ""),
                chunk_count=meta.get("chunk_count", 0),
                ingested_at=meta.get("ingested_at", 0),
            ))
        docs.sort(key=lambda d: d.ingested_at, reverse=True)
        return docs

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        collection = self._get_collection()
        doc_collection = self._get_doc_collection()

        try:
            # Delete chunks belonging to this document
            collection.delete(where={"document_id": document_id})
            # Delete the document record
            doc_collection.delete(ids=[document_id])
            return True
        except Exception:
            return False

    async def stats(self) -> dict[str, Any]:
        """Return knowledge base statistics."""
        collection = self._get_collection()
        doc_collection = self._get_doc_collection()
        return {
            "total_chunks": collection.count(),
            "total_documents": doc_collection.count(),
        }
