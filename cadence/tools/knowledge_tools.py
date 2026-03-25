"""Knowledge base tools — let agents ingest and search documents."""

from __future__ import annotations

from cadence.core.types import PermissionTier
from cadence.knowledge.store import KnowledgeStore
from cadence.tools.base import Tool


class KBIngestTool(Tool):
    name = "kb_ingest"
    description = (
        "Ingest a document into the knowledge base for later semantic search. "
        "Supports PDFs, DOCX files, emails (.eml), web pages (URLs), and plain text. "
        "The source_type is auto-detected from the path/URL if omitted."
    )
    parameters = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": (
                    "File path, URL, or raw text content to ingest. "
                    "For web pages, provide the full URL (https://...). "
                    "For files, provide the absolute path."
                ),
            },
            "title": {
                "type": "string",
                "description": "A descriptive title for this document.",
            },
            "source_type": {
                "type": "string",
                "enum": ["pdf", "docx", "email", "web", "text"],
                "description": (
                    "Document type. Auto-detected if omitted. "
                    "Use 'text' to ingest raw text content directly."
                ),
            },
        },
        "required": ["source", "title"],
    }
    permission_tier = PermissionTier.PRIVILEGED

    def __init__(self, store: KnowledgeStore):
        self._store = store

    async def execute(
        self, source: str, title: str, source_type: str | None = None
    ) -> str:
        from cadence.knowledge.parsers import PARSERS, detect_source_type

        stype = source_type or detect_source_type(source)
        parser = PARSERS.get(stype)
        if not parser:
            return f"Error: unsupported source type '{stype}'"

        try:
            text, metadata = parser(source)
        except ImportError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error parsing {stype}: {e}"

        if not text or not text.strip():
            return "Error: no text content could be extracted from the source."

        doc = await self._store.ingest(
            title=title,
            content=text,
            source=stype,
            origin=source if stype != "text" else "",
            metadata=metadata,
        )
        return (
            f"Ingested '{title}' ({stype}) — {doc.chunk_count} chunks indexed. "
            f"Document ID: {doc.id[:8]}"
        )


class KBSearchTool(Tool):
    name = "kb_search"
    description = (
        "Search the knowledge base for relevant information across all ingested "
        "documents (PDFs, emails, docs, web pages). Returns the most relevant "
        "text chunks ranked by semantic similarity."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 5,
            },
            "source_filter": {
                "type": "string",
                "enum": ["pdf", "docx", "email", "web", "text"],
                "description": "Optionally filter results to a specific document type.",
            },
        },
        "required": ["query"],
    }
    permission_tier = PermissionTier.READ_ONLY

    def __init__(self, store: KnowledgeStore):
        self._store = store

    async def execute(
        self, query: str, max_results: int = 5, source_filter: str | None = None
    ) -> str:
        results = await self._store.search(
            query=query, max_results=max_results, source_filter=source_filter
        )
        if not results:
            return "No relevant results found in the knowledge base."

        lines = []
        for r in results:
            doc_info = ""
            if r.document:
                doc_info = f" | source: {r.document.title} ({r.document.source})"
            lines.append(
                f"[{r.chunk.id[:8]}] (relevance: {r.relevance:.2f}){doc_info}\n"
                f"{r.chunk.content}"
            )
        return "\n\n---\n\n".join(lines)


class KBListTool(Tool):
    name = "kb_list"
    description = "List all documents in the knowledge base."
    parameters = {
        "type": "object",
        "properties": {},
    }
    permission_tier = PermissionTier.READ_ONLY

    def __init__(self, store: KnowledgeStore):
        self._store = store

    async def execute(self) -> str:
        docs = await self._store.list_documents()
        if not docs:
            return "Knowledge base is empty. Use kb_ingest to add documents."

        lines = []
        for doc in docs:
            lines.append(
                f"[{doc.id[:8]}] {doc.title} ({doc.source}) — "
                f"{doc.chunk_count} chunks, origin: {doc.origin or 'N/A'}"
            )
        return "\n".join(lines)


class KBDeleteTool(Tool):
    name = "kb_delete"
    description = "Delete a document from the knowledge base by its ID."
    parameters = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "The document ID (or first 8 chars) to delete.",
            },
        },
        "required": ["document_id"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(self, store: KnowledgeStore):
        self._store = store

    async def execute(self, document_id: str) -> str:
        # Support short IDs by finding the full ID
        docs = await self._store.list_documents()
        full_id = document_id
        for doc in docs:
            if doc.id.startswith(document_id):
                full_id = doc.id
                break

        ok = await self._store.delete_document(full_id)
        return (
            f"Deleted document {document_id}" if ok
            else f"Failed to delete document {document_id}"
        )
