"""Memory tools — let agents save, query, and manage memories."""

from __future__ import annotations

from sentinel.core.types import PermissionTier
from sentinel.memory.store import MemoryStore
from sentinel.tools.base import Tool


class MemorySaveTool(Tool):
    name = "memory_save"
    description = (
        "Save information to long-term memory for later retrieval. "
        "Use for facts, decisions, user preferences, or task results worth remembering."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember."},
            "importance": {
                "type": "number",
                "description": "How important is this? 0.0 (trivial) to 1.0 (critical).",
                "default": 0.5,
            },
            "namespace": {
                "type": "string",
                "description": "Memory namespace (e.g., 'shared', 'project:myapp').",
                "default": "shared",
            },
        },
        "required": ["content"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(self, store: MemoryStore):
        self._store = store

    async def execute(self, content: str, importance: float = 0.5, namespace: str = "shared") -> str:
        entry = await self._store.save(content=content, importance=importance, namespace=namespace)
        return f"Saved memory [{entry.id[:8]}] (importance: {importance}) in namespace '{namespace}'"


class MemoryQueryTool(Tool):
    name = "memory_query"
    description = (
        "Search long-term memory for relevant information. "
        "Returns memories ranked by relevance (similarity * importance * recency)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "namespace": {
                "type": "string",
                "description": "Namespace to search. Default searches 'shared'.",
                "default": "shared",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results.",
                "default": 5,
            },
        },
        "required": ["query"],
    }
    permission_tier = PermissionTier.READ_ONLY

    def __init__(self, store: MemoryStore):
        self._store = store

    async def execute(self, query: str, namespace: str = "shared", max_results: int = 5) -> str:
        results = await self._store.query(query=query, namespace=namespace, max_results=max_results)
        if not results:
            return "No relevant memories found."

        lines = []
        for r in results:
            lines.append(
                f"[{r.entry.id[:8]}] (relevance: {r.relevance:.2f}) "
                f"{r.entry.content}"
            )
        return "\n".join(lines)


class MemoryDeleteTool(Tool):
    name = "memory_delete"
    description = "Delete a specific memory by its ID."
    parameters = {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "The ID of the memory to delete."},
            "namespace": {"type": "string", "default": "shared"},
        },
        "required": ["memory_id"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(self, store: MemoryStore):
        self._store = store

    async def execute(self, memory_id: str, namespace: str = "shared") -> str:
        ok = await self._store.delete(memory_id, namespace)
        return f"Deleted memory {memory_id}" if ok else f"Failed to delete memory {memory_id}"
