"""Memory tools — let agents save, query, and manage memories."""

from __future__ import annotations

from sentinel.core.types import PermissionTier
from sentinel.memory.store import MemoryStore
from sentinel.tools.base import Tool


def _resolve_namespace(namespace: str | None, agent_id: str | None) -> str:
    """Determine the effective namespace.

    If an explicit namespace is provided, use it directly.
    If the agent has an ID and no explicit namespace is given, scope to ``agent:<id>``.
    Falls back to ``"shared"``.
    """
    if namespace is not None:
        return namespace
    if agent_id:
        return f"agent:{agent_id}"
    return "shared"


class MemorySaveTool(Tool):
    name = "memory_save"
    description = (
        "Save information to long-term memory for later retrieval. "
        "Use for facts, decisions, user preferences, or task results worth remembering. "
        "Omit 'namespace' to save to your private agent memory, or set 'shared' for cross-agent access."
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
                "description": (
                    "Memory namespace. Omit to use your private agent namespace. "
                    "Use 'shared' for memories accessible to all agents."
                ),
            },
        },
        "required": ["content"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(self, store: MemoryStore, agent_id: str | None = None):
        self._store = store
        self._agent_id = agent_id

    def with_agent_id(self, agent_id: str) -> "MemorySaveTool":
        """Return a copy of this tool scoped to a specific agent."""
        return MemorySaveTool(store=self._store, agent_id=agent_id)

    async def execute(self, content: str, importance: float = 0.5, namespace: str | None = None) -> str:
        ns = _resolve_namespace(namespace, self._agent_id)
        entry = await self._store.save(content=content, importance=importance, namespace=ns)
        return f"Saved memory [{entry.id[:8]}] (importance: {importance}) in namespace '{ns}'"


class MemoryQueryTool(Tool):
    name = "memory_query"
    description = (
        "Search long-term memory for relevant information. "
        "Returns memories ranked by relevance (similarity * importance * recency). "
        "Omit 'namespace' to search your private agent memory, or set 'shared' for cross-agent memories."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "namespace": {
                "type": "string",
                "description": (
                    "Namespace to search. Omit to search your private agent namespace. "
                    "Use 'shared' for cross-agent memories."
                ),
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

    def __init__(self, store: MemoryStore, agent_id: str | None = None):
        self._store = store
        self._agent_id = agent_id

    def with_agent_id(self, agent_id: str) -> "MemoryQueryTool":
        """Return a copy of this tool scoped to a specific agent."""
        return MemoryQueryTool(store=self._store, agent_id=agent_id)

    async def execute(self, query: str, namespace: str | None = None, max_results: int = 5) -> str:
        ns = _resolve_namespace(namespace, self._agent_id)
        results = await self._store.query(query=query, namespace=ns, max_results=max_results)
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
            "namespace": {
                "type": "string",
                "description": "Namespace containing the memory. Omit to use your private agent namespace.",
            },
        },
        "required": ["memory_id"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(self, store: MemoryStore, agent_id: str | None = None):
        self._store = store
        self._agent_id = agent_id

    def with_agent_id(self, agent_id: str) -> "MemoryDeleteTool":
        """Return a copy of this tool scoped to a specific agent."""
        return MemoryDeleteTool(store=self._store, agent_id=agent_id)

    async def execute(self, memory_id: str, namespace: str | None = None) -> str:
        ns = _resolve_namespace(namespace, self._agent_id)
        ok = await self._store.delete(memory_id, ns)
        return f"Deleted memory {memory_id}" if ok else f"Failed to delete memory {memory_id}"
