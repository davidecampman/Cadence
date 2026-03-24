"""Scratchpad tools — ephemeral key-value store for sharing intermediate results between agents."""

from __future__ import annotations

import json
import time
from typing import Any

from sentinel.core.types import PermissionTier
from sentinel.tools.base import Tool


class ScratchpadStore:
    """In-memory key-value store with TTL support."""

    def __init__(self):
        self._data: dict[str, dict[str, Any]] = {}

    def write(self, key: str, value: str, ttl_seconds: int = 0) -> None:
        self._data[key] = {
            "value": value,
            "created_at": time.time(),
            "ttl": ttl_seconds,
        }

    def read(self, key: str) -> str | None:
        entry = self._data.get(key)
        if entry is None:
            return None

        # Check TTL
        if entry["ttl"] > 0:
            age = time.time() - entry["created_at"]
            if age > entry["ttl"]:
                del self._data[key]
                return None

        return entry["value"]

    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def keys(self) -> list[str]:
        # Prune expired entries
        now = time.time()
        expired = [
            k for k, v in self._data.items()
            if v["ttl"] > 0 and (now - v["created_at"]) > v["ttl"]
        ]
        for k in expired:
            del self._data[k]
        return list(self._data.keys())


# Shared singleton instance
_store = ScratchpadStore()


class ScratchWriteTool(Tool):
    name = "scratch_write"
    description = (
        "Write a value to the ephemeral scratchpad. Use for sharing intermediate "
        "results between agents or storing temporary data during multi-step tasks."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Key to store the value under."},
            "value": {"type": "string", "description": "Value to store (any text or JSON)."},
            "ttl_seconds": {
                "type": "integer",
                "description": "Time-to-live in seconds. 0 = no expiry.",
                "default": 0,
            },
        },
        "required": ["key", "value"],
    }
    permission_tier = PermissionTier.STANDARD

    async def execute(self, key: str, value: str, ttl_seconds: int = 0) -> str:
        _store.write(key, value, ttl_seconds)
        ttl_info = f" (TTL: {ttl_seconds}s)" if ttl_seconds > 0 else ""
        return f"Stored '{key}'{ttl_info} — {len(value)} chars"


class ScratchReadTool(Tool):
    name = "scratch_read"
    description = (
        "Read a value from the ephemeral scratchpad by key. "
        "Use key='*' to list all available keys."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Key to read, or '*' to list all keys.",
            },
        },
        "required": ["key"],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, key: str) -> str:
        if key == "*":
            keys = _store.keys()
            if not keys:
                return "(scratchpad is empty)"
            return "Keys: " + ", ".join(keys)

        value = _store.read(key)
        if value is None:
            return f"Key '{key}' not found (may have expired)."
        return value
