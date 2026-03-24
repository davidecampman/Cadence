"""Memory export/import — portable JSON format for cross-machine migration.

Supports exporting memories from any backend to a single JSON file and
importing them back into any backend.  This enables:
    - Migrating between macOS <-> Windows <-> Linux
    - Switching backends (ChromaDB <-> SQLite) without data loss
    - Creating backups / snapshots
    - Sharing memory between team members
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from sentinel.memory.paths import get_exports_dir


class MemoryExportEntry(BaseModel):
    """Portable representation of a single memory."""

    id: str
    content: str
    namespace: str
    importance: float
    created_at: float
    last_accessed: float
    access_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_agent: str | None = None


class MemoryExportBundle(BaseModel):
    """A complete export bundle with metadata."""

    version: int = 1
    exported_at: float = Field(default_factory=time.time)
    source_backend: str = "unknown"
    namespaces: dict[str, list[MemoryExportEntry]] = Field(default_factory=dict)
    total_entries: int = 0


class MemoryBackend(Protocol):
    """Protocol that any memory backend must satisfy for export/import."""

    async def list_namespaces(self) -> list[str]: ...
    async def get_all(self, namespace: str | None = None) -> list[Any]: ...
    async def save(
        self,
        content: str,
        namespace: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        source_agent: str | None = None,
    ) -> Any: ...


async def export_memories(
    backend: MemoryBackend,
    output_path: str | Path | None = None,
    namespaces: list[str] | None = None,
    source_backend_name: str = "unknown",
) -> Path:
    """Export all memories (or specific namespaces) to a portable JSON file.

    Args:
        backend: The memory store to export from.
        output_path: Where to write the JSON file. Defaults to exports dir.
        namespaces: Specific namespaces to export. None = all.
        source_backend_name: Label for the source backend type.

    Returns:
        Path to the written export file.
    """
    if namespaces is None:
        namespaces = await backend.list_namespaces()

    bundle = MemoryExportBundle(
        source_backend=source_backend_name,
    )

    total = 0
    for ns in namespaces:
        entries = await backend.get_all(namespace=ns)
        export_entries = []
        for e in entries:
            export_entries.append(MemoryExportEntry(
                id=e.id,
                content=e.content,
                namespace=getattr(e, "namespace", ns),
                importance=e.importance,
                created_at=e.created_at,
                last_accessed=getattr(e, "last_accessed", e.created_at),
                access_count=getattr(e, "access_count", 0),
                metadata=getattr(e, "metadata", {}),
                source_agent=getattr(e, "source_agent", None),
            ))
        bundle.namespaces[ns] = export_entries
        total += len(export_entries)

    bundle.total_entries = total

    if output_path is None:
        export_dir = get_exports_dir()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = export_dir / f"memory_export_{timestamp}.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        bundle.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return output_path


async def import_memories(
    backend: MemoryBackend,
    input_path: str | Path,
    namespaces: list[str] | None = None,
    skip_duplicates: bool = True,
) -> int:
    """Import memories from a JSON export file into a backend.

    Args:
        backend: The memory store to import into.
        input_path: Path to the export JSON file.
        namespaces: Only import these namespaces. None = all.
        skip_duplicates: If True, skip entries whose IDs already exist.

    Returns:
        Number of memories imported.
    """
    input_path = Path(input_path)
    raw = input_path.read_text(encoding="utf-8")
    bundle = MemoryExportBundle.model_validate_json(raw)

    imported = 0
    for ns, entries in bundle.namespaces.items():
        if namespaces and ns not in namespaces:
            continue

        for entry in entries:
            try:
                await backend.save(
                    content=entry.content,
                    namespace=ns,
                    importance=entry.importance,
                    metadata={
                        **entry.metadata,
                        "_imported_from": str(input_path.name),
                        "_original_id": entry.id,
                    },
                    source_agent=entry.source_agent,
                )
                imported += 1
            except Exception:
                # Skip entries that fail (e.g. duplicate constraint)
                if not skip_duplicates:
                    raise
    return imported
