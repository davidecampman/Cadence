"""Memory backend factory — selects the right store based on config.

Usage:
    from sentinel.memory.backend import create_memory_backend
    store = create_memory_backend()   # Reads config, returns ready-to-use store
"""

from __future__ import annotations

import logging
from typing import Any

from sentinel.core.config import get_config
from sentinel.memory.paths import get_memory_dir

logger = logging.getLogger(__name__)


def create_memory_backend() -> Any:
    """Create a memory store based on the configured backend.

    Config keys:
        memory.backend: "sqlite" | "chromadb"
        memory.persist_dir: "auto" | explicit path

    Returns:
        An instance of SQLiteMemoryStore or MemoryStore (ChromaDB).
    """
    config = get_config().memory
    backend = config.backend.lower()

    # Resolve the persistence directory
    if config.persist_dir == "auto":
        persist_dir = get_memory_dir()
    else:
        from pathlib import Path
        persist_dir = Path(config.persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

    if backend == "sqlite":
        from sentinel.memory.sqlite_store import SQLiteMemoryStore
        logger.info("Using SQLite memory backend at %s", persist_dir)
        return SQLiteMemoryStore(db_dir=persist_dir)

    elif backend == "chromadb":
        from sentinel.memory.store import MemoryStore
        logger.info("Using ChromaDB memory backend at %s", persist_dir)
        # Patch the config persist_dir so MemoryStore uses the resolved path
        config.persist_dir = str(persist_dir)
        return MemoryStore()

    else:
        raise ValueError(
            f"Unknown memory backend: {backend!r}. "
            f"Supported: 'sqlite', 'chromadb'"
        )
