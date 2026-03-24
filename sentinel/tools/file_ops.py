"""File operation tools — read, write, list, search."""

from __future__ import annotations

import glob as glob_mod
import os
from pathlib import Path

from sentinel.core.types import PermissionTier
from sentinel.tools.base import Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file. Returns the text content with line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
            "offset": {"type": "integer", "description": "Line number to start from (0-based).", "default": 0},
            "limit": {"type": "integer", "description": "Max lines to return. 0 = all.", "default": 0},
        },
        "required": ["path"],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, path: str, offset: int = 0, limit: int = 0) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"File not found: {path}"
        if not p.is_file():
            return f"Not a file: {path}"

        text = p.read_text(errors="replace")
        lines = text.splitlines()

        if offset:
            lines = lines[offset:]
        if limit:
            lines = lines[:limit]

        numbered = [f"{i + offset + 1:>5}\t{line}" for i, line in enumerate(lines)]
        return "\n".join(numbered) or "(empty file)"


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write or overwrite a file with the given content. Creates parent directories as needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write to."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["path", "content"],
    }
    permission_tier = PermissionTier.STANDARD

    async def execute(self, path: str, content: str) -> str:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        abs_path = str(p.resolve())
        return (
            f"Wrote {len(content)} bytes to {path}\n"
            f"[[FILE:{abs_path}]]"
        )


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files matching a glob pattern, or list a directory's contents."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., '**/*.py') or directory path.",
            },
        },
        "required": ["pattern"],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, pattern: str) -> str:
        p = Path(pattern).expanduser()
        if p.is_dir():
            entries = sorted(p.iterdir())
            lines = []
            for e in entries[:200]:
                prefix = "📁 " if e.is_dir() else "  "
                lines.append(f"{prefix}{e.name}")
            return "\n".join(lines) or "(empty directory)"

        matches = sorted(glob_mod.glob(pattern, recursive=True))[:200]
        if not matches:
            return f"No files matching: {pattern}"
        return "\n".join(matches)


class SearchFilesTool(Tool):
    name = "search_files"
    description = "Search file contents using a text pattern (substring match). Returns matching lines with context."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Text pattern to search for."},
            "path": {"type": "string", "description": "Directory or file to search in.", "default": "."},
            "glob": {"type": "string", "description": "File glob filter (e.g., '*.py').", "default": "*"},
        },
        "required": ["pattern"],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, pattern: str, path: str = ".", glob: str = "*") -> str:
        base = Path(path).expanduser()
        if base.is_file():
            files = [base]
        else:
            files = sorted(base.rglob(glob))[:500]

        results = []
        for fp in files:
            if not fp.is_file():
                continue
            try:
                lines = fp.read_text(errors="replace").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines):
                if pattern.lower() in line.lower():
                    results.append(f"{fp}:{i+1}: {line.rstrip()}")
                    if len(results) >= 50:
                        results.append("... (truncated at 50 matches)")
                        return "\n".join(results)

        return "\n".join(results) or f"No matches for '{pattern}'"
