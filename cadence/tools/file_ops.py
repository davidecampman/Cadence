"""File operation tools — read, write, list, search, grep, edit."""

from __future__ import annotations

import glob as glob_mod
import os
import re
from pathlib import Path

from cadence.core.types import PermissionTier
from cadence.tools.base import Tool

# Sensitive paths that file tools should never write to.
_SENSITIVE_PATHS = {"/etc", "/usr", "/bin", "/sbin", "/boot", "/var", "/root", "/proc", "/sys"}


def _is_write_safe(p: Path) -> bool:
    """Return True if *p* is not inside a sensitive system directory."""
    resolved = str(p.resolve())
    return not any(resolved == s or resolved.startswith(s + "/") for s in _SENSITIVE_PATHS)


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

        text = p.read_text(encoding="utf-8", errors="replace")
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
        if not _is_write_safe(p):
            return f"Write blocked: path resolves inside a protected system directory ({p.resolve()})"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
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


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents using a regular expression. Returns matching lines "
        "with file path and line number. Supports recursive directory search, "
        "context lines, and file-type filtering."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {"type": "string", "description": "File or directory to search.", "default": "."},
            "glob": {
                "type": "string",
                "description": "File glob filter (e.g., '*.py', '**/*.ts'). Default searches all files.",
                "default": "**/*",
            },
            "context_lines": {
                "type": "integer",
                "description": "Lines of context to show before and after each match.",
                "default": 0,
            },
            "ignore_case": {"type": "boolean", "description": "Case-insensitive search.", "default": False},
            "max_matches": {"type": "integer", "description": "Maximum number of matches to return.", "default": 100},
        },
        "required": ["pattern"],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str = "**/*",
        context_lines: int = 0,
        ignore_case: bool = False,
        max_matches: int = 100,
    ) -> str:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        base = Path(path).expanduser()
        if base.is_file():
            files = [base]
        else:
            files = sorted(base.rglob(glob.lstrip("**/") if glob == "**/*" else glob))[:1000]

        results: list[str] = []
        total = 0

        for fp in files:
            if not fp.is_file():
                continue
            try:
                lines = fp.read_text(errors="replace").splitlines()
            except Exception:
                continue

            for i, line in enumerate(lines):
                if not rx.search(line):
                    continue

                if context_lines:
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    for j in range(start, end):
                        marker = ">" if j == i else " "
                        results.append(f"{fp}:{j + 1}{marker} {lines[j]}")
                    results.append("--")
                else:
                    results.append(f"{fp}:{i + 1}: {line.rstrip()}")

                total += 1
                if total >= max_matches:
                    results.append(f"... (truncated — {max_matches} matches shown)")
                    return "\n".join(results)

        return "\n".join(results) or f"No matches for '{pattern}'"


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Make a precise string replacement in a file. Replaces old_string with new_string. "
        "old_string must appear exactly once in the file (or use replace_all=true). "
        "Prefer this over write_file for targeted edits — it is safer and more efficient."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path of the file to edit."},
            "old_string": {"type": "string", "description": "Exact text to find and replace."},
            "new_string": {"type": "string", "description": "Text to replace it with."},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring uniqueness.",
                "default": False,
            },
        },
        "required": ["path", "old_string", "new_string"],
    }
    permission_tier = PermissionTier.STANDARD

    async def execute(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"File not found: {path}"
        if not p.is_file():
            return f"Not a file: {path}"
        if not _is_write_safe(p):
            return f"Write blocked: path resolves inside a protected system directory ({p.resolve()})"

        content = p.read_text(encoding="utf-8", errors="replace")

        if old_string not in content:
            return f"old_string not found in {path}"

        count = content.count(old_string)
        if not replace_all and count > 1:
            return (
                f"old_string appears {count} times in {path} — it must be unique. "
                "Add more surrounding context to make it unique, or use replace_all=true."
            )

        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        n = count if replace_all else 1

        p.write_text(new_content, encoding="utf-8")
        return f"Replaced {n} occurrence(s) in {path}"
