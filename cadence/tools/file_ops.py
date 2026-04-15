"""File operation tools — read, write, list, search, grep, edit, diff."""

from __future__ import annotations

import base64
import difflib
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


class WriteBinaryFileTool(Tool):
    name = "write_binary_file"
    description = (
        "Write binary content to a file using base64 encoding. "
        "Use this for non-text files like ZIP archives, images, PDFs, etc. "
        "The content must be base64-encoded. Creates parent directories as needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write to."},
            "content_base64": {
                "type": "string",
                "description": "Base64-encoded binary content to write.",
            },
        },
        "required": ["path", "content_base64"],
    }
    permission_tier = PermissionTier.STANDARD

    async def execute(self, path: str, content_base64: str) -> str:
        p = Path(path).expanduser()
        if not _is_write_safe(p):
            return f"Write blocked: path resolves inside a protected system directory ({p.resolve()})"
        try:
            data = base64.b64decode(content_base64)
        except Exception as e:
            return f"Invalid base64 content: {e}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        abs_path = str(p.resolve())
        return (
            f"Wrote {len(data)} bytes (binary) to {path}\n"
            f"[[FILE:{abs_path}]]"
        )


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files matching a glob pattern, or list a directory's contents. Supports max_depth to limit recursion."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., '**/*.py') or directory path.",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum directory depth to traverse. 0 = current dir only, -1 = unlimited.",
                "default": -1,
            },
        },
        "required": ["pattern"],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, pattern: str, max_depth: int = -1) -> str:
        p = Path(pattern).expanduser()
        if p.is_dir():
            entries = sorted(p.iterdir())
            lines = []
            for e in entries[:200]:
                if e.is_symlink():
                    prefix = "🔗 "
                elif e.is_dir():
                    prefix = "📁 "
                else:
                    prefix = "  "
                lines.append(f"{prefix}{e.name}")
            return "\n".join(lines) or "(empty directory)"

        matches = sorted(glob_mod.glob(pattern, recursive=True))

        # Apply max_depth filter if specified
        if max_depth >= 0:
            base_parts = len(Path(pattern.split("*")[0].rstrip("/")).parts) if "*" in pattern else 0
            filtered = []
            for m in matches:
                depth = len(Path(m).parts) - base_parts
                if depth <= max_depth:
                    filtered.append(m)
            matches = filtered

        # Skip symlink loops
        seen: set[str] = set()
        safe_matches = []
        for m in matches[:200]:
            try:
                resolved = str(Path(m).resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    safe_matches.append(m)
            except OSError:
                continue

        if not safe_matches:
            return f"No files matching: {pattern}"
        return "\n".join(safe_matches)


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
            # rglob("*") matches all files recursively; strip leading "**/" prefix properly
            rglob_pattern = "*" if glob in ("**/*", "**/") else glob.removeprefix("**/")
            files = sorted(base.rglob(rglob_pattern))[:1000]

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


class ApplyDiffTool(Tool):
    name = "apply_diff"
    description = (
        "Apply a unified diff patch to a file. Supports fuzzy matching when exact "
        "line content doesn't match (e.g., whitespace differences). Safer than "
        "write_file for multi-line edits — shows exactly what changed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to patch."},
            "diff": {
                "type": "string",
                "description": (
                    "Unified diff content. Lines starting with '-' are removed, "
                    "'+' are added, ' ' (space) are context. Example:\n"
                    " def hello():\n"
                    '-    print("old")\n'
                    '+    print("new")\n'
                ),
            },
            "fuzzy_threshold": {
                "type": "number",
                "description": "Similarity threshold (0.0–1.0) for fuzzy line matching. Default 0.8.",
                "default": 0.8,
            },
        },
        "required": ["path", "diff"],
    }
    permission_tier = PermissionTier.STANDARD

    async def execute(self, path: str, diff: str, fuzzy_threshold: float = 0.8) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"File not found: {path}"
        if not p.is_file():
            return f"Not a file: {path}"
        if not _is_write_safe(p):
            return f"Write blocked: path resolves inside a protected system directory ({p.resolve()})"

        original = p.read_text(encoding="utf-8", errors="replace")
        lines = original.splitlines(keepends=True)

        # Parse the diff into hunks
        hunks = _parse_unified_diff(diff)
        if not hunks:
            return "No valid diff hunks found. Ensure diff uses unified format (lines starting with -, +, or space)."

        # Apply hunks in reverse order to preserve line numbers
        result_lines = list(lines)
        applied = 0
        errors = []

        for hunk in reversed(hunks):
            success, result_lines, msg = _apply_hunk(result_lines, hunk, fuzzy_threshold)
            if success:
                applied += 1
            else:
                errors.append(msg)

        if not applied:
            return "Failed to apply any hunks:\n" + "\n".join(errors)

        new_content = "".join(result_lines)
        p.write_text(new_content, encoding="utf-8")

        # Generate a summary of what changed
        diff_summary = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=2,
        ))
        summary = "".join(diff_summary[:50]) if diff_summary else "(no visible changes)"
        result = f"Applied {applied}/{len(hunks)} hunk(s) to {path}"
        if errors:
            result += f"\nWarnings: {'; '.join(errors)}"
        result += f"\n\n{summary}"
        return result


class FileDiffTool(Tool):
    name = "file_diff"
    description = (
        "Show differences between two files, or between the current version "
        "and a provided 'before' text. Returns a unified diff."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "other_path": {
                "type": "string",
                "description": "Path to another file to diff against. If omitted, use 'before_text'.",
            },
            "before_text": {
                "type": "string",
                "description": "Previous content to diff against the current file.",
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of context lines around changes.",
                "default": 3,
            },
        },
        "required": ["path"],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(
        self,
        path: str,
        other_path: str | None = None,
        before_text: str | None = None,
        context_lines: int = 3,
    ) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"File not found: {path}"

        current = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

        if other_path:
            op = Path(other_path).expanduser()
            if not op.exists():
                return f"File not found: {other_path}"
            other = op.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            label_a, label_b = f"a/{other_path}", f"b/{path}"
        elif before_text is not None:
            other = before_text.splitlines(keepends=True)
            label_a, label_b = "a/before", f"b/{path}"
        else:
            return "Provide either 'other_path' or 'before_text' to diff against."

        diff_lines = list(difflib.unified_diff(
            other, current, fromfile=label_a, tofile=label_b, n=context_lines,
        ))
        if not diff_lines:
            return "No differences found."
        return "".join(diff_lines[:200])


# --- Diff helper functions ---

def _parse_unified_diff(diff_text: str) -> list[dict]:
    """Parse a unified diff into hunks.

    Each hunk is a dict with:
    - context_before: list of context lines expected before the change
    - removals: list of lines to remove
    - additions: list of lines to add
    - context_after: list of context lines expected after the change
    """
    hunks: list[dict] = []
    current_hunk: dict | None = None

    for raw_line in diff_text.splitlines():
        # Skip diff headers
        if raw_line.startswith("---") or raw_line.startswith("+++") or raw_line.startswith("@@"):
            if current_hunk and (current_hunk["removals"] or current_hunk["additions"]):
                hunks.append(current_hunk)
            current_hunk = {
                "context_before": [], "removals": [],
                "additions": [], "context_after": [],
            }
            continue

        if current_hunk is None:
            current_hunk = {
                "context_before": [], "removals": [],
                "additions": [], "context_after": [],
            }

        if raw_line.startswith("-"):
            # If we already have additions, this starts a new change block
            if current_hunk["context_after"]:
                hunks.append(current_hunk)
                current_hunk = {
                    "context_before": list(current_hunk["context_after"]),
                    "removals": [], "additions": [], "context_after": [],
                }
            current_hunk["removals"].append(raw_line[1:])
        elif raw_line.startswith("+"):
            current_hunk["additions"].append(raw_line[1:])
        elif raw_line.startswith(" "):
            ctx_line = raw_line[1:]
            if current_hunk["removals"] or current_hunk["additions"]:
                current_hunk["context_after"].append(ctx_line)
            else:
                current_hunk["context_before"].append(ctx_line)
        else:
            # Unrecognized line — treat as context
            if current_hunk["removals"] or current_hunk["additions"]:
                current_hunk["context_after"].append(raw_line)
            else:
                current_hunk["context_before"].append(raw_line)

    if current_hunk and (current_hunk["removals"] or current_hunk["additions"]):
        hunks.append(current_hunk)

    return hunks


def _apply_hunk(
    lines: list[str],
    hunk: dict,
    fuzzy_threshold: float,
) -> tuple[bool, list[str], str]:
    """Apply a single diff hunk to a list of lines.

    Returns (success, modified_lines, error_message).
    Uses fuzzy matching when exact context doesn't match.
    """
    context = hunk["context_before"]
    removals = hunk["removals"]
    additions = hunk["additions"]

    # Build the expected block: context + removals
    expected = context + removals

    if not expected:
        # Pure addition — append at end
        return True, lines + [line + "\n" for line in additions], ""

    # Normalize lines for matching
    def _normalize(s: str) -> str:
        return s.rstrip("\r\n")

    # Try exact match first
    for i in range(len(lines)):
        if i + len(expected) > len(lines):
            break
        match = all(
            _normalize(lines[i + j]) == _normalize(expected[j])
            for j in range(len(expected))
        )
        if match:
            # Replace: keep lines before, skip context+removals, add context+additions, keep lines after
            new_lines = (
                lines[:i]
                + [line + "\n" if not line.endswith("\n") else line for line in context]
                + [line + "\n" if not line.endswith("\n") else line for line in additions]
                + lines[i + len(expected):]
            )
            return True, new_lines, ""

    # Fuzzy match: find the best matching location
    best_score = 0.0
    best_idx = -1
    for i in range(len(lines)):
        if i + len(expected) > len(lines):
            break
        scores = []
        for j in range(len(expected)):
            ratio = difflib.SequenceMatcher(
                None, _normalize(lines[i + j]), _normalize(expected[j]),
            ).ratio()
            scores.append(ratio)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if avg_score > best_score:
            best_score = avg_score
            best_idx = i

    if best_score >= fuzzy_threshold and best_idx >= 0:
        new_lines = (
            lines[:best_idx]
            + [line + "\n" if not line.endswith("\n") else line for line in context]
            + [line + "\n" if not line.endswith("\n") else line for line in additions]
            + lines[best_idx + len(expected):]
        )
        return True, new_lines, f"Applied with fuzzy match (score={best_score:.2f})"

    return False, lines, f"Could not find matching location (best score={best_score:.2f})"
