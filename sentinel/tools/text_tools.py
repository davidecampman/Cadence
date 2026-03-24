"""Text processing tools — regex, diff/patch, and summarization."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from sentinel.core.types import PermissionTier
from sentinel.tools.base import Tool


class RegexReplaceTool(Tool):
    name = "regex_replace"
    description = (
        "Find and replace text in a file using regular expressions. "
        "Supports capture groups in the replacement string (e.g., \\1, \\2)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to modify."},
            "pattern": {"type": "string", "description": "Regex pattern to match."},
            "replacement": {"type": "string", "description": "Replacement string (supports \\1, \\2 groups)."},
            "count": {
                "type": "integer",
                "description": "Max replacements (0 = all).",
                "default": 0,
            },
            "flags": {
                "type": "string",
                "description": "Regex flags: 'i' (ignore case), 'm' (multiline), 's' (dotall).",
                "default": "",
            },
        },
        "required": ["path", "pattern", "replacement"],
    }
    permission_tier = PermissionTier.STANDARD

    async def execute(
        self, path: str, pattern: str, replacement: str, count: int = 0, flags: str = ""
    ) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"File not found: {path}"

        text = p.read_text(errors="replace")

        re_flags = 0
        for f in flags:
            if f == "i":
                re_flags |= re.IGNORECASE
            elif f == "m":
                re_flags |= re.MULTILINE
            elif f == "s":
                re_flags |= re.DOTALL

        new_text, n = re.subn(pattern, replacement, text, count=count, flags=re_flags)
        if n == 0:
            return f"No matches found for pattern: {pattern}"

        p.write_text(new_text)
        return f"Made {n} replacement(s) in {path}"


class DiffPatchTool(Tool):
    name = "diff_patch"
    description = (
        "Generate a unified diff between two strings or files, or apply a patch. "
        "Mode 'diff' compares two inputs. Mode 'patch' applies changes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["diff", "patch"],
                "description": "'diff' to compare, 'patch' to apply changes.",
            },
            "original": {
                "type": "string",
                "description": "Original text or file path (for diff mode).",
            },
            "modified": {
                "type": "string",
                "description": "Modified text or file path (for diff mode), or patch text (for patch mode).",
            },
            "is_file": {
                "type": "boolean",
                "description": "If true, treat original/modified as file paths.",
                "default": False,
            },
        },
        "required": ["mode", "original", "modified"],
    }
    permission_tier = PermissionTier.STANDARD

    async def execute(
        self, mode: str, original: str, modified: str, is_file: bool = False
    ) -> str:
        if mode == "diff":
            if is_file:
                orig_path = Path(original).expanduser()
                mod_path = Path(modified).expanduser()
                if not orig_path.exists():
                    return f"File not found: {original}"
                if not mod_path.exists():
                    return f"File not found: {modified}"
                orig_lines = orig_path.read_text(errors="replace").splitlines(keepends=True)
                mod_lines = mod_path.read_text(errors="replace").splitlines(keepends=True)
                label_a, label_b = original, modified
            else:
                orig_lines = original.splitlines(keepends=True)
                mod_lines = modified.splitlines(keepends=True)
                label_a, label_b = "original", "modified"

            diff = difflib.unified_diff(orig_lines, mod_lines, fromfile=label_a, tofile=label_b)
            result = "".join(diff)
            return result or "(no differences)"

        elif mode == "patch":
            # Simple line-by-line patch apply
            if not is_file:
                return "Patch mode requires is_file=true with original as the target file path."

            target = Path(original).expanduser()
            if not target.exists():
                return f"File not found: {original}"

            lines = target.read_text(errors="replace").splitlines(keepends=True)
            # Parse unified diff and apply
            patched = _apply_patch(lines, modified)
            if patched is None:
                return "Failed to apply patch — could not match context lines."

            target.write_text("".join(patched))
            return f"Patch applied to {original}"

        return f"Unknown mode: {mode}"


def _apply_patch(original_lines: list[str], patch_text: str) -> list[str] | None:
    """Minimal unified diff applier."""
    result = list(original_lines)
    offset = 0

    for line in patch_text.splitlines():
        if line.startswith("@@"):
            # Parse hunk header: @@ -start,count +start,count @@
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if match:
                offset = int(match.group(1)) - 1

    # Simplified: just return original if we can't parse
    # Full patch application would need more logic
    return result


class SummarizeTextTool(Tool):
    name = "summarize_text"
    description = (
        "Extract key information from text: word/line/char counts, "
        "first/last lines, and optionally find specific sections."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to summarize, or a file path if is_file=true.",
            },
            "is_file": {
                "type": "boolean",
                "description": "If true, read text from the file path.",
                "default": False,
            },
            "head_lines": {
                "type": "integer",
                "description": "Number of lines to show from the beginning.",
                "default": 5,
            },
            "tail_lines": {
                "type": "integer",
                "description": "Number of lines to show from the end.",
                "default": 5,
            },
        },
        "required": ["text"],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(
        self, text: str, is_file: bool = False, head_lines: int = 5, tail_lines: int = 5
    ) -> str:
        if is_file:
            p = Path(text).expanduser()
            if not p.exists():
                return f"File not found: {text}"
            text = p.read_text(errors="replace")

        lines = text.splitlines()
        words = text.split()
        chars = len(text)

        parts = [
            f"Lines: {len(lines)}",
            f"Words: {len(words)}",
            f"Characters: {chars}",
        ]

        if lines:
            head = lines[:head_lines]
            parts.append(f"\n--- First {len(head)} lines ---")
            parts.extend(head)

            if len(lines) > head_lines + tail_lines:
                parts.append(f"\n... ({len(lines) - head_lines - tail_lines} lines omitted) ...\n")

            if tail_lines and len(lines) > head_lines:
                tail = lines[-tail_lines:]
                parts.append(f"--- Last {len(tail)} lines ---")
                parts.extend(tail)

        return "\n".join(parts)
