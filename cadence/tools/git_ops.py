"""Git tools — version control operations for agents."""

from __future__ import annotations

import asyncio
from pathlib import Path

from cadence.core.types import PermissionTier
from cadence.tools.base import Tool


async def _run_git(args: list[str], cwd: str = ".") -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    return (
        proc.returncode,
        stdout.decode(errors="replace").strip(),
        stderr.decode(errors="replace").strip(),
    )


class GitStatusTool(Tool):
    name = "git_status"
    description = "Show the working tree status — modified, staged, and untracked files."
    parameters = {
        "type": "object",
        "properties": {
            "working_dir": {
                "type": "string",
                "description": "Repository directory. Defaults to current directory.",
                "default": ".",
            },
        },
        "required": [],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, working_dir: str = ".") -> str:
        rc, out, err = await _run_git(["status", "--short"], cwd=working_dir)
        if rc != 0:
            return f"git status failed: {err}"
        return out or "(working tree clean)"


class GitDiffTool(Tool):
    name = "git_diff"
    description = (
        "Show changes between commits, working tree, and staging area. "
        "By default shows unstaged changes. Use staged=true for staged changes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Limit diff to a specific file or directory.",
                "default": "",
            },
            "staged": {
                "type": "boolean",
                "description": "Show staged (cached) changes instead of unstaged.",
                "default": False,
            },
            "working_dir": {
                "type": "string",
                "description": "Repository directory.",
                "default": ".",
            },
        },
        "required": [],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, path: str = "", staged: bool = False, working_dir: str = ".") -> str:
        args = ["diff"]
        if staged:
            args.append("--cached")
        if path:
            args.extend(["--", path])
        rc, out, err = await _run_git(args, cwd=working_dir)
        if rc != 0:
            return f"git diff failed: {err}"
        if len(out) > 20000:
            out = out[:20000] + "\n... (truncated)"
        return out or "(no changes)"


class GitCommitTool(Tool):
    name = "git_commit"
    description = (
        "Stage files and create a git commit. "
        "Specify files to stage, or use add_all=true to stage everything."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Commit message.",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of files to stage before committing.",
                "default": [],
            },
            "add_all": {
                "type": "boolean",
                "description": "Stage all modified and untracked files (git add -A).",
                "default": False,
            },
            "working_dir": {
                "type": "string",
                "description": "Repository directory.",
                "default": ".",
            },
        },
        "required": ["message"],
    }
    permission_tier = PermissionTier.STANDARD

    async def execute(
        self, message: str, files: list[str] | None = None, add_all: bool = False, working_dir: str = "."
    ) -> str:
        # Stage files
        if add_all:
            rc, _, err = await _run_git(["add", "-A"], cwd=working_dir)
            if rc != 0:
                return f"git add -A failed: {err}"
        elif files:
            rc, _, err = await _run_git(["add", "--"] + files, cwd=working_dir)
            if rc != 0:
                return f"git add failed: {err}"

        # Commit
        rc, out, err = await _run_git(["commit", "-m", message], cwd=working_dir)
        if rc != 0:
            return f"git commit failed: {err}"
        return out


class GitLogTool(Tool):
    name = "git_log"
    description = "Show recent commit history."
    parameters = {
        "type": "object",
        "properties": {
            "max_count": {
                "type": "integer",
                "description": "Number of commits to show.",
                "default": 10,
            },
            "oneline": {
                "type": "boolean",
                "description": "Use compact one-line format.",
                "default": True,
            },
            "working_dir": {
                "type": "string",
                "description": "Repository directory.",
                "default": ".",
            },
        },
        "required": [],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, max_count: int = 10, oneline: bool = True, working_dir: str = ".") -> str:
        args = ["log", f"--max-count={max_count}"]
        if oneline:
            args.append("--oneline")
        rc, out, err = await _run_git(args, cwd=working_dir)
        if rc != 0:
            return f"git log failed: {err}"
        return out or "(no commits)"
