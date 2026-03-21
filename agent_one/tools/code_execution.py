"""Sandboxed code execution tool."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

from agent_one.core.config import get_config
from agent_one.core.types import PermissionTier
from agent_one.tools.base import Tool


class CodeExecutionTool(Tool):
    name = "execute_code"
    description = (
        "Execute code in a sandboxed environment. Supports Python, bash, and JavaScript (Node). "
        "Returns stdout/stderr. Use this for computation, data processing, or running scripts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "enum": ["python", "bash", "javascript"],
                "description": "Programming language to execute.",
            },
            "code": {
                "type": "string",
                "description": "The code to execute.",
            },
        },
        "required": ["language", "code"],
    }
    permission_tier = PermissionTier.STANDARD

    _INTERPRETERS = {
        "python": ["python3", "-c"],
        "bash": ["bash", "-c"],
        "javascript": ["node", "-e"],
    }

    async def execute(self, language: str, code: str) -> str:
        cfg = get_config().execution
        interpreter = self._INTERPRETERS.get(language)
        if not interpreter:
            return f"Unsupported language: {language}"

        try:
            proc = await asyncio.create_subprocess_exec(
                *interpreter, code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir(),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=cfg.timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return f"Execution timed out after {cfg.timeout_seconds}s"

        output = ""
        if stdout:
            out = stdout.decode(errors="replace")[:cfg.max_output_bytes]
            output += out
        if stderr:
            err = stderr.decode(errors="replace")[:cfg.max_output_bytes]
            output += f"\n[stderr]\n{err}"
        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"

        return output.strip() or "(no output)"


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run a shell command directly. Use for system operations, file manipulation, "
        "git commands, package management, etc. Output is captured and returned."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run.",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory. Defaults to current directory.",
                "default": ".",
            },
        },
        "required": ["command"],
    }
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(self, command: str, working_dir: str = ".") -> str:
        cfg = get_config().execution
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=cfg.timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return f"Command timed out after {cfg.timeout_seconds}s"

        output = ""
        if stdout:
            output += stdout.decode(errors="replace")[:cfg.max_output_bytes]
        if stderr:
            output += f"\n[stderr]\n{stderr.decode(errors='replace')[:cfg.max_output_bytes]}"
        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"

        return output.strip() or "(no output)"
