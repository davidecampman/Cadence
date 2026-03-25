"""Hardened code execution tools with OS-level sandboxing."""

from __future__ import annotations

import asyncio
import platform
import shutil
import tempfile

from cadence.core.config import get_config
from cadence.core.types import PermissionTier
from cadence.tools.base import Tool

# Whether we can use Linux-specific sandboxing (unshare, ulimit wrappers)
_IS_LINUX = platform.system() == "Linux"


def _build_resource_limits(cfg) -> str:
    """Build a shell preamble that enforces resource limits via ulimit."""
    lines = []
    if cfg.max_memory_mb > 0:
        kb = cfg.max_memory_mb * 1024
        lines.append(f"ulimit -v {kb}")
    if cfg.max_cpu_seconds > 0:
        lines.append(f"ulimit -t {cfg.max_cpu_seconds}")
    if cfg.max_file_descriptors > 0:
        lines.append(f"ulimit -n {cfg.max_file_descriptors}")
    return " && ".join(lines)


def _wrap_with_sandbox(command: str, cfg) -> str:
    """Wrap a command with OS-level resource limits and optional network isolation."""
    parts = []

    # Network isolation via Linux namespaces
    if cfg.restrict_network and _IS_LINUX and shutil.which("unshare"):
        parts.append("unshare --net --map-root-user")

    # Resource limits preamble
    limits = _build_resource_limits(cfg)
    if limits:
        inner = f"{limits} && {command}"
    else:
        inner = command

    if parts:
        prefix = " ".join(parts)
        return f"{prefix} bash -c {_shell_quote(inner)}"

    return inner


def _shell_quote(s: str) -> str:
    """Single-quote a string for safe shell embedding."""
    return "'" + s.replace("'", "'\\''") + "'"


def _check_blocked(command: str, blocked: list[str]) -> str | None:
    """Return the matched blocked pattern if the command is disallowed, else None."""
    cmd_lower = command.lower().strip()
    for pattern in blocked:
        if pattern.lower() in cmd_lower:
            return pattern
    return None


class CodeExecutionTool(Tool):
    name = "execute_code"
    description = (
        "Execute code in a hardened subprocess with resource limits. "
        "Supports Python, bash, and JavaScript (Node). "
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
        "python": "python3 -c",
        "bash": "bash -c",
        "javascript": "node -e",
    }

    async def execute(self, language: str, code: str) -> str:
        cfg = get_config().execution
        interpreter = self._INTERPRETERS.get(language)
        if not interpreter:
            return f"Unsupported language: {language}"

        # Build the raw command
        raw_command = f"{interpreter} {_shell_quote(code)}"

        # Apply sandboxing
        sandboxed = _wrap_with_sandbox(raw_command, cfg)

        try:
            proc = await asyncio.create_subprocess_shell(
                sandboxed,
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
        "Run a shell command with resource limits. Use for system operations, "
        "file manipulation, git commands, package management, etc. "
        "Dangerous commands are blocked by a configurable blocklist."
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

        # Check blocklist
        blocked = _check_blocked(command, cfg.blocked_commands)
        if blocked:
            raise PermissionError(f"Blocked: command matches dangerous pattern '{blocked}'")

        # Apply sandboxing
        sandboxed = _wrap_with_sandbox(command, cfg)

        try:
            proc = await asyncio.create_subprocess_shell(
                sandboxed,
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
