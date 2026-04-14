"""Hardened code execution tools with OS-level sandboxing."""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import signal
import tempfile
import time

from cadence.core.config import get_config
from cadence.core.types import PermissionTier
from cadence.tools.base import Tool

# Whether we can use Linux-specific sandboxing (unshare, ulimit wrappers)
_IS_LINUX = platform.system() == "Linux"

# Grace period before SIGKILL after SIGTERM (seconds)
_GRACEFUL_SHUTDOWN_TIMEOUT = 5


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


async def _graceful_terminate(
    proc: asyncio.subprocess.Process,
    partial_stdout: bytes,
    partial_stderr: bytes,
    timeout_seconds: int,
) -> tuple[bytes, bytes, str]:
    """Gracefully terminate a process: SIGTERM → wait → SIGKILL.

    Returns (stdout, stderr, termination_reason).
    Preserves any partial output captured before the timeout.
    """
    reason = f"timed out after {timeout_seconds}s"

    # Try SIGTERM first for graceful shutdown
    try:
        proc.terminate()
    except ProcessLookupError:
        return partial_stdout, partial_stderr, "process already exited"

    # Wait briefly for graceful shutdown
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_GRACEFUL_SHUTDOWN_TIMEOUT,
        )
        return stdout or partial_stdout, stderr or partial_stderr, reason + " (terminated gracefully)"
    except asyncio.TimeoutError:
        pass

    # Force kill
    try:
        proc.kill()
    except ProcessLookupError:
        pass

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=2)
        return stdout or partial_stdout, stderr or partial_stderr, reason + " (killed)"
    except (asyncio.TimeoutError, Exception):
        return partial_stdout, partial_stderr, reason + " (killed, output lost)"


def _format_output(
    stdout: bytes,
    stderr: bytes,
    returncode: int | None,
    max_bytes: int,
    duration_ms: float,
    termination_reason: str = "",
) -> str:
    """Format process output with separated stdout/stderr and telemetry."""
    parts = []

    if stdout:
        out = stdout.decode(errors="replace")[:max_bytes]
        parts.append(out)

    if stderr:
        err = stderr.decode(errors="replace")[:max_bytes]
        parts.append(f"[stderr]\n{err}")

    if returncode is not None and returncode != 0:
        parts.append(f"[exit code: {returncode}]")

    if termination_reason:
        parts.append(f"[{termination_reason}]")

    # Telemetry footer
    parts.append(f"[duration: {duration_ms:.0f}ms]")

    return "\n".join(parts).strip() or "(no output)"


class CodeExecutionTool(Tool):
    name = "execute_code"
    description = (
        "Execute code in a hardened subprocess with resource limits. "
        "Supports Python, bash, and JavaScript (Node). "
        "Returns stdout and stderr separately with execution telemetry. "
        "On timeout, partial output is preserved via graceful shutdown."
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
            "timeout": {
                "type": "integer",
                "description": "Override timeout in seconds. Defaults to config value.",
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

    async def execute(self, language: str, code: str, timeout: int | None = None) -> str:
        cfg = get_config().execution
        interpreter = self._INTERPRETERS.get(language)
        if not interpreter:
            return f"Unsupported language: {language}"

        effective_timeout = timeout or cfg.timeout_seconds

        # Build the raw command
        raw_command = f"{interpreter} {_shell_quote(code)}"

        # Apply sandboxing
        sandboxed = _wrap_with_sandbox(raw_command, cfg)

        start_time = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_shell(
                sandboxed,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir(),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=effective_timeout,
            )
            duration_ms = (time.monotonic() - start_time) * 1000
            return _format_output(
                stdout, stderr, proc.returncode,
                cfg.max_output_bytes, duration_ms,
            )
        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - start_time) * 1000
            # Graceful termination preserving partial output
            stdout, stderr, reason = await _graceful_terminate(
                proc, b"", b"", effective_timeout,
            )
            return _format_output(
                stdout, stderr, None,
                cfg.max_output_bytes, duration_ms,
                termination_reason=reason,
            )


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run a shell command with resource limits. Use for system operations, "
        "file manipulation, git commands, package management, etc. "
        "Dangerous commands are blocked by a configurable blocklist. "
        "Returns stdout and stderr separately with execution telemetry."
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
            "timeout": {
                "type": "integer",
                "description": "Override timeout in seconds. Defaults to config value.",
            },
        },
        "required": ["command"],
    }
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(self, command: str, working_dir: str = ".", timeout: int | None = None) -> str:
        cfg = get_config().execution

        # Check blocklist
        blocked = _check_blocked(command, cfg.blocked_commands)
        if blocked:
            raise PermissionError(f"Blocked: command matches dangerous pattern '{blocked}'")

        effective_timeout = timeout or cfg.timeout_seconds

        # Apply sandboxing
        sandboxed = _wrap_with_sandbox(command, cfg)

        start_time = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_shell(
                sandboxed,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=effective_timeout,
            )
            duration_ms = (time.monotonic() - start_time) * 1000
            return _format_output(
                stdout, stderr, proc.returncode,
                cfg.max_output_bytes, duration_ms,
            )
        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - start_time) * 1000
            stdout, stderr, reason = await _graceful_terminate(
                proc, b"", b"", effective_timeout,
            )
            return _format_output(
                stdout, stderr, None,
                cfg.max_output_bytes, duration_ms,
                termination_reason=reason,
            )
