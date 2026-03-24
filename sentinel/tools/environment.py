"""Environment tools — inspect runtime, manage packages, check dependencies."""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import sys

from sentinel.core.types import PermissionTier
from sentinel.tools.base import Tool


class EnvInfoTool(Tool):
    name = "env_info"
    description = (
        "Get information about the runtime environment: OS, Python version, "
        "available tools, environment variables, and disk space."
    )
    parameters = {
        "type": "object",
        "properties": {
            "show_env_vars": {
                "type": "boolean",
                "description": "Include environment variable names (values are hidden for security).",
                "default": False,
            },
        },
        "required": [],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, show_env_vars: bool = False) -> str:
        lines = [
            f"OS: {platform.system()} {platform.release()}",
            f"Architecture: {platform.machine()}",
            f"Python: {sys.version}",
            f"Working directory: {os.getcwd()}",
        ]

        # Check for common tools
        tools_to_check = ["git", "node", "npm", "docker", "pip", "curl", "wget", "gcc", "make"]
        available = [t for t in tools_to_check if shutil.which(t)]
        lines.append(f"Available tools: {', '.join(available) or 'none checked'}")

        # Disk space
        try:
            stat = os.statvfs(".")
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
            lines.append(f"Disk space: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
        except (OSError, AttributeError):
            pass

        if show_env_vars:
            env_names = sorted(os.environ.keys())
            lines.append(f"\nEnvironment variables ({len(env_names)}):")
            for name in env_names:
                # Mask values for security
                lines.append(f"  {name}=***")

        return "\n".join(lines)


class InstallPackageTool(Tool):
    name = "install_package"
    description = (
        "Install a Python package using pip. "
        "Can also install from requirements.txt or a local path."
    )
    parameters = {
        "type": "object",
        "properties": {
            "package": {
                "type": "string",
                "description": "Package name (e.g., 'requests'), version spec (e.g., 'requests>=2.28'), or path to requirements.txt.",
            },
            "upgrade": {
                "type": "boolean",
                "description": "Upgrade if already installed.",
                "default": False,
            },
        },
        "required": ["package"],
    }
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(self, package: str, upgrade: bool = False) -> str:
        args = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            args.append("--upgrade")

        if package.endswith(".txt"):
            args.extend(["-r", package])
        else:
            args.append(package)

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        out = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()

        if proc.returncode != 0:
            return f"Install failed:\n{err}"

        # Extract the key success line
        for line in out.splitlines():
            if "Successfully installed" in line or "Requirement already satisfied" in line:
                return line
        return out[-500:] if out else "(installed)"


class CheckDependencyTool(Tool):
    name = "check_dependency"
    description = (
        "Check if a command-line tool or Python package is available. "
        "Returns version info if found."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the tool or package to check.",
            },
            "type": {
                "type": "string",
                "enum": ["command", "python"],
                "description": "'command' checks CLI tools, 'python' checks pip packages.",
                "default": "command",
            },
        },
        "required": ["name"],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, name: str, type: str = "command") -> str:
        if type == "command":
            path = shutil.which(name)
            if not path:
                return f"Command '{name}' not found."

            # Try to get version
            for flag in ["--version", "-v", "version"]:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        name, flag,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
                    version_out = (stdout or stderr).decode(errors="replace").strip()
                    if version_out and proc.returncode == 0:
                        first_line = version_out.splitlines()[0]
                        return f"Found: {path}\nVersion: {first_line}"
                except (asyncio.TimeoutError, OSError):
                    continue

            return f"Found: {path} (version unknown)"

        elif type == "python":
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "show", name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            out = stdout.decode(errors="replace").strip()
            if proc.returncode != 0 or not out:
                return f"Python package '{name}' is not installed."

            # Extract name and version
            info = {}
            for line in out.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
            return f"Installed: {info.get('Name', name)} {info.get('Version', '?')}\nLocation: {info.get('Location', '?')}"

        return f"Unknown type: {type}"
