"""Vision tools — screenshot capture and image analysis."""

from __future__ import annotations

import asyncio
import base64
import os
import shlex
from pathlib import Path

from cadence.core.types import PermissionTier
from cadence.tools.base import Tool


class ScreenshotTool(Tool):
    name = "screenshot"
    description = (
        "Capture a screenshot of the current screen or a specific window. "
        "Saves to a file and returns the path. Requires a display and screenshot utility."
    )
    parameters = {
        "type": "object",
        "properties": {
            "output_path": {
                "type": "string",
                "description": "File path to save the screenshot. Defaults to /tmp/screenshot.png.",
                "default": "/tmp/screenshot.png",
            },
            "delay_seconds": {
                "type": "integer",
                "description": "Delay before capturing (seconds).",
                "default": 0,
            },
        },
        "required": [],
    }
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(self, output_path: str = "/tmp/screenshot.png", delay_seconds: int = 0) -> str:
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Try common screenshot tools in order, using exec (not shell) to
        # prevent command injection via the output_path argument.
        safe_path = str(Path(output_path).resolve())
        tool_cmds = [
            ["import", "-window", "root", safe_path],   # ImageMagick
            ["scrot", safe_path],                         # scrot
            ["gnome-screenshot", "-f", safe_path],        # GNOME
            ["screencapture", safe_path],                 # macOS
        ]
        for cmd_args in tool_cmds:
            tool_name = cmd_args[0]
            proc = await asyncio.create_subprocess_exec(
                "which", tool_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode == 0:
                proc = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, err = await asyncio.wait_for(proc.communicate(), timeout=15)
                if proc.returncode == 0 and Path(output_path).exists():
                    size = Path(output_path).stat().st_size
                    return f"Screenshot saved to {output_path} ({size} bytes)"
                return f"Screenshot failed: {err.decode(errors='replace')}"

        return "No screenshot tool available (install scrot, ImageMagick, or gnome-screenshot)"


class ImageDescribeTool(Tool):
    name = "image_describe"
    description = (
        "Read an image file and return its base64-encoded content for analysis. "
        "Returns image metadata and base64 data that can be passed to a vision-capable model."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the image file.",
            },
            "max_size_mb": {
                "type": "number",
                "description": "Maximum file size in MB to process.",
                "default": 5.0,
            },
        },
        "required": ["path"],
    }
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, path: str, max_size_mb: float = 5.0) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Image not found: {path}"

        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return f"Image too large: {size_mb:.1f}MB (max {max_size_mb}MB)"

        suffix = p.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime = mime_types.get(suffix)
        if not mime:
            return f"Unsupported image format: {suffix}"

        data = p.read_bytes()
        b64 = base64.b64encode(data).decode()

        return (
            f"Image: {p.name} ({size_mb:.2f}MB, {mime})\n"
            f"Base64 length: {len(b64)} chars\n"
            f"data:{mime};base64,{b64}"
        )
