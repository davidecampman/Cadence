"""Platform-aware data directory resolution for macOS, Windows, and Linux."""

from __future__ import annotations

import os
import platform
from pathlib import Path

APP_NAME = "sentinel"


def get_data_dir(subdir: str | None = None) -> Path:
    """Return the platform-appropriate persistent data directory.

    Resolution order:
        1. SENTINEL_DATA_DIR environment variable (explicit override)
        2. Platform default:
            - macOS:   ~/Library/Application Support/sentinel
            - Windows: %LOCALAPPDATA%/sentinel
            - Linux:   $XDG_DATA_HOME/sentinel  (default: ~/.local/share/sentinel)

    Args:
        subdir: Optional subdirectory under the data root (e.g. "memory", "sessions").

    Returns:
        Resolved Path (created if it doesn't exist).
    """
    explicit = os.environ.get("SENTINEL_DATA_DIR")
    if explicit:
        base = Path(explicit)
    else:
        system = platform.system()
        if system == "Darwin":
            base = Path.home() / "Library" / "Application Support" / APP_NAME
        elif system == "Windows":
            local_appdata = os.environ.get("LOCALAPPDATA")
            if local_appdata:
                base = Path(local_appdata) / APP_NAME
            else:
                base = Path.home() / "AppData" / "Local" / APP_NAME
        else:
            # Linux / other POSIX — follow XDG Base Directory spec
            xdg = os.environ.get("XDG_DATA_HOME")
            if xdg:
                base = Path(xdg) / APP_NAME
            else:
                base = Path.home() / ".local" / "share" / APP_NAME

    if subdir:
        base = base / subdir

    base.mkdir(parents=True, exist_ok=True)
    return base


def get_memory_dir() -> Path:
    """Return the memory persistence directory."""
    return get_data_dir("memory")


def get_sessions_dir() -> Path:
    """Return the session persistence directory."""
    return get_data_dir("sessions")


def get_exports_dir() -> Path:
    """Return the default export directory."""
    return get_data_dir("exports")
