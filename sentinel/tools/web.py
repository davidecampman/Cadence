"""Web tools — HTTP fetch and search (provider-agnostic)."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from html.parser import HTMLParser

from sentinel.core.types import PermissionTier
from sentinel.tools.base import Tool


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor."""

    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a URL and return its text content. Strips HTML tags. "
        "Use for reading documentation, API responses, or web pages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch."},
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return.",
                "default": 20000,
            },
        },
        "required": ["url"],
    }
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(self, url: str, max_chars: int = 20000) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "Sentinel/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read().decode(errors="replace")
        except urllib.error.URLError as e:
            return f"Fetch failed: {e}"

        if "html" in content_type.lower():
            extractor = _TextExtractor()
            extractor.feed(raw)
            text = extractor.get_text()
        elif "json" in content_type.lower():
            try:
                text = json.dumps(json.loads(raw), indent=2)
            except json.JSONDecodeError:
                text = raw
        else:
            text = raw

        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text or "(empty response)"
