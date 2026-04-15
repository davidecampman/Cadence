"""Web tools — HTTP fetch and search (provider-agnostic)."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import urllib.parse
import urllib.request
import urllib.error
from html.parser import HTMLParser

from cadence.core.types import PermissionTier
from cadence.tools.base import Tool


def _validate_url(url: str) -> str | None:
    """Validate a URL is safe to fetch. Returns error message or None if safe."""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return "Invalid URL"

    if parsed.scheme not in ("http", "https"):
        return f"Unsupported scheme: {parsed.scheme!r}. Only http/https allowed."

    hostname = parsed.hostname
    if not hostname:
        return "Missing hostname in URL"

    # Block private/internal IPs
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_link_local:
            return f"Blocked: {hostname} is a private/reserved address"
    except ValueError:
        # hostname is a DNS name, resolve it
        try:
            resolved = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
            for _, _, _, _, sockaddr in resolved:
                addr = ipaddress.ip_address(sockaddr[0])
                if addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_link_local:
                    return f"Blocked: {hostname} resolves to private address {sockaddr[0]}"
        except socket.gaierror:
            pass  # DNS resolution failure will be caught by the actual fetch

    return None


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
        validation_error = _validate_url(url)
        if validation_error:
            return f"URL validation failed: {validation_error}"

        def _blocking_fetch():
            req = urllib.request.Request(url, headers={"User-Agent": "Cadence/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type", "")
                # Cap read size to avoid OOM from malicious servers
                raw = resp.read(max_chars * 4).decode(errors="replace")
            return content_type, raw

        try:
            content_type, raw = await asyncio.to_thread(_blocking_fetch)
        except urllib.error.URLError as e:
            return f"Fetch failed: {e}"
        except Exception as e:
            return f"Fetch failed: {type(e).__name__}: {e}"

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
