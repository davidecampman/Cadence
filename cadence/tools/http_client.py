"""HTTP client tools — make REST API calls."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from cadence.core.types import PermissionTier
from cadence.tools.base import Tool


class HttpRequestTool(Tool):
    name = "http_request"
    description = (
        "Make an HTTP request to a REST API. Supports GET, POST, PUT, PATCH, DELETE. "
        "Returns status code, headers, and response body. "
        "Use for interacting with external APIs and services."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to request."},
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                "description": "HTTP method.",
                "default": "GET",
            },
            "headers": {
                "type": "object",
                "description": "Request headers as key-value pairs.",
                "default": {},
            },
            "body": {
                "type": "string",
                "description": "Request body (for POST/PUT/PATCH). JSON string or plain text.",
                "default": "",
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds.",
                "default": 30,
            },
            "max_response_chars": {
                "type": "integer",
                "description": "Max characters to return from response body.",
                "default": 20000,
            },
        },
        "required": ["url"],
    }
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(
        self,
        url: str,
        method: str = "GET",
        headers: dict | None = None,
        body: str = "",
        timeout: int = 30,
        max_response_chars: int = 20000,
    ) -> str:
        headers = headers or {}
        if "User-Agent" not in headers:
            headers["User-Agent"] = "Cadence/0.1"

        data = body.encode() if body else None
        if data and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                resp_headers = dict(resp.headers)
                raw = resp.read().decode(errors="replace")
        except urllib.error.HTTPError as e:
            status = e.code
            resp_headers = dict(e.headers)
            raw = e.read().decode(errors="replace")
        except urllib.error.URLError as e:
            return f"Request failed: {e}"

        # Try to pretty-print JSON responses
        content_type = resp_headers.get("Content-Type", "")
        if "json" in content_type.lower():
            try:
                raw = json.dumps(json.loads(raw), indent=2)
            except json.JSONDecodeError:
                pass

        if len(raw) > max_response_chars:
            raw = raw[:max_response_chars] + "\n... (truncated)"

        return f"HTTP {status}\n\n{raw}"
