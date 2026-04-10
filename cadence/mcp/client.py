"""MCP client — connects to MCP servers via stdio or SSE transports.

Implements the Model Context Protocol JSON-RPC interface for:
- Server initialization and capability negotiation
- Tool listing (tools/list)
- Tool invocation (tools/call)

Supports two transports:
- **stdio**: Launch a subprocess and communicate via stdin/stdout
- **sse**: Connect to an HTTP+SSE server endpoint
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class MCPTransport(str, Enum):
    STDIO = "stdio"
    SSE = "sse"


@dataclass
class MCPToolSchema:
    """Schema for a tool exposed by an MCP server."""
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPServerInfo:
    """Information about a connected MCP server."""
    name: str
    version: str
    capabilities: dict[str, Any] = field(default_factory=dict)


class MCPClient:
    """Client for communicating with a single MCP server.

    Usage::

        client = MCPClient(name="github", command="npx", args=["-y", "@modelcontextprotocol/server-github"])
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("search_repositories", {"query": "cadence"})
        await client.disconnect()
    """

    def __init__(
        self,
        name: str,
        *,
        # stdio transport
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        # SSE transport
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.name = name
        self._command = command
        self._args = args or []
        self._env = env or {}
        self._url = url
        self._headers = headers or {}

        self._transport: MCPTransport | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._request_id: int = 0
        self._connected: bool = False
        self._server_info: MCPServerInfo | None = None
        self._tools: list[MCPToolSchema] = []

        # Pending response futures for stdio transport
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def server_info(self) -> MCPServerInfo | None:
        return self._server_info

    @property
    def tools(self) -> list[MCPToolSchema]:
        return list(self._tools)

    async def connect(self) -> MCPServerInfo:
        """Connect to the MCP server and perform initialization handshake."""
        if self._url:
            self._transport = MCPTransport.SSE
            await self._connect_sse()
        elif self._command:
            self._transport = MCPTransport.STDIO
            await self._connect_stdio()
        else:
            raise ValueError(f"MCP server '{self.name}': must specify 'command' (stdio) or 'url' (SSE)")

        # Initialize
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "cadence", "version": "0.1.0"},
        })

        self._server_info = MCPServerInfo(
            name=result.get("serverInfo", {}).get("name", self.name),
            version=result.get("serverInfo", {}).get("version", "unknown"),
            capabilities=result.get("capabilities", {}),
        )

        # Send initialized notification
        await self._send_notification("notifications/initialized", {})

        self._connected = True
        logger.info(
            "MCP server '%s' connected: %s v%s",
            self.name, self._server_info.name, self._server_info.version,
        )

        # Pre-fetch tools
        await self.list_tools()

        return self._server_info

    async def disconnect(self) -> None:
        """Gracefully disconnect from the MCP server."""
        self._connected = False

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._process:
            self._process.stdin.close()  # type: ignore[union-attr]
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        # Reject any pending futures
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("MCP client disconnected"))
        self._pending.clear()

        logger.info("MCP server '%s' disconnected", self.name)

    async def list_tools(self) -> list[MCPToolSchema]:
        """Fetch the list of tools from the MCP server."""
        result = await self._send_request("tools/list", {})
        self._tools = [
            MCPToolSchema(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in result.get("tools", [])
        ]
        return list(self._tools)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool on the MCP server and return the text result."""
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        # MCP returns content as a list of content blocks
        content_blocks = result.get("content", [])
        texts = []
        for block in content_blocks:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "image":
                texts.append(f"[image: {block.get('mimeType', 'unknown')}]")
            else:
                texts.append(str(block))

        is_error = result.get("isError", False)
        output = "\n".join(texts) if texts else "(no output)"
        if is_error:
            output = f"MCP tool error: {output}"

        return output

    # ------------------------------------------------------------------
    # stdio transport
    # ------------------------------------------------------------------

    async def _connect_stdio(self) -> None:
        """Launch the MCP server subprocess."""
        import os

        env = {**os.environ, **self._env}
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._reader_task = asyncio.create_task(self._stdio_reader())

    async def _stdio_reader(self) -> None:
        """Background task that reads JSON-RPC responses from stdout."""
        assert self._process and self._process.stdout
        buffer = b""
        while True:
            try:
                chunk = await self._process.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk

                # Process complete lines (JSON-RPC messages are newline-delimited)
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        self._handle_message(msg)
                    except json.JSONDecodeError:
                        logger.debug("MCP '%s': non-JSON line: %s", self.name, line[:100])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("MCP '%s' reader error: %s", self.name, e)
                break

    def _handle_message(self, msg: dict[str, Any]) -> None:
        """Route an incoming JSON-RPC message to the appropriate handler."""
        msg_id = msg.get("id")
        if msg_id is not None and msg_id in self._pending:
            fut = self._pending.pop(msg_id)
            if "error" in msg:
                fut.set_exception(MCPError(
                    msg["error"].get("code", -1),
                    msg["error"].get("message", "Unknown MCP error"),
                ))
            else:
                fut.set_result(msg.get("result", {}))

    async def _send_stdio(self, message: dict[str, Any]) -> None:
        """Write a JSON-RPC message to the subprocess stdin."""
        assert self._process and self._process.stdin
        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

    # ------------------------------------------------------------------
    # SSE transport
    # ------------------------------------------------------------------

    async def _connect_sse(self) -> None:
        """Set up the HTTP client for SSE transport."""
        self._http_client = httpx.AsyncClient(
            headers=self._headers,
            timeout=httpx.Timeout(30.0, read=120.0),
        )

    async def _send_sse(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request via HTTP POST and return the result."""
        assert self._http_client and self._url
        response = await self._http_client.post(
            self._url,
            json=message,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise MCPError(
                data["error"].get("code", -1),
                data["error"].get("message", "Unknown MCP error"),
            )
        return data.get("result", {})

    # ------------------------------------------------------------------
    # JSON-RPC helpers
    # ------------------------------------------------------------------

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        self._request_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        if self._transport == MCPTransport.SSE:
            return await self._send_sse(msg)

        # stdio transport — use pending futures
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[self._request_id] = fut
        await self._send_stdio(msg)

        try:
            return await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending.pop(self._request_id, None)
            raise MCPError(-1, f"MCP request timed out: {method}")

    async def _send_notification(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        if self._transport == MCPTransport.SSE:
            assert self._http_client and self._url
            await self._http_client.post(
                self._url,
                json=msg,
                headers={"Content-Type": "application/json"},
            )
        else:
            await self._send_stdio(msg)


class MCPError(Exception):
    """Error returned by an MCP server."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"MCP error {code}: {message}")
