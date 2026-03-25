"""MCP client — connects to external MCP servers via stdio or SSE transports."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class MCPClient:
    """Manages a connection to a single MCP server.

    Supports two transports:
      - stdio: launches a subprocess (e.g., ``npx -y @modelcontextprotocol/server-filesystem /tmp``)
      - sse: connects to an HTTP SSE endpoint (e.g., ``http://localhost:3000/sse``)
    """

    def __init__(
        self,
        name: str,
        transport: str = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
    ):
        self.name = name
        self.transport = transport
        self.command = command
        self.args = args or []
        self.env = env
        self.url = url

        self._session: ClientSession | None = None
        self._cm_stack: list[Any] = []  # context managers to clean up
        self._tools: list[dict[str, Any]] = []

    async def connect(self) -> None:
        """Establish connection to the MCP server and discover tools."""
        if self.transport == "stdio":
            await self._connect_stdio()
        elif self.transport == "sse":
            await self._connect_sse()
        else:
            raise ValueError(f"Unsupported MCP transport: {self.transport}")

        # Discover available tools
        await self._discover_tools()
        logger.info(
            "MCP server '%s' connected — %d tools available",
            self.name,
            len(self._tools),
        )

    async def _connect_stdio(self) -> None:
        if not self.command:
            raise ValueError(f"MCP server '{self.name}': stdio transport requires 'command'")

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        # stdio_client is an async context manager that yields (read, write) streams
        transport_cm = stdio_client(server_params)
        read_stream, write_stream = await transport_cm.__aenter__()
        self._cm_stack.append(transport_cm)

        # ClientSession wraps the streams into an MCP session
        session_cm = ClientSession(read_stream, write_stream)
        self._session = await session_cm.__aenter__()
        self._cm_stack.append(session_cm)

        await self._session.initialize()

    async def _connect_sse(self) -> None:
        if not self.url:
            raise ValueError(f"MCP server '{self.name}': sse transport requires 'url'")

        transport_cm = sse_client(self.url)
        read_stream, write_stream = await transport_cm.__aenter__()
        self._cm_stack.append(transport_cm)

        session_cm = ClientSession(read_stream, write_stream)
        self._session = await session_cm.__aenter__()
        self._cm_stack.append(session_cm)

        await self._session.initialize()

    async def _discover_tools(self) -> None:
        """List tools from the connected MCP server."""
        if not self._session:
            return
        result = await self._session.list_tools()
        self._tools = []
        for tool in result.tools:
            self._tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            })

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return discovered tool definitions."""
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool on the MCP server and return the text result."""
        if not self._session:
            raise RuntimeError(f"MCP server '{self.name}' is not connected")

        result = await self._session.call_tool(tool_name, arguments)
        # MCP returns a list of content blocks; concatenate text parts
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts)

    async def disconnect(self) -> None:
        """Cleanly shut down the MCP connection."""
        for cm in reversed(self._cm_stack):
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                logger.debug("Error closing MCP context manager for '%s'", self.name, exc_info=True)
        self._cm_stack.clear()
        self._session = None
        self._tools = []


class MCPManager:
    """Manages multiple MCP server connections."""

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}

    async def add_server(
        self,
        name: str,
        transport: str = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
    ) -> MCPClient:
        """Add and connect to an MCP server."""
        client = MCPClient(
            name=name,
            transport=transport,
            command=command,
            args=args,
            env=env,
            url=url,
        )
        await client.connect()
        self._clients[name] = client
        return client

    def get_client(self, name: str) -> MCPClient | None:
        return self._clients.get(name)

    @property
    def clients(self) -> dict[str, MCPClient]:
        return self._clients

    def all_tools(self) -> list[tuple[str, dict[str, Any]]]:
        """Return all tools from all servers as (server_name, tool_def) pairs."""
        result = []
        for server_name, client in self._clients.items():
            for tool in client.tools:
                result.append((server_name, tool))
        return result

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()
