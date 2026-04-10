"""MCP server manager — lifecycle management for multiple MCP servers.

Reads MCP server definitions from the Cadence config and manages connections,
tool bridging, and graceful shutdown.
"""

from __future__ import annotations

import logging
from typing import Any

from cadence.core.types import PermissionTier
from cadence.mcp.bridge import MCPBridgedTool, bridge_mcp_tools
from cadence.mcp.client import MCPClient
from cadence.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages a fleet of MCP server connections.

    Usage::

        manager = MCPManager()
        manager.add_server("github", command="npx", args=["-y", "@modelcontextprotocol/server-github"],
                           env={"GITHUB_TOKEN": "..."})
        manager.add_server("slack", url="http://localhost:3001/mcp")

        await manager.connect_all(registry)  # bridges tools into the Cadence registry
        # ... agents can now use mcp_github_* and mcp_slack_* tools ...
        await manager.disconnect_all()
    """

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}
        self._bridged_tools: dict[str, list[MCPBridgedTool]] = {}

    @property
    def servers(self) -> list[str]:
        """Names of all registered MCP servers."""
        return list(self._clients.keys())

    def add_server(
        self,
        name: str,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Register an MCP server configuration (does not connect yet)."""
        if name in self._clients:
            logger.warning("MCP server '%s' already registered, replacing", name)
            # Will be disconnected on next connect_all or explicit disconnect

        self._clients[name] = MCPClient(
            name=name,
            command=command,
            args=args,
            env=env,
            url=url,
            headers=headers,
        )

    def add_servers_from_config(self, servers_config: list[dict[str, Any]]) -> None:
        """Register MCP servers from config-style dicts.

        Each dict should have:
          - name (str): Server identifier
          - command (str, optional): Executable for stdio transport
          - args (list[str], optional): Arguments for the command
          - env (dict, optional): Environment variables
          - url (str, optional): URL for SSE transport
          - headers (dict, optional): HTTP headers for SSE
        """
        for server_def in servers_config:
            name = server_def.get("name")
            if not name:
                logger.warning("MCP server config missing 'name', skipping: %s", server_def)
                continue

            self.add_server(
                name=name,
                command=server_def.get("command"),
                args=server_def.get("args"),
                env=server_def.get("env"),
                url=server_def.get("url"),
                headers=server_def.get("headers"),
            )

    async def connect_all(
        self,
        registry: ToolRegistry,
        permission_tier: PermissionTier = PermissionTier.PRIVILEGED,
    ) -> dict[str, int]:
        """Connect all registered MCP servers and bridge their tools.

        Returns a dict of ``{server_name: num_tools_registered}``.
        """
        results: dict[str, int] = {}
        for name, client in self._clients.items():
            try:
                tools = await bridge_mcp_tools(client, registry, permission_tier)
                self._bridged_tools[name] = tools
                results[name] = len(tools)
            except Exception as e:
                logger.error("Failed to connect MCP server '%s': %s", name, e)
                results[name] = 0
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all connected MCP servers."""
        for name, client in self._clients.items():
            if client.connected:
                try:
                    await client.disconnect()
                except Exception as e:
                    logger.error("Error disconnecting MCP server '%s': %s", name, e)
        self._bridged_tools.clear()

    async def disconnect_server(self, name: str) -> None:
        """Disconnect a specific MCP server."""
        client = self._clients.get(name)
        if client and client.connected:
            await client.disconnect()
            self._bridged_tools.pop(name, None)

    def get_client(self, name: str) -> MCPClient | None:
        """Get the MCPClient for a specific server."""
        return self._clients.get(name)

    def status(self) -> dict[str, dict[str, Any]]:
        """Return connection status for all servers."""
        return {
            name: {
                "connected": client.connected,
                "tools": len(self._bridged_tools.get(name, [])),
                "server_info": {
                    "name": client.server_info.name,
                    "version": client.server_info.version,
                } if client.server_info else None,
            }
            for name, client in self._clients.items()
        }
