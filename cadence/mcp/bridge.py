"""MCP tool bridge — adapts MCP server tools into Cadence Tool instances.

For each tool exposed by an MCP server, this module creates a Cadence ``Tool``
subclass that delegates execution to the MCP client. The bridged tools are
registered in the normal ToolRegistry and become available to all agents.
"""

from __future__ import annotations

import logging
from typing import Any

from cadence.core.types import PermissionTier
from cadence.mcp.client import MCPClient, MCPToolSchema
from cadence.tools.base import Tool, ToolRegistry

logger = logging.getLogger(__name__)


class MCPBridgedTool(Tool):
    """A Cadence Tool backed by an MCP server tool.

    Each instance wraps a single MCP tool and delegates ``execute()`` to the
    MCP client's ``call_tool()`` method.
    """

    def __init__(
        self,
        client: MCPClient,
        schema: MCPToolSchema,
        server_name: str,
        permission_tier: PermissionTier = PermissionTier.PRIVILEGED,
    ):
        self._client = client
        self._schema = schema
        self._server_name = server_name

        # Cadence Tool interface
        self.name = f"mcp_{server_name}_{schema.name}"
        self.description = (
            f"[MCP:{server_name}] {schema.description}" if schema.description
            else f"MCP tool '{schema.name}' from server '{server_name}'"
        )
        self.parameters = schema.input_schema or {
            "type": "object",
            "properties": {},
        }
        self.permission_tier = permission_tier

    async def execute(self, **kwargs) -> str:
        """Forward execution to the MCP server."""
        if not self._client.connected:
            return f"Error: MCP server '{self._server_name}' is not connected"
        return await self._client.call_tool(self._schema.name, kwargs)


async def bridge_mcp_tools(
    client: MCPClient,
    registry: ToolRegistry,
    permission_tier: PermissionTier = PermissionTier.PRIVILEGED,
) -> list[MCPBridgedTool]:
    """Connect to an MCP server, fetch its tools, and register them in the Cadence registry.

    Returns the list of bridged tools that were registered.
    """
    tools: list[MCPBridgedTool] = []

    if not client.connected:
        try:
            await client.connect()
        except Exception as e:
            logger.error("Failed to connect MCP server '%s': %s", client.name, e)
            return tools

    for schema in client.tools:
        bridged = MCPBridgedTool(
            client=client,
            schema=schema,
            server_name=client.name,
            permission_tier=permission_tier,
        )

        # Avoid collisions with existing tools
        if registry.get(bridged.name):
            logger.warning(
                "MCP tool name collision: '%s' already registered, skipping",
                bridged.name,
            )
            continue

        registry.register(bridged)
        tools.append(bridged)
        logger.debug("Registered MCP tool: %s", bridged.name)

    logger.info(
        "Bridged %d tools from MCP server '%s'",
        len(tools), client.name,
    )
    return tools
