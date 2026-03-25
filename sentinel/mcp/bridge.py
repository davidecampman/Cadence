"""Bridge MCP server tools into Sentinel's ToolRegistry."""

from __future__ import annotations

import logging
from typing import Any

from sentinel.core.types import PermissionTier
from sentinel.mcp.client import MCPClient, MCPManager
from sentinel.tools.base import Tool, ToolRegistry

logger = logging.getLogger(__name__)


class MCPTool(Tool):
    """Wraps a single MCP server tool as a Sentinel Tool.

    Tool names are prefixed with ``mcp_<server>__`` to avoid collisions
    with built-in Sentinel tools.
    """

    def __init__(
        self,
        server_name: str,
        tool_name: str,
        tool_description: str,
        tool_parameters: dict[str, Any],
        client: MCPClient,
    ):
        self._server_name = server_name
        self._tool_name = tool_name  # original MCP tool name
        self._client = client

        # Sentinel Tool attributes
        self.name = f"mcp_{server_name}__{tool_name}"
        self.description = f"[MCP:{server_name}] {tool_description}"
        self.parameters = tool_parameters
        self.permission_tier = PermissionTier.PRIVILEGED  # MCP tools access external systems

    async def execute(self, **kwargs) -> str:
        """Forward the call to the MCP server."""
        return await self._client.call_tool(self._tool_name, kwargs)


def register_mcp_tools(
    manager: MCPManager,
    registry: ToolRegistry,
) -> int:
    """Register all MCP server tools into a Sentinel ToolRegistry.

    Returns the number of tools registered.
    """
    count = 0
    for server_name, tool_def in manager.all_tools():
        client = manager.get_client(server_name)
        if not client:
            continue
        mcp_tool = MCPTool(
            server_name=server_name,
            tool_name=tool_def["name"],
            tool_description=tool_def["description"],
            tool_parameters=tool_def.get("input_schema", {}),
            client=client,
        )
        registry.register(mcp_tool)
        count += 1
        logger.debug("Registered MCP tool: %s", mcp_tool.name)

    logger.info("Registered %d MCP tools from %d servers", count, len(manager.clients))
    return count
