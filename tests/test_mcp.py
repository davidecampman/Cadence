"""Tests for MCP (Model Context Protocol) integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentinel.core.types import PermissionTier
from sentinel.tools.base import ToolRegistry
from sentinel.mcp.client import MCPClient, MCPManager
from sentinel.mcp.bridge import MCPTool, register_mcp_tools
from sentinel.core.config import Config, MCPConfig, MCPServerConfig


# --- MCPTool bridge tests ---

class TestMCPTool:
    def test_tool_naming(self):
        """MCP tools are prefixed with mcp_<server>__ to avoid collisions."""
        client = MagicMock(spec=MCPClient)
        tool = MCPTool(
            server_name="cursor",
            tool_name="read_file",
            tool_description="Read a file from disk",
            tool_parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            client=client,
        )
        assert tool.name == "mcp_cursor__read_file"
        assert "cursor" in tool.description
        assert tool.permission_tier == PermissionTier.PRIVILEGED

    def test_tool_definition(self):
        """MCPTool produces a valid ToolDefinition."""
        client = MagicMock(spec=MCPClient)
        tool = MCPTool(
            server_name="test",
            tool_name="my_tool",
            tool_description="A test tool",
            tool_parameters={"type": "object", "properties": {}},
            client=client,
        )
        defn = tool.definition()
        assert defn.name == "mcp_test__my_tool"
        assert defn.description == "[MCP:test] A test tool"

    @pytest.mark.asyncio
    async def test_execute_forwards_to_client(self):
        """execute() delegates to the MCP client's call_tool."""
        client = AsyncMock(spec=MCPClient)
        client.call_tool.return_value = "result text"

        tool = MCPTool(
            server_name="srv",
            tool_name="do_thing",
            tool_description="Does a thing",
            tool_parameters={},
            client=client,
        )
        result = await tool.execute(arg1="value1")
        client.call_tool.assert_awaited_once_with("do_thing", {"arg1": "value1"})
        assert result == "result text"


# --- register_mcp_tools tests ---

class TestRegisterMCPTools:
    def test_registers_all_tools(self):
        """All tools from all servers get registered in the ToolRegistry."""
        manager = MagicMock(spec=MCPManager)
        client = MagicMock(spec=MCPClient)

        manager.all_tools.return_value = [
            ("server_a", {"name": "tool1", "description": "Tool 1", "input_schema": {}}),
            ("server_a", {"name": "tool2", "description": "Tool 2", "input_schema": {}}),
            ("server_b", {"name": "tool3", "description": "Tool 3", "input_schema": {}}),
        ]
        manager.get_client.return_value = client
        manager.clients = {"server_a": client, "server_b": client}

        registry = ToolRegistry()
        count = register_mcp_tools(manager, registry)

        assert count == 3
        assert registry.get("mcp_server_a__tool1") is not None
        assert registry.get("mcp_server_a__tool2") is not None
        assert registry.get("mcp_server_b__tool3") is not None

    def test_empty_manager_registers_nothing(self):
        """No tools registered when no MCP servers are connected."""
        manager = MagicMock(spec=MCPManager)
        manager.all_tools.return_value = []
        manager.clients = {}

        registry = ToolRegistry()
        count = register_mcp_tools(manager, registry)
        assert count == 0


# --- MCPManager tests ---

class TestMCPManager:
    def test_initial_state(self):
        manager = MCPManager()
        assert len(manager.clients) == 0
        assert manager.all_tools() == []

    def test_get_client_missing(self):
        manager = MCPManager()
        assert manager.get_client("nonexistent") is None


# --- Config tests ---

class TestMCPConfig:
    def test_default_config_disabled(self):
        """MCP is disabled by default."""
        config = Config()
        assert config.mcp.enabled is False
        assert config.mcp.servers == {}

    def test_config_with_servers(self):
        """MCP config can define multiple servers."""
        config = Config(
            mcp=MCPConfig(
                enabled=True,
                servers={
                    "cursor-fs": MCPServerConfig(
                        transport="stdio",
                        command="npx",
                        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    ),
                    "custom-sse": MCPServerConfig(
                        transport="sse",
                        url="http://localhost:3000/sse",
                    ),
                    "disabled-server": MCPServerConfig(
                        transport="stdio",
                        command="echo",
                        enabled=False,
                    ),
                },
            )
        )
        assert config.mcp.enabled is True
        assert len(config.mcp.servers) == 3
        assert config.mcp.servers["cursor-fs"].transport == "stdio"
        assert config.mcp.servers["custom-sse"].url == "http://localhost:3000/sse"
        assert config.mcp.servers["disabled-server"].enabled is False
