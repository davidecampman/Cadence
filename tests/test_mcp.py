"""Tests for MCP (Model Context Protocol) server integration."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cadence.core.config import Config, MCPConfig, MCPServerDef
from cadence.core.types import PermissionTier
from cadence.mcp.bridge import MCPBridgedTool, bridge_mcp_tools
from cadence.mcp.client import MCPClient, MCPError, MCPServerInfo, MCPToolSchema, MCPTransport
from cadence.mcp.manager import MCPManager
from cadence.tools.base import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> ToolRegistry:
    return ToolRegistry()


def _mock_tool_schemas() -> list[MCPToolSchema]:
    return [
        MCPToolSchema(
            name="search_repos",
            description="Search GitHub repositories",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        ),
        MCPToolSchema(
            name="get_issue",
            description="Get a GitHub issue by number",
            input_schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "number": {"type": "integer"},
                },
                "required": ["repo", "number"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# MCPToolSchema tests
# ---------------------------------------------------------------------------

class TestMCPToolSchema:
    def test_schema_fields(self):
        schema = MCPToolSchema(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
        )
        assert schema.name == "test_tool"
        assert schema.description == "A test tool"
        assert schema.input_schema["type"] == "object"

    def test_schema_defaults(self):
        schema = MCPToolSchema(name="minimal", description="")
        assert schema.input_schema == {}


# ---------------------------------------------------------------------------
# MCPClient tests
# ---------------------------------------------------------------------------

class TestMCPClient:
    def test_init_stdio(self):
        client = MCPClient(name="test", command="echo", args=["hello"])
        assert client.name == "test"
        assert not client.connected
        assert client.server_info is None
        assert client.tools == []

    def test_init_sse(self):
        client = MCPClient(name="test", url="http://localhost:3001/mcp")
        assert client.name == "test"
        assert not client.connected

    @pytest.mark.asyncio
    async def test_connect_requires_command_or_url(self):
        client = MCPClient(name="test")
        with pytest.raises(ValueError, match="must specify"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_list_tools_parses_response(self):
        client = MCPClient(name="test", command="echo")
        client._connected = True

        tool_response = {
            "tools": [
                {
                    "name": "my_tool",
                    "description": "Does things",
                    "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            ]
        }

        with patch.object(client, "_send_request", new=AsyncMock(return_value=tool_response)):
            tools = await client.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "my_tool"
        assert tools[0].description == "Does things"
        assert "x" in tools[0].input_schema["properties"]

    @pytest.mark.asyncio
    async def test_call_tool_returns_text(self):
        client = MCPClient(name="test", command="echo")
        client._connected = True

        call_response = {
            "content": [
                {"type": "text", "text": "Found 42 results"},
            ],
            "isError": False,
        }

        with patch.object(client, "_send_request", new=AsyncMock(return_value=call_response)):
            result = await client.call_tool("search", {"query": "test"})

        assert result == "Found 42 results"

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self):
        client = MCPClient(name="test", command="echo")
        client._connected = True

        call_response = {
            "content": [
                {"type": "text", "text": "Invalid query"},
            ],
            "isError": True,
        }

        with patch.object(client, "_send_request", new=AsyncMock(return_value=call_response)):
            result = await client.call_tool("search", {"query": ""})

        assert "MCP tool error" in result
        assert "Invalid query" in result

    @pytest.mark.asyncio
    async def test_call_tool_multiple_content_blocks(self):
        client = MCPClient(name="test", command="echo")
        client._connected = True

        call_response = {
            "content": [
                {"type": "text", "text": "Line 1"},
                {"type": "text", "text": "Line 2"},
                {"type": "image", "mimeType": "image/png"},
            ],
        }

        with patch.object(client, "_send_request", new=AsyncMock(return_value=call_response)):
            result = await client.call_tool("multi", {})

        assert "Line 1" in result
        assert "Line 2" in result
        assert "[image: image/png]" in result

    @pytest.mark.asyncio
    async def test_call_tool_empty_content(self):
        client = MCPClient(name="test", command="echo")
        client._connected = True

        call_response = {"content": []}

        with patch.object(client, "_send_request", new=AsyncMock(return_value=call_response)):
            result = await client.call_tool("empty", {})

        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_disconnect(self):
        client = MCPClient(name="test", command="echo")
        client._connected = True

        # No process/http to clean up — just verify state change
        await client.disconnect()
        assert not client.connected

    def test_handle_message_resolves_pending(self):
        client = MCPClient(name="test", command="echo")
        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        client._pending[1] = fut

        client._handle_message({"id": 1, "result": {"data": "ok"}})
        assert fut.done()
        assert fut.result() == {"data": "ok"}
        loop.close()

    def test_handle_message_error(self):
        client = MCPClient(name="test", command="echo")
        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        client._pending[2] = fut

        client._handle_message({
            "id": 2,
            "error": {"code": -32600, "message": "Invalid request"},
        })
        assert fut.done()
        with pytest.raises(MCPError, match="Invalid request"):
            fut.result()
        loop.close()


# ---------------------------------------------------------------------------
# MCPError tests
# ---------------------------------------------------------------------------

class TestMCPError:
    def test_error_fields(self):
        err = MCPError(code=-32600, message="Invalid request")
        assert err.code == -32600
        assert err.message == "Invalid request"
        assert "MCP error -32600" in str(err)


# ---------------------------------------------------------------------------
# MCPBridgedTool tests
# ---------------------------------------------------------------------------

class TestMCPBridgedTool:
    def test_tool_naming(self):
        client = MCPClient(name="github", command="echo")
        schema = MCPToolSchema(
            name="search_repos",
            description="Search repositories",
            input_schema={"type": "object", "properties": {}},
        )
        tool = MCPBridgedTool(client, schema, "github")

        assert tool.name == "mcp_github_search_repos"
        assert "[MCP:github]" in tool.description
        assert tool.permission_tier == PermissionTier.PRIVILEGED

    def test_tool_parameters(self):
        client = MCPClient(name="fs", command="echo")
        schema = MCPToolSchema(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        tool = MCPBridgedTool(client, schema, "fs")

        assert tool.parameters["properties"]["path"]["type"] == "string"
        assert "path" in tool.parameters["required"]

    @pytest.mark.asyncio
    async def test_tool_execute_delegates_to_client(self):
        client = MCPClient(name="test", command="echo")
        client._connected = True

        schema = MCPToolSchema(name="my_tool", description="test")
        tool = MCPBridgedTool(client, schema, "test")

        mock_call = AsyncMock(return_value="tool output")
        with patch.object(client, "call_tool", mock_call):
            result = await tool.execute(arg1="value1")

        mock_call.assert_called_once_with("my_tool", {"arg1": "value1"})
        assert result == "tool output"

    @pytest.mark.asyncio
    async def test_tool_execute_when_disconnected(self):
        client = MCPClient(name="offline", command="echo")
        client._connected = False

        schema = MCPToolSchema(name="my_tool", description="test")
        tool = MCPBridgedTool(client, schema, "offline")

        result = await tool.execute(arg="val")
        assert "not connected" in result


# ---------------------------------------------------------------------------
# bridge_mcp_tools tests
# ---------------------------------------------------------------------------

class TestBridgeMCPTools:
    @pytest.mark.asyncio
    async def test_bridge_registers_tools(self):
        client = MCPClient(name="github", command="echo")
        client._connected = True
        client._tools = _mock_tool_schemas()

        registry = _make_registry()
        tools = await bridge_mcp_tools(client, registry)

        assert len(tools) == 2
        assert registry.get("mcp_github_search_repos") is not None
        assert registry.get("mcp_github_get_issue") is not None

    @pytest.mark.asyncio
    async def test_bridge_connects_if_not_connected(self):
        client = MCPClient(name="test", command="echo")
        client._connected = False

        async def fake_connect():
            client._connected = True
            client._tools = [MCPToolSchema(name="tool1", description="test")]
            client._server_info = MCPServerInfo(name="test", version="1.0")
            return client._server_info

        with patch.object(client, "connect", side_effect=fake_connect):
            registry = _make_registry()
            tools = await bridge_mcp_tools(client, registry)

        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_bridge_handles_connection_failure(self):
        client = MCPClient(name="broken", command="echo")
        client._connected = False

        with patch.object(client, "connect", side_effect=ConnectionError("refused")):
            registry = _make_registry()
            tools = await bridge_mcp_tools(client, registry)

        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_bridge_skips_name_collisions(self):
        """If a tool name already exists in the registry, skip it."""
        client = MCPClient(name="test", command="echo")
        client._connected = True
        client._tools = [MCPToolSchema(name="existing", description="test")]

        registry = _make_registry()
        # Pre-register a tool with the same bridged name
        existing = MagicMock(spec=Tool)
        existing.name = "mcp_test_existing"
        registry._tools["mcp_test_existing"] = existing

        tools = await bridge_mcp_tools(client, registry)
        assert len(tools) == 0  # Skipped due to collision

    @pytest.mark.asyncio
    async def test_bridge_custom_permission_tier(self):
        client = MCPClient(name="safe", command="echo")
        client._connected = True
        client._tools = [MCPToolSchema(name="read_only_tool", description="test")]

        registry = _make_registry()
        tools = await bridge_mcp_tools(
            client, registry,
            permission_tier=PermissionTier.READ_ONLY,
        )

        assert len(tools) == 1
        assert tools[0].permission_tier == PermissionTier.READ_ONLY


# ---------------------------------------------------------------------------
# MCPManager tests
# ---------------------------------------------------------------------------

class TestMCPManager:
    def test_add_server(self):
        manager = MCPManager()
        manager.add_server("github", command="npx", args=["-y", "server-github"])
        assert "github" in manager.servers

    def test_add_server_replaces_existing(self):
        manager = MCPManager()
        manager.add_server("github", command="old")
        manager.add_server("github", command="new")
        assert len(manager.servers) == 1

    def test_add_servers_from_config(self):
        manager = MCPManager()
        manager.add_servers_from_config([
            {"name": "github", "command": "npx", "args": ["-y", "server-github"]},
            {"name": "slack", "url": "http://localhost:3001/mcp"},
        ])
        assert set(manager.servers) == {"github", "slack"}

    def test_add_servers_from_config_skips_nameless(self):
        manager = MCPManager()
        manager.add_servers_from_config([
            {"command": "npx"},  # no name
            {"name": "valid", "command": "echo"},
        ])
        assert manager.servers == ["valid"]

    @pytest.mark.asyncio
    async def test_connect_all(self):
        manager = MCPManager()
        manager.add_server("test1", command="echo")
        manager.add_server("test2", command="echo")

        # Mock bridge_mcp_tools to return fake tools
        async def fake_bridge(client, registry, permission_tier):
            client._connected = True
            return [MagicMock(spec=MCPBridgedTool)]

        with patch("cadence.mcp.manager.bridge_mcp_tools", side_effect=fake_bridge):
            registry = _make_registry()
            results = await manager.connect_all(registry)

        assert results == {"test1": 1, "test2": 1}

    @pytest.mark.asyncio
    async def test_connect_all_handles_failure(self):
        manager = MCPManager()
        manager.add_server("broken", command="echo")

        async def failing_bridge(client, registry, permission_tier):
            raise ConnectionError("refused")

        with patch("cadence.mcp.manager.bridge_mcp_tools", side_effect=failing_bridge):
            registry = _make_registry()
            results = await manager.connect_all(registry)

        assert results == {"broken": 0}

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        manager = MCPManager()
        manager.add_server("test", command="echo")

        client = manager.get_client("test")
        client._connected = True

        mock_disconnect = AsyncMock()
        with patch.object(client, "disconnect", mock_disconnect):
            await manager.disconnect_all()

        mock_disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_server(self):
        manager = MCPManager()
        manager.add_server("test", command="echo")

        client = manager.get_client("test")
        client._connected = True

        mock_disconnect = AsyncMock()
        with patch.object(client, "disconnect", mock_disconnect):
            await manager.disconnect_server("test")

        mock_disconnect.assert_called_once()

    def test_status(self):
        manager = MCPManager()
        manager.add_server("test", command="echo")

        status = manager.status()
        assert "test" in status
        assert status["test"]["connected"] is False
        assert status["test"]["tools"] == 0
        assert status["test"]["server_info"] is None

    def test_get_client(self):
        manager = MCPManager()
        manager.add_server("test", command="echo")

        client = manager.get_client("test")
        assert client is not None
        assert client.name == "test"

        assert manager.get_client("nonexistent") is None


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------

class TestMCPConfig:
    def test_mcp_config_defaults(self):
        config = Config()
        assert config.mcp.enabled is False
        assert config.mcp.servers == []

    def test_mcp_config_with_servers(self):
        config = Config(mcp=MCPConfig(
            enabled=True,
            servers=[
                MCPServerDef(
                    name="github",
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-github"],
                    env={"GITHUB_TOKEN": "test-token"},
                ),
                MCPServerDef(
                    name="remote",
                    url="http://localhost:3001/mcp",
                    headers={"Authorization": "Bearer token"},
                ),
            ],
        ))
        assert config.mcp.enabled is True
        assert len(config.mcp.servers) == 2
        assert config.mcp.servers[0].name == "github"
        assert config.mcp.servers[0].command == "npx"
        assert config.mcp.servers[0].env["GITHUB_TOKEN"] == "test-token"
        assert config.mcp.servers[1].url == "http://localhost:3001/mcp"

    def test_mcp_server_def_defaults(self):
        server = MCPServerDef(name="test")
        assert server.command is None
        assert server.args == []
        assert server.env == {}
        assert server.url is None
        assert server.headers == {}


# ---------------------------------------------------------------------------
# App integration tests
# ---------------------------------------------------------------------------

class TestAppMCPIntegration:
    def test_app_creates_mcp_manager(self):
        """CadenceApp creates an MCPManager even when MCP is disabled."""
        from cadence.app import CadenceApp

        with patch("cadence.app.load_config", return_value=Config()):
            app = CadenceApp.__new__(CadenceApp)
            app.config = Config()
            app.config.mcp.enabled = False
            app.mcp_manager = MCPManager()

        assert app.mcp_manager is not None
        assert app.mcp_manager.servers == []

    @pytest.mark.asyncio
    async def test_connect_mcp_servers_noop_when_disabled(self):
        """connect_mcp_servers returns empty dict when MCP is disabled."""
        from cadence.app import CadenceApp

        app = MagicMock(spec=CadenceApp)
        app.config = Config()
        app.config.mcp.enabled = False
        app.mcp_manager = MCPManager()
        app.connect_mcp_servers = CadenceApp.connect_mcp_servers.__get__(app)

        result = await app.connect_mcp_servers()
        assert result == {}
