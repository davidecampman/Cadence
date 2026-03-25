"""Integration tests for tool execution — file ops, code execution, shell, memory."""

import asyncio
import os
import platform

import pytest

from cadence.core.types import PermissionTier
from cadence.tools.file_ops import (
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from cadence.tools.code_execution import (
    CodeExecutionTool,
    ShellTool,
    _build_resource_limits,
    _check_blocked,
    _shell_quote,
    _wrap_with_sandbox,
)
from cadence.tools.base import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# File operation tools
# ---------------------------------------------------------------------------

class TestReadFileTool:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line one\nline two\nline three\n")

        tool = ReadFileTool()
        result = await tool.run("r1", {"path": str(f)})
        assert result.success
        assert "line one" in result.output
        assert "line two" in result.output
        assert "line three" in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        tool = ReadFileTool()
        result = await tool.run("r2", {"path": "/nonexistent/file.txt"})
        assert result.success  # Tool returns error message as output, not exception
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_read_with_offset_and_limit(self, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("\n".join(f"line {i}" for i in range(20)))

        tool = ReadFileTool()
        result = await tool.run("r3", {"path": str(f), "offset": 5, "limit": 3})
        assert result.success
        assert "line 5" in result.output
        assert "line 7" in result.output
        assert "line 8" not in result.output

    @pytest.mark.asyncio
    async def test_read_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")

        tool = ReadFileTool()
        result = await tool.run("r4", {"path": str(f)})
        assert result.success
        assert "empty file" in result.output.lower()

    @pytest.mark.asyncio
    async def test_read_directory_not_file(self, tmp_path):
        tool = ReadFileTool()
        result = await tool.run("r5", {"path": str(tmp_path)})
        assert result.success
        assert "not a file" in result.output.lower()


class TestWriteFileTool:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        target = tmp_path / "output.txt"

        tool = WriteFileTool()
        result = await tool.run("w1", {"path": str(target), "content": "hello world"})
        assert result.success
        assert "wrote" in result.output.lower()
        assert target.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "file.txt"

        tool = WriteFileTool()
        result = await tool.run("w2", {"path": str(target), "content": "nested content"})
        assert result.success
        assert target.exists()
        assert target.read_text() == "nested content"

    @pytest.mark.asyncio
    async def test_write_blocked_for_sensitive_path(self):
        tool = WriteFileTool()
        result = await tool.run("w3", {"path": "/etc/passwd", "content": "bad"})
        assert result.success  # Returns error message, not exception
        assert "blocked" in result.output.lower() or "protected" in result.output.lower()

    @pytest.mark.asyncio
    async def test_write_includes_file_path_tag(self, tmp_path):
        target = tmp_path / "tagged.txt"

        tool = WriteFileTool()
        result = await tool.run("w4", {"path": str(target), "content": "data"})
        assert result.success
        assert "[[FILE:" in result.output

    @pytest.mark.asyncio
    async def test_overwrite_existing_file(self, tmp_path):
        target = tmp_path / "overwrite.txt"
        target.write_text("original")

        tool = WriteFileTool()
        await tool.run("w5", {"path": str(target), "content": "replaced"})
        assert target.read_text() == "replaced"


class TestListFilesTool:
    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "subdir").mkdir()

        tool = ListFilesTool()
        result = await tool.run("l1", {"pattern": str(tmp_path)})
        assert result.success
        assert "a.txt" in result.output
        assert "b.py" in result.output
        assert "subdir" in result.output

    @pytest.mark.asyncio
    async def test_list_glob_pattern(self, tmp_path):
        (tmp_path / "one.py").write_text("1")
        (tmp_path / "two.py").write_text("2")
        (tmp_path / "three.txt").write_text("3")

        tool = ListFilesTool()
        result = await tool.run("l2", {"pattern": str(tmp_path / "*.py")})
        assert result.success
        assert "one.py" in result.output
        assert "two.py" in result.output
        assert "three.txt" not in result.output

    @pytest.mark.asyncio
    async def test_list_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()

        tool = ListFilesTool()
        result = await tool.run("l3", {"pattern": str(empty)})
        assert result.success
        assert "empty" in result.output.lower()

    @pytest.mark.asyncio
    async def test_list_no_matches(self, tmp_path):
        tool = ListFilesTool()
        result = await tool.run("l4", {"pattern": str(tmp_path / "*.zzz")})
        assert result.success
        assert "no files" in result.output.lower()


class TestSearchFilesTool:
    @pytest.mark.asyncio
    async def test_search_finds_pattern(self, tmp_path):
        (tmp_path / "code.py").write_text("def hello():\n    return 'world'\n")
        (tmp_path / "readme.md").write_text("# Project\nNothing here\n")

        tool = SearchFilesTool()
        result = await tool.run("s1", {"pattern": "hello", "path": str(tmp_path)})
        assert result.success
        assert "hello" in result.output
        assert "code.py" in result.output

    @pytest.mark.asyncio
    async def test_search_no_matches(self, tmp_path):
        (tmp_path / "file.txt").write_text("nothing special here")

        tool = SearchFilesTool()
        result = await tool.run("s2", {"pattern": "zzzzz", "path": str(tmp_path)})
        assert result.success
        assert "no matches" in result.output.lower()

    @pytest.mark.asyncio
    async def test_search_with_glob_filter(self, tmp_path):
        (tmp_path / "code.py").write_text("TODO: fix this")
        (tmp_path / "notes.txt").write_text("TODO: remember this")

        tool = SearchFilesTool()
        result = await tool.run("s3", {
            "pattern": "TODO",
            "path": str(tmp_path),
            "glob": "*.py",
        })
        assert result.success
        assert "code.py" in result.output
        assert "notes.txt" not in result.output

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, tmp_path):
        (tmp_path / "file.txt").write_text("Hello World")

        tool = SearchFilesTool()
        result = await tool.run("s4", {"pattern": "hello world", "path": str(tmp_path)})
        assert result.success
        assert "Hello World" in result.output


# ---------------------------------------------------------------------------
# Code execution tools
# ---------------------------------------------------------------------------

class TestCodeExecutionTool:
    @pytest.mark.asyncio
    async def test_execute_python(self):
        tool = CodeExecutionTool()
        result = await tool.run("c1", {"language": "python", "code": "print(2 + 3)"})
        assert result.success
        assert "5" in result.output

    @pytest.mark.asyncio
    async def test_execute_python_error(self):
        tool = CodeExecutionTool()
        result = await tool.run("c2", {"language": "python", "code": "raise ValueError('boom')"})
        assert result.success  # Tool catches errors and returns them
        assert "ValueError" in result.output or "boom" in result.output

    @pytest.mark.asyncio
    async def test_execute_bash(self):
        tool = CodeExecutionTool()
        result = await tool.run("c3", {"language": "bash", "code": "echo hello"})
        assert result.success
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_execute_unsupported_language(self):
        tool = CodeExecutionTool()
        result = await tool.run("c4", {"language": "cobol", "code": "DISPLAY 'HI'"})
        assert result.success
        assert "unsupported" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_python_multiline(self):
        code = "for i in range(3):\n    print(i)"
        tool = CodeExecutionTool()
        result = await tool.run("c5", {"language": "python", "code": code})
        assert result.success
        assert "0" in result.output
        assert "1" in result.output
        assert "2" in result.output


class TestShellTool:
    @pytest.mark.asyncio
    async def test_shell_basic_command(self, tmp_path):
        tool = ShellTool()
        result = await tool.run("sh1", {
            "command": "echo 'shell works'",
            "working_dir": str(tmp_path),
        })
        assert result.success
        assert "shell works" in result.output

    @pytest.mark.asyncio
    async def test_shell_blocked_command(self):
        tool = ShellTool()
        result = await tool.run("sh2", {"command": "rm -rf /"})
        assert not result.success
        assert "blocked" in result.output.lower() or "error" in result.output.lower()

    @pytest.mark.asyncio
    async def test_shell_working_directory(self, tmp_path):
        tool = ShellTool()
        result = await tool.run("sh3", {
            "command": "pwd",
            "working_dir": str(tmp_path),
        })
        assert result.success
        assert str(tmp_path) in result.output


# ---------------------------------------------------------------------------
# Code execution helpers
# ---------------------------------------------------------------------------

class TestCodeExecutionHelpers:
    def test_shell_quote_basic(self):
        assert _shell_quote("hello") == "'hello'"

    def test_shell_quote_with_single_quotes(self):
        quoted = _shell_quote("it's a test")
        assert "it" in quoted
        assert "'" not in quoted.strip("'") or "\\'" in quoted

    def test_check_blocked_matches(self):
        blocked = ["rm -rf", "mkfs", "dd if="]
        assert _check_blocked("rm -rf /home", blocked) == "rm -rf"
        assert _check_blocked("sudo mkfs.ext4", blocked) == "mkfs"

    def test_check_blocked_no_match(self):
        blocked = ["rm -rf", "mkfs"]
        assert _check_blocked("ls -la", blocked) is None
        assert _check_blocked("echo hello", blocked) is None

    def test_build_resource_limits(self):
        class FakeCfg:
            max_memory_mb = 512
            max_cpu_seconds = 30
            max_file_descriptors = 256
            restrict_network = False

        limits = _build_resource_limits(FakeCfg())
        assert "ulimit -v" in limits
        assert "ulimit -t 30" in limits
        assert "ulimit -n 256" in limits

    def test_build_resource_limits_disabled(self):
        class FakeCfg:
            max_memory_mb = 0
            max_cpu_seconds = 0
            max_file_descriptors = 0
            restrict_network = False

        limits = _build_resource_limits(FakeCfg())
        assert limits == ""


# ---------------------------------------------------------------------------
# Tool registry integration
# ---------------------------------------------------------------------------

class TestToolRegistryIntegration:
    def test_scoped_copy_preserves_non_memory_tools(self):
        reg = ToolRegistry()
        reg.register(ReadFileTool())
        reg.register(WriteFileTool())

        scoped = reg.scoped_copy("agent-123")
        assert scoped.get("read_file") is not None
        assert scoped.get("write_file") is not None

    def test_tool_run_error_handling(self):
        """Tool.run() should catch exceptions and return a failed ToolResult."""
        class BrokenTool(Tool):
            name = "broken"
            description = "Always fails"
            parameters = {}

            async def execute(self, **kwargs) -> str:
                raise RuntimeError("intentional failure")

        async def _test():
            tool = BrokenTool()
            result = await tool.run("t1", {})
            assert not result.success
            assert "RuntimeError" in result.output
            assert "intentional failure" in result.output

        asyncio.get_event_loop().run_until_complete(_test())

    def test_tool_definition(self):
        tool = ReadFileTool()
        defn = tool.definition()
        assert defn.name == "read_file"
        assert defn.permission_tier == PermissionTier.READ_ONLY
        assert "path" in str(defn.parameters)
