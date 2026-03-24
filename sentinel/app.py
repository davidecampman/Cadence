"""Application bootstrap — wires all components together."""

from __future__ import annotations

from sentinel.core.config import Config, load_config
from sentinel.core.trace import TraceLogger
from sentinel.agents.orchestrator import Orchestrator
from sentinel.memory.store import MemoryStore
from sentinel.skills.loader import SkillLoader
from sentinel.routing.router import SmartRouter
from sentinel.tools.base import ToolRegistry
from sentinel.tools.code_execution import CodeExecutionTool, ShellTool
from sentinel.tools.database import SqlQueryTool
from sentinel.tools.delegate import DelegateTool
from sentinel.tools.environment import CheckDependencyTool, EnvInfoTool, InstallPackageTool
from sentinel.tools.file_ops import ListFilesTool, ReadFileTool, SearchFilesTool, WriteFileTool
from sentinel.tools.git_ops import GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool
from sentinel.tools.http_client import HttpRequestTool
from sentinel.tools.memory_tools import MemoryDeleteTool, MemoryQueryTool, MemorySaveTool
from sentinel.tools.scratchpad import ScratchReadTool, ScratchWriteTool
from sentinel.tools.text_tools import DiffPatchTool, RegexReplaceTool, SummarizeTextTool
from sentinel.tools.vision import ImageDescribeTool, ScreenshotTool
from sentinel.tools.web import WebFetchTool


class SentinelApp:
    """Main application — bootstraps all components and provides the run interface."""

    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path)
        self.trace = TraceLogger(
            trace_file=self.config.logging.trace_file,
            console=self.config.logging.rich_console,
        )
        self.memory = MemoryStore()
        self.router = SmartRouter(self.config)
        self.skills = SkillLoader(self.config.skills.directories)
        self.tools = self._build_tool_registry()
        self.orchestrator = Orchestrator(
            tool_registry=self.tools,
            trace=self.trace,
            config=self.config,
            skill_loader=self.skills,
        )

    def _build_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()

        # File operations
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(ListFilesTool())
        registry.register(SearchFilesTool())

        # Code execution
        registry.register(CodeExecutionTool())
        registry.register(ShellTool())

        # Web & HTTP
        registry.register(WebFetchTool())
        registry.register(HttpRequestTool())

        # Git
        registry.register(GitStatusTool())
        registry.register(GitDiffTool())
        registry.register(GitCommitTool())
        registry.register(GitLogTool())

        # Database
        registry.register(SqlQueryTool())

        # Text processing
        registry.register(RegexReplaceTool())
        registry.register(DiffPatchTool())
        registry.register(SummarizeTextTool())

        # Vision
        registry.register(ScreenshotTool())
        registry.register(ImageDescribeTool())

        # Environment
        registry.register(EnvInfoTool())
        registry.register(InstallPackageTool())
        registry.register(CheckDependencyTool())

        # Scratchpad
        registry.register(ScratchWriteTool())
        registry.register(ScratchReadTool())

        # Memory
        registry.register(MemorySaveTool(self.memory))
        registry.register(MemoryQueryTool(self.memory))
        registry.register(MemoryDeleteTool(self.memory))

        # Delegation
        registry.register(DelegateTool(
            tool_registry=registry,
            trace=self.trace,
            config=self.config,
            max_depth=self.config.agents.max_depth,
            skill_loader=self.skills,
        ))

        return registry

    async def run(
        self,
        user_input: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Process a user request through the orchestrator."""
        return await self.orchestrator.run(
            user_input, conversation_history=conversation_history or []
        )

    def discover_skills(self) -> int:
        """Load skills from configured directories."""
        skills = self.skills.discover()
        return len(skills)
