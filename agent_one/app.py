"""Application bootstrap — wires all components together."""

from __future__ import annotations

from agent_one.core.config import Config, load_config
from agent_one.core.trace import TraceLogger
from agent_one.agents.orchestrator import Orchestrator
from agent_one.memory.backend import create_memory_backend
from agent_one.memory.sessions import SessionStore
from agent_one.skills.loader import SkillLoader
from agent_one.routing.router import SmartRouter
from agent_one.tools.base import ToolRegistry
from agent_one.tools.code_execution import CodeExecutionTool, ShellTool
from agent_one.tools.delegate import DelegateTool
from agent_one.tools.file_ops import ListFilesTool, ReadFileTool, SearchFilesTool, WriteFileTool
from agent_one.tools.memory_tools import MemoryDeleteTool, MemoryQueryTool, MemorySaveTool
from agent_one.tools.web import WebFetchTool


class AgentOneApp:
    """Main application — bootstraps all components and provides the run interface."""

    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path)
        self.trace = TraceLogger(
            trace_file=self.config.logging.trace_file,
            console=self.config.logging.rich_console,
        )
        self.memory = create_memory_backend()
        self.sessions = SessionStore() if self.config.memory.session_persistence else None
        self.router = SmartRouter(self.config)
        self.skills = SkillLoader(self.config.skills.directories)
        self.tools = self._build_tool_registry()
        self.orchestrator = Orchestrator(
            tool_registry=self.tools,
            trace=self.trace,
            config=self.config,
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

        # Web
        registry.register(WebFetchTool())

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
        ))

        return registry

    async def run(self, user_input: str) -> str:
        """Process a user request through the orchestrator."""
        return await self.orchestrator.run(user_input)

    def discover_skills(self) -> int:
        """Load skills from configured directories."""
        skills = self.skills.discover()
        return len(skills)
