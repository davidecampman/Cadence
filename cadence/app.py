"""Application bootstrap — wires all components together."""

from __future__ import annotations

import logging

from cadence.core.config import Config, load_config
from cadence.core.checkpoint import CheckpointManager
from cadence.core.message_bus import MessageBus
from cadence.core.trace import TraceLogger
from cadence.agents.orchestrator import Orchestrator
from cadence.knowledge.graph import KnowledgeGraph
from cadence.knowledge.store import KnowledgeStore
from cadence.learning.store import LearningStore
from cadence.mcp.manager import MCPManager
from cadence.memory.store import MemoryStore
from cadence.prompts.evolution import PromptEvolver
from cadence.prompts.store import PromptEvolutionStore
from cadence.skills.loader import SkillLoader
from cadence.routing.router import SmartRouter
from cadence.tools.base import ToolRegistry

logger = logging.getLogger(__name__)
from cadence.tools.checkpoint_tools import RequestApprovalTool
from cadence.tools.code_execution import CodeExecutionTool, ShellTool
from cadence.tools.database import SqlQueryTool
from cadence.tools.delegate import DelegateTool
from cadence.tools.environment import CheckDependencyTool, EnvInfoTool, InstallPackageTool
from cadence.tools.file_ops import EditFileTool, GrepTool, ListFilesTool, ReadFileTool, SearchFilesTool, WriteFileTool
from cadence.tools.git_ops import GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool
from cadence.tools.graph_tools import GraphAddEntityTool, GraphAddRelationTool, GraphQueryTool
from cadence.tools.http_client import HttpRequestTool
from cadence.tools.learning_tools import LearningInsightsTool, LearningStatsTool
from cadence.tools.memory_tools import MemoryDeleteTool, MemoryQueryTool, MemorySaveTool
from cadence.tools.message_bus_tools import BusPeekTool, BusPublishTool
from cadence.tools.prompt_tools import (
    PromptHistoryTool,
    PromptModifyTool,
    PromptRollbackTool,
    PromptViewTool,
)
from cadence.tools.scratchpad import ScratchReadTool, ScratchWriteTool
from cadence.tools.text_tools import DiffPatchTool, RegexReplaceTool, SummarizeTextTool
from cadence.tools.vision import ImageDescribeTool, ScreenshotTool
from cadence.tools.browser import (
    BrowseWebTool,
    BrowserClickTool,
    BrowserExtractTool,
    BrowserFormTool,
    BrowserScreenshotTool,
)
from cadence.tools.knowledge_tools import KBDeleteTool, KBIngestTool, KBListTool, KBSearchTool
from cadence.tools.web import WebFetchTool


class CadenceApp:
    """Main application — bootstraps all components and provides the run interface."""

    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path)
        self.trace = TraceLogger(
            trace_file=self.config.logging.trace_file,
            console=self.config.logging.rich_console,
        )
        self.memory = MemoryStore()
        self.knowledge = KnowledgeStore()
        self.router = SmartRouter(self.config)
        self.skills = SkillLoader(self.config.skills.directories)

        # Initialize prompt evolution system
        if self.config.prompt_evolution.enabled:
            self.prompt_evolution_store = PromptEvolutionStore(
                db_path=self.config.prompt_evolution.persist_dir,
            )
            self.prompt_evolver = PromptEvolver(
                store=self.prompt_evolution_store,
                config=self.config,
            )
        else:
            self.prompt_evolution_store = None
            self.prompt_evolver = None

        # Initialize message bus
        self.message_bus = MessageBus(
            history_limit=self.config.message_bus.history_limit,
        ) if self.config.message_bus.enabled else None

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager() if self.config.checkpoints.enabled else None

        # Initialize cross-session learning
        self.learning_store = LearningStore(
            db_path=self.config.learning.persist_dir,
        ) if self.config.learning.enabled else None

        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph(
            persist_path=self.config.knowledge_graph.persist_path,
        ) if self.config.knowledge_graph.enabled else None

        # Initialize MCP manager
        self.mcp_manager = MCPManager()
        if self.config.mcp.enabled and self.config.mcp.servers:
            self.mcp_manager.add_servers_from_config(
                [s.model_dump() for s in self.config.mcp.servers]
            )

        self.tools = self._build_tool_registry()
        self.orchestrator = Orchestrator(
            tool_registry=self.tools,
            trace=self.trace,
            config=self.config,
            skill_loader=self.skills,
            prompt_evolver=self.prompt_evolver,
            message_bus=self.message_bus,
            checkpoint_manager=self.checkpoint_manager,
            learning_store=self.learning_store,
        )

    def _build_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()

        # File operations
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(EditFileTool())
        registry.register(ListFilesTool())
        registry.register(SearchFilesTool())
        registry.register(GrepTool())

        # Code execution
        registry.register(CodeExecutionTool())
        registry.register(ShellTool())

        # Web & HTTP
        registry.register(WebFetchTool())
        registry.register(HttpRequestTool())

        # Browser (Playwright)
        registry.register(BrowseWebTool())
        registry.register(BrowserClickTool())
        registry.register(BrowserFormTool())
        registry.register(BrowserScreenshotTool())
        registry.register(BrowserExtractTool())

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

        # Knowledge base
        registry.register(KBIngestTool(self.knowledge))
        registry.register(KBSearchTool(self.knowledge))
        registry.register(KBListTool(self.knowledge))
        registry.register(KBDeleteTool(self.knowledge))

        # Prompt evolution (self-modifying prompts)
        if self.prompt_evolution_store:
            registry.register(PromptViewTool(self.prompt_evolution_store))
            registry.register(PromptModifyTool(self.prompt_evolution_store))
            registry.register(PromptRollbackTool(self.prompt_evolution_store))
            registry.register(PromptHistoryTool(self.prompt_evolution_store))

        # Message bus (inter-agent communication)
        if self.message_bus:
            registry.register(BusPublishTool(self.message_bus))
            registry.register(BusPeekTool(self.message_bus))

        # Checkpoints (human-in-the-loop)
        if self.checkpoint_manager:
            registry.register(RequestApprovalTool(self.checkpoint_manager))

        # Knowledge graph
        if self.knowledge_graph:
            registry.register(GraphAddEntityTool(self.knowledge_graph))
            registry.register(GraphAddRelationTool(self.knowledge_graph))
            registry.register(GraphQueryTool(self.knowledge_graph))

        # Cross-session learning
        if self.learning_store:
            registry.register(LearningInsightsTool(self.learning_store))
            registry.register(LearningStatsTool(self.learning_store))

        # Delegation
        registry.register(DelegateTool(
            tool_registry=registry,
            trace=self.trace,
            config=self.config,
            max_depth=self.config.agents.max_depth,
            skill_loader=self.skills,
        ))

        return registry

    async def connect_mcp_servers(self) -> dict[str, int]:
        """Connect to configured MCP servers and bridge their tools into the registry.

        Should be called once after construction (requires async). Returns a dict
        of ``{server_name: num_tools_registered}``.
        """
        if not self.config.mcp.enabled or not self.mcp_manager.servers:
            return {}

        results = await self.mcp_manager.connect_all(self.tools)
        for name, count in results.items():
            if count > 0:
                logger.info("MCP server '%s': registered %d tools", name, count)
            else:
                logger.warning("MCP server '%s': no tools registered", name)
        return results

    async def disconnect_mcp_servers(self) -> None:
        """Gracefully disconnect all MCP servers."""
        await self.mcp_manager.disconnect_all()

    async def run(
        self,
        user_input: str,
        conversation_history: list[dict[str, str]] | None = None,
        images: list[dict] | None = None,
    ) -> str:
        """Process a user request through the orchestrator."""
        return await self.orchestrator.run(
            user_input, conversation_history=conversation_history or [], images=images
        )

    def discover_skills(self) -> int:
        """Load skills from configured directories."""
        skills = self.skills.discover()
        return len(skills)
