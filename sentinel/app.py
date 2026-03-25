"""Application bootstrap — wires all components together."""

from __future__ import annotations

from sentinel.core.config import Config, load_config
from sentinel.core.checkpoint import CheckpointManager
from sentinel.core.message_bus import MessageBus
from sentinel.core.trace import TraceLogger
from sentinel.agents.orchestrator import Orchestrator
from sentinel.knowledge.graph import KnowledgeGraph
from sentinel.knowledge.store import KnowledgeStore
from sentinel.learning.store import LearningStore
from sentinel.memory.store import MemoryStore
from sentinel.prompts.evolution import PromptEvolver
from sentinel.prompts.store import PromptEvolutionStore
from sentinel.skills.loader import SkillLoader
from sentinel.routing.router import SmartRouter
from sentinel.tools.base import ToolRegistry
from sentinel.tools.checkpoint_tools import RequestApprovalTool
from sentinel.tools.code_execution import CodeExecutionTool, ShellTool
from sentinel.tools.database import SqlQueryTool
from sentinel.tools.delegate import DelegateTool
from sentinel.tools.environment import CheckDependencyTool, EnvInfoTool, InstallPackageTool
from sentinel.tools.file_ops import ListFilesTool, ReadFileTool, SearchFilesTool, WriteFileTool
from sentinel.tools.git_ops import GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool
from sentinel.tools.graph_tools import GraphAddEntityTool, GraphAddRelationTool, GraphQueryTool
from sentinel.tools.http_client import HttpRequestTool
from sentinel.tools.learning_tools import LearningInsightsTool, LearningStatsTool
from sentinel.tools.memory_tools import MemoryDeleteTool, MemoryQueryTool, MemorySaveTool
from sentinel.tools.message_bus_tools import BusPeekTool, BusPublishTool
from sentinel.tools.prompt_tools import (
    PromptHistoryTool,
    PromptModifyTool,
    PromptRollbackTool,
    PromptViewTool,
)
from sentinel.tools.scratchpad import ScratchReadTool, ScratchWriteTool
from sentinel.tools.text_tools import DiffPatchTool, RegexReplaceTool, SummarizeTextTool
from sentinel.tools.vision import ImageDescribeTool, ScreenshotTool
from sentinel.tools.browser import (
    BrowseWebTool,
    BrowserClickTool,
    BrowserExtractTool,
    BrowserFormTool,
    BrowserScreenshotTool,
)
from sentinel.tools.knowledge_tools import KBDeleteTool, KBIngestTool, KBListTool, KBSearchTool
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
        registry.register(ListFilesTool())
        registry.register(SearchFilesTool())

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
