"""Delegate tool — lets agents spawn specialist sub-agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_one.core.types import AgentRole, PermissionTier
from agent_one.tools.base import Tool

if TYPE_CHECKING:
    from agent_one.core.agent import Agent
    from agent_one.core.config import Config
    from agent_one.core.trace import TraceLogger
    from agent_one.tools.base import ToolRegistry


class DelegateTool(Tool):
    name = "delegate"
    description = (
        "Delegate a subtask to a specialist agent. Choose a role:\n"
        "  - researcher: gathers information, reads files, searches\n"
        "  - coder: writes and debugs code\n"
        "  - reviewer: reviews code/output for correctness\n"
        "  - general: can use all tools\n"
        "The sub-agent works independently and returns its result."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Clear description of what the sub-agent should do.",
            },
            "role": {
                "type": "string",
                "enum": ["researcher", "coder", "reviewer", "general"],
                "description": "Which specialist to delegate to.",
                "default": "general",
            },
        },
        "required": ["task"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        trace: "TraceLogger",
        config: "Config",
        parent_depth: int = 0,
        max_depth: int = 5,
    ):
        self._tool_registry = tool_registry
        self._trace = trace
        self._config = config
        self._parent_depth = parent_depth
        self._max_depth = max_depth

    async def execute(self, task: str, role: str = "general") -> str:
        if self._parent_depth >= self._max_depth:
            return f"Cannot delegate: max agent depth ({self._max_depth}) reached."

        from agent_one.agents.orchestrator import ROLES
        from agent_one.core.agent import Agent

        agent_role = ROLES.get(role, ROLES["general"])
        agent = Agent(
            role=agent_role,
            tool_registry=self._tool_registry,
            trace=self._trace,
            config=self._config,
            depth=self._parent_depth + 1,
        )

        self._trace.action(
            f"delegate-{agent.id}",
            f"Sub-agent [{role}] depth={self._parent_depth + 1}: {task[:100]}",
        )

        result = await agent.run(task)
        return result
