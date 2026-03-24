"""Delegate tool — lets agents spawn specialist sub-agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sentinel.core.types import AgentRole, PermissionTier
from sentinel.tools.base import Tool

if TYPE_CHECKING:
    from sentinel.core.agent import Agent
    from sentinel.core.config import Config
    from sentinel.core.trace import TraceLogger
    from sentinel.skills.loader import SkillLoader
    from sentinel.tools.base import ToolRegistry


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
        skill_loader: "SkillLoader | None" = None,
    ):
        self._tool_registry = tool_registry
        self._trace = trace
        self._config = config
        self._parent_depth = parent_depth
        self._max_depth = max_depth
        self._skill_loader = skill_loader

    async def execute(self, task: str, role: str = "general") -> str:
        if self._parent_depth >= self._max_depth:
            return f"Cannot delegate: max agent depth ({self._max_depth}) reached."

        from sentinel.agents.orchestrator import ROLES
        from sentinel.core.agent import Agent

        agent_role = ROLES.get(role, ROLES["general"])
        agent = Agent(
            role=agent_role,
            tool_registry=self._tool_registry,
            trace=self._trace,
            config=self._config,
            depth=self._parent_depth + 1,
            skill_loader=self._skill_loader,
        )

        self._trace.action(
            f"delegate-{agent.id}",
            f"Sub-agent [{role}] depth={self._parent_depth + 1}: {task[:100]}",
        )

        result = await agent.run(task)
        return result
