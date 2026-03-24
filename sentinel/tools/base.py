"""Base tool class — all tools inherit from this."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from sentinel.core.types import PermissionTier, ToolDefinition, ToolResult


class Tool(ABC):
    """Abstract base for all Sentinel tools.

    Subclasses define:
      - name, description, parameters (the schema the LLM sees)
      - permission_tier (what level of access is required)
      - execute() (the actual implementation)
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    permission_tier: PermissionTier = PermissionTier.STANDARD

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            permission_tier=self.permission_tier,
        )

    async def run(self, tool_call_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute with timing and error handling."""
        start = time.time()
        try:
            output = await self.execute(**arguments)
            duration_ms = (time.time() - start) * 1000
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                output=str(output),
                success=True,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                output=f"Error: {type(e).__name__}: {e}",
                success=False,
                duration_ms=duration_ms,
            )

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Implement the tool's logic. Return a string result."""
        ...


class ToolRegistry:
    """Manages available tools, enforces permission tiers."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def definitions(
        self,
        max_tier: PermissionTier = PermissionTier.UNRESTRICTED,
        allowed_names: list[str] | None = None,
    ) -> list[ToolDefinition]:
        """Return tool definitions filtered by permission tier and allowlist."""
        tier_order = list(PermissionTier)
        max_idx = tier_order.index(max_tier)
        result = []
        for tool in self._tools.values():
            if tier_order.index(tool.permission_tier) > max_idx:
                continue
            if allowed_names is not None and tool.name not in allowed_names:
                continue
            result.append(tool.definition())
        return result

    def names(self) -> list[str]:
        return list(self._tools.keys())
