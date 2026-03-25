"""Prompt evolution tools — let agents view, modify, and manage their own prompts."""

from __future__ import annotations

from sentinel.core.types import PermissionTier
from sentinel.prompts.store import (
    ModificationType,
    PromptEvolutionStore,
    PromptModification,
)
from sentinel.tools.base import Tool


class PromptViewTool(Tool):
    """View the current active prompt modifications for a role."""

    name = "prompt_view"
    description = (
        "View active prompt modifications for an agent role. "
        "Shows all learned strategies, constraints, and additions "
        "that have been applied through self-reflection."
    )
    parameters = {
        "type": "object",
        "properties": {
            "role_name": {
                "type": "string",
                "description": (
                    "The agent role to inspect (e.g. 'researcher', 'coder', 'general'). "
                    "Omit to view your own role's modifications."
                ),
            },
        },
        "required": [],
    }
    permission_tier = PermissionTier.READ_ONLY

    def __init__(self, store: PromptEvolutionStore, agent_role: str | None = None):
        self._store = store
        self._agent_role = agent_role

    def with_agent_id(self, agent_id: str) -> "PromptViewTool":
        # Extract role name from agent_id (format: "role-uuid")
        role = agent_id.rsplit("-", 1)[0] if "-" in agent_id else agent_id
        return PromptViewTool(store=self._store, agent_role=role)

    async def execute(self, role_name: str | None = None) -> str:
        role = role_name or self._agent_role or "general"
        modifications = self._store.get_active(role)
        if not modifications:
            return f"No active prompt modifications for role '{role}'."

        lines = [f"Active prompt modifications for '{role}':"]
        for mod in modifications:
            lines.append(
                f"  [{mod.id[:8]}] v{mod.version} ({mod.modification_type.value}): "
                f"{mod.content[:120]}"
            )
            if mod.reasoning:
                lines.append(f"    Reason: {mod.reasoning[:100]}")
        return "\n".join(lines)


class PromptModifyTool(Tool):
    """Add a new prompt modification for a role."""

    name = "prompt_modify"
    description = (
        "Add a learned instruction or strategy to an agent role's prompt. "
        "Use this to teach the agent new behaviors based on task experience. "
        "Types: 'strategy' (behavioral), 'constraint' (guardrail), 'append' (context)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The instruction or strategy to add to the prompt.",
            },
            "modification_type": {
                "type": "string",
                "enum": ["strategy", "constraint", "append"],
                "description": "Type of modification: strategy, constraint, or append.",
                "default": "strategy",
            },
            "reasoning": {
                "type": "string",
                "description": "Why this modification is being added.",
            },
            "role_name": {
                "type": "string",
                "description": "Target role. Omit to modify your own role.",
            },
        },
        "required": ["content"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(self, store: PromptEvolutionStore, agent_role: str | None = None):
        self._store = store
        self._agent_role = agent_role

    def with_agent_id(self, agent_id: str) -> "PromptModifyTool":
        role = agent_id.rsplit("-", 1)[0] if "-" in agent_id else agent_id
        return PromptModifyTool(store=self._store, agent_role=role)

    async def execute(
        self,
        content: str,
        modification_type: str = "strategy",
        reasoning: str = "",
        role_name: str | None = None,
    ) -> str:
        role = role_name or self._agent_role or "general"
        mod_type = ModificationType(modification_type)

        modification = PromptModification(
            role_name=role,
            modification_type=mod_type,
            content=content,
            reasoning=reasoning,
        )
        saved = self._store.save(modification)
        return (
            f"Added prompt modification [{saved.id[:8]}] v{saved.version} "
            f"({mod_type.value}) for role '{role}': {content[:80]}"
        )


class PromptRollbackTool(Tool):
    """Rollback prompt modifications to a previous version."""

    name = "prompt_rollback"
    description = (
        "Roll back prompt modifications for a role to a specific version. "
        "All modifications after the given version will be deactivated. "
        "Use prompt_view to see current versions before rolling back."
    )
    parameters = {
        "type": "object",
        "properties": {
            "version": {
                "type": "integer",
                "description": "Roll back to this version (deactivates all later modifications).",
            },
            "role_name": {
                "type": "string",
                "description": "Target role. Omit to roll back your own role.",
            },
        },
        "required": ["version"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(self, store: PromptEvolutionStore, agent_role: str | None = None):
        self._store = store
        self._agent_role = agent_role

    def with_agent_id(self, agent_id: str) -> "PromptRollbackTool":
        role = agent_id.rsplit("-", 1)[0] if "-" in agent_id else agent_id
        return PromptRollbackTool(store=self._store, agent_role=role)

    async def execute(self, version: int, role_name: str | None = None) -> str:
        role = role_name or self._agent_role or "general"
        count = self._store.rollback_to_version(role, version)
        if count == 0:
            return f"No modifications to roll back (role '{role}' already at or before v{version})."
        return f"Rolled back role '{role}' to v{version}: deactivated {count} modification(s)."


class PromptHistoryTool(Tool):
    """View the full prompt modification history for a role."""

    name = "prompt_history"
    description = (
        "View the full history of prompt modifications for a role, "
        "including deactivated ones. Useful for understanding how the "
        "prompt has evolved over time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "role_name": {
                "type": "string",
                "description": "Target role. Omit to view your own role.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of entries to return.",
                "default": 20,
            },
        },
        "required": [],
    }
    permission_tier = PermissionTier.READ_ONLY

    def __init__(self, store: PromptEvolutionStore, agent_role: str | None = None):
        self._store = store
        self._agent_role = agent_role

    def with_agent_id(self, agent_id: str) -> "PromptHistoryTool":
        role = agent_id.rsplit("-", 1)[0] if "-" in agent_id else agent_id
        return PromptHistoryTool(store=self._store, agent_role=role)

    async def execute(self, role_name: str | None = None, limit: int = 20) -> str:
        role = role_name or self._agent_role or "general"
        history = self._store.get_history(role, limit=limit)
        if not history:
            return f"No prompt modification history for role '{role}'."

        lines = [f"Prompt modification history for '{role}' (newest first):"]
        for mod in history:
            status = "ACTIVE" if mod.active else "inactive"
            lines.append(
                f"  v{mod.version} [{status}] ({mod.modification_type.value}): "
                f"{mod.content[:100]}"
            )
            if mod.source_task:
                lines.append(f"    Source: {mod.source_task[:80]}")
        return "\n".join(lines)
