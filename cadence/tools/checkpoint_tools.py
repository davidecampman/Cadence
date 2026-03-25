"""Tools for human-in-the-loop checkpoints."""

from __future__ import annotations

from typing import Any

from cadence.core.checkpoint import CheckpointManager, CheckpointType
from cadence.core.types import PermissionTier
from cadence.tools.base import Tool


class RequestApprovalTool(Tool):
    """Request human approval before proceeding with an action."""

    name = "request_approval"
    description = (
        "Pause and request human approval before proceeding. "
        "Use this for destructive operations, sensitive actions, or when you need "
        "clarification from the user. The agent will wait until the user responds."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for the checkpoint (e.g., 'Delete database table')",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of what you want to do and why approval is needed",
            },
            "checkpoint_type": {
                "type": "string",
                "enum": ["approval", "clarification", "confirmation"],
                "description": (
                    "Type: 'approval' for approve/reject, "
                    "'clarification' when you need more info, "
                    "'confirmation' before destructive actions"
                ),
            },
        },
        "required": ["title", "description"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(self, manager: CheckpointManager, agent_id: str = ""):
        self._manager = manager
        self._agent_id = agent_id

    def with_agent_id(self, agent_id: str) -> "RequestApprovalTool":
        return RequestApprovalTool(manager=self._manager, agent_id=agent_id)

    async def execute(
        self,
        title: str,
        description: str,
        checkpoint_type: str = "approval",
    ) -> str:
        cp_type = CheckpointType(checkpoint_type)
        approved, response = await self._manager.request_approval(
            agent_id=self._agent_id,
            title=title,
            description=description,
            checkpoint_type=cp_type,
        )

        if approved:
            result = "APPROVED"
            if response:
                result += f" — User response: {response}"
        else:
            result = "REJECTED"
            if response:
                result += f" — User response: {response}"

        return result
