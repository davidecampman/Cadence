"""Human-in-the-loop checkpoint system for agent approval workflows."""

from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class CheckpointType(str, Enum):
    APPROVAL = "approval"          # Simple approve/reject
    CLARIFICATION = "clarification"  # Agent needs more info from user
    CONFIRMATION = "confirmation"   # Confirm before destructive action


class Checkpoint(BaseModel):
    """A point where an agent pauses for human input."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str
    checkpoint_type: CheckpointType = CheckpointType.APPROVAL
    title: str
    description: str
    context: dict[str, Any] = Field(default_factory=dict)
    status: CheckpointStatus = CheckpointStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    resolved_at: float | None = None
    response: str | None = None  # User's response text
    timeout_seconds: float = 300.0  # 5 minute default timeout
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointManager:
    """Manages human-in-the-loop checkpoints.

    Agents call `request_approval()` to pause and wait for human input.
    The API layer resolves checkpoints via `resolve()`.

    Usage in agent code:
        approved, response = await checkpoint_mgr.request_approval(
            agent_id="coder-abc123",
            title="Delete database table",
            description="About to run DROP TABLE users. Approve?",
        )
        if not approved:
            return "Operation cancelled by user."
    """

    # Maximum number of resolved checkpoints to retain in memory
    _MAX_RESOLVED = 200

    def __init__(self):
        self._checkpoints: dict[str, Checkpoint] = {}
        self._waiters: dict[str, asyncio.Event] = {}

    async def request_approval(
        self,
        agent_id: str,
        title: str,
        description: str,
        checkpoint_type: CheckpointType = CheckpointType.APPROVAL,
        context: dict[str, Any] | None = None,
        timeout: float = 300.0,
    ) -> tuple[bool, str | None]:
        """Create a checkpoint and block until resolved or timeout.

        Returns (approved: bool, response: str | None).
        """
        checkpoint = Checkpoint(
            agent_id=agent_id,
            checkpoint_type=checkpoint_type,
            title=title,
            description=description,
            context=context or {},
            timeout_seconds=timeout,
        )

        event = asyncio.Event()
        self._checkpoints[checkpoint.id] = checkpoint
        self._waiters[checkpoint.id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            checkpoint.status = CheckpointStatus.EXPIRED
            checkpoint.resolved_at = time.time()
            self._waiters.pop(checkpoint.id, None)
            return False, "Checkpoint expired (timeout)"

        self._waiters.pop(checkpoint.id, None)
        cp = self._checkpoints[checkpoint.id]
        return cp.status == CheckpointStatus.APPROVED, cp.response

    def resolve(
        self,
        checkpoint_id: str,
        approved: bool,
        response: str | None = None,
    ) -> Checkpoint | None:
        """Resolve a pending checkpoint. Returns the updated checkpoint or None."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint or checkpoint.status != CheckpointStatus.PENDING:
            return None

        checkpoint.status = (
            CheckpointStatus.APPROVED if approved else CheckpointStatus.REJECTED
        )
        checkpoint.response = response
        checkpoint.resolved_at = time.time()

        # Wake the waiting agent
        event = self._waiters.get(checkpoint_id)
        if event:
            event.set()

        # Evict oldest resolved checkpoints to prevent unbounded memory growth
        self._evict_resolved()

        return checkpoint

    def _evict_resolved(self) -> None:
        """Remove oldest resolved checkpoints if over the retention limit."""
        resolved = [
            cp for cp in self._checkpoints.values()
            if cp.status != CheckpointStatus.PENDING
        ]
        if len(resolved) > self._MAX_RESOLVED:
            resolved.sort(key=lambda c: c.created_at)
            for cp in resolved[: len(resolved) - self._MAX_RESOLVED]:
                self._checkpoints.pop(cp.id, None)

    def get_pending(self) -> list[Checkpoint]:
        """Return all pending checkpoints (for UI display)."""
        return [
            cp for cp in self._checkpoints.values()
            if cp.status == CheckpointStatus.PENDING
        ]

    def get_all(self, limit: int = 50) -> list[Checkpoint]:
        """Return recent checkpoints of any status."""
        checkpoints = sorted(
            self._checkpoints.values(),
            key=lambda c: c.created_at,
            reverse=True,
        )
        return checkpoints[:limit]

    def get(self, checkpoint_id: str) -> Checkpoint | None:
        """Get a specific checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)
