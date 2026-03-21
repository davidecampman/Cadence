"""Core type definitions for Agent One."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --- Message types ---

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in the conversation history."""
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    timestamp: float = Field(default_factory=time.time)


# --- Tool types ---

class ToolCall(BaseModel):
    """A structured tool invocation from the LLM."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of executing a tool."""
    tool_call_id: str
    name: str
    output: str
    success: bool = True
    duration_ms: float = 0.0


class ToolDefinition(BaseModel):
    """Schema for a tool that can be offered to the LLM."""
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON Schema
    permission_tier: PermissionTier = "standard"


# --- Permission tiers ---

class PermissionTier(str, Enum):
    READ_ONLY = "read_only"       # Can read files, search, query memory
    STANDARD = "standard"         # Can write files, run sandboxed code
    PRIVILEGED = "privileged"     # Can access network, shell, Docker
    UNRESTRICTED = "unrestricted" # Full OS access (requires explicit opt-in)


# --- Agent types ---

class AgentRole(BaseModel):
    """Defines what an agent specializes in."""
    name: str                                   # e.g., "researcher", "coder", "reviewer"
    description: str                            # What this agent does
    system_prompt_file: str | None = None       # Override prompt from prompts/ dir
    allowed_tools: list[str] | None = None      # None = all tools
    permission_tier: PermissionTier = PermissionTier.STANDARD
    model_override: str | None = None           # Use specific model for this role


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Task(BaseModel):
    """A unit of work in the task DAG."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str | None = None           # Agent ID
    dependencies: list[str] = Field(default_factory=list)  # Task IDs
    result: str | None = None
    subtasks: list[str] = Field(default_factory=list)       # Child task IDs
    parent_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Reasoning trace ---

class TraceStep(BaseModel):
    """One step in the observation-thought-action trace."""
    agent_id: str
    task_id: str | None = None
    step_type: str  # "observation" | "thought" | "action" | "result" | "error"
    content: str
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = 0
