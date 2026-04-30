"""Streaming response support — SSE (Server-Sent Events) for real-time output."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field


class StreamEvent(BaseModel):
    """A single event in the SSE stream."""
    event: str  # "token", "tool_start", "tool_result", "thinking", "done", "error"
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    def to_sse(self) -> str:
        """Format as an SSE message string."""
        payload = json.dumps({"event": self.event, **self.data})
        return f"event: {self.event}\ndata: {payload}\n\n"


class StreamCollector:
    """Collects streaming events from an agent run and exposes them as an async iterator.

    Usage:
        collector = StreamCollector()

        # In the agent loop:
        await collector.emit_token("Hello")
        await collector.emit_token(" world")
        await collector.emit_thinking("Analyzing the request...")
        await collector.emit_done(full_response="Hello world")

        # In the API endpoint:
        async for event in collector:
            yield event.to_sse()
    """

    def __init__(self):
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._closed = False
        self._full_response: str = ""

    async def emit(self, event: StreamEvent) -> None:
        """Emit a stream event."""
        if not self._closed:
            await self._queue.put(event)

    async def emit_token(self, token: str, agent_id: str = "") -> None:
        """Emit a token (partial text) event."""
        await self.emit(StreamEvent(
            event="token",
            data={"token": token, "agent_id": agent_id},
        ))

    async def emit_thinking(self, thought: str, agent_id: str = "") -> None:
        """Emit a thinking/reasoning step."""
        await self.emit(StreamEvent(
            event="thinking",
            data={"content": thought, "agent_id": agent_id},
        ))

    async def emit_tool_start(self, tool_name: str, args: dict[str, Any], agent_id: str = "") -> None:
        """Emit a tool execution start event."""
        await self.emit(StreamEvent(
            event="tool_start",
            data={"tool": tool_name, "arguments": args, "agent_id": agent_id},
        ))

    async def emit_tool_result(self, tool_name: str, result: str, success: bool, agent_id: str = "") -> None:
        """Emit a tool execution result."""
        await self.emit(StreamEvent(
            event="tool_result",
            data={
                "tool": tool_name,
                "result": result[:1000],  # Cap result size for streaming
                "success": success,
                "agent_id": agent_id,
            },
        ))

    async def emit_status(self, status: str, agent_id: str = "") -> None:
        """Emit a status update (e.g., phase transitions)."""
        await self.emit(StreamEvent(
            event="status",
            data={"status": status, "agent_id": agent_id},
        ))

    async def emit_done(
        self,
        full_response: str,
        session_id: str = "",
        duration_ms: float = 0.0,
        trace_steps: list[dict[str, Any]] | None = None,
        context_turns: int = 0,
        max_context_turns: int = 50,
    ) -> None:
        """Emit the final done event and close the stream."""
        self._full_response = full_response
        await self.emit(StreamEvent(
            event="done",
            data={
                "response": full_response,
                "session_id": session_id,
                "duration_ms": duration_ms,
                "trace_steps": trace_steps or [],
                "context_turns": context_turns,
                "max_context_turns": max_context_turns,
            },
        ))
        await self.close()

    async def emit_error(self, error: str) -> None:
        """Emit an error event and close the stream."""
        await self.emit(StreamEvent(
            event="error",
            data={"error": error},
        ))
        await self.close()

    async def close(self) -> None:
        """Signal end of stream. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __anext__(self) -> StreamEvent:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item
