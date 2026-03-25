"""Agent-to-agent message bus — pub/sub communication between running agents."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field


class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class BusMessage(BaseModel):
    """A message sent between agents via the message bus."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str
    sender_id: str
    content: str
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    reply_to: str | None = None  # Message ID this is replying to


# Type alias for subscriber callbacks
Subscriber = Callable[[BusMessage], Awaitable[None]]


class MessageBus:
    """Async pub/sub message bus for inter-agent communication.

    Agents can:
      - publish(topic, content) to broadcast discoveries
      - subscribe(topic, callback) to receive messages on a topic
      - request(topic, content) to send and wait for a reply
      - peek(topic) to read recent messages without subscribing

    Built-in topics:
      - "discovery"    — agents share findings mid-execution
      - "status"       — agents broadcast progress updates
      - "coordination" — agents negotiate task ownership
      - "error"        — agents report failures
    """

    def __init__(self, history_limit: int = 100):
        self._subscribers: dict[str, list[tuple[str, Subscriber]]] = defaultdict(list)
        self._history: dict[str, list[BusMessage]] = defaultdict(list)
        self._history_limit = history_limit
        self._pending_replies: dict[str, asyncio.Future[BusMessage]] = {}
        self._lock = asyncio.Lock()

    async def publish(
        self,
        topic: str,
        sender_id: str,
        content: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        reply_to: str | None = None,
    ) -> BusMessage:
        """Publish a message to a topic. All subscribers are notified."""
        msg = BusMessage(
            topic=topic,
            sender_id=sender_id,
            content=content,
            priority=priority,
            metadata=metadata or {},
            reply_to=reply_to,
        )

        async with self._lock:
            # Store in history
            self._history[topic].append(msg)
            if len(self._history[topic]) > self._history_limit:
                self._history[topic] = self._history[topic][-self._history_limit:]

        # Resolve any pending reply futures
        if reply_to and reply_to in self._pending_replies:
            future = self._pending_replies.pop(reply_to)
            if not future.done():
                future.set_result(msg)

        # Notify subscribers (fire and forget, don't block publisher)
        subscribers = list(self._subscribers.get(topic, []))
        for _agent_id, callback in subscribers:
            try:
                asyncio.create_task(callback(msg))
            except Exception:
                pass  # Don't let subscriber errors affect the publisher

        return msg

    def subscribe(self, topic: str, agent_id: str, callback: Subscriber) -> None:
        """Subscribe an agent to a topic."""
        # Prevent duplicate subscriptions
        for existing_id, _ in self._subscribers[topic]:
            if existing_id == agent_id:
                return
        self._subscribers[topic].append((agent_id, callback))

    def unsubscribe(self, topic: str, agent_id: str) -> None:
        """Unsubscribe an agent from a topic."""
        self._subscribers[topic] = [
            (aid, cb) for aid, cb in self._subscribers[topic]
            if aid != agent_id
        ]

    def unsubscribe_all(self, agent_id: str) -> None:
        """Remove an agent from all topic subscriptions."""
        for topic in list(self._subscribers.keys()):
            self.unsubscribe(topic, agent_id)

    async def request(
        self,
        topic: str,
        sender_id: str,
        content: str,
        timeout: float = 30.0,
        metadata: dict[str, Any] | None = None,
    ) -> BusMessage | None:
        """Send a message and wait for a reply. Returns None on timeout."""
        msg = await self.publish(
            topic=topic,
            sender_id=sender_id,
            content=content,
            metadata=metadata,
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[BusMessage] = loop.create_future()
        self._pending_replies[msg.id] = future

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_replies.pop(msg.id, None)
            return None

    def peek(
        self,
        topic: str,
        limit: int = 10,
        since: float | None = None,
    ) -> list[BusMessage]:
        """Read recent messages on a topic without subscribing."""
        messages = self._history.get(topic, [])
        if since is not None:
            messages = [m for m in messages if m.timestamp >= since]
        return messages[-limit:]

    def topics(self) -> list[str]:
        """List all topics that have messages or subscribers."""
        all_topics = set(self._history.keys()) | set(self._subscribers.keys())
        return sorted(all_topics)

    def stats(self) -> dict[str, Any]:
        """Return bus statistics."""
        return {
            "topics": len(self.topics()),
            "total_messages": sum(len(msgs) for msgs in self._history.values()),
            "active_subscriptions": sum(
                len(subs) for subs in self._subscribers.values()
            ),
            "pending_replies": len(self._pending_replies),
            "topic_details": {
                topic: {
                    "messages": len(self._history.get(topic, [])),
                    "subscribers": len(self._subscribers.get(topic, [])),
                }
                for topic in self.topics()
            },
        }
