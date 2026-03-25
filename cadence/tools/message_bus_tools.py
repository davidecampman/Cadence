"""Tools for agent-to-agent communication via the message bus."""

from __future__ import annotations

from typing import Any

from cadence.core.message_bus import MessageBus, MessagePriority
from cadence.core.types import PermissionTier
from cadence.tools.base import Tool


class BusPublishTool(Tool):
    """Publish a message to the inter-agent message bus."""

    name = "bus_publish"
    description = (
        "Publish a message to a topic on the agent message bus. "
        "Use this to share discoveries, status updates, or coordinate with other agents. "
        "Topics: 'discovery', 'status', 'coordination', 'error', or any custom topic."
    )
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic to publish to (e.g., 'discovery', 'status', 'coordination')",
            },
            "content": {
                "type": "string",
                "description": "The message content to publish",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high", "urgent"],
                "description": "Message priority (default: normal)",
            },
        },
        "required": ["topic", "content"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(self, bus: MessageBus, agent_id: str = ""):
        self._bus = bus
        self._agent_id = agent_id

    def with_agent_id(self, agent_id: str) -> "BusPublishTool":
        return BusPublishTool(bus=self._bus, agent_id=agent_id)

    async def execute(
        self,
        topic: str,
        content: str,
        priority: str = "normal",
    ) -> str:
        prio = MessagePriority(priority)
        msg = await self._bus.publish(
            topic=topic,
            sender_id=self._agent_id,
            content=content,
            priority=prio,
        )
        return f"Published message {msg.id} to topic '{topic}'"


class BusPeekTool(Tool):
    """Read recent messages from a topic on the message bus."""

    name = "bus_peek"
    description = (
        "Read recent messages on a message bus topic without subscribing. "
        "Use this to check what other agents have shared."
    )
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic to read from",
            },
            "limit": {
                "type": "integer",
                "description": "Max messages to return (default: 5)",
            },
        },
        "required": ["topic"],
    }
    permission_tier = PermissionTier.READ_ONLY

    def __init__(self, bus: MessageBus, agent_id: str = ""):
        self._bus = bus
        self._agent_id = agent_id

    def with_agent_id(self, agent_id: str) -> "BusPeekTool":
        return BusPeekTool(bus=self._bus, agent_id=agent_id)

    async def execute(self, topic: str, limit: int = 5) -> str:
        messages = self._bus.peek(topic=topic, limit=limit)
        if not messages:
            return f"No messages on topic '{topic}'"

        lines = []
        for msg in messages:
            lines.append(
                f"[{msg.sender_id}] ({msg.priority.value}) {msg.content[:200]}"
            )
        return f"Messages on '{topic}' ({len(messages)}):\n" + "\n".join(lines)
