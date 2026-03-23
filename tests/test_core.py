"""Basic tests — verify imports, types, config, skills, and tool registry."""

import asyncio

import pytest

from agent_one.core.types import (
    AgentRole,
    Message,
    PermissionTier,
    Role,
    Task,
    TaskStatus,
    ToolCall,
    ToolDefinition,
    TraceStep,
)
from agent_one.core.config import Config, BedrockConfig, load_config
from agent_one.core.llm import supports_native_tools, _is_bedrock_model
from agent_one.core.trace import TraceLogger
from agent_one.tools.base import Tool, ToolRegistry
from agent_one.skills.loader import SkillLoader, SkillDefinition
from agent_one.routing.router import ModelStats, SmartRouter


# --- Types ---

def test_message_creation():
    msg = Message(role=Role.USER, content="hello")
    assert msg.role == Role.USER
    assert msg.content == "hello"
    assert msg.timestamp > 0


def test_tool_call():
    tc = ToolCall(name="read_file", arguments={"path": "test.py"})
    assert tc.name == "read_file"
    assert tc.arguments["path"] == "test.py"
    assert len(tc.id) > 0


def test_task_lifecycle():
    t = Task(description="do something")
    assert t.status == TaskStatus.PENDING
    t.status = TaskStatus.RUNNING
    assert t.status == TaskStatus.RUNNING


def test_permission_tier_ordering():
    tiers = list(PermissionTier)
    assert tiers.index(PermissionTier.READ_ONLY) < tiers.index(PermissionTier.PRIVILEGED)


# --- Config ---

def test_default_config():
    cfg = Config()
    assert cfg.models.strong
    assert cfg.agents.max_depth == 5
    assert cfg.memory.decay_rate == 0.05


# --- Tool Registry ---

class DummyTool(Tool):
    name = "dummy"
    description = "A test tool"
    parameters = {"type": "object", "properties": {}}
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, **kwargs) -> str:
        return "dummy result"


class PrivilegedDummyTool(Tool):
    name = "priv_dummy"
    description = "A privileged test tool"
    parameters = {"type": "object", "properties": {}}
    permission_tier = PermissionTier.PRIVILEGED

    async def execute(self, **kwargs) -> str:
        return "privileged result"


def test_tool_registry():
    reg = ToolRegistry()
    reg.register(DummyTool())
    reg.register(PrivilegedDummyTool())

    assert "dummy" in reg.names()
    assert reg.get("dummy") is not None
    assert reg.get("nonexistent") is None


def test_tool_permission_filtering():
    reg = ToolRegistry()
    reg.register(DummyTool())
    reg.register(PrivilegedDummyTool())

    # READ_ONLY tier should only see the dummy tool
    defs = reg.definitions(max_tier=PermissionTier.READ_ONLY)
    names = [d.name for d in defs]
    assert "dummy" in names
    assert "priv_dummy" not in names

    # PRIVILEGED tier should see both
    defs = reg.definitions(max_tier=PermissionTier.PRIVILEGED)
    names = [d.name for d in defs]
    assert "dummy" in names
    assert "priv_dummy" in names


def test_tool_allowlist_filtering():
    reg = ToolRegistry()
    reg.register(DummyTool())
    reg.register(PrivilegedDummyTool())

    defs = reg.definitions(allowed_names=["dummy"])
    assert len(defs) == 1
    assert defs[0].name == "dummy"


@pytest.mark.asyncio
async def test_tool_execution():
    tool = DummyTool()
    result = await tool.run("test-id", {})
    assert result.success
    assert result.output == "dummy result"
    assert result.duration_ms >= 0


# --- Trace ---

def test_trace_logger(tmp_path):
    trace_file = tmp_path / "trace.jsonl"
    logger = TraceLogger(trace_file=str(trace_file), console=False)

    logger.observation("agent-1", "saw something")
    logger.thought("agent-1", "thinking about it")
    logger.action("agent-1", "doing something")
    logger.result("agent-1", "done")

    assert len(logger.steps) == 4
    assert trace_file.exists()
    lines = trace_file.read_text().strip().splitlines()
    assert len(lines) == 4


# --- Skills ---

def test_skill_parsing(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""---
name: test-skill
version: 2.0.0
description: A test skill
dependencies:
  - other-skill
tags:
  - test
---

# Instructions

Do the thing.

## Examples

- Example one
- Example two
""")

    loader = SkillLoader(directories=[str(tmp_path)])
    skills = loader.discover()
    assert len(skills) == 1

    skill = skills[0]
    assert skill.name == "test-skill"
    assert skill.version == "2.0.0"
    assert "other-skill" in skill.dependencies
    assert len(skill.examples) == 2


def test_skill_versioning(tmp_path):
    """Higher version wins when duplicate names exist."""
    for ver, subdir in [("1.0.0", "a"), ("2.0.0", "b")]:
        d = tmp_path / subdir
        d.mkdir()
        (d / "SKILL.md").write_text(f"---\nname: dupe\nversion: {ver}\n---\nContent")

    loader = SkillLoader(directories=[str(tmp_path)])
    loader.discover()
    assert loader.get("dupe").version == "2.0.0"


# --- Router ---

def test_model_stats():
    stats = ModelStats()
    stats.record("gpt-4o", True, 500.0, 100)
    stats.record("gpt-4o", True, 600.0, 120)
    stats.record("gpt-4o", False, 300.0, 0)

    assert stats.success_rate("gpt-4o") == pytest.approx(2 / 3)
    assert stats.avg_latency("gpt-4o") == pytest.approx(1400 / 3)


def test_smart_router_selection():
    router = SmartRouter()
    # Fast tasks → fast model
    assert router.select_model("plan") == router.config.models.fast
    assert router.select_model("classify") == router.config.models.fast
    # Strong tasks → strong model
    assert router.select_model("code") == router.config.models.strong
    assert router.select_model("reason") == router.config.models.strong


# --- Bedrock ---

def test_bedrock_config_defaults():
    cfg = BedrockConfig()
    assert cfg.enabled is False
    assert cfg.region == "us-east-1"
    assert cfg.profile is None
    assert cfg.role_arn is None


def test_bedrock_config_in_models():
    cfg = Config()
    assert cfg.models.bedrock.enabled is False
    # Bedrock config should be customizable
    cfg2 = Config(models={"bedrock": {"enabled": True, "region": "eu-west-1"}})
    assert cfg2.models.bedrock.enabled is True
    assert cfg2.models.bedrock.region == "eu-west-1"


def test_is_bedrock_model():
    assert _is_bedrock_model("bedrock/anthropic.claude-sonnet-4-20250514-v1:0")
    assert _is_bedrock_model("Bedrock/anthropic.claude-3-haiku")
    assert not _is_bedrock_model("claude-sonnet-4-20250514")
    assert not _is_bedrock_model("gpt-4o")


def test_bedrock_native_tool_support():
    assert supports_native_tools("bedrock/anthropic.claude-sonnet-4-20250514-v1:0")
    assert supports_native_tools("bedrock/anthropic.claude-3-haiku-20240307-v1:0")
    # Non-bedrock models should still work
    assert supports_native_tools("claude-sonnet-4-20250514")
    assert supports_native_tools("gpt-4o")
