"""Basic tests — verify imports, types, config, skills, and tool registry."""

import asyncio

import pytest

from sentinel.core.types import (
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
from sentinel.core.config import Config, BedrockConfig, load_config
from sentinel.core.llm import supports_native_tools, _is_bedrock_model, _to_bedrock_model, _region_to_inference_prefix
from sentinel.core.trace import TraceLogger
from sentinel.tools.base import Tool, ToolRegistry
from sentinel.skills.loader import SkillLoader, SkillDefinition
from sentinel.routing.router import ModelStats, SmartRouter
from sentinel.tools.code_execution import (
    _build_resource_limits,
    _check_blocked,
    _shell_quote,
    _wrap_with_sandbox,
    CodeExecutionTool,
    ShellTool,
)


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
    assert cfg.access_key_id is None
    assert cfg.secret_access_key is None
    assert cfg.api_key is None


def test_bedrock_config_with_api_key():
    cfg = BedrockConfig(api_key="br-key-123", enabled=True)
    assert cfg.api_key == "br-key-123"
    assert cfg.enabled is True
    assert cfg.access_key_id is None


def test_bedrock_config_with_access_keys():
    cfg = BedrockConfig(
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        enabled=True,
    )
    assert cfg.access_key_id == "AKIAIOSFODNN7EXAMPLE"
    assert cfg.secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    assert cfg.api_key is None


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


def test_region_to_inference_prefix():
    assert _region_to_inference_prefix("us-east-1") == "us"
    assert _region_to_inference_prefix("us-west-2") == "us"
    assert _region_to_inference_prefix("eu-west-1") == "eu"
    assert _region_to_inference_prefix("eu-central-1") == "eu"
    assert _region_to_inference_prefix("ap-northeast-1") == "ap"
    assert _region_to_inference_prefix("ap-southeast-2") == "ap"
    assert _region_to_inference_prefix("me-south-1") == "eu"
    # Unknown region defaults to "us"
    assert _region_to_inference_prefix("unknown-region") == "us"


def test_to_bedrock_model_uses_inference_profile():
    # Known model in map should get inference profile prefix
    result = _to_bedrock_model("claude-haiku-4-5-20251001", region="us-east-1")
    assert result == "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0"

    # EU region should use eu. prefix
    result = _to_bedrock_model("claude-sonnet-4-5-20250514", region="eu-west-1")
    assert result == "bedrock/converse/eu.anthropic.claude-sonnet-4-5-20250514-v1:0"

    # AP region
    result = _to_bedrock_model("claude-3-opus-20240229", region="ap-northeast-1")
    assert result == "bedrock/converse/ap.anthropic.claude-3-opus-20240229-v1:0"

    # Already bedrock-prefixed model should be returned as-is
    result = _to_bedrock_model("bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert result == "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0"

    # Unknown claude model not in map should still get inference profile prefix
    result = _to_bedrock_model("claude-future-model-20260101", region="us-east-1")
    assert result == "bedrock/converse/us.anthropic.claude-future-model-20260101-v1:0"

    # Non-claude model should not get inference profile prefix
    result = _to_bedrock_model("some-other-model", region="us-east-1")
    assert result == "bedrock/converse/some-other-model"


# --- Skill Install / Uninstall ---

def _make_skill_zip(skill_dir_name: str, skill_md_content: str) -> bytes:
    """Helper to create an in-memory zip containing a SKILL.md."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{skill_dir_name}/SKILL.md", skill_md_content)
    return buf.getvalue()


def test_skill_install_from_zip(tmp_path):
    """Installing a skill from a zip creates the directory and registers it."""
    skill_content = """---
name: my-new-skill
version: 1.0.0
description: A brand new skill
tags:
  - test
---

# Instructions

Do the new thing.
"""
    zip_data = _make_skill_zip("my-new-skill", skill_content)

    loader = SkillLoader(directories=[str(tmp_path)])
    loader.discover()
    assert loader.get("my-new-skill") is None

    skill = loader.install_from_zip(zip_data)
    assert skill.name == "my-new-skill"
    assert skill.version == "1.0.0"
    assert skill.description == "A brand new skill"

    # Verify it's on disk
    assert (tmp_path / "my-new-skill" / "SKILL.md").exists()

    # Verify it's in the registry
    assert loader.get("my-new-skill") is not None


def test_skill_install_overwrites_existing(tmp_path):
    """Installing a skill with the same name overwrites the previous version."""
    v1 = _make_skill_zip("upd-skill", "---\nname: upd-skill\nversion: 1.0.0\n---\nOld")
    v2 = _make_skill_zip("upd-skill", "---\nname: upd-skill\nversion: 2.0.0\n---\nNew")

    loader = SkillLoader(directories=[str(tmp_path)])
    loader.discover()
    loader.install_from_zip(v1)
    assert loader.get("upd-skill").version == "1.0.0"

    loader.install_from_zip(v2)
    assert loader.get("upd-skill").version == "2.0.0"


def test_skill_install_invalid_zip(tmp_path):
    """Installing from invalid data raises ValueError."""
    loader = SkillLoader(directories=[str(tmp_path)])
    loader.discover()
    with pytest.raises(ValueError, match="not a valid zip"):
        loader.install_from_zip(b"not a zip file")


def test_skill_install_no_skill_md(tmp_path):
    """Zip without SKILL.md raises ValueError."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("README.md", "No skill here")
    loader = SkillLoader(directories=[str(tmp_path)])
    loader.discover()
    with pytest.raises(ValueError, match="does not contain a SKILL.md"):
        loader.install_from_zip(buf.getvalue())


def test_skill_uninstall(tmp_path):
    """Uninstalling a skill removes it from disk and registry."""
    skill_dir = tmp_path / "removable"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: removable\nversion: 1.0.0\n---\nContent")

    loader = SkillLoader(directories=[str(tmp_path)])
    loader.discover()
    assert loader.get("removable") is not None

    result = loader.uninstall("removable")
    assert result is True
    assert loader.get("removable") is None
    assert not skill_dir.exists()


def test_skill_uninstall_not_found(tmp_path):
    """Uninstalling a nonexistent skill returns False."""
    loader = SkillLoader(directories=[str(tmp_path)])
    loader.discover()
    assert loader.uninstall("ghost-skill") is False
