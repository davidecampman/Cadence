"""Tests for the self-modifying prompt evolution system."""

import pytest

from sentinel.core.config import Config, PromptEvolutionConfig
from sentinel.prompts.store import (
    ModificationType,
    PromptEvolutionStore,
    PromptModification,
)
from sentinel.prompts.evolution import PromptEvolver
from sentinel.tools.prompt_tools import (
    PromptHistoryTool,
    PromptModifyTool,
    PromptRollbackTool,
    PromptViewTool,
)


# --- PromptEvolutionStore ---


def test_store_save_and_retrieve(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)

    mod = PromptModification(
        role_name="coder",
        modification_type=ModificationType.STRATEGY,
        content="Always write tests before implementation.",
        reasoning="TDD improves code quality.",
    )
    saved = store.save(mod)
    assert saved.version == 1
    assert saved.active is True

    active = store.get_active("coder")
    assert len(active) == 1
    assert active[0].content == "Always write tests before implementation."


def test_store_auto_increments_version(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)

    for i in range(3):
        mod = PromptModification(
            role_name="researcher",
            modification_type=ModificationType.APPEND,
            content=f"Modification {i}",
        )
        saved = store.save(mod)
        assert saved.version == i + 1


def test_store_deactivate(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)

    mod = PromptModification(
        role_name="general",
        modification_type=ModificationType.CONSTRAINT,
        content="Never use deprecated APIs.",
    )
    saved = store.save(mod)
    assert len(store.get_active("general")) == 1

    store.deactivate(saved.id)
    assert len(store.get_active("general")) == 0


def test_store_rollback(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)

    # Create 5 modifications
    for i in range(5):
        store.save(PromptModification(
            role_name="coder",
            modification_type=ModificationType.STRATEGY,
            content=f"Strategy {i}",
        ))

    assert len(store.get_active("coder")) == 5

    # Roll back to version 3
    count = store.rollback_to_version("coder", 3)
    assert count == 2  # v4 and v5 deactivated
    assert len(store.get_active("coder")) == 3


def test_store_clear_role(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)

    for i in range(3):
        store.save(PromptModification(
            role_name="reviewer",
            modification_type=ModificationType.APPEND,
            content=f"Rule {i}",
        ))

    count = store.clear_role("reviewer")
    assert count == 3
    assert len(store.get_active("reviewer")) == 0

    # History should still be available
    history = store.get_history("reviewer")
    assert len(history) == 3
    assert all(not h.active for h in history)


def test_store_reactivate(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)

    mod = PromptModification(
        role_name="coder",
        modification_type=ModificationType.STRATEGY,
        content="Use type hints.",
    )
    saved = store.save(mod)
    store.deactivate(saved.id)
    assert len(store.get_active("coder")) == 0

    store.reactivate(saved.id)
    assert len(store.get_active("coder")) == 1


def test_store_get_by_id(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)

    mod = PromptModification(
        role_name="general",
        modification_type=ModificationType.CONSTRAINT,
        content="Check for XSS vulnerabilities.",
    )
    saved = store.save(mod)

    retrieved = store.get_by_id(saved.id)
    assert retrieved is not None
    assert retrieved.content == "Check for XSS vulnerabilities."
    assert store.get_by_id("nonexistent") is None


def test_store_isolates_roles(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)

    store.save(PromptModification(
        role_name="coder", modification_type=ModificationType.STRATEGY, content="A",
    ))
    store.save(PromptModification(
        role_name="researcher", modification_type=ModificationType.STRATEGY, content="B",
    ))

    assert len(store.get_active("coder")) == 1
    assert len(store.get_active("researcher")) == 1
    assert store.get_active("coder")[0].content == "A"
    assert store.get_active("researcher")[0].content == "B"


# --- PromptEvolver ---


def test_build_evolved_prompt_empty(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)
    evolver = PromptEvolver(store=store)

    base = "You are a coder agent."
    assert evolver.build_evolved_prompt("coder", base) == base


def test_build_evolved_prompt_with_modifications(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)
    evolver = PromptEvolver(store=store)

    store.save(PromptModification(
        role_name="coder",
        modification_type=ModificationType.STRATEGY,
        content="Always validate input parameters.",
    ))
    store.save(PromptModification(
        role_name="coder",
        modification_type=ModificationType.CONSTRAINT,
        content="Never use eval() or exec().",
    ))
    store.save(PromptModification(
        role_name="coder",
        modification_type=ModificationType.APPEND,
        content="The project uses Python 3.11+.",
    ))

    base = "You are a coder agent."
    evolved = evolver.build_evolved_prompt("coder", base)

    assert "Learned Strategies" in evolved
    assert "Always validate input parameters." in evolved
    assert "Learned Constraints" in evolved
    assert "Never use eval() or exec()." in evolved
    assert "Additional Context" in evolved
    assert "The project uses Python 3.11+." in evolved


def test_build_evolved_prompt_replace(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)
    evolver = PromptEvolver(store=store)

    store.save(PromptModification(
        role_name="coder",
        modification_type=ModificationType.REPLACE,
        content="Be verbose and explain your reasoning.",
        metadata={"target": "Be concise."},
    ))

    base = "You are a coder. Be concise. Write clean code."
    evolved = evolver.build_evolved_prompt("coder", base)
    assert "Be verbose and explain your reasoning." in evolved
    assert "Be concise." not in evolved


def test_evolution_summary_empty(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)
    evolver = PromptEvolver(store=store)

    summary = evolver.get_evolution_summary("coder")
    assert "No prompt evolution history" in summary


def test_evolution_summary_with_data(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)
    evolver = PromptEvolver(store=store)

    store.save(PromptModification(
        role_name="coder",
        modification_type=ModificationType.STRATEGY,
        content="Test first approach.",
    ))

    summary = evolver.get_evolution_summary("coder")
    assert "Active modifications: 1" in summary
    assert "strategy" in summary


# --- Prompt Tools ---


@pytest.mark.asyncio
async def test_prompt_view_tool_empty(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)
    tool = PromptViewTool(store=store, agent_role="coder")

    result = await tool.execute()
    assert "No active" in result


@pytest.mark.asyncio
async def test_prompt_view_tool_with_data(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)

    store.save(PromptModification(
        role_name="coder",
        modification_type=ModificationType.STRATEGY,
        content="Write unit tests.",
        reasoning="Improves reliability.",
    ))

    tool = PromptViewTool(store=store, agent_role="coder")
    result = await tool.execute()
    assert "Write unit tests." in result
    assert "Improves reliability." in result


@pytest.mark.asyncio
async def test_prompt_modify_tool(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)
    tool = PromptModifyTool(store=store, agent_role="coder")

    result = await tool.execute(
        content="Prefer list comprehensions over map/filter.",
        modification_type="strategy",
        reasoning="More Pythonic.",
    )
    assert "Added prompt modification" in result
    assert "v1" in result

    active = store.get_active("coder")
    assert len(active) == 1
    assert active[0].content == "Prefer list comprehensions over map/filter."


@pytest.mark.asyncio
async def test_prompt_rollback_tool(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)

    for i in range(5):
        store.save(PromptModification(
            role_name="general",
            modification_type=ModificationType.STRATEGY,
            content=f"Strategy {i}",
        ))

    tool = PromptRollbackTool(store=store, agent_role="general")
    result = await tool.execute(version=2)
    assert "deactivated 3" in result
    assert len(store.get_active("general")) == 2


@pytest.mark.asyncio
async def test_prompt_history_tool(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = PromptEvolutionStore(db_path=db_path)

    store.save(PromptModification(
        role_name="reviewer",
        modification_type=ModificationType.CONSTRAINT,
        content="Check for SQL injection.",
        source_task="Code review task",
    ))

    tool = PromptHistoryTool(store=store, agent_role="reviewer")
    result = await tool.execute()
    assert "Check for SQL injection." in result
    assert "Code review task" in result


# --- Config ---


def test_prompt_evolution_config_defaults():
    cfg = Config()
    assert cfg.prompt_evolution.enabled is True
    assert cfg.prompt_evolution.reflect_after_task is True
    assert cfg.prompt_evolution.max_active_modifications == 10


def test_prompt_evolution_config_custom():
    cfg = Config(prompt_evolution={
        "enabled": False,
        "reflect_after_task": False,
        "max_active_modifications": 5,
    })
    assert cfg.prompt_evolution.enabled is False
    assert cfg.prompt_evolution.max_active_modifications == 5
