"""Tests for agent collaboration patterns (debate, peer review, consensus)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cadence.agents.collaboration import (
    CollaborationEngine,
    CollaborationMode,
    CollaborationResult,
)
from cadence.core.config import Config
from cadence.core.trace import TraceLogger
from cadence.core.types import AgentRole, PermissionTier, ToolCall
from cadence.tools.base import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class NoopTool(Tool):
    name = "noop"
    description = "Does nothing"
    parameters = {"type": "object", "properties": {}}
    permission_tier = PermissionTier.READ_ONLY

    async def execute(self, **kwargs) -> str:
        return "ok"


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(NoopTool())
    return reg


def _make_config() -> Config:
    cfg = Config()
    cfg.prompt_evolution.enabled = False
    return cfg


def _make_engine() -> CollaborationEngine:
    return CollaborationEngine(
        tool_registry=_make_registry(),
        trace=TraceLogger(trace_file=None, console=False),
        config=_make_config(),
    )


def _mock_chat_completion(responses: list[str]):
    """Return an AsyncMock that cycles through predefined responses."""
    call_count = 0

    async def _fake_chat(*args, **kwargs):
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        return responses[idx], [], {"total_tokens": 100}

    return _fake_chat


# ---------------------------------------------------------------------------
# Debate pattern tests
# ---------------------------------------------------------------------------

class TestDebatePattern:
    @pytest.mark.asyncio
    async def test_debate_basic_flow(self):
        """Debate runs the correct number of rounds and produces a final answer."""
        engine = _make_engine()

        responses = [
            "Proponent arg round 1",   # proponent r1
            "Opponent arg round 1",    # opponent r1
            "Proponent arg round 2",   # proponent r2
            "Opponent arg round 2",    # opponent r2
            "Final synthesized answer",  # judge
        ]

        with patch("cadence.core.llm.chat_completion", new=_mock_chat_completion(responses)):
            with patch.object(engine, "_spawn_agent") as mock_spawn:
                # Create mock agents that return canned responses
                agent_call_idx = 0

                def make_mock_agent(role, agent_id=None):
                    nonlocal agent_call_idx
                    agent = MagicMock()
                    agent._total_tokens = 50
                    agent._iterations = 0
                    agent._history = []

                    captured_idx = agent_call_idx
                    agent_call_idx += 1

                    async def mock_run(task, conversation_history=None, images=None):
                        return responses[captured_idx]

                    agent.run = mock_run
                    return agent

                mock_spawn.side_effect = make_mock_agent

                # Use _judge_debate directly after running agent rounds
                with patch.object(engine, "_judge_debate", new=AsyncMock(return_value="Final synthesized answer")):
                    result = await engine.run(
                        CollaborationMode.DEBATE,
                        "Should we use microservices or monolith?",
                        max_rounds=2,
                    )

        assert isinstance(result, CollaborationResult)
        assert result.mode == CollaborationMode.DEBATE
        assert result.rounds == 2
        assert result.final_answer == "Final synthesized answer"
        assert len(result.contributions) == 4  # 2 rounds x 2 agents

    @pytest.mark.asyncio
    async def test_debate_contributions_have_correct_roles(self):
        """Each contribution is tagged with the correct role."""
        engine = _make_engine()

        with patch.object(engine, "_spawn_agent") as mock_spawn:
            call_idx = 0

            def make_agent(role, agent_id=None):
                nonlocal call_idx
                agent = MagicMock()
                agent._total_tokens = 10
                agent._iterations = 0
                agent._history = []

                idx = call_idx
                call_idx += 1
                texts = ["pro1", "opp1"]

                async def mock_run(task, conversation_history=None, images=None):
                    return texts[idx] if idx < len(texts) else "x"

                agent.run = mock_run
                return agent

            mock_spawn.side_effect = make_agent

            with patch.object(engine, "_judge_debate", new=AsyncMock(return_value="judged")):
                result = await engine.run(
                    CollaborationMode.DEBATE,
                    "test debate",
                    max_rounds=1,
                )

        assert result.contributions[0]["role"] == "proponent"
        assert result.contributions[1]["role"] == "opponent"

    @pytest.mark.asyncio
    async def test_judge_debate_calls_strong_model(self):
        """The judge uses the strong model for synthesis."""
        engine = _make_engine()

        with patch("cadence.agents.collaboration.chat_completion") as mock_cc:
            mock_cc.return_value = ("The best approach is...", [], {"total_tokens": 200})
            result = await engine._judge_debate("test task", "debate transcript")

        assert result == "The best approach is..."
        call_args = mock_cc.call_args
        assert call_args.kwargs["model"] == engine.config.models.strong


# ---------------------------------------------------------------------------
# Peer Review pattern tests
# ---------------------------------------------------------------------------

class TestPeerReviewPattern:
    @pytest.mark.asyncio
    async def test_peer_review_approved_first_round(self):
        """If the reviewer says APPROVED on round 1, stop immediately."""
        engine = _make_engine()

        with patch.object(engine, "_spawn_agent") as mock_spawn:
            call_idx = 0

            def make_agent(role, agent_id=None):
                nonlocal call_idx
                agent = MagicMock()
                agent._total_tokens = 10
                agent._iterations = 0
                agent._history = []

                idx = call_idx
                call_idx += 1
                texts = ["produced work", "APPROVED - looks great"]

                async def mock_run(task, conversation_history=None, images=None):
                    return texts[idx] if idx < len(texts) else "x"

                agent.run = mock_run
                return agent

            mock_spawn.side_effect = make_agent

            result = await engine.run(
                CollaborationMode.PEER_REVIEW,
                "Write a sorting function",
                max_rounds=3,
            )

        assert result.mode == CollaborationMode.PEER_REVIEW
        assert result.rounds == 1
        assert result.final_answer == "produced work"
        assert result.metadata.get("approved") is True

    @pytest.mark.asyncio
    async def test_peer_review_iterates_until_approved(self):
        """Peer review iterates when reviewer gives feedback, stops on APPROVED."""
        engine = _make_engine()

        # Peer review spawns producer and reviewer ONCE, then calls .run()
        # on each repeatedly (resetting state between rounds).
        # Producer returns different drafts on successive calls.
        # Reviewer rejects round 1, approves round 2.
        producer_responses = iter(["first draft", "revised with error handling"])
        reviewer_responses = iter(["Needs error handling", "APPROVED"])

        def make_agent(role, agent_id=None):
            agent = MagicMock()
            agent._total_tokens = 10
            agent._iterations = 0
            agent._history = []

            # Pick the right response iterator based on the role
            is_producer = role.name == "producer"
            responses = producer_responses if is_producer else reviewer_responses

            async def mock_run(task, conversation_history=None, images=None):
                return next(responses, "fallback")

            agent.run = mock_run
            return agent

        with patch.object(engine, "_spawn_agent", side_effect=make_agent):
            result = await engine.run(
                CollaborationMode.PEER_REVIEW,
                "Write a sorting function",
                max_rounds=3,
            )

        assert result.rounds == 2
        assert result.metadata.get("approved") is True
        assert len(result.contributions) == 4  # 2 producer + 2 reviewer

    @pytest.mark.asyncio
    async def test_peer_review_max_rounds_without_approval(self):
        """Returns last work product with approved=False when max rounds exhausted."""
        engine = _make_engine()

        with patch.object(engine, "_spawn_agent") as mock_spawn:
            call_idx = 0

            def make_agent(role, agent_id=None):
                nonlocal call_idx
                agent = MagicMock()
                agent._total_tokens = 10
                agent._iterations = 0
                agent._history = []

                idx = call_idx
                call_idx += 1
                texts = [
                    "draft 1", "fix X",
                    "draft 2", "fix Y",
                ]

                async def mock_run(task, conversation_history=None, images=None):
                    return texts[idx] if idx < len(texts) else "still not right"

                agent.run = mock_run
                return agent

            mock_spawn.side_effect = make_agent

            result = await engine.run(
                CollaborationMode.PEER_REVIEW,
                "Write a function",
                max_rounds=2,
            )

        assert result.rounds == 2
        assert result.metadata.get("approved") is False


# ---------------------------------------------------------------------------
# Consensus pattern tests
# ---------------------------------------------------------------------------

class TestConsensusPattern:
    @pytest.mark.asyncio
    async def test_consensus_runs_proposers_and_selects_best(self):
        """Consensus spawns N proposers, then a judge picks the best."""
        engine = _make_engine()

        with patch.object(engine, "_spawn_agent") as mock_spawn:
            call_idx = 0

            def make_agent(role, agent_id=None):
                nonlocal call_idx
                agent = MagicMock()
                agent._total_tokens = 10
                agent._iterations = 0
                agent._history = []

                idx = call_idx
                call_idx += 1
                proposals = ["Solution A", "Solution B", "Solution C"]

                async def mock_run(task, conversation_history=None, images=None):
                    return proposals[idx] if idx < len(proposals) else "fallback"

                agent.run = mock_run
                return agent

            mock_spawn.side_effect = make_agent

            with patch.object(
                engine, "_judge_consensus",
                new=AsyncMock(return_value="Best combined solution"),
            ):
                result = await engine.run(
                    CollaborationMode.CONSENSUS,
                    "Design an API",
                    num_proposers=3,
                )

        assert result.mode == CollaborationMode.CONSENSUS
        assert result.rounds == 1
        assert len(result.contributions) == 3
        assert result.final_answer == "Best combined solution"
        assert result.metadata.get("num_proposers") == 3

    @pytest.mark.asyncio
    async def test_consensus_single_proposer(self):
        """Consensus works with a single proposer."""
        engine = _make_engine()

        with patch.object(engine, "_spawn_agent") as mock_spawn:
            agent = MagicMock()
            agent._total_tokens = 10
            agent._iterations = 0
            agent._history = []

            async def mock_run(task, conversation_history=None, images=None):
                return "Only solution"

            agent.run = mock_run
            mock_spawn.return_value = agent

            with patch.object(
                engine, "_judge_consensus",
                new=AsyncMock(return_value="Only solution"),
            ):
                result = await engine.run(
                    CollaborationMode.CONSENSUS,
                    "Design an API",
                    num_proposers=1,
                )

        assert len(result.contributions) == 1

    @pytest.mark.asyncio
    async def test_judge_consensus_calls_strong_model(self):
        """The consensus judge uses the strong model."""
        engine = _make_engine()

        with patch("cadence.agents.collaboration.chat_completion") as mock_cc:
            mock_cc.return_value = ("Best solution is...", [], {"total_tokens": 200})
            result = await engine._judge_consensus(
                "Design an API",
                ["Solution A", "Solution B"],
            )

        assert result == "Best solution is..."
        call_args = mock_cc.call_args
        assert call_args.kwargs["model"] == engine.config.models.strong


# ---------------------------------------------------------------------------
# CollaborationMode enum tests
# ---------------------------------------------------------------------------

class TestCollaborationMode:
    def test_enum_values(self):
        assert CollaborationMode.DEBATE.value == "debate"
        assert CollaborationMode.PEER_REVIEW.value == "peer_review"
        assert CollaborationMode.CONSENSUS.value == "consensus"

    def test_enum_from_string(self):
        assert CollaborationMode("debate") == CollaborationMode.DEBATE
        assert CollaborationMode("peer_review") == CollaborationMode.PEER_REVIEW
        assert CollaborationMode("consensus") == CollaborationMode.CONSENSUS


# ---------------------------------------------------------------------------
# CollaborationResult tests
# ---------------------------------------------------------------------------

class TestCollaborationResult:
    def test_result_fields(self):
        result = CollaborationResult(
            mode=CollaborationMode.DEBATE,
            final_answer="answer",
            rounds=2,
            contributions=[{"role": "proponent", "content": "arg"}],
            metadata={"tokens_used": 500},
        )
        assert result.mode == CollaborationMode.DEBATE
        assert result.final_answer == "answer"
        assert result.rounds == 2
        assert len(result.contributions) == 1
        assert result.metadata["tokens_used"] == 500

    def test_result_defaults(self):
        result = CollaborationResult(
            mode=CollaborationMode.CONSENSUS,
            final_answer="x",
            rounds=1,
        )
        assert result.contributions == []
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# Engine error handling
# ---------------------------------------------------------------------------

class TestCollaborationEngine:
    @pytest.mark.asyncio
    async def test_invalid_mode_raises(self):
        engine = _make_engine()
        with pytest.raises(ValueError, match="Unknown collaboration mode"):
            await engine.run("invalid_mode", "test")  # type: ignore

    @pytest.mark.asyncio
    async def test_token_tracking(self):
        """Engine tracks total tokens across all agent calls."""
        engine = _make_engine()
        assert engine._total_tokens == 0

        with patch.object(engine, "_spawn_agent") as mock_spawn:
            agent = MagicMock()
            agent._total_tokens = 100
            agent._iterations = 0
            agent._history = []

            async def mock_run(task, conversation_history=None, images=None):
                return "done"

            agent.run = mock_run
            mock_spawn.return_value = agent

            with patch.object(
                engine, "_judge_consensus",
                new=AsyncMock(return_value="final"),
            ):
                await engine.run(
                    CollaborationMode.CONSENSUS,
                    "test",
                    num_proposers=2,
                )

        # 2 proposers x 100 tokens each
        assert engine._total_tokens >= 200


# ---------------------------------------------------------------------------
# Orchestrator integration — collaboration in _plan
# ---------------------------------------------------------------------------

class TestOrchestratorCollaboration:
    @pytest.mark.asyncio
    async def test_plan_detects_collaboration_mode(self):
        """When the planner returns a collaboration JSON, the orchestrator runs it."""
        from cadence.agents.orchestrator import Orchestrator

        registry = _make_registry()
        config = _make_config()
        trace = TraceLogger(trace_file=None, console=False)

        orch = Orchestrator(
            tool_registry=registry,
            trace=trace,
            config=config,
        )
        orch._conversation_history = []

        # Mock the planning LLM to return a collaboration response
        plan_response = '{"collaboration": "debate"}'

        with patch("cadence.agents.orchestrator.chat_completion") as mock_cc:
            mock_cc.return_value = (plan_response, [], {"total_tokens": 50})

            # Mock the collaboration engine's run method
            collab_result = CollaborationResult(
                mode=CollaborationMode.DEBATE,
                final_answer="Debated answer",
                rounds=3,
            )
            with patch.object(
                orch.collaboration, "run",
                new=AsyncMock(return_value=collab_result),
            ):
                result = await orch._plan("Design the best architecture")

        assert isinstance(result, CollaborationResult)
        assert result.final_answer == "Debated answer"

    @pytest.mark.asyncio
    async def test_plan_falls_through_on_invalid_collaboration(self):
        """Invalid collaboration JSON falls through to normal task parsing."""
        from cadence.agents.orchestrator import Orchestrator

        registry = _make_registry()
        config = _make_config()
        trace = TraceLogger(trace_file=None, console=False)

        orch = Orchestrator(
            tool_registry=registry,
            trace=trace,
            config=config,
        )
        orch._conversation_history = []

        # Return invalid collaboration (unknown mode)
        plan_response = '{"collaboration": "unknown_mode"}'

        with patch("cadence.agents.orchestrator.chat_completion") as mock_cc:
            mock_cc.return_value = (plan_response, [], {"total_tokens": 50})
            result = await orch._plan("Do something")

        # Should fall through and return empty task list
        assert result == []
