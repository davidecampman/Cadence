"""Agent collaboration patterns — structured multi-agent interactions.

Provides three collaboration strategies that go beyond simple task delegation:

- **Debate**: Two agents argue opposing positions; a judge synthesizes the best answer.
- **Peer Review**: One agent produces output, another reviews and critiques it,
  then the producer revises. Iterates until the reviewer approves.
- **Consensus**: Multiple agents independently propose solutions; the best is
  selected by LLM-based voting.

Each pattern is invoked by the orchestrator when the planner deems structured
interaction more effective than independent parallel tasks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cadence.core.agent import Agent
from cadence.core.config import Config, get_config
from cadence.core.llm import chat_completion
from cadence.core.trace import TraceLogger
from cadence.core.types import AgentRole, Message, PermissionTier, Role
from cadence.prompts.evolution import PromptEvolver
from cadence.skills.loader import SkillLoader
from cadence.tools.base import ToolRegistry


class CollaborationMode(str, Enum):
    DEBATE = "debate"
    PEER_REVIEW = "peer_review"
    CONSENSUS = "consensus"


@dataclass
class CollaborationResult:
    """Outcome of a collaboration pattern."""
    mode: CollaborationMode
    final_answer: str
    rounds: int
    contributions: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Specialist roles for collaboration ---

_PROPONENT_ROLE = AgentRole(
    name="proponent",
    description=(
        "You argue FOR a position or approach. Present the strongest possible case "
        "with evidence and reasoning. Be persuasive but honest."
    ),
    permission_tier=PermissionTier.PRIVILEGED,
)

_OPPONENT_ROLE = AgentRole(
    name="opponent",
    description=(
        "You argue AGAINST a position or approach. Find weaknesses, counter-arguments, "
        "and alternative viewpoints. Be rigorous and constructive."
    ),
    permission_tier=PermissionTier.PRIVILEGED,
)

_REVIEWER_ROLE = AgentRole(
    name="critic",
    description=(
        "You review work for correctness, completeness, security, and quality. "
        "Provide specific, actionable feedback. Say APPROVED if the work meets "
        "all requirements with no issues."
    ),
    allowed_tools=["read_file", "list_files", "search_files", "execute_code", "memory_query"],
)

_PRODUCER_ROLE = AgentRole(
    name="producer",
    description=(
        "You produce high-quality work product (code, analysis, documentation). "
        "When given review feedback, you address every point carefully."
    ),
    permission_tier=PermissionTier.PRIVILEGED,
)


class CollaborationEngine:
    """Runs structured multi-agent collaboration patterns."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        trace: TraceLogger,
        config: Config | None = None,
        skill_loader: SkillLoader | None = None,
        prompt_evolver: PromptEvolver | None = None,
    ):
        self.tools = tool_registry
        self.trace = trace
        self.config = config or get_config()
        self.skill_loader = skill_loader
        self.prompt_evolver = prompt_evolver
        self._total_tokens: int = 0

    async def run(
        self,
        mode: CollaborationMode,
        task: str,
        *,
        max_rounds: int = 3,
        num_proposers: int = 3,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> CollaborationResult:
        """Execute a collaboration pattern and return the result."""
        if mode == CollaborationMode.DEBATE:
            return await self._debate(task, max_rounds, conversation_history)
        elif mode == CollaborationMode.PEER_REVIEW:
            return await self._peer_review(task, max_rounds, conversation_history)
        elif mode == CollaborationMode.CONSENSUS:
            return await self._consensus(task, num_proposers, conversation_history)
        else:
            raise ValueError(f"Unknown collaboration mode: {mode}")

    # ------------------------------------------------------------------
    # Debate pattern
    # ------------------------------------------------------------------

    async def _debate(
        self,
        task: str,
        max_rounds: int,
        conversation_history: list[dict[str, str]] | None,
    ) -> CollaborationResult:
        """Two agents argue opposing sides; a judge synthesizes the best answer.

        Flow per round:
          1. Proponent argues FOR
          2. Opponent argues AGAINST (seeing proponent's argument)
          3. After all rounds, a judge LLM call synthesizes the final answer
        """
        self.trace.thought("collaboration", f"Starting DEBATE on: {task[:100]}")

        proponent = self._spawn_agent(_PROPONENT_ROLE)
        opponent = self._spawn_agent(_OPPONENT_ROLE)

        contributions: list[dict[str, str]] = []
        debate_context = ""

        for round_num in range(1, max_rounds + 1):
            self.trace.action("collaboration", f"Debate round {round_num}/{max_rounds}")

            # Proponent argues
            pro_prompt = (
                f"TASK: {task}\n\n"
                f"You are in round {round_num} of a structured debate.\n"
            )
            if debate_context:
                pro_prompt += f"Previous exchanges:\n{debate_context}\n\n"
            pro_prompt += (
                "Present your strongest argument FOR this approach/position. "
                "Address any counter-arguments raised so far."
            )

            pro_result = await proponent.run(pro_prompt, conversation_history=conversation_history)
            self._total_tokens += proponent._total_tokens
            proponent._total_tokens = 0
            proponent._iterations = 0
            proponent._history = []

            contributions.append({"role": "proponent", "round": str(round_num), "content": pro_result})
            debate_context += f"\n[Round {round_num} — Proponent]: {pro_result}\n"

            # Opponent argues
            opp_prompt = (
                f"TASK: {task}\n\n"
                f"You are in round {round_num} of a structured debate.\n"
                f"Previous exchanges:\n{debate_context}\n\n"
                "Present your strongest argument AGAINST the proponent's position. "
                "Find weaknesses, risks, and better alternatives."
            )

            opp_result = await opponent.run(opp_prompt, conversation_history=conversation_history)
            self._total_tokens += opponent._total_tokens
            opponent._total_tokens = 0
            opponent._iterations = 0
            opponent._history = []

            contributions.append({"role": "opponent", "round": str(round_num), "content": opp_result})
            debate_context += f"\n[Round {round_num} — Opponent]: {opp_result}\n"

        # Judge synthesizes
        final = await self._judge_debate(task, debate_context)

        self.trace.result("collaboration", f"Debate completed after {max_rounds} rounds")
        return CollaborationResult(
            mode=CollaborationMode.DEBATE,
            final_answer=final,
            rounds=max_rounds,
            contributions=contributions,
            metadata={"tokens_used": self._total_tokens},
        )

    async def _judge_debate(self, task: str, debate_context: str) -> str:
        """LLM call to synthesize the debate into a final answer."""
        prompt = (
            f"You are a fair judge evaluating a structured debate.\n\n"
            f"ORIGINAL TASK: {task}\n\n"
            f"DEBATE TRANSCRIPT:\n{debate_context}\n\n"
            "Synthesize the strongest points from both sides into a single, "
            "well-reasoned final answer. Give credit where arguments were strongest. "
            "Be decisive — provide a clear recommendation or solution."
        )

        text, _, usage = await chat_completion(
            model=self.config.models.strong,
            messages=[Message(role=Role.USER, content=prompt)],
            temperature=0.4,
            bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
            local_config=self.config.models.local if self.config.models.local.enabled else None,
        )
        self._total_tokens += usage.get("total_tokens", 0)
        return text

    # ------------------------------------------------------------------
    # Peer Review pattern
    # ------------------------------------------------------------------

    async def _peer_review(
        self,
        task: str,
        max_rounds: int,
        conversation_history: list[dict[str, str]] | None,
    ) -> CollaborationResult:
        """One agent produces, another reviews. Iterate until approved or max rounds.

        Flow per round:
          1. Producer creates/revises the work product
          2. Reviewer critiques it
          3. If reviewer says APPROVED, stop
          4. Otherwise, producer revises based on feedback
        """
        self.trace.thought("collaboration", f"Starting PEER REVIEW on: {task[:100]}")

        producer = self._spawn_agent(_PRODUCER_ROLE)
        reviewer = self._spawn_agent(_REVIEWER_ROLE)

        contributions: list[dict[str, str]] = []
        review_feedback = ""

        for round_num in range(1, max_rounds + 1):
            self.trace.action("collaboration", f"Peer review round {round_num}/{max_rounds}")

            # Producer creates/revises
            prod_prompt = f"TASK: {task}\n\n"
            if review_feedback:
                prod_prompt += (
                    f"Round {round_num}: You received this review feedback on your previous work:\n"
                    f"{review_feedback}\n\n"
                    "Revise your work to address ALL feedback points."
                )
            else:
                prod_prompt += "Produce your best work for this task."

            prod_result = await producer.run(prod_prompt, conversation_history=conversation_history)
            self._total_tokens += producer._total_tokens
            producer._total_tokens = 0
            producer._iterations = 0
            producer._history = []

            contributions.append({"role": "producer", "round": str(round_num), "content": prod_result})

            # Reviewer critiques
            rev_prompt = (
                f"ORIGINAL TASK: {task}\n\n"
                f"WORK PRODUCT (round {round_num}):\n{prod_result}\n\n"
                "Review this work for correctness, completeness, quality, and security.\n"
                "If it fully meets all requirements with no issues, respond with: APPROVED\n"
                "Otherwise, provide specific, actionable feedback for improvement."
            )

            rev_result = await reviewer.run(rev_prompt, conversation_history=conversation_history)
            self._total_tokens += reviewer._total_tokens
            reviewer._total_tokens = 0
            reviewer._iterations = 0
            reviewer._history = []

            contributions.append({"role": "reviewer", "round": str(round_num), "content": rev_result})

            # Check if approved
            if "APPROVED" in rev_result.upper():
                self.trace.result(
                    "collaboration",
                    f"Peer review APPROVED at round {round_num}",
                )
                return CollaborationResult(
                    mode=CollaborationMode.PEER_REVIEW,
                    final_answer=prod_result,
                    rounds=round_num,
                    contributions=contributions,
                    metadata={"approved": True, "tokens_used": self._total_tokens},
                )

            review_feedback = rev_result

        # Max rounds reached without approval — return last production with note
        self.trace.thought(
            "collaboration",
            f"Peer review reached max rounds ({max_rounds}) without full approval",
        )
        return CollaborationResult(
            mode=CollaborationMode.PEER_REVIEW,
            final_answer=prod_result,
            rounds=max_rounds,
            contributions=contributions,
            metadata={"approved": False, "tokens_used": self._total_tokens},
        )

    # ------------------------------------------------------------------
    # Consensus pattern
    # ------------------------------------------------------------------

    async def _consensus(
        self,
        task: str,
        num_proposers: int,
        conversation_history: list[dict[str, str]] | None,
    ) -> CollaborationResult:
        """Multiple agents independently propose solutions; the best is selected.

        Flow:
          1. N agents work on the same task independently (in parallel)
          2. An LLM judge evaluates all proposals and picks the best one
        """
        self.trace.thought(
            "collaboration",
            f"Starting CONSENSUS ({num_proposers} proposers) on: {task[:100]}",
        )

        # Create proposer agents with distinct IDs
        general_role = AgentRole(
            name="proposer",
            description=(
                "You independently solve the given task. Provide a complete, "
                "well-reasoned solution. Do your best work."
            ),
            permission_tier=PermissionTier.PRIVILEGED,
        )

        async def _run_proposer(idx: int) -> str:
            agent = self._spawn_agent(general_role, agent_id=f"proposer-{idx}")
            prompt = (
                f"TASK: {task}\n\n"
                f"You are proposer #{idx + 1} of {num_proposers}. "
                "Provide your independent solution. Other proposers are working "
                "on the same task — aim for the best possible answer."
            )
            result = await agent.run(prompt, conversation_history=conversation_history)
            self._total_tokens += agent._total_tokens
            return result

        # Run all proposers in parallel
        proposals = await asyncio.gather(
            *[_run_proposer(i) for i in range(num_proposers)]
        )

        contributions = [
            {"role": "proposer", "round": str(i + 1), "content": p}
            for i, p in enumerate(proposals)
        ]

        self.trace.action(
            "collaboration",
            f"Received {num_proposers} proposals, selecting best via judge",
        )

        # Judge selects the best
        final = await self._judge_consensus(task, proposals)

        self.trace.result("collaboration", "Consensus selection completed")
        return CollaborationResult(
            mode=CollaborationMode.CONSENSUS,
            final_answer=final,
            rounds=1,
            contributions=contributions,
            metadata={"num_proposers": num_proposers, "tokens_used": self._total_tokens},
        )

    async def _judge_consensus(self, task: str, proposals: list[str]) -> str:
        """LLM call to select and synthesize the best proposal."""
        proposals_block = "\n\n---\n\n".join(
            f"### Proposal {i + 1}\n{p}" for i, p in enumerate(proposals)
        )
        prompt = (
            f"You are evaluating {len(proposals)} independent proposals for a task.\n\n"
            f"ORIGINAL TASK: {task}\n\n"
            f"PROPOSALS:\n{proposals_block}\n\n"
            "Select the best proposal. You may combine strengths from multiple proposals. "
            "Provide the final, definitive answer. Do NOT list the proposals — give the "
            "synthesized best answer directly."
        )

        text, _, usage = await chat_completion(
            model=self.config.models.strong,
            messages=[Message(role=Role.USER, content=prompt)],
            temperature=0.3,
            bedrock_config=self.config.models.bedrock if self.config.models.bedrock.enabled else None,
            local_config=self.config.models.local if self.config.models.local.enabled else None,
        )
        self._total_tokens += usage.get("total_tokens", 0)
        return text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _spawn_agent(
        self,
        role: AgentRole,
        agent_id: str | None = None,
    ) -> Agent:
        """Create a specialist agent for collaboration."""
        return Agent(
            role=role,
            tool_registry=self.tools,
            trace=self.trace,
            config=self.config,
            agent_id=agent_id,
            skill_loader=self.skill_loader,
            prompt_evolver=self.prompt_evolver,
        )
