"""Prompt evolver — uses LLM reflection to generate self-modifying prompt improvements."""

from __future__ import annotations

import json
import re
from typing import Any

from cadence.core.config import Config, get_config
from cadence.core.llm import chat_completion
from cadence.core.types import Message, Role
from cadence.prompts.store import (
    ModificationType,
    PromptEvolutionStore,
    PromptModification,
)


class PromptEvolver:
    """Analyzes agent performance and generates prompt modifications.

    After a task completes, the evolver:
      1. Reflects on the agent's performance (quality, efficiency, errors)
      2. Proposes specific prompt modifications (if warranted)
      3. Persists accepted modifications for future runs

    Modifications are applied to the agent's system prompt on subsequent runs,
    making the agent progressively better at its role.
    """

    def __init__(
        self,
        store: PromptEvolutionStore | None = None,
        config: Config | None = None,
    ):
        self.store = store or PromptEvolutionStore()
        self.config = config or get_config()

    def build_evolved_prompt(self, role_name: str, base_prompt: str) -> str:
        """Apply all active modifications to a base system prompt.

        Returns the modified prompt with active evolutions appended.
        """
        modifications = self.store.get_active(role_name)
        if not modifications:
            return base_prompt

        # Group by type for organized injection
        strategies = []
        constraints = []
        appendages = []

        for mod in modifications:
            if mod.modification_type == ModificationType.STRATEGY:
                strategies.append(mod)
            elif mod.modification_type == ModificationType.CONSTRAINT:
                constraints.append(mod)
            elif mod.modification_type == ModificationType.APPEND:
                appendages.append(mod)
            elif mod.modification_type == ModificationType.REPLACE:
                # Direct text replacement in the base prompt
                base_prompt = base_prompt.replace(
                    mod.metadata.get("target", ""),
                    mod.content,
                )

        evolved_sections = []

        if strategies:
            evolved_sections.append("## Learned Strategies")
            for mod in strategies:
                evolved_sections.append(f"- {mod.content}")

        if constraints:
            evolved_sections.append("## Learned Constraints")
            for mod in constraints:
                evolved_sections.append(f"- {mod.content}")

        if appendages:
            evolved_sections.append("## Additional Context")
            for mod in appendages:
                evolved_sections.append(mod.content)

        if evolved_sections:
            base_prompt += "\n\n" + "\n".join(evolved_sections) + "\n"

        return base_prompt

    async def reflect_and_evolve(
        self,
        role_name: str,
        task_description: str,
        task_result: str,
        iterations_used: int,
        max_iterations: int,
        errors: list[str] | None = None,
    ) -> list[PromptModification]:
        """Reflect on task execution and propose prompt modifications.

        Uses the fast model to analyze performance and suggest improvements.
        Only creates modifications when there's clear evidence of improvement needed.

        Returns list of new modifications (may be empty if performance was good).
        """
        # Build reflection prompt
        error_block = ""
        if errors:
            error_block = f"\nErrors encountered:\n" + "\n".join(f"- {e}" for e in errors)

        efficiency = 1.0 - (iterations_used / max_iterations)
        performance_score = max(0.0, min(1.0, efficiency))

        # Get existing modifications for context
        existing = self.store.get_active(role_name)
        existing_block = ""
        if existing:
            existing_block = "\nCurrent learned modifications:\n" + "\n".join(
                f"- [{mod.modification_type.value}] {mod.content}" for mod in existing
            )

        reflection_prompt = (
            "You are a prompt optimization expert. Analyze this agent's task execution "
            "and decide if the agent's system prompt should be modified to improve future performance.\n\n"
            f"Agent role: {role_name}\n"
            f"Task: {task_description}\n"
            f"Result (truncated): {task_result[:1500]}\n"
            f"Iterations used: {iterations_used}/{max_iterations}\n"
            f"Efficiency score: {performance_score:.2f}\n"
            f"{error_block}\n"
            f"{existing_block}\n\n"
            "Rules:\n"
            "- Only suggest modifications if there's a CLEAR pattern to improve.\n"
            "- Don't suggest changes for one-off issues.\n"
            "- Prefer minimal, targeted modifications over broad changes.\n"
            "- Never suggest more than 2 modifications at once.\n"
            "- If performance was good (efficiency > 0.7, no errors), respond with NO_CHANGES.\n\n"
            "If changes are needed, respond with a JSON array:\n"
            "```json\n"
            '[\n  {\n    "type": "strategy|constraint|append",\n'
            '    "content": "The specific instruction to add",\n'
            '    "reasoning": "Why this will help"\n  }\n]\n'
            "```\n\n"
            "If no changes needed, respond with: NO_CHANGES"
        )

        text, _, usage = await chat_completion(
            model=self.config.models.fast,
            messages=[Message(role=Role.USER, content=reflection_prompt)],
            temperature=0.3,
            max_tokens=1024,
            bedrock_config=(
                self.config.models.bedrock
                if self.config.models.bedrock.enabled
                else None
            ),
            local_config=(
                self.config.models.local
                if self.config.models.local.enabled
                else None
            ),
        )

        if "NO_CHANGES" in text.upper():
            return []

        # Parse proposed modifications
        modifications = self._parse_modifications(
            text=text,
            role_name=role_name,
            task_description=task_description,
            performance_score=performance_score,
        )

        # Save accepted modifications
        saved = []
        for mod in modifications:
            saved.append(self.store.save(mod))

        return saved

    def _parse_modifications(
        self,
        text: str,
        role_name: str,
        task_description: str,
        performance_score: float,
    ) -> list[PromptModification]:
        """Parse LLM output into PromptModification objects."""
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            return []

        try:
            proposals = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        modifications = []
        type_map = {
            "strategy": ModificationType.STRATEGY,
            "constraint": ModificationType.CONSTRAINT,
            "append": ModificationType.APPEND,
            "replace": ModificationType.REPLACE,
        }

        for proposal in proposals[:2]:  # Cap at 2 per reflection
            mod_type = type_map.get(
                proposal.get("type", "append"),
                ModificationType.APPEND,
            )
            content = proposal.get("content", "").strip()
            if not content:
                continue

            modifications.append(PromptModification(
                role_name=role_name,
                modification_type=mod_type,
                content=content,
                reasoning=proposal.get("reasoning", ""),
                source_task=task_description[:200],
                performance_score=performance_score,
            ))

        return modifications

    def get_evolution_summary(self, role_name: str) -> str:
        """Human-readable summary of a role's prompt evolution history."""
        active = self.store.get_active(role_name)
        history = self.store.get_history(role_name)

        if not history:
            return f"No prompt evolution history for role '{role_name}'."

        lines = [
            f"Prompt evolution for '{role_name}':",
            f"  Active modifications: {len(active)}",
            f"  Total modifications (all time): {len(history)}",
        ]

        if active:
            lines.append("\n  Active modifications:")
            for mod in active:
                status = "ON" if mod.active else "OFF"
                lines.append(
                    f"    v{mod.version} [{status}] ({mod.modification_type.value}) "
                    f"{mod.content[:80]}"
                )

        return "\n".join(lines)
