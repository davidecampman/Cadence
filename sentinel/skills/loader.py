"""SKILL.md loader — discovers, parses, and manages skills with dependencies and versioning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SkillDefinition(BaseModel):
    """A parsed SKILL.md file."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    dependencies: list[str] = Field(default_factory=list)   # Other skill names
    tags: list[str] = Field(default_factory=list)
    instructions: str = ""                                    # The main skill body
    examples: list[str] = Field(default_factory=list)
    file_path: str = ""

    @property
    def semver(self) -> tuple[int, ...]:
        """Parse version as tuple for comparison."""
        return tuple(int(x) for x in self.version.split(".") if x.isdigit())


class SkillLoader:
    """Discovers and loads SKILL.md files from configured directories.

    SKILL.md format:
    ```
    ---
    name: skill-name
    version: 1.0.0
    description: What this skill does
    author: someone
    dependencies:
      - other-skill
    tags:
      - category
    ---

    # Instructions
    The actual skill instructions go here...

    ## Examples
    - Example 1
    - Example 2
    ```
    """

    def __init__(self, directories: list[str] | None = None):
        self._dirs = [Path(d) for d in (directories or ["./skills"])]
        self._skills: dict[str, SkillDefinition] = {}

    def discover(self) -> list[SkillDefinition]:
        """Scan all configured directories for SKILL.md files."""
        self._skills.clear()

        for skill_dir in self._dirs:
            if not skill_dir.exists():
                continue

            # Look for SKILL.md files (direct or in subdirectories)
            for skill_file in skill_dir.rglob("SKILL.md"):
                try:
                    skill = self._parse_skill_file(skill_file)
                    if skill:
                        # If duplicate, keep the higher version
                        existing = self._skills.get(skill.name)
                        if existing and existing.semver >= skill.semver:
                            continue
                        self._skills[skill.name] = skill
                except Exception:
                    continue  # Skip malformed files

        return list(self._skills.values())

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def resolve_dependencies(self, skill_name: str) -> list[SkillDefinition]:
        """Return a skill and all its dependencies in dependency order (topological sort)."""
        visited: set[str] = set()
        result: list[SkillDefinition] = []

        def _visit(name: str):
            if name in visited:
                return
            visited.add(name)
            skill = self._skills.get(name)
            if not skill:
                return
            for dep in skill.dependencies:
                _visit(dep)
            result.append(skill)

        _visit(skill_name)
        return result

    def get_skill_prompt(self, skill_name: str) -> str | None:
        """Get the full prompt for a skill, including resolved dependencies."""
        skills = self.resolve_dependencies(skill_name)
        if not skills:
            return None

        parts = []
        for skill in skills:
            parts.append(f"## Skill: {skill.name} (v{skill.version})")
            parts.append(skill.instructions)
            if skill.examples:
                parts.append("\n### Examples")
                for ex in skill.examples:
                    parts.append(f"- {ex}")
            parts.append("")

        return "\n".join(parts)

    @property
    def all_skills(self) -> dict[str, SkillDefinition]:
        return dict(self._skills)

    @staticmethod
    def _parse_skill_file(path: Path) -> SkillDefinition | None:
        """Parse a SKILL.md file with YAML frontmatter."""
        text = path.read_text(errors="replace")

        # Split frontmatter from body
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not frontmatter_match:
            # No frontmatter — treat entire file as instructions with filename as name
            name = path.parent.name or path.stem
            return SkillDefinition(
                name=name,
                instructions=text.strip(),
                file_path=str(path),
            )

        fm_text = frontmatter_match.group(1)
        body = frontmatter_match.group(2)

        try:
            fm = yaml.safe_load(fm_text) or {}
        except yaml.YAMLError:
            return None

        # Extract examples from body
        examples = []
        example_match = re.search(r"##\s*Examples?\s*\n(.*?)(?=\n##|\Z)", body, re.DOTALL)
        if example_match:
            for line in example_match.group(1).strip().splitlines():
                line = line.strip()
                if line.startswith("- "):
                    examples.append(line[2:])

        # Instructions = body minus examples section
        instructions = body
        if example_match:
            instructions = body[:example_match.start()].strip()

        return SkillDefinition(
            name=fm.get("name", path.parent.name),
            version=str(fm.get("version", "1.0.0")),
            description=fm.get("description", ""),
            author=fm.get("author", ""),
            dependencies=fm.get("dependencies", []),
            tags=fm.get("tags", []),
            instructions=instructions,
            examples=examples,
            file_path=str(path),
        )
