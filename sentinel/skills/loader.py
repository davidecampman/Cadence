"""SKILL.md loader — discovers, parses, and manages skills with dependencies and versioning."""

from __future__ import annotations

import re
import shutil
import zipfile
from io import BytesIO
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

    def install_from_zip(self, data: bytes) -> SkillDefinition:
        """Install a skill from a zip file containing a SKILL.md.

        The zip must contain a SKILL.md at the root or inside a single
        top-level directory. The skill is extracted into the first
        configured skills directory.

        Returns the installed SkillDefinition.
        Raises ValueError on invalid zip or missing SKILL.md.
        """
        buf = BytesIO(data)
        if not zipfile.is_zipfile(buf):
            raise ValueError("Uploaded file is not a valid zip archive")

        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()

            # Find the SKILL.md inside the zip
            skill_md_path: str | None = None
            for n in names:
                if n.endswith("SKILL.md") and "__MACOSX" not in n:
                    skill_md_path = n
                    break

            if not skill_md_path:
                raise ValueError("Zip archive does not contain a SKILL.md file")

            # Parse the skill to validate before extracting
            skill_md_content = zf.read(skill_md_path).decode("utf-8", errors="replace")
            # Determine the skill directory name from the zip structure
            parts = skill_md_path.split("/")
            if len(parts) > 1:
                # SKILL.md is inside a subdirectory — use that as the skill folder name
                top_dir = parts[0]
            else:
                # SKILL.md at root of zip — parse name from frontmatter
                top_dir = None

            # Write a temp file to parse it
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_skill_dir = Path(tmpdir) / "skill"
                tmp_skill_dir.mkdir()
                tmp_file = tmp_skill_dir / "SKILL.md"
                tmp_file.write_text(skill_md_content)
                skill = self._parse_skill_file(tmp_file)
                if not skill:
                    raise ValueError("Failed to parse SKILL.md — invalid format")

            # Determine the install target directory
            install_base = self._dirs[0]
            install_base.mkdir(parents=True, exist_ok=True)

            skill_folder = install_base / skill.name
            # Remove existing version if present
            if skill_folder.exists():
                shutil.rmtree(skill_folder)
            skill_folder.mkdir(parents=True, exist_ok=True)

            # Extract files into the skill folder
            for member in names:
                if "__MACOSX" in member or member.endswith("/"):
                    continue
                # Strip the top-level directory prefix if present
                if top_dir and member.startswith(top_dir + "/"):
                    relative = member[len(top_dir) + 1:]
                else:
                    relative = member
                if not relative:
                    continue
                dest = skill_folder / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(member))

            # Re-parse the installed skill from its final location
            installed_skill_file = skill_folder / "SKILL.md"
            installed = self._parse_skill_file(installed_skill_file)
            if installed:
                self._skills[installed.name] = installed
                return installed
            raise ValueError("Skill installed but failed to re-parse")

    def uninstall(self, skill_name: str) -> bool:
        """Remove a skill by name.

        Deletes the skill directory from disk and removes it from the
        in-memory registry. Returns True if the skill was found and removed.
        """
        skill = self._skills.get(skill_name)
        if not skill:
            return False

        # Remove from disk
        if skill.file_path:
            skill_file = Path(skill.file_path)
            skill_dir = skill_file.parent
            # Only remove the directory if it's inside one of the configured skill dirs
            for base_dir in self._dirs:
                try:
                    skill_dir.relative_to(base_dir.resolve())
                    if skill_dir.exists():
                        shutil.rmtree(skill_dir)
                    break
                except ValueError:
                    continue

        # Remove from in-memory registry
        self._skills.pop(skill_name, None)
        return True

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
