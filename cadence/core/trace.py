"""Structured reasoning trace — logs every observation/thought/action/result."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cadence.core.types import TraceStep

logger = logging.getLogger(__name__)


class TraceLogger:
    """Appends structured trace steps to a JSONL file and optional console."""

    def __init__(self, trace_file: str | None = None, console: bool = True):
        self._file_path = Path(trace_file) if trace_file else None
        self._console = console
        self._steps: list[TraceStep] = []

        if self._file_path:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: TraceStep) -> None:
        self._steps.append(step)

        if self._file_path:
            with open(self._file_path, "a", encoding="utf-8") as f:
                f.write(step.model_dump_json() + "\n")

        if self._console:
            _print_step(step)

    def observation(self, agent_id: str, content: str, task_id: str | None = None, **meta):
        self.log(TraceStep(
            agent_id=agent_id, task_id=task_id,
            step_type="observation", content=content, metadata=meta,
        ))

    def thought(self, agent_id: str, content: str, task_id: str | None = None, **meta):
        self.log(TraceStep(
            agent_id=agent_id, task_id=task_id,
            step_type="thought", content=content, metadata=meta,
        ))

    def action(self, agent_id: str, content: str, task_id: str | None = None, **meta):
        self.log(TraceStep(
            agent_id=agent_id, task_id=task_id,
            step_type="action", content=content, metadata=meta,
        ))

    def result(self, agent_id: str, content: str, task_id: str | None = None, **meta):
        self.log(TraceStep(
            agent_id=agent_id, task_id=task_id,
            step_type="result", content=content, metadata=meta,
        ))

    def error(self, agent_id: str, content: str, task_id: str | None = None, **meta):
        self.log(TraceStep(
            agent_id=agent_id, task_id=task_id,
            step_type="error", content=content, metadata=meta,
        ))

    @property
    def steps(self) -> list[TraceStep]:
        return list(self._steps)

    def export_json(self) -> str:
        return json.dumps([s.model_dump() for s in self._steps], indent=2)


# --- Console formatting ---

_STEP_ICONS = {
    "observation": "👁 ",
    "thought": "💭",
    "action": "⚡",
    "result": "✅",
    "error": "❌",
}


def _print_step(step: TraceStep) -> None:
    icon = _STEP_ICONS.get(step.step_type, "•")
    agent = step.agent_id[:12]
    preview = step.content[:200].replace("\n", " ")
    try:
        print(f"  {icon} [{agent}] {preview}")
    except UnicodeEncodeError:
        safe = f"  {icon} [{agent}] {preview}".encode("ascii", errors="replace").decode("ascii")
        print(safe)
