"""Configuration loading with YAML defaults + environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BedrockConfig(BaseModel):
    """AWS Bedrock provider configuration."""
    enabled: bool = False
    region: str = "us-east-1"
    profile: str | None = None
    role_arn: str | None = None


class ModelConfig(BaseModel):
    strong: str = "claude-sonnet-4-20250514"
    fast: str = "claude-haiku-4-5-20251001"
    embedding: str = "text-embedding-3-small"
    fallback_chain: list[str] = Field(default_factory=lambda: ["gpt-4o", "claude-sonnet-4-20250514"])
    bedrock: BedrockConfig = Field(default_factory=BedrockConfig)


class BudgetConfig(BaseModel):
    max_tokens_per_task: int = 100_000
    max_tokens_per_session: int = 500_000
    warn_at_percentage: int = 80


class AgentLimits(BaseModel):
    max_depth: int = 5
    max_parallel: int = 4
    loop_detection_window: int = 5
    max_iterations_per_task: int = 25


class MemoryConfig(BaseModel):
    backend: str = "chromadb"
    persist_dir: str = "./data/memory"
    default_namespace: str = "shared"
    decay_rate: float = 0.05
    max_results: int = 10
    similarity_threshold: float = 0.7


class ExecutionConfig(BaseModel):
    sandbox: str = "subprocess"
    timeout_seconds: int = 120
    max_output_bytes: int = 1_048_576


class SkillsConfig(BaseModel):
    directories: list[str] = Field(default_factory=lambda: ["./skills"])
    auto_discover: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    trace_file: str = "./data/traces.jsonl"
    rich_console: bool = True


class Config(BaseModel):
    """Root configuration for Sentinel."""

    models: ModelConfig = Field(default_factory=ModelConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    agents: AgentLimits = Field(default_factory=AgentLimits)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


ENV_PREFIX = "SENTINEL_"


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Override nested config values from environment variables.

    SENTINEL_MODELS_STRONG=gpt-4o -> data["models"]["strong"] = "gpt-4o"
    """
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = key[len(ENV_PREFIX) :].lower().split("_", 1)
        if len(parts) == 2:
            section, field = parts
            if section in data and isinstance(data[section], dict):
                # Attempt type coercion for numeric values
                existing = data[section].get(field)
                if isinstance(existing, int):
                    value = int(value)
                elif isinstance(existing, float):
                    value = float(value)
                elif isinstance(existing, bool):
                    value = value.lower() in ("true", "1", "yes")
                data[section][field] = value
    return data


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML file with environment variable overrides."""
    if path is None:
        path = Path(__file__).parent.parent / "config" / "default.yaml"
    else:
        path = Path(path)

    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    data = _apply_env_overrides(data)
    return Config(**data)
