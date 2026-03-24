"""Configuration loader with YAML defaults + environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default.yaml"


class BedrockConfig(BaseModel):
    """AWS Bedrock provider configuration."""
    enabled: bool = False
    region: str = "us-east-1"
    profile: str | None = None  # AWS profile name (uses default chain if None)
    role_arn: str | None = None  # Optional IAM role to assume
    access_key_id: str | None = None  # Explicit AWS access key ID
    secret_access_key: str | None = None  # Explicit AWS secret access key
    api_key: str | None = None  # Long-term Bedrock API key (alternative to IAM credentials)


class ModelsConfig(BaseModel):
    strong: str = "claude-sonnet-4-20250514"
    fast: str = "claude-haiku-4-5-20251001"
    embedding: str = "text-embedding-3-small"
    fallback_chain: list[str] = Field(default_factory=lambda: ["gpt-4o", "claude-sonnet-4-20250514"])
    bedrock: BedrockConfig = Field(default_factory=BedrockConfig)


class BudgetConfig(BaseModel):
    max_tokens_per_task: int = 100_000
    max_tokens_per_session: int = 500_000
    warn_at_percentage: int = 80


class AgentsConfig(BaseModel):
    max_depth: int = 5
    max_parallel: int = 4
    loop_detection_window: int = 5
    max_iterations_per_task: int = 25


class MemoryConfig(BaseModel):
    backend: str = "sqlite"  # "sqlite" (cross-platform) | "chromadb" (vector search)
    persist_dir: str = "auto"  # "auto" = platform-aware default, or explicit path
    default_namespace: str = "shared"
    decay_rate: float = 0.05
    max_results: int = 10
    similarity_threshold: float = 0.7
    session_persistence: bool = True  # Persist conversation history across restarts


class ExecutionConfig(BaseModel):
    timeout_seconds: int = 120
    max_output_bytes: int = 1_048_576
    restrict_network: bool = False
    max_memory_mb: int = 512
    max_cpu_seconds: int = 60
    max_file_descriptors: int = 256
    blocked_commands: list[str] = Field(default_factory=lambda: [
        "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){", "fork bomb",
        "chmod -R 777 /", "shutdown", "reboot", "halt", "poweroff",
    ])


class SkillsConfig(BaseModel):
    directories: list[str] = Field(default_factory=lambda: ["./skills"])
    auto_discover: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    trace_file: str = "./data/traces.jsonl"
    rich_console: bool = True


class Config(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Override config values with SENTINEL_<SECTION>_<KEY> env vars."""
    prefix = "SENTINEL_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("_", 1)
        if len(parts) == 2:
            section, field = parts
            if section in data and isinstance(data[section], dict):
                # Attempt type coercion based on existing value
                existing = data[section].get(field)
                if isinstance(existing, bool):
                    data[section][field] = value.lower() in ("true", "1", "yes")
                elif isinstance(existing, int):
                    data[section][field] = int(value)
                elif isinstance(existing, float):
                    data[section][field] = float(value)
                else:
                    data[section][field] = value
    return data


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML file with env var overrides."""
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    data = _apply_env_overrides(data)
    return Config(**data)


# Singleton for convenience
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def update_config(updates: dict[str, Any]) -> Config:
    """Apply partial updates to the current config and persist to YAML."""
    global _config
    current = get_config()
    current_data = current.model_dump()

    # Deep merge updates into current config
    for section, values in updates.items():
        if section in current_data and isinstance(values, dict) and isinstance(current_data[section], dict):
            current_data[section].update(values)
        else:
            current_data[section] = values

    _config = Config(**current_data)

    # Persist to default.yaml
    if _DEFAULT_CONFIG_PATH.parent.exists():
        with open(_DEFAULT_CONFIG_PATH, "w") as f:
            yaml.safe_dump(current_data, f, default_flow_style=False, sort_keys=False)

    return _config
