"""Configuration loader with YAML defaults + environment variable overrides."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

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


class LocalModelsConfig(BaseModel):
    """Local model provider configuration (Ollama, LM Studio, vLLM, etc.)."""
    enabled: bool = False
    base_url: str = "http://localhost:11434/v1"  # Ollama default OpenAI-compatible endpoint
    api_key: str = "local"  # Placeholder; most local servers ignore this
    supports_tool_use: bool = False  # Whether the local model supports native tool calling


class ChatGPTOAuthConfig(BaseModel):
    """ChatGPT OAuth configuration for subscription-based access."""
    enabled: bool = False
    callback_port: int = 18756  # Local port for the OAuth callback server

    @field_validator("callback_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"callback_port must be 1-65535, got {v}")
        return v


class ModelsConfig(BaseModel):
    strong: str = "claude-sonnet-4-5-20250514"
    fast: str = "claude-haiku-4-5-20251001"
    embedding: str = "text-embedding-3-small"
    fallback_chain: list[str] = Field(default_factory=lambda: ["gpt-4o", "claude-sonnet-4-5-20250514"])
    bedrock: BedrockConfig = Field(default_factory=BedrockConfig)
    chatgpt_oauth: ChatGPTOAuthConfig = Field(default_factory=ChatGPTOAuthConfig)
    local: LocalModelsConfig = Field(default_factory=LocalModelsConfig)

    @field_validator("strong", "fast", "embedding")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class BudgetConfig(BaseModel):
    max_tokens_per_task: int = 100_000
    max_tokens_per_session: int = 500_000
    warn_at_percentage: int = 80

    @field_validator("max_tokens_per_task", "max_tokens_per_session")
    @classmethod
    def validate_positive_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Token budget must be positive, got {v}")
        return v

    @field_validator("warn_at_percentage")
    @classmethod
    def validate_percentage(cls, v: int) -> int:
        if not (0 <= v <= 100):
            raise ValueError(f"warn_at_percentage must be 0-100, got {v}")
        return v


class AgentsConfig(BaseModel):
    max_depth: int = 5
    max_parallel: int = 4
    loop_detection_window: int = 5
    max_iterations_per_task: int = 25
    max_tool_result_chars: int = 16_000  # Truncate tool outputs beyond this in agent history
    prune_threshold: int = 40            # Prune older tool results when history exceeds this many messages
    max_loop_iterations: int = 5         # Hard cap on retry/loop iterations in task DAG

    @field_validator("max_depth", "max_parallel", "max_iterations_per_task", "max_loop_iterations")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator("loop_detection_window")
    @classmethod
    def validate_window(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"loop_detection_window must be >= 2, got {v}")
        return v

    @field_validator("prune_threshold")
    @classmethod
    def validate_prune_threshold(cls, v: int) -> int:
        if v < 10:
            raise ValueError(f"prune_threshold must be >= 10, got {v}")
        return v


class MemoryConfig(BaseModel):
    backend: str = "chromadb"
    persist_dir: str = "./data/memory"
    default_namespace: str = "shared"
    decay_rate: float = 0.05
    max_results: int = 10
    similarity_threshold: float = 0.7

    @field_validator("decay_rate")
    @classmethod
    def validate_decay_rate(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"decay_rate must be 0.0-1.0, got {v}")
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"similarity_threshold must be 0.0-1.0, got {v}")
        return v

    @field_validator("max_results")
    @classmethod
    def validate_max_results(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"max_results must be positive, got {v}")
        return v


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

    @field_validator("timeout_seconds", "max_cpu_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Timeout must be positive, got {v}")
        return v

    @field_validator("max_memory_mb")
    @classmethod
    def validate_memory(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_memory_mb must be non-negative, got {v}")
        return v


class SkillsConfig(BaseModel):
    directories: list[str] = Field(default_factory=lambda: ["./skills"])
    auto_discover: bool = True


class ConversationConfig(BaseModel):
    """Settings for conversation context persistence and compression."""
    max_history_turns: int = 50          # Max user+assistant pairs to keep
    compression_enabled: bool = True     # Summarize old context when threshold hit
    compression_threshold: int = 30      # Compress when turns exceed this count

    @field_validator("max_history_turns")
    @classmethod
    def validate_max_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_history_turns must be >= 1, got {v}")
        return v

    @field_validator("compression_threshold")
    @classmethod
    def validate_compression_threshold(cls, v: int) -> int:
        if v < 5:
            raise ValueError(f"compression_threshold must be >= 5, got {v}")
        return v


class LoggingConfig(BaseModel):
    level: str = "INFO"
    trace_file: str = "./data/traces.jsonl"
    rich_console: bool = True

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log level must be one of {valid}, got '{v}'")
        return v.upper()


class PromptEvolutionConfig(BaseModel):
    """Settings for self-modifying prompt evolution."""
    enabled: bool = True                  # Master switch for prompt evolution
    reflect_after_task: bool = True       # Trigger reflection after each task
    max_active_modifications: int = 10    # Cap active modifications per role
    persist_dir: str = "./data/prompt_evolution.db"  # SQLite database path

    @field_validator("max_active_modifications")
    @classmethod
    def validate_max_mods(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_active_modifications must be non-negative, got {v}")
        return v


class MessageBusConfig(BaseModel):
    """Settings for the inter-agent message bus."""
    enabled: bool = True
    history_limit: int = 100              # Max messages per topic

    @field_validator("history_limit")
    @classmethod
    def validate_history_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"history_limit must be >= 1, got {v}")
        return v


class CheckpointConfig(BaseModel):
    """Settings for human-in-the-loop checkpoints."""
    enabled: bool = True
    default_timeout: float = 300.0        # Default timeout in seconds (5 min)
    auto_approve_read_only: bool = True   # Auto-approve read-only operations

    @field_validator("default_timeout")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"default_timeout must be positive, got {v}")
        return v


class LearningConfig(BaseModel):
    """Settings for cross-session learning."""
    enabled: bool = True
    persist_dir: str = "./data/learning.db"


class KnowledgeGraphConfig(BaseModel):
    """Settings for the structured knowledge graph."""
    enabled: bool = True
    persist_path: str = "./data/knowledge_graph.json"


class MCPServerDef(BaseModel):
    """Definition of a single MCP server to connect to."""
    name: str
    command: str | None = None              # Executable for stdio transport
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None                  # URL for SSE transport
    headers: dict[str, str] = Field(default_factory=dict)


class MCPConfig(BaseModel):
    """Settings for MCP (Model Context Protocol) server integration."""
    enabled: bool = False
    servers: list[MCPServerDef] = Field(default_factory=list)


class MultiModalConfig(BaseModel):
    """Settings for multi-modal (image) input."""
    enabled: bool = True
    max_image_size_mb: float = 10.0       # Max image size in MB
    max_images_per_message: int = 5       # Max images per chat message

    @field_validator("max_image_size_mb")
    @classmethod
    def validate_image_size(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"max_image_size_mb must be positive, got {v}")
        return v

    @field_validator("max_images_per_message")
    @classmethod
    def validate_max_images(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_images_per_message must be >= 1, got {v}")
        return v


class Config(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    prompt_evolution: PromptEvolutionConfig = Field(default_factory=PromptEvolutionConfig)
    message_bus: MessageBusConfig = Field(default_factory=MessageBusConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    multimodal: MultiModalConfig = Field(default_factory=MultiModalConfig)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Override config values with CADENCE_<SECTION>_<KEY> env vars.

    Supports JSON-encoded values for complex types:
    e.g., CADENCE_MCP='{"enabled": true, "servers": [...]}'
    """
    import json as _json

    prefix = "CADENCE_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("_", 1)
        if len(parts) == 2:
            section, field = parts
            if section in data and isinstance(data[section], dict):
                existing = data[section].get(field)
                if isinstance(existing, bool):
                    data[section][field] = value.lower() in ("true", "1", "yes")
                elif isinstance(existing, int):
                    data[section][field] = int(value)
                elif isinstance(existing, float):
                    data[section][field] = float(value)
                else:
                    data[section][field] = value
        elif len(parts) == 1:
            section = parts[0]
            # Try JSON parse for whole-section overrides
            if section in data:
                try:
                    data[section] = _json.loads(value)
                except (ValueError, _json.JSONDecodeError):
                    pass
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


def validate_config(updates: dict[str, Any]) -> Config:
    """Validate config updates without applying them (dry-run).

    Returns the would-be Config object if valid, raises ValueError if not.
    """
    current = get_config()
    current_data = current.model_dump()

    for section, values in updates.items():
        if section in current_data and isinstance(values, dict) and isinstance(current_data[section], dict):
            current_data[section].update(values)
        else:
            current_data[section] = values

    return Config(**current_data)  # Raises ValidationError if invalid


def update_config(updates: dict[str, Any]) -> Config:
    """Apply partial updates to the current config and persist to YAML.

    Uses atomic file writes (write to temp file, then rename) to prevent
    data corruption on crash. Validates the entire config before persisting.
    """
    global _config

    current = get_config()
    current_data = current.model_dump()

    # Deep merge updates into current config
    for section, values in updates.items():
        if section in current_data and isinstance(values, dict) and isinstance(current_data[section], dict):
            current_data[section].update(values)
        else:
            current_data[section] = values

    # Validate BEFORE persisting — fail fast on invalid config
    new_config = Config(**current_data)

    # Atomic write: temp file → rename
    if _DEFAULT_CONFIG_PATH.parent.exists():
        try:
            fd, temp_path = tempfile.mkstemp(
                dir=_DEFAULT_CONFIG_PATH.parent,
                suffix=".yaml.tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    yaml.safe_dump(current_data, f, default_flow_style=False, sort_keys=False)
                os.replace(temp_path, str(_DEFAULT_CONFIG_PATH))
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
        except Exception:
            # If atomic write fails, fall back to direct write
            with open(_DEFAULT_CONFIG_PATH, "w") as f:
                yaml.safe_dump(current_data, f, default_flow_style=False, sort_keys=False)

    _config = new_config
    return _config
