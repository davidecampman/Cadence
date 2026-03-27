"""LLM interface layer using openai + anthropic SDKs directly.

Supports two provider paths:
- **OpenRouter / OpenAI-compatible**: via the ``openai`` SDK
- **AWS Bedrock**: via the ``anthropic`` SDK with Bedrock transport
- **Anthropic direct**: via the ``anthropic`` SDK
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import anthropic
import openai

from cadence.core.types import Message, Role, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bedrock environment / credential helpers
# ---------------------------------------------------------------------------

def _configure_bedrock_env(bedrock_config) -> dict[str, Any]:
    """Apply bedrock_config values to environment variables.

    Returns a dict of extra kwargs for the Anthropic Bedrock client.
    """
    if bedrock_config.region:
        os.environ.setdefault("AWS_REGION_NAME", bedrock_config.region)
    if bedrock_config.profile:
        os.environ.setdefault("AWS_PROFILE", bedrock_config.profile)
    if bedrock_config.role_arn:
        os.environ.setdefault("AWS_ROLE_ARN", bedrock_config.role_arn)
    if bedrock_config.access_key_id:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", bedrock_config.access_key_id)
    if bedrock_config.secret_access_key:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", bedrock_config.secret_access_key)

    extra: dict[str, Any] = {}
    bedrock_api_key = bedrock_config.api_key or os.environ.get("BEDROCK_API_KEY")
    if bedrock_api_key:
        extra["api_key"] = bedrock_api_key
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "bedrock-api-key-auth")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "bedrock-api-key-auth")
    return extra


def _configure_bedrock_from_env() -> dict[str, Any]:
    """Build bedrock extra kwargs from environment variables alone."""
    extra: dict[str, Any] = {}
    bedrock_api_key = os.environ.get("BEDROCK_API_KEY")
    if bedrock_api_key:
        extra["api_key"] = bedrock_api_key
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "bedrock-api-key-auth")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "bedrock-api-key-auth")
    os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
    return extra


# ---------------------------------------------------------------------------
# Message / tool conversion — OpenAI format (used for OpenRouter + OpenAI)
# ---------------------------------------------------------------------------

def _messages_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert Message objects to OpenAI-compatible dicts."""
    result = []
    for msg in messages:
        content = msg.content
        if hasattr(msg, "content_blocks") and msg.content_blocks:
            content = msg.content_blocks

        d: dict[str, Any] = {"role": msg.role.value, "content": content}
        if msg.name:
            d["name"] = msg.name
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        result.append(d)
    return result


def _tools_to_dicts(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert ToolDefinitions to OpenAI-compatible function schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Message / tool conversion — Anthropic format (used for Bedrock + Anthropic)
# ---------------------------------------------------------------------------

def _messages_to_anthropic(
    messages: list[Message],
) -> tuple[str | anthropic.NotGiven, list[dict[str, Any]]]:
    """Convert Message objects to Anthropic SDK format.

    Returns ``(system_prompt, messages)`` — Anthropic takes the system prompt
    as a separate parameter rather than a system message in the list.
    """
    system: str | anthropic.NotGiven = anthropic.NOT_GIVEN
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system = msg.content
            continue

        if msg.role == Role.TOOL:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content,
            }
            # Anthropic expects tool_result blocks inside a user message.
            # Group consecutive tool results into the same user message.
            if (
                result
                and result[-1]["role"] == "user"
                and isinstance(result[-1]["content"], list)
                and result[-1]["content"]
                and result[-1]["content"][-1].get("type") == "tool_result"
            ):
                result[-1]["content"].append(block)
            else:
                result.append({"role": "user", "content": [block]})
            continue

        if msg.role == Role.ASSISTANT and msg.tool_calls:
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            result.append({"role": "assistant", "content": content})
            continue

        # Regular user/assistant message
        content_val: Any = msg.content
        if hasattr(msg, "content_blocks") and msg.content_blocks:
            content_val = msg.content_blocks
        result.append({"role": msg.role.value, "content": content_val})

    return system, result


def _tools_to_anthropic(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert ToolDefinitions to Anthropic SDK tool format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Freetext tool-call fallback (unchanged)
# ---------------------------------------------------------------------------

def _extract_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Fallback: parse tool calls from freetext for models without native tool_use.

    Looks for JSON blocks like:
    ```tool
    {"name": "tool_name", "arguments": {...}}
    ```
    Or inline: <tool>{"name": "...", "arguments": {...}}</tool>
    """
    calls = []

    for match in re.finditer(r"```tool\s*\n(.*?)\n```", text, re.DOTALL):
        try:
            data = json.loads(match.group(1).strip())
            calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            continue

    for match in re.finditer(r"<tool>(.*?)</tool>", text, re.DOTALL):
        try:
            data = json.loads(match.group(1).strip())
            calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            continue

    return calls


# ---------------------------------------------------------------------------
# Model routing helpers
# ---------------------------------------------------------------------------

_MODEL_PROVIDER_ENV: list[tuple[str, str]] = [
    ("claude-", "ANTHROPIC_API_KEY"),
    ("gpt-", "OPENAI_API_KEY"),
    ("gemini-", "GEMINI_API_KEY"),
    ("mistral", "MISTRAL_API_KEY"),
    ("command-r", "COHERE_API_KEY"),
    ("deepseek", "DEEPSEEK_API_KEY"),
]


def _maybe_reroute_model(model: str) -> str:
    """Auto-prefix a model with ``openrouter/`` when the direct provider key
    is missing but an OpenRouter key is available.
    """
    if "/" in model:
        return model

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        return model

    model_lower = model.lower()
    for prefix, env_var in _MODEL_PROVIDER_ENV:
        if model_lower.startswith(prefix):
            if not os.environ.get(env_var):
                return f"openrouter/{model}"
            break
    return model


_NATIVE_TOOL_USE_PREFIXES = (
    "gpt-4", "gpt-3.5", "claude-", "gemini-", "mistral",
    "command-r", "deepseek",
    "openrouter/",
    "bedrock/",
)


def _is_bedrock_model(model: str) -> bool:
    """Check if a model string targets AWS Bedrock."""
    return model.lower().startswith("bedrock/")


_BEDROCK_MODEL_MAP: dict[str, str] = {
    # Claude 4.6 family
    "claude-opus-4-6-20250610": "anthropic.claude-opus-4-6-20250610-v1:0",
    "claude-sonnet-4-6-20250610": "anthropic.claude-sonnet-4-6-20250610-v1:0",
    # Claude 4.5 family
    "claude-sonnet-4-5-20250514": "anthropic.claude-sonnet-4-5-20250514-v1:0",
    "claude-haiku-4-5-20251001": "anthropic.claude-haiku-4-5-20251001-v1:0",
    # Claude 4 family
    "claude-sonnet-4-20250514": "anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4-20250514": "anthropic.claude-opus-4-20250514-v1:0",
    # Claude 3.5 family
    "claude-3-5-sonnet-20241022": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-haiku-20241022": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    # Claude 3 family
    "claude-3-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0",
    "claude-3-sonnet-20240229": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku-20240307": "anthropic.claude-3-haiku-20240307-v1:0",
}


def _region_to_inference_prefix(region: str) -> str:
    """Map an AWS region to the Bedrock cross-region inference profile prefix."""
    if region.startswith("us-"):
        return "us"
    if region.startswith("eu-") or region.startswith("me-"):
        return "eu"
    if region.startswith("ap-"):
        return "ap"
    return "us"


def _to_bedrock_model(model: str, region: str = "us-east-1") -> str:
    """Convert a standard model name to a Bedrock-prefixed model ID.

    Returns a string like ``bedrock/converse/us.anthropic.claude-...-v1:0``.
    The ``bedrock/converse/`` prefix is an internal routing marker stripped
    before making the actual SDK call.
    """
    if _is_bedrock_model(model):
        return model
    prefix = _region_to_inference_prefix(region)
    if model in _BEDROCK_MODEL_MAP:
        return f"bedrock/converse/{prefix}.{_BEDROCK_MODEL_MAP[model]}"
    if model.startswith("claude-"):
        return f"bedrock/converse/{prefix}.anthropic.{model}-v1:0"
    return f"bedrock/converse/{model}"


def _strip_bedrock_prefix(model: str) -> str:
    """Strip the internal ``bedrock/`` or ``bedrock/converse/`` routing prefix
    to get the bare Bedrock model ID for the Anthropic SDK."""
    if model.startswith("bedrock/converse/"):
        return model[len("bedrock/converse/"):]
    if model.startswith("bedrock/"):
        return model[len("bedrock/"):]
    return model


def supports_native_tools(model: str) -> bool:
    """Check if a model likely supports native tool_use API."""
    model_lower = model.lower()
    return any(model_lower.startswith(p) for p in _NATIVE_TOOL_USE_PREFIXES)


def _get_provider(model: str) -> str:
    """Determine which SDK path to use for a given model string."""
    if model.startswith("bedrock/"):
        return "bedrock"
    if model.startswith("claude-"):
        return "anthropic"
    # openrouter/, gpt-*, or anything else → openai-compatible
    return "openai"


# ---------------------------------------------------------------------------
# OpenAI-compatible completion (OpenRouter, OpenAI, etc.)
# ---------------------------------------------------------------------------

async def _openai_completion(
    model: str,
    messages: list[Message],
    tools: list[ToolDefinition] | None,
    temperature: float,
    max_tokens: int,
) -> tuple[str, list[ToolCall], dict[str, Any]]:
    """Call an OpenAI-compatible endpoint and return (text, tool_calls, usage)."""

    # Determine base URL and API key
    if model.startswith("openrouter/"):
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        api_model = model[len("openrouter/"):]
    else:
        # Default: direct OpenAI
        base_url = None  # uses sdk default
        api_key = os.environ.get("OPENAI_API_KEY", "")
        api_model = model

    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    kwargs: dict[str, Any] = {
        "model": api_model,
        "messages": _messages_to_dicts(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    use_native_tools = tools and supports_native_tools(model)
    if use_native_tools:
        kwargs["tools"] = _tools_to_dicts(tools)
        kwargs["tool_choice"] = "auto"

    response = await client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message

    text = message.content or ""
    tool_calls: list[ToolCall] = []

    if use_native_tools and message.tool_calls:
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {"raw": tc.function.arguments}
            tool_calls.append(ToolCall(
                id=tc.id or "",
                name=tc.function.name,
                arguments=args,
            ))
    elif not use_native_tools and text:
        tool_calls = _extract_tool_calls_from_text(text)

    usage = {
        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
        "model": model,
    }

    return text, tool_calls, usage


# ---------------------------------------------------------------------------
# Anthropic completion (direct + Bedrock)
# ---------------------------------------------------------------------------

def _build_anthropic_client(
    provider: str,
    bedrock_config=None,
) -> anthropic.AsyncAnthropic | anthropic.AsyncAnthropicBedrock:
    """Construct the right Anthropic async client for the provider."""
    if provider == "bedrock":
        region = os.environ.get("AWS_REGION_NAME", "us-east-1")
        if bedrock_config and bedrock_config.region:
            region = bedrock_config.region
        return anthropic.AsyncAnthropicBedrock(aws_region=region)
    # Direct Anthropic
    return anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )


async def _anthropic_completion(
    model: str,
    messages: list[Message],
    tools: list[ToolDefinition] | None,
    temperature: float,
    max_tokens: int,
    provider: str,
    bedrock_config=None,
) -> tuple[str, list[ToolCall], dict[str, Any]]:
    """Call the Anthropic API (direct or Bedrock) and return (text, tool_calls, usage)."""

    if provider == "bedrock":
        api_model = _strip_bedrock_prefix(model)
    else:
        api_model = model

    client = _build_anthropic_client(provider, bedrock_config)
    system, api_messages = _messages_to_anthropic(messages)

    kwargs: dict[str, Any] = {
        "model": api_model,
        "messages": api_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if system is not anthropic.NOT_GIVEN:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = _tools_to_anthropic(tools)

    response = await client.messages.create(**kwargs)

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input if isinstance(block.input, dict) else {},
            ))

    text = "\n".join(text_parts) if text_parts else ""

    usage = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        "model": model,
    }

    return text, tool_calls, usage


# ---------------------------------------------------------------------------
# Public API — single entry point
# ---------------------------------------------------------------------------

async def chat_completion(
    model: str,
    messages: list[Message],
    tools: list[ToolDefinition] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    bedrock_config=None,
) -> tuple[str, list[ToolCall], dict[str, Any]]:
    """Call an LLM and return (text_response, tool_calls, usage_metadata).

    Automatically dispatches to the correct SDK based on the model string:
    - ``bedrock/...`` → Anthropic SDK with Bedrock transport
    - ``claude-...`` with ANTHROPIC_API_KEY → Anthropic SDK directly
    - ``openrouter/...`` → OpenAI SDK with OpenRouter base URL
    - anything else → OpenAI SDK (direct OpenAI or compatible)
    """
    # Auto-reroute through OpenRouter when direct provider key is missing
    model = _maybe_reroute_model(model)

    # Apply Bedrock env vars if a config is provided
    if bedrock_config and bedrock_config.enabled:
        model = _to_bedrock_model(model, region=bedrock_config.region)
        _configure_bedrock_env(bedrock_config)
    elif _is_bedrock_model(model):
        _configure_bedrock_from_env()

    provider = _get_provider(model)

    if provider in ("bedrock", "anthropic"):
        return await _anthropic_completion(
            model, messages, tools, temperature, max_tokens,
            provider=provider, bedrock_config=bedrock_config,
        )
    return await _openai_completion(model, messages, tools, temperature, max_tokens)
