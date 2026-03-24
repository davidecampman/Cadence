"""LLM interface layer using LiteLLM for model-agnostic access."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import litellm

from sentinel.core.types import Message, Role, ToolCall, ToolDefinition

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True


def _configure_bedrock_env(bedrock_config) -> dict[str, Any]:
    """Set environment variables for LiteLLM's Bedrock integration.

    LiteLLM reads AWS credentials from env vars. This applies the config
    values only if they aren't already set (env vars take precedence).

    Returns a dict of extra kwargs to pass to litellm.acompletion (e.g. api_key).
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

    extra_kwargs: dict[str, Any] = {}
    bedrock_api_key = bedrock_config.api_key or os.environ.get("BEDROCK_API_KEY")
    if bedrock_api_key:
        extra_kwargs["api_key"] = bedrock_api_key
        # LiteLLM's bedrock handler requires boto3 credentials even when using
        # Bearer token auth (api_key). Set placeholder AWS creds so boto3
        # session initialization doesn't fail.
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "bedrock-api-key-auth")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "bedrock-api-key-auth")
    return extra_kwargs


def _configure_bedrock_from_env() -> dict[str, Any]:
    """Build bedrock extra kwargs from environment variables alone.

    Used when a model has a bedrock/ prefix but no explicit bedrock config.
    """
    extra_kwargs: dict[str, Any] = {}
    bedrock_api_key = os.environ.get("BEDROCK_API_KEY")
    if bedrock_api_key:
        extra_kwargs["api_key"] = bedrock_api_key
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "bedrock-api-key-auth")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "bedrock-api-key-auth")
    os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
    return extra_kwargs


def _messages_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert our Message objects to the format LiteLLM expects."""
    result = []
    for msg in messages:
        d: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
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


def _extract_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Fallback: parse tool calls from freetext for models without native tool_use.

    Looks for JSON blocks like:
    ```tool
    {"name": "tool_name", "arguments": {...}}
    ```
    Or inline: <tool>{"name": "...", "arguments": {...}}</tool>
    """
    calls = []

    # Pattern 1: ```tool ... ``` blocks
    for match in re.finditer(r"```tool\s*\n(.*?)\n```", text, re.DOTALL):
        try:
            data = json.loads(match.group(1).strip())
            calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            continue

    # Pattern 2: <tool>...</tool> tags
    for match in re.finditer(r"<tool>(.*?)</tool>", text, re.DOTALL):
        try:
            data = json.loads(match.group(1).strip())
            calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            continue

    return calls


# Models known to support native function calling / tool_use
_NATIVE_TOOL_USE_PREFIXES = (
    "gpt-4", "gpt-3.5", "claude-", "gemini-", "mistral",
    "command-r", "deepseek",
    # Bedrock-hosted models (LiteLLM uses "bedrock/" prefix)
    "bedrock/",
)


def _is_bedrock_model(model: str) -> bool:
    """Check if a model string targets AWS Bedrock."""
    return model.lower().startswith("bedrock/")


# Mapping from standard Anthropic model names to Bedrock model IDs.
# When bedrock is enabled and users specify a bare model name (e.g. "claude-sonnet-4-20250514"),
# we need to convert it to the Bedrock-specific ID (e.g. "anthropic.claude-sonnet-4-20250514-v1:0").
_BEDROCK_MODEL_MAP: dict[str, str] = {
    "claude-sonnet-4-20250514": "anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4-20250514": "anthropic.claude-opus-4-20250514-v1:0",
    "claude-haiku-4-5-20251001": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-3-5-sonnet-20241022": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-haiku-20241022": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0",
    "claude-3-sonnet-20240229": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku-20240307": "anthropic.claude-3-haiku-20240307-v1:0",
}


def _to_bedrock_model(model: str) -> str:
    """Convert a standard model name to a Bedrock-prefixed model ID.

    If the model is already bedrock-prefixed, return as-is.
    If the model name is in our mapping, convert it.
    Otherwise, prepend "bedrock/" and hope litellm recognises it.
    """
    if _is_bedrock_model(model):
        return model
    if model in _BEDROCK_MODEL_MAP:
        return f"bedrock/{_BEDROCK_MODEL_MAP[model]}"
    return f"bedrock/{model}"


def supports_native_tools(model: str) -> bool:
    """Check if a model likely supports native tool_use API."""
    model_lower = model.lower()
    return any(model_lower.startswith(p) for p in _NATIVE_TOOL_USE_PREFIXES)


async def chat_completion(
    model: str,
    messages: list[Message],
    tools: list[ToolDefinition] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    bedrock_config=None,
) -> tuple[str, list[ToolCall], dict[str, Any]]:
    """Call an LLM and return (text_response, tool_calls, usage_metadata).

    Uses native tool_use when the model supports it, otherwise falls back
    to freetext tool parsing.

    For Bedrock models, either pass a bedrock_config or use the ``bedrock/``
    model prefix with AWS credentials configured via environment variables.
    """
    # Apply Bedrock env vars if a config is provided
    bedrock_extra: dict[str, Any] = {}
    if bedrock_config and bedrock_config.enabled:
        model = _to_bedrock_model(model)
        bedrock_extra = _configure_bedrock_env(bedrock_config)
    elif _is_bedrock_model(model):
        # Model is explicitly bedrock-prefixed; configure from env vars / keystore
        bedrock_extra = _configure_bedrock_from_env()

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _messages_to_dicts(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        **bedrock_extra,
    }

    use_native_tools = tools and supports_native_tools(model)
    if use_native_tools:
        kwargs["tools"] = _tools_to_dicts(tools)
        kwargs["tool_choice"] = "auto"

    response = await litellm.acompletion(**kwargs)
    choice = response.choices[0]
    message = choice.message

    text = message.content or ""
    tool_calls: list[ToolCall] = []

    # Extract tool calls from native API response
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
        # Fallback parsing for models without native tool support
        tool_calls = _extract_tool_calls_from_text(text)

    usage = {
        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
        "total_tokens": getattr(response.usage, "total_tokens", 0),
        "model": model,
    }

    return text, tool_calls, usage
