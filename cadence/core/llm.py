"""LLM interface layer using openai + anthropic SDKs directly.

Supports two provider paths:
- **OpenRouter / OpenAI-compatible**: via the ``openai`` SDK
- **AWS Bedrock**: via the ``anthropic`` SDK with Bedrock transport
- **Anthropic direct**: via the ``anthropic`` SDK
- **ChatGPT OAuth**: via the ``openai`` SDK with OAuth access tokens
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
    if "/" in model:  # Already prefixed (openrouter/, local/, bedrock/, etc.)
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


def supports_native_tools(model: str, local_config=None) -> bool:
    """Check if a model likely supports native tool_use API."""
    if model.startswith("local/"):
        return local_config.supports_tool_use if local_config else False
    model_lower = model.lower()
    return any(model_lower.startswith(p) for p in _NATIVE_TOOL_USE_PREFIXES)


def _get_provider(model: str) -> str:
    """Determine which SDK path to use for a given model string."""
    if model.startswith("local/"):
        return "local"
    if model.startswith("bedrock/"):
        return "bedrock"
    if model.startswith("claude-"):
        return "anthropic"
    # openrouter/, gpt-*, or anything else → openai-compatible
    return "openai"


# ---------------------------------------------------------------------------
# ChatGPT OAuth / Codex helpers
# ---------------------------------------------------------------------------

async def _get_chatgpt_oauth_token() -> str | None:
    """Return a valid ChatGPT OAuth access token if configured, else None.

    Handles automatic token refresh when the stored token is expired.
    """
    try:
        from cadence.core.chatgpt_oauth import get_access_token
        return await get_access_token()
    except Exception as e:
        logger.debug("ChatGPT OAuth token retrieval failed: %s", e)
        return None


def _messages_to_responses_api(
    messages: list[Message],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert Message objects to the OpenAI Responses API format.

    The Responses API uses ``instructions`` for the system prompt and
    ``input`` for the conversation — different from Chat Completions'
    ``messages`` array.

    Returns ``(instructions, input_items)``.
    """
    instructions = ""
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            instructions = msg.content
            continue

        if msg.role == Role.USER:
            input_items.append({
                "type": "message",
                "role": "user",
                "content": msg.content,
            })
        elif msg.role == Role.ASSISTANT:
            if msg.tool_calls:
                item: dict[str, Any] = {
                    "type": "message",
                    "role": "assistant",
                    "content": msg.content or "",
                }
                input_items.append(item)
                for tc in msg.tool_calls:
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    })
            else:
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": msg.content or "",
                })
        elif msg.role == Role.TOOL:
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.tool_call_id or "",
                "output": msg.content,
            })

    return instructions, input_items


def _tools_to_responses_api(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert ToolDefinitions to Responses API function tool format."""
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tools
    ]


class CodexQuotaExhaustedError(Exception):
    """Raised when the Codex subscription quota is exhausted."""


async def _codex_oauth_completion(
    model: str,
    messages: list[Message],
    tools: list[ToolDefinition] | None,
    temperature: float,
    max_tokens: int,
    oauth_token: str,
) -> tuple[str, list[ToolCall], dict[str, Any]]:
    """Call the Codex Responses API using a ChatGPT OAuth token.

    Uses ``https://chatgpt.com/backend-api/codex/responses`` — the
    subscription-based endpoint — with the Responses API request format
    (``input`` + ``instructions``), NOT Chat Completions.

    Raises ``CodexQuotaExhaustedError`` if the subscription quota is
    exhausted so the caller can fall back to the regular API key path.
    """
    import httpx
    from cadence.core.chatgpt_oauth import CODEX_API_BASE_URL, CODEX_RESPONSES_PATH

    url = f"{CODEX_API_BASE_URL}{CODEX_RESPONSES_PATH}"
    instructions, input_items = _messages_to_responses_api(messages)

    payload: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "store": False,
        "stream": True,
    }
    payload["instructions"] = instructions or "You are a helpful assistant."
    if tools:
        payload["tools"] = _tools_to_responses_api(tools)

    from cadence.core.chatgpt_oauth import get_chatgpt_account_id
    chatgpt_acct = get_chatgpt_account_id()

    logger.info(
        "Codex request: model=%s, account_id=%s, input_items=%d, url=%s",
        model, chatgpt_acct or "(none)", len(input_items), url,
    )

    headers = {
        "Authorization": f"Bearer {oauth_token}",
        "Content-Type": "application/json",
    }
    if chatgpt_acct:
        headers["ChatGPT-Account-Id"] = chatgpt_acct

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=payload, headers=headers)

        # Detect quota exhaustion or unsupported model (429, 400, 401, 403)
        if resp.status_code == 429:
            logger.warning("Codex quota exhausted (429). Will fall back to API key.")
            raise CodexQuotaExhaustedError("Codex subscription quota exhausted.")
        if resp.status_code == 400:
            body_text = resp.text[:500]
            logger.warning("Codex returned 400 for model %s: %s", model, body_text)
            raise CodexQuotaExhaustedError(f"Codex 400 for model {model}: {body_text}")
        if resp.status_code in (401, 403):
            body_text = resp.text
            if "quota" in body_text.lower() or "rate" in body_text.lower():
                logger.warning("Codex quota/auth error (%d). Will fall back.", resp.status_code)
                raise CodexQuotaExhaustedError(f"Codex auth error ({resp.status_code}).")
            resp.raise_for_status()

        resp.raise_for_status()

    # Parse SSE stream — collect the final response.completed event
    # which contains the full output and usage data.
    data: dict[str, Any] = {}
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for line in resp.text.split("\n"):
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        # The response.completed event has the full output
        if event_type == "response.completed":
            data = event.get("response", {})
            break

        # Accumulate text deltas as they arrive
        if event_type == "response.output_text.delta":
            delta = event.get("delta", "")
            if delta:
                text_parts.append(delta)

    # Parse output from the completed response (if we got one)
    if data.get("output"):
        text_parts = []  # Reset — use the final output instead of deltas
        for item in data["output"]:
            item_type = item.get("type", "")
            if item_type == "message":
                for content_block in item.get("content", []):
                    if content_block.get("type") in ("output_text", "text"):
                        text_parts.append(content_block.get("text", ""))
            elif item_type == "function_call":
                try:
                    args = json.loads(item.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {"raw": item.get("arguments", "")}
                tool_calls.append(ToolCall(
                    id=item.get("call_id", item.get("id", "")),
                    name=item.get("name", ""),
                    arguments=args,
                ))

    text = "\n".join(text_parts) if text_parts else ""

    resp_usage = data.get("usage", {})
    usage = {
        "prompt_tokens": resp_usage.get("input_tokens", 0),
        "completion_tokens": resp_usage.get("output_tokens", 0),
        "total_tokens": resp_usage.get("total_tokens", 0),
        "model": model,
        "via": "codex-oauth",
    }

    return text, tool_calls, usage


class ChatGPTConversationError(Exception):
    """Raised when the ChatGPT conversation endpoint fails."""


async def _chatgpt_conversation_completion(
    model: str,
    messages: list[Message],
    tools: list[ToolDefinition] | None,
    temperature: float,
    max_tokens: int,
    oauth_token: str,
) -> tuple[str, list[ToolCall], dict[str, Any]]:
    """Call the ChatGPT conversation endpoint (same as the web/desktop client).

    Uses ``https://chatgpt.com/backend-api/conversation`` — the same endpoint
    that powers the ChatGPT UI.  This has a **separate quota pool** from Codex,
    so it can still work when the Codex quota is exhausted.

    The request format uses ``action``, ``messages`` with ``parts``, and
    ``parent_message_id``.  The response is an EventStream (SSE).

    Raises ``ChatGPTConversationError`` on non-quota failures so the caller
    can fall back further.
    """
    import uuid
    from cadence.core.chatgpt_oauth import CODEX_API_BASE_URL

    url = f"{CODEX_API_BASE_URL}/conversation"

    # Build messages in the ChatGPT backend-api format
    system_message = ""
    conv_messages = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system_message = msg.content
            continue

        msg_id = str(uuid.uuid4())

        if msg.role == Role.USER:
            conv_messages.append({
                "id": msg_id,
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [msg.content]},
                "metadata": {},
            })
        elif msg.role == Role.ASSISTANT:
            conv_messages.append({
                "id": msg_id,
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [msg.content or ""]},
                "metadata": {},
            })

    # Use the last message only (ChatGPT conversation API expects
    # the new user message; prior context is in conversation_id/parent)
    # For a stateless call, we send all messages as new input.
    parent_id = str(uuid.uuid4())

    payload: dict[str, Any] = {
        "action": "next",
        "messages": conv_messages if conv_messages else [{
            "id": str(uuid.uuid4()),
            "author": {"role": "user"},
            "content": {"content_type": "text", "parts": [""]},
            "metadata": {},
        }],
        "model": model,
        "parent_message_id": parent_id,
        "timezone_offset_min": 0,
        "history_and_training_disabled": True,
        "conversation_mode": {"kind": "primary_assistant"},
        "force_paragen": False,
        "force_paragen_model_slug": "",
        "force_rate_limit": False,
    }

    # Inject system message if present
    if system_message:
        system_id = str(uuid.uuid4())
        payload["messages"].insert(0, {
            "id": system_id,
            "author": {"role": "system"},
            "content": {"content_type": "text", "parts": [system_message]},
            "metadata": {},
        })

    from cadence.core.chatgpt_oauth import get_chatgpt_account_id
    chatgpt_acct = get_chatgpt_account_id()

    # The backend-api/conversation endpoint has Cloudflare bot protection.
    # We need browser-like headers to get past it — matching what the
    # ChatGPT desktop/web client sends.
    device_id = str(uuid.uuid4())
    headers = {
        "Authorization": f"Bearer {oauth_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "oai-device-id": device_id,
        "oai-language": "en-US",
        "Referer": "https://chatgpt.com/",
        "Origin": "https://chatgpt.com",
    }
    if chatgpt_acct:
        headers["ChatGPT-Account-Id"] = chatgpt_acct

    logger.info(
        "ChatGPT conversation: model=%s, account_id=%s, messages=%d",
        model, chatgpt_acct or "(none)", len(conv_messages),
    )

    # Use curl_cffi to impersonate a browser TLS fingerprint — the
    # /backend-api/conversation endpoint is behind Cloudflare bot protection
    # that blocks Python HTTP clients based on their TLS handshake.
    try:
        from curl_cffi.requests import AsyncSession as CurlAsyncSession
    except ImportError:
        raise ChatGPTConversationError(
            "curl_cffi is required for the ChatGPT conversation endpoint. "
            "Install it with: pip install curl_cffi"
        )

    final_text = ""
    final_message_id = ""

    try:
        async with CurlAsyncSession(impersonate="chrome") as session:
            resp = await session.post(
                url, json=payload, headers=headers, timeout=120,
            )

            if resp.status_code >= 400:
                body_text = resp.text[:500]
                logger.warning(
                    "ChatGPT conversation endpoint returned %d: %s",
                    resp.status_code, body_text,
                )
                raise ChatGPTConversationError(
                    f"ChatGPT conversation error ({resp.status_code}): {body_text}"
                )

            # Parse SSE lines from the response body
            for line in resp.text.split("\n"):
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                msg_data = data.get("message")
                if not msg_data:
                    continue

                author = msg_data.get("author", {})
                if author.get("role") != "assistant":
                    continue

                content = msg_data.get("content", {})
                if content.get("content_type") == "text":
                    parts = content.get("parts", [])
                    if parts:
                        final_text = parts[0] if isinstance(parts[0], str) else str(parts[0])
                final_message_id = msg_data.get("id", "")

    except ChatGPTConversationError:
        raise
    except Exception as e:
        raise ChatGPTConversationError(f"curl_cffi error: {e}")

    if not final_text and not final_message_id:
        raise ChatGPTConversationError("No response received from ChatGPT conversation endpoint.")

    # The conversation endpoint doesn't return structured tool calls —
    # fall back to freetext extraction if the model output contains them
    tool_calls: list[ToolCall] = []
    if tools and final_text:
        tool_calls = _extract_tool_calls_from_text(final_text)

    usage = {
        "prompt_tokens": 0,  # Not provided by this endpoint
        "completion_tokens": 0,
        "total_tokens": 0,
        "model": model,
        "via": "chatgpt-conversation",
    }

    return final_text, tool_calls, usage


# ---------------------------------------------------------------------------
# OpenAI-compatible completion (OpenRouter, OpenAI, etc.)
# ---------------------------------------------------------------------------

async def _openai_completion(
    model: str,
    messages: list[Message],
    tools: list[ToolDefinition] | None,
    temperature: float,
    max_tokens: int,
    local_config=None,
) -> tuple[str, list[ToolCall], dict[str, Any]]:
    """Call an OpenAI-compatible endpoint and return (text, tool_calls, usage)."""

    # Determine base URL and API key
    if model.startswith("local/"):
        base_url = local_config.base_url if local_config else "http://localhost:11434/v1"
        api_key = local_config.api_key if local_config else "local"
        api_model = model[len("local/"):]
    elif model.startswith("openrouter/"):
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        api_model = model[len("openrouter/"):]
    else:
        # Default: direct OpenAI (API key only — Codex OAuth is handled separately)
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

    use_native_tools = tools and supports_native_tools(model, local_config=local_config)
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

def _is_openai_model(model: str) -> bool:
    """Return True if the model targets OpenAI directly (not local, bedrock, etc.)."""
    return not any(model.startswith(p) for p in ("local/", "bedrock/", "openrouter/", "claude-"))


async def chat_completion(
    model: str,
    messages: list[Message],
    tools: list[ToolDefinition] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    bedrock_config=None,
    local_config=None,
) -> tuple[str, list[ToolCall], dict[str, Any]]:
    """Call an LLM and return (text_response, tool_calls, usage_metadata).

    Automatically dispatches to the correct SDK based on the model string:
    - ``local/...`` → OpenAI SDK with local base URL (Ollama, LM Studio, vLLM, etc.)
    - ``bedrock/...`` → Anthropic SDK with Bedrock transport
    - ``claude-...`` with ANTHROPIC_API_KEY → Anthropic SDK directly
    - ``openrouter/...`` → OpenAI SDK with OpenRouter base URL
    - anything else → OpenAI SDK (direct OpenAI or compatible)

    For OpenAI models with ChatGPT OAuth configured, the fallback chain is:
    1. ChatGPT Conversation (``chatgpt.com/backend-api/conversation``) — uses
       the same endpoint as the ChatGPT web/desktop app (separate quota pool)
    2. Codex Responses API (``chatgpt.com/backend-api/codex/responses``)
    3. OpenAI API key (``api.openai.com/v1/chat/completions``) — per-token billing
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

    # For OpenAI models: three-tier fallback with ChatGPT OAuth
    # 1) ChatGPT Conversation → 2) Codex Responses API → 3) API key
    if _is_openai_model(model):
        oauth_token = await _get_chatgpt_oauth_token()
        if oauth_token:
            # --- Tier 1: ChatGPT Conversation endpoint ---
            try:
                result = await _chatgpt_conversation_completion(
                    model, messages, tools, temperature, max_tokens,
                    oauth_token=oauth_token,
                )
                logger.info("Request served via ChatGPT conversation endpoint (tier 1).")
                return result
            except ChatGPTConversationError as e:
                logger.info(
                    "ChatGPT conversation endpoint failed (%s), trying Codex...", e,
                )

            # --- Tier 2: Codex Responses API ---
            try:
                result = await _codex_oauth_completion(
                    model, messages, tools, temperature, max_tokens,
                    oauth_token=oauth_token,
                )
                logger.info("Request served via Codex OAuth endpoint (tier 2).")
                return result
            except CodexQuotaExhaustedError:
                logger.info(
                    "Codex quota also exhausted, trying API key for model %s...",
                    model,
                )

            # --- Tier 3: fall through to API key ---
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "Both ChatGPT conversation and Codex quotas are exhausted, "
                    "and no OPENAI_API_KEY is configured as a fallback. Either "
                    "wait for your quota to reset or add an OpenAI API key in "
                    "Config > Providers."
                )
            logger.info("Falling back to OpenAI API key for model %s (tier 3).", model)

    return await _openai_completion(
        model, messages, tools, temperature, max_tokens,
        local_config=local_config,
    )
