"""Tests for the LLM provider layer (cadence.core.llm).

Covers provider dispatch, message conversion, tool formatting, and the
chat_completion entry point with mocked SDK clients.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cadence.core.types import Message, Role, ToolCall, ToolDefinition
from cadence.core.llm import (
    _extract_tool_calls_from_text,
    _get_provider,
    _is_bedrock_model,
    _maybe_reroute_model,
    _messages_to_anthropic,
    _messages_to_dicts,
    _region_to_inference_prefix,
    _strip_bedrock_prefix,
    _to_bedrock_model,
    _tools_to_anthropic,
    _tools_to_dicts,
    chat_completion,
    supports_native_tools,
)


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

class TestGetProvider:
    def test_local_prefix(self):
        assert _get_provider("local/llama3.1:8b") == "local"
        assert _get_provider("local/mistral:7b") == "local"

    def test_bedrock_prefix(self):
        assert _get_provider("bedrock/converse/us.anthropic.claude-v1:0") == "bedrock"
        assert _get_provider("bedrock/anthropic.claude-3-haiku") == "bedrock"

    def test_anthropic_prefix(self):
        assert _get_provider("claude-sonnet-4-5-20250514") == "anthropic"
        assert _get_provider("claude-3-opus-20240229") == "anthropic"

    def test_openai_default(self):
        assert _get_provider("gpt-4o") == "openai"
        assert _get_provider("openrouter/anthropic/claude-3-haiku") == "openai"
        assert _get_provider("mistral-large") == "openai"


class TestMaybeRerouteModel:
    def test_already_prefixed_unchanged(self):
        assert _maybe_reroute_model("openrouter/anthropic/claude-3-haiku") == "openrouter/anthropic/claude-3-haiku"
        assert _maybe_reroute_model("bedrock/converse/model") == "bedrock/converse/model"
        assert _maybe_reroute_model("local/llama3.1:8b") == "local/llama3.1:8b"

    def test_reroute_when_openrouter_key_present(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _maybe_reroute_model("claude-sonnet-4-5-20250514") == "openrouter/claude-sonnet-4-5-20250514"

    def test_no_reroute_when_direct_key_exists(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-key")
        assert _maybe_reroute_model("claude-sonnet-4-5-20250514") == "claude-sonnet-4-5-20250514"

    def test_no_reroute_without_openrouter_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _maybe_reroute_model("claude-sonnet-4-5-20250514") == "claude-sonnet-4-5-20250514"


class TestStripBedrockPrefix:
    def test_converse_prefix(self):
        assert _strip_bedrock_prefix("bedrock/converse/us.anthropic.claude-v1:0") == "us.anthropic.claude-v1:0"

    def test_bedrock_prefix(self):
        assert _strip_bedrock_prefix("bedrock/some-model") == "some-model"

    def test_no_prefix(self):
        assert _strip_bedrock_prefix("claude-sonnet-4-5-20250514") == "claude-sonnet-4-5-20250514"


# ---------------------------------------------------------------------------
# Message conversion — OpenAI format
# ---------------------------------------------------------------------------

class TestMessagesToDicts:
    def test_simple_messages(self):
        msgs = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Hello!"),
        ]
        result = _messages_to_dicts(msgs)
        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "Hi"}
        assert result[2] == {"role": "assistant", "content": "Hello!"}

    def test_tool_call_message(self):
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        msg = Message(role=Role.ASSISTANT, content="", tool_calls=[tc])
        result = _messages_to_dicts([msg])
        assert len(result) == 1
        assert result[0]["tool_calls"][0]["id"] == "tc1"
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"query": "test"}

    def test_tool_result_message(self):
        msg = Message(role=Role.TOOL, content="result data", tool_call_id="tc1")
        result = _messages_to_dicts([msg])
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tc1"

    def test_multimodal_content_blocks(self):
        blocks = [{"type": "text", "text": "desc"}, {"type": "image_url", "image_url": {"url": "data:..."}}]
        msg = Message(role=Role.USER, content="desc", content_blocks=blocks)
        result = _messages_to_dicts([msg])
        assert result[0]["content"] == blocks


class TestToolsToDicts:
    def test_basic_conversion(self):
        tools = [
            ToolDefinition(name="search", description="Search the web", parameters={"type": "object", "properties": {"q": {"type": "string"}}}),
        ]
        result = _tools_to_dicts(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["description"] == "Search the web"


# ---------------------------------------------------------------------------
# Message conversion — Anthropic format
# ---------------------------------------------------------------------------

class TestMessagesToAnthropic:
    def test_system_extracted(self):
        msgs = [
            Message(role=Role.SYSTEM, content="Be concise."),
            Message(role=Role.USER, content="Hi"),
        ]
        system, result = _messages_to_anthropic(msgs)
        assert system == "Be concise."
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hi"}

    def test_no_system(self):
        msgs = [Message(role=Role.USER, content="Hi")]
        system, result = _messages_to_anthropic(msgs)
        import anthropic as _anth
        assert system is _anth.NOT_GIVEN
        assert len(result) == 1

    def test_tool_calls_converted(self):
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        msgs = [
            Message(role=Role.ASSISTANT, content="Let me search.", tool_calls=[tc]),
        ]
        _, result = _messages_to_anthropic(msgs)
        assert result[0]["role"] == "assistant"
        blocks = result[0]["content"]
        assert blocks[0] == {"type": "text", "text": "Let me search."}
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["id"] == "tc1"
        assert blocks[1]["name"] == "search"
        assert blocks[1]["input"] == {"query": "test"}

    def test_tool_results_grouped(self):
        msgs = [
            Message(role=Role.TOOL, content="result1", tool_call_id="tc1"),
            Message(role=Role.TOOL, content="result2", tool_call_id="tc2"),
        ]
        _, result = _messages_to_anthropic(msgs)
        # Both tool results should be grouped into a single user message
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["tool_use_id"] == "tc1"
        assert result[0]["content"][1]["tool_use_id"] == "tc2"

    def test_tool_result_after_user_not_grouped(self):
        msgs = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.TOOL, content="result1", tool_call_id="tc1"),
        ]
        _, result = _messages_to_anthropic(msgs)
        # User message has string content, not tool_result — should NOT be grouped
        assert len(result) == 2


class TestToolsToAnthropic:
    def test_uses_input_schema(self):
        tools = [
            ToolDefinition(name="read", description="Read a file", parameters={"type": "object"}),
        ]
        result = _tools_to_anthropic(tools)
        assert result[0]["name"] == "read"
        assert result[0]["input_schema"] == {"type": "object"}
        assert "parameters" not in result[0]


# ---------------------------------------------------------------------------
# Freetext tool extraction (unchanged, but verify it still works)
# ---------------------------------------------------------------------------

class TestExtractToolCallsFromText:
    def test_code_block_format(self):
        text = '```tool\n{"name": "search", "arguments": {"q": "hello"}}\n```'
        calls = _extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments == {"q": "hello"}

    def test_xml_tag_format(self):
        text = '<tool>{"name": "read", "arguments": {"path": "f.py"}}</tool>'
        calls = _extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "read"

    def test_invalid_json_skipped(self):
        text = '<tool>not json</tool>'
        calls = _extract_tool_calls_from_text(text)
        assert calls == []


# ---------------------------------------------------------------------------
# chat_completion — OpenAI path (mocked)
# ---------------------------------------------------------------------------

def _mock_openai_response(content="Hello!", tool_calls=None, prompt_tokens=10, completion_tokens=5):
    """Build a mock that looks like an OpenAI ChatCompletion response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


class TestChatCompletionOpenAI:
    @pytest.mark.asyncio
    async def test_openrouter_text_response(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        mock_create = AsyncMock(return_value=_mock_openai_response("The answer is 42."))

        with patch("cadence.core.llm.openai.AsyncOpenAI") as MockClient:
            MockClient.return_value.chat.completions.create = mock_create

            text, tool_calls, usage = await chat_completion(
                model="openrouter/openai/gpt-4o",
                messages=[Message(role=Role.USER, content="What is 6*7?")],
            )

        assert text == "The answer is 42."
        assert tool_calls == []
        assert usage["prompt_tokens"] == 10
        assert usage["model"] == "openrouter/openai/gpt-4o"

        # Verify OpenAI client was constructed with OpenRouter base URL
        MockClient.assert_called_once()
        call_kwargs = MockClient.call_args
        assert call_kwargs[1]["base_url"] == "https://openrouter.ai/api/v1"

    @pytest.mark.asyncio
    async def test_openrouter_tool_call_response(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        tc = SimpleNamespace(
            id="call_123",
            function=SimpleNamespace(name="search", arguments='{"query": "test"}'),
        )
        mock_resp = _mock_openai_response(content="", tool_calls=[tc])
        mock_create = AsyncMock(return_value=mock_resp)

        tools = [ToolDefinition(name="search", description="Search", parameters={"type": "object"})]

        with patch("cadence.core.llm.openai.AsyncOpenAI") as MockClient:
            MockClient.return_value.chat.completions.create = mock_create

            text, tool_calls, usage = await chat_completion(
                model="openrouter/openai/gpt-4o",
                messages=[Message(role=Role.USER, content="search for test")],
                tools=tools,
            )

        assert text == ""
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search"
        assert tool_calls[0].arguments == {"query": "test"}
        assert tool_calls[0].id == "call_123"

    @pytest.mark.asyncio
    async def test_reroute_bare_model_to_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        mock_create = AsyncMock(return_value=_mock_openai_response("Done"))

        with patch("cadence.core.llm.openai.AsyncOpenAI") as MockClient:
            MockClient.return_value.chat.completions.create = mock_create

            text, _, _ = await chat_completion(
                model="gpt-4o",
                messages=[Message(role=Role.USER, content="Hi")],
            )

        assert text == "Done"
        # Should have been rerouted — check base_url is OpenRouter
        assert MockClient.call_args[1]["base_url"] == "https://openrouter.ai/api/v1"
        # Model sent to API should NOT have the openrouter/ prefix
        create_kwargs = mock_create.call_args[1]
        assert create_kwargs["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# chat_completion — Anthropic path (mocked)
# ---------------------------------------------------------------------------

def _mock_anthropic_response(text="Hello!", tool_blocks=None, input_tokens=10, output_tokens=5):
    """Build a mock that looks like an Anthropic Messages response."""
    content = []
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    if tool_blocks:
        content.extend(tool_blocks)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=content, usage=usage, model="claude-sonnet-4-5-20250514")


class TestChatCompletionAnthropic:
    @pytest.mark.asyncio
    async def test_direct_anthropic_text(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        mock_create = AsyncMock(return_value=_mock_anthropic_response("I'm Claude."))

        with patch("cadence.core.llm.anthropic.AsyncAnthropic") as MockClient:
            MockClient.return_value.messages.create = mock_create

            text, tool_calls, usage = await chat_completion(
                model="claude-sonnet-4-5-20250514",
                messages=[Message(role=Role.USER, content="Who are you?")],
            )

        assert text == "I'm Claude."
        assert tool_calls == []
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_anthropic_tool_use(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        tool_block = SimpleNamespace(type="tool_use", id="tu_1", name="read_file", input={"path": "main.py"})
        mock_resp = _mock_anthropic_response(text="", tool_blocks=[tool_block])
        mock_create = AsyncMock(return_value=mock_resp)

        tools = [ToolDefinition(name="read_file", description="Read", parameters={"type": "object"})]

        with patch("cadence.core.llm.anthropic.AsyncAnthropic") as MockClient:
            MockClient.return_value.messages.create = mock_create

            text, tool_calls, usage = await chat_completion(
                model="claude-sonnet-4-5-20250514",
                messages=[Message(role=Role.USER, content="Read main.py")],
                tools=tools,
            )

        assert text == ""
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "read_file"
        assert tool_calls[0].arguments == {"path": "main.py"}
        assert tool_calls[0].id == "tu_1"

    @pytest.mark.asyncio
    async def test_anthropic_mixed_text_and_tools(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        tool_block = SimpleNamespace(type="tool_use", id="tu_1", name="search", input={"q": "test"})
        mock_resp = _mock_anthropic_response(text="Let me search.", tool_blocks=[tool_block])
        mock_create = AsyncMock(return_value=mock_resp)

        tools = [ToolDefinition(name="search", description="Search", parameters={"type": "object"})]

        with patch("cadence.core.llm.anthropic.AsyncAnthropic") as MockClient:
            MockClient.return_value.messages.create = mock_create

            text, tool_calls, _ = await chat_completion(
                model="claude-sonnet-4-5-20250514",
                messages=[Message(role=Role.USER, content="search test")],
                tools=tools,
            )

        assert text == "Let me search."
        assert len(tool_calls) == 1


# ---------------------------------------------------------------------------
# chat_completion — Bedrock path (mocked)
# ---------------------------------------------------------------------------

class TestChatCompletionBedrock:
    @pytest.mark.asyncio
    async def test_bedrock_explicit_prefix(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        mock_create = AsyncMock(return_value=_mock_anthropic_response("Bedrock says hi."))

        with patch("cadence.core.llm.anthropic.AsyncAnthropicBedrock") as MockClient:
            MockClient.return_value.messages.create = mock_create

            text, _, usage = await chat_completion(
                model="bedrock/converse/us.anthropic.claude-sonnet-4-5-20250514-v1:0",
                messages=[Message(role=Role.USER, content="Hi")],
            )

        assert text == "Bedrock says hi."
        # Verify Bedrock client was used
        MockClient.assert_called_once()
        # Verify the model sent to the API has the prefix stripped
        create_kwargs = mock_create.call_args[1]
        assert create_kwargs["model"] == "us.anthropic.claude-sonnet-4-5-20250514-v1:0"

    @pytest.mark.asyncio
    async def test_bedrock_via_config(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        mock_create = AsyncMock(return_value=_mock_anthropic_response("Hello from bedrock"))

        bedrock_cfg = SimpleNamespace(
            enabled=True,
            region="eu-west-1",
            profile=None,
            role_arn=None,
            access_key_id=None,
            secret_access_key=None,
            api_key=None,
        )

        with patch("cadence.core.llm.anthropic.AsyncAnthropicBedrock") as MockClient:
            MockClient.return_value.messages.create = mock_create

            text, _, _ = await chat_completion(
                model="claude-sonnet-4-5-20250514",
                messages=[Message(role=Role.USER, content="Hi")],
                bedrock_config=bedrock_cfg,
            )

        assert text == "Hello from bedrock"
        # Bedrock client should be constructed with the right region
        assert MockClient.call_args[1]["aws_region"] == "eu-west-1"

    @pytest.mark.asyncio
    async def test_bedrock_system_message_handling(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        mock_create = AsyncMock(return_value=_mock_anthropic_response("Done"))

        with patch("cadence.core.llm.anthropic.AsyncAnthropicBedrock") as MockClient:
            MockClient.return_value.messages.create = mock_create

            await chat_completion(
                model="bedrock/converse/us.anthropic.claude-3-haiku-20240307-v1:0",
                messages=[
                    Message(role=Role.SYSTEM, content="Be brief."),
                    Message(role=Role.USER, content="Hi"),
                ],
            )

        create_kwargs = mock_create.call_args[1]
        assert create_kwargs["system"] == "Be brief."
        # System message should NOT appear in the messages list
        assert all(m["role"] != "system" for m in create_kwargs["messages"])


# ---------------------------------------------------------------------------
# Local model provider
# ---------------------------------------------------------------------------

class TestLocalModelProvider:
    """Tests for local model (Ollama, LM Studio, vLLM, etc.) support."""

    def test_local_model_not_rerouted(self, monkeypatch):
        """local/ prefix should never be rerouted to OpenRouter."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        assert _maybe_reroute_model("local/llama3.1:8b") == "local/llama3.1:8b"

    def test_local_model_provider_detection(self):
        assert _get_provider("local/llama3.1:8b") == "local"
        assert _get_provider("local/codellama:13b") == "local"
        assert _get_provider("local/mistral:7b-instruct") == "local"

    def test_local_model_no_native_tools_by_default(self):
        """Local models should not claim native tool support without config."""
        assert not supports_native_tools("local/llama3.1:8b")

    def test_local_model_native_tools_with_config(self):
        """Local models with supports_tool_use=True should report native tool support."""
        from cadence.core.config import LocalModelsConfig
        config = LocalModelsConfig(enabled=True, supports_tool_use=True)
        assert supports_native_tools("local/llama3.1:8b", local_config=config)

    @pytest.mark.asyncio
    async def test_local_model_completion(self):
        """Local model calls should use the configured base_url and strip the local/ prefix."""
        from cadence.core.config import LocalModelsConfig

        local_config = LocalModelsConfig(
            enabled=True,
            base_url="http://localhost:11434/v1",
            api_key="local",
        )

        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Hello from local!", tool_calls=None),
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_create = AsyncMock(return_value=mock_response)

        with patch("cadence.core.llm.openai.AsyncOpenAI") as MockClient:
            MockClient.return_value.chat.completions.create = mock_create

            text, tool_calls, usage = await chat_completion(
                model="local/llama3.1:8b",
                messages=[Message(role=Role.USER, content="Hi")],
                local_config=local_config,
            )

        # Verify the client was created with the local base_url
        MockClient.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="local",
        )

        # Verify the model name had the local/ prefix stripped
        create_kwargs = mock_create.call_args[1]
        assert create_kwargs["model"] == "llama3.1:8b"

        assert text == "Hello from local!"
        assert tool_calls == []
        assert usage["total_tokens"] == 15
