# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that the provider finish_reason is propagated into LLMResponse.finish_reason.

Regression coverage for #327: LLMResponse.finish_reason used to be hardcoded to
"stop"/"tool_calls" at every client return site, so the provider's real
finish_reason (notably "length" for output-token exhaustion) never reached
callers and CodeAct's max-tokens abort branch was dead in production.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from litellm.types.utils import ChatCompletionMessageToolCall, Function

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.errors import GenerationError
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import CompletionClient, ResponsesClient, RetryConfig
from nooa.unifiedllm.unifiedllm import (
    _map_completion_finish_reason,
    _map_responses_finish_reason,
)


def make_mock_response(
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[ChatCompletionMessageToolCall] | None = None,
    finish_reason: str | None = None,
) -> litellm.ModelResponse:
    """Create a litellm.ModelResponse for testing with an explicit finish_reason."""
    msg = litellm.Message(
        content=content,
        role="assistant",
        tool_calls=tool_calls,
        reasoning_content=reasoning,
    )
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_calls else "stop"
    choice = litellm.Choices(message=msg, index=0, finish_reason=finish_reason)
    return litellm.ModelResponse(choices=[choice], model="test-model")


def make_tool_call(id: str, name: str, arguments: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=id, function=Function(name=name, arguments=arguments), type="function"
    )


class TestMapCompletionFinishReason:
    """Unit tests for the Chat-Completions finish_reason mapper."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("stop", "stop"),
            ("length", "length"),
            ("tool_calls", "tool_calls"),
            ("content_filter", "content_filter"),
            ("insufficient_system_resource", "insufficient_system_resource"),
            ("error", "error"),
            (None, "unknown"),
            ("something_new", "unknown"),
        ],
    )
    def test_mapping(self, raw, expected):
        resp = SimpleNamespace(choices=[SimpleNamespace(finish_reason=raw)])
        assert _map_completion_finish_reason(resp) == expected

    def test_malformed_response_fails_closed(self):
        assert _map_completion_finish_reason(SimpleNamespace()) == "unknown"
        assert _map_completion_finish_reason(None) == "unknown"

    def test_native_finish_reason_wins_over_litellm_normalization(self):
        choice = litellm.Choices(
            message=litellm.Message(content="partial", role="assistant"),
            index=0,
            finish_reason="insufficient_system_resource",
        )
        assert choice.finish_reason == "stop"
        assert choice.provider_specific_fields == {
            "native_finish_reason": "insufficient_system_resource"
        }
        response = litellm.ModelResponse(choices=[choice], model="test-model")
        assert _map_completion_finish_reason(response) == "insufficient_system_resource"


class TestMapResponsesFinishReason:
    """Unit tests for the Responses-API finish_reason mapper."""

    def test_completed_maps_to_stop(self):
        resp = SimpleNamespace(status="completed", incomplete_details=None)
        assert _map_responses_finish_reason(resp) == "stop"

    def test_incomplete_max_output_tokens_maps_to_length(self):
        resp = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        )
        assert _map_responses_finish_reason(resp) == "length"

    def test_incomplete_dict_details_maps_to_length(self):
        resp = SimpleNamespace(
            status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
        assert _map_responses_finish_reason(resp) == "length"

    def test_incomplete_content_filter_is_explicit(self):
        resp = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="content_filter"),
        )
        assert _map_responses_finish_reason(resp) == "content_filter"

    def test_failed_maps_to_error(self):
        resp = SimpleNamespace(status="failed", incomplete_details=None)
        assert _map_responses_finish_reason(resp) == "error"

    def test_missing_status_fails_closed(self):
        assert _map_responses_finish_reason(SimpleNamespace()) == "unknown"


class TestCompletionClientPropagation:
    """The real CompletionClient return path surfaces the provider finish_reason."""

    @pytest.fixture
    def client(self):
        return CompletionClient(model="test-model")

    def test_sync_length_propagates(self, client):
        resp = make_mock_response(content="partial", finish_reason="length")
        with patch("litellm.completion", return_value=resp):
            out = client.call([{"role": "user", "content": "Hi"}])
        assert out.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_async_length_propagates(self, client):
        resp = make_mock_response(content="partial", finish_reason="length")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
            out = await client.acall([{"role": "user", "content": "Hi"}])
        assert out.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_insufficient_resource_retries_identical_request_without_leaking_partial(self):
        interrupted = make_mock_response(
            content="partial",
            reasoning="discard me",
            tool_calls=[make_tool_call("partial_call", "do_thing", '{"unsafe":true}')],
        )
        interrupted.choices[0].finish_reason = "insufficient_system_resource"
        completed = make_mock_response(content="done", finish_reason="stop")
        client = CompletionClient(
            model="test-model",
            retry_config=RetryConfig(max_retries=1, base_delay=0, jitter_factor=0),
        )
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=[interrupted, completed],
        ) as mock_acompletion:
            out = await client.acall(
                [{"role": "user", "content": "Hi"}],
                tools=[],
            )

        assert out.finish_reason == "stop"
        assert out.content == "done"
        assert out.reasoning is None
        assert out.tool_calls == []
        assert mock_acompletion.call_count == 2
        assert (
            mock_acompletion.call_args_list[0].kwargs == mock_acompletion.call_args_list[1].kwargs
        )

    def test_sync_stop_stays_stop(self, client):
        resp = make_mock_response(content="done", finish_reason="stop")
        with patch("litellm.completion", return_value=resp):
            out = client.call([{"role": "user", "content": "Hi"}])
        assert out.finish_reason == "stop"

    def test_content_filter_is_preserved(self, client):
        resp = make_mock_response(content="", finish_reason="content_filter")
        with patch("litellm.completion", return_value=resp):
            out = client.call([{"role": "user", "content": "Hi"}])
        assert out.finish_reason == "content_filter"

    def test_tool_calls_preserved_even_if_provider_reports_length(self, client):
        # When tool calls are present the client keeps "tool_calls" regardless of
        # the provider's raw finish_reason.
        tc = make_tool_call("call_1", "do_thing", "{}")
        resp = make_mock_response(content=None, tool_calls=[tc], finish_reason="length")
        with patch("litellm.completion", return_value=resp):
            out = client.call([{"role": "user", "content": "Hi"}])
        assert out.finish_reason == "tool_calls"
        assert len(out.tool_calls) == 1


def _make_responses_api_response(status: str, reason: str | None = None):
    """Fake a litellm Responses-API response with a given status/reason."""
    details = SimpleNamespace(reason=reason) if reason is not None else None
    return SimpleNamespace(
        output=[],  # no message/tool items -> empty text, no tool calls
        usage=None,
        status=status,
        incomplete_details=details,
    )


class TestResponsesClientPropagation:
    """The real ResponsesClient return path surfaces the derived finish_reason."""

    @pytest.fixture
    def client(self):
        return ResponsesClient(model="test-model")

    def test_sync_incomplete_max_output_tokens_maps_to_length(self, client):
        resp = _make_responses_api_response("incomplete", "max_output_tokens")
        with patch("litellm.responses", return_value=resp):
            out = client.call([{"role": "user", "content": "Hi"}])
        assert out.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_async_incomplete_max_output_tokens_maps_to_length(self, client):
        resp = _make_responses_api_response("incomplete", "max_output_tokens")
        with patch("litellm.aresponses", new_callable=AsyncMock, return_value=resp):
            out = await client.acall([{"role": "user", "content": "Hi"}])
        assert out.finish_reason == "length"

    def test_sync_completed_maps_to_stop(self, client):
        resp = _make_responses_api_response("completed")
        with patch("litellm.responses", return_value=resp):
            out = client.call([{"role": "user", "content": "Hi"}])
        assert out.finish_reason == "stop"

    def test_sync_failed_maps_to_error(self, client):
        resp = _make_responses_api_response("failed")
        with patch("litellm.responses", return_value=resp):
            out = client.call([{"role": "user", "content": "Hi"}])
        assert out.finish_reason == "error"


class TestCodeActAbortOnRealLengthPath:
    """End-to-end: a real CompletionClient truncation triggers CodeAct's abort."""

    @pytest.mark.asyncio
    async def test_length_from_real_client_triggers_generation_error(self):
        # Empty content + provider finish_reason="length": the reasoning model
        # burned all output tokens. The mapped finish_reason must reach CodeAct's
        # abort branch instead of the generic empty-response retry loop.
        length_response = make_mock_response(content="", finish_reason="length")

        real_llm = CompletionClient(model="test-model")

        class TestAgent(Agent, llm=real_llm):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=3, max_iterations=10)))
            async def my_task(self) -> str:
                """A task."""
                ...

        agent_instance = TestAgent(llm=real_llm)

        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=length_response
        ) as mock_acompletion:
            with pytest.raises(GenerationError, match="max_tokens"):
                await agent_instance.my_task()

        # Abort must be immediate: a single call, no retry loop.
        assert mock_acompletion.call_count == 1
