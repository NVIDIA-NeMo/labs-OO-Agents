# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for empty content retry and reasoning fallback in CompletionClient."""

from unittest.mock import AsyncMock, patch

import litellm
import pytest
from litellm.types.utils import ChatCompletionMessageToolCall, Function
from pydantic import BaseModel

from nooa.unifiedllm import CompletionClient, EmptyContentError, RetryConfig


def make_mock_response(
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[ChatCompletionMessageToolCall] | None = None,
) -> litellm.ModelResponse:
    """Create a litellm.ModelResponse for testing."""
    msg = litellm.Message(
        content=content,
        role="assistant",
        tool_calls=tool_calls,
        reasoning_content=reasoning,
    )
    finish_reason = "tool_calls" if tool_calls else "stop"
    choice = litellm.Choices(message=msg, index=0, finish_reason=finish_reason)
    return litellm.ModelResponse(choices=[choice], model="test-model")


def make_tool_call(id: str, name: str, arguments: str) -> ChatCompletionMessageToolCall:
    """Create a litellm tool call for testing."""
    return ChatCompletionMessageToolCall(
        id=id, function=Function(name=name, arguments=arguments), type="function"
    )


class TestEmptyContentError:
    """Tests for EmptyContentError exception."""

    def test_error_with_reasoning(self):
        """Test exception stores reasoning."""
        error = EmptyContentError("This is the reasoning")
        assert error.reasoning == "This is the reasoning"
        assert "This is the reasoning" in str(error)

    def test_error_without_reasoning(self):
        """Test exception without reasoning."""
        error = EmptyContentError(None)
        assert error.reasoning is None
        assert "Empty content" in str(error)


class TestRetryConfigEmptyContent:
    """Tests for retry_on_empty_content flag in RetryConfig."""

    def test_default_disabled(self):
        """Test retry_on_empty_content is disabled by default."""
        config = RetryConfig()
        assert config.retry_on_empty_content is False

    def test_enable_empty_content_retry(self):
        """Test enabling retry_on_empty_content."""
        config = RetryConfig(retry_on_empty_content=True)
        assert config.retry_on_empty_content is True


class TestCompletionClientEmptyContentRetry:
    """Tests for empty content retry in CompletionClient."""

    @pytest.fixture
    def client_with_retry(self):
        """Create client with empty content retry enabled."""
        return CompletionClient(
            model="test-model",
            retry_config=RetryConfig(max_retries=2, base_delay=0.01, retry_on_empty_content=True),
        )

    @pytest.fixture
    def client_default_retry(self):
        """Create client with default endpoint retry config."""
        return CompletionClient(model="test-model")

    @pytest.fixture
    def client_retry_disabled(self):
        """Create client with retry config but empty content retry disabled."""
        return CompletionClient(
            model="test-model",
            retry_config=RetryConfig(max_retries=2, base_delay=0.01, retry_on_empty_content=False),
        )

    @pytest.mark.asyncio
    async def test_successful_response_no_retry(self, client_with_retry):
        """Test that successful responses don't trigger retry."""
        mock_response = make_mock_response(content="Hello, world!")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            response = await client_with_retry.acall([{"role": "user", "content": "Hi"}])

            assert response.content == "Hello, world!"
            assert mock_acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_content_without_reasoning_no_retry(self, client_with_retry):
        """Test that empty content without reasoning doesn't retry."""
        mock_response = make_mock_response(content="")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            response = await client_with_retry.acall([{"role": "user", "content": "Hi"}])

            assert response.content == ""
            assert mock_acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_content_with_reasoning_triggers_retry(self, client_with_retry):
        """Test that empty content with reasoning triggers retry."""
        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return make_mock_response(content="", reasoning="I'm thinking...")
            return make_mock_response(content="Final answer")

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            response = await client_with_retry.acall([{"role": "user", "content": "Hi"}])

            assert response.content == "Final answer"
            assert call_count == 3  # max_retries=2 means 3 total attempts

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_error(self, client_with_retry):
        """Test that exhausted retries raise EmptyContentError."""
        mock_response = make_mock_response(content="", reasoning="Still thinking...")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            with pytest.raises(EmptyContentError) as exc_info:
                await client_with_retry.acall([{"role": "user", "content": "Hi"}])

            assert exc_info.value.reasoning == "Still thinking..."
            assert mock_acompletion.call_count == 3  # 1 + 2 retries

    @pytest.mark.asyncio
    async def test_no_retry_when_disabled(self, client_retry_disabled):
        """Test that retry is disabled when retry_on_empty_content=False."""
        mock_response = make_mock_response(content="", reasoning="I'm thinking...")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            response = await client_retry_disabled.acall([{"role": "user", "content": "Hi"}])

            assert response.content == ""
            assert mock_acompletion.call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_default_config_does_not_retry_empty_content(self, client_default_retry):
        """Default endpoint retries do not retry empty content unless explicitly enabled."""
        mock_response = make_mock_response(content="", reasoning="I'm thinking...")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            response = await client_default_retry.acall([{"role": "user", "content": "Hi"}])

            assert response.content == ""
            assert mock_acompletion.call_count == 1  # Empty-content retry remains opt-in

    def test_completion_client_default_retries_bad_gateway_sync(self):
        """CompletionClient.call retries endpoint errors by default."""
        client = CompletionClient(model="test-model")
        with (
            patch("nooa.unifiedllm.retry.time.sleep"),
            patch(
                "litellm.completion",
                side_effect=[Exception("status 502 bad gateway"), make_mock_response(content="ok")],
            ) as mock_completion,
        ):
            response = client.call([{"role": "user", "content": "Hi"}])

        assert response.content == "ok"
        assert mock_completion.call_count == 2

    def test_completion_client_zero_retry_config_disables_sync_retries(self):
        """RetryConfig(max_retries=0, rate_limit_extra_retries=0) opts out for call()."""
        client = CompletionClient(
            model="test-model", retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0)
        )
        with patch(
            "litellm.completion", side_effect=Exception("status 502 bad gateway")
        ) as mock_completion:
            with pytest.raises(Exception, match="status 502"):
                client.call([{"role": "user", "content": "Hi"}])

        assert mock_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_completion_client_default_retries_bad_gateway_async(self):
        """CompletionClient.acall retries endpoint errors by default."""
        client = CompletionClient(model="test-model")
        sleep = AsyncMock()
        mock_acompletion = AsyncMock(
            side_effect=[Exception("status 502 bad gateway"), make_mock_response(content="ok")]
        )
        with (
            patch("nooa.unifiedllm.retry.asyncio.sleep", sleep),
            patch("litellm.acompletion", mock_acompletion),
        ):
            response = await client.acall([{"role": "user", "content": "Hi"}])

        assert response.content == "ok"
        assert mock_acompletion.call_count == 2
        sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_completion_client_zero_retry_config_disables_async_retries(self):
        """RetryConfig(max_retries=0, rate_limit_extra_retries=0) opts out for acall()."""
        client = CompletionClient(
            model="test-model", retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0)
        )
        mock_acompletion = AsyncMock(side_effect=Exception("status 502 bad gateway"))
        with patch("litellm.acompletion", mock_acompletion):
            with pytest.raises(Exception, match="status 502"):
                await client.acall([{"role": "user", "content": "Hi"}])

        assert mock_acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_calls_bypass_retry(self, client_with_retry):
        """Test that tool call responses don't trigger retry even with empty content."""
        mock_tool_call = make_tool_call(
            id="call_123", name="test_function", arguments='{"arg": "value"}'
        )
        mock_response = make_mock_response(content="", tool_calls=[mock_tool_call])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            response = await client_with_retry.acall([{"role": "user", "content": "Hi"}])

            assert response.finish_reason == "tool_calls"
            assert len(response.tool_calls) == 1
            assert mock_acompletion.call_count == 1  # No retry for tool calls

    @pytest.mark.asyncio
    async def test_tool_calls_with_none_content_stores_empty_string_in_assistant_message(
        self, client_with_retry
    ):
        """When API returns tool_calls with message.content None, assistant_message['content'] must be ''."""
        mock_tool_call = make_tool_call(
            id="call_123", name="test_function", arguments='{"arg": "value"}'
        )
        # Simulate API returning None for content on tool-call-only response
        mock_response = make_mock_response(content=None, tool_calls=[mock_tool_call])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            response = await client_with_retry.acall([{"role": "user", "content": "Hi"}])

            assert response.finish_reason == "tool_calls"
            assert response.assistant_message["content"] == ""
            assert response.content == ""


class TestCompletionClientSyncEmptyContentRetry:
    """Tests for sync empty content retry in CompletionClient.call()."""

    @pytest.fixture
    def client_with_retry(self):
        """Create client with empty content retry enabled."""
        return CompletionClient(
            model="test-model",
            retry_config=RetryConfig(max_retries=2, base_delay=0.01, retry_on_empty_content=True),
        )

    def test_sync_empty_content_with_reasoning_triggers_retry(self, client_with_retry):
        """Test that sync call retries on empty content with reasoning."""
        call_count = 0

        def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return make_mock_response(content="", reasoning="Thinking...")
            return make_mock_response(content="Done!")

        with patch("litellm.completion", side_effect=mock_completion):
            response = client_with_retry.call([{"role": "user", "content": "Hi"}])

            assert response.content == "Done!"
            assert call_count == 3


class SentimentResponse(BaseModel):
    value: str


class TestOutputModelReasoningFallback:
    """Tests for structured output fallback to reasoning_content.

    Some models (e.g. Nemotron) put structured output JSON in reasoning_content
    instead of content. The client should fall back to parsing reasoning when
    content is empty and output_model is set.
    """

    @pytest.fixture
    def client(self):
        return CompletionClient(model="test-model")

    @pytest.mark.asyncio
    async def test_async_output_model_falls_back_to_reasoning(self, client):
        """When content is empty but reasoning has JSON, parse reasoning."""
        mock_response = make_mock_response(content="", reasoning='{"value": "positive"}')

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            response = await client.acall(
                [{"role": "user", "content": "classify"}],
                output_model=SentimentResponse,
            )

            assert isinstance(response.content, SentimentResponse)
            assert response.content.value == "positive"
            # When reasoning was consumed as content, it should be cleared
            assert response.reasoning is None
            assert mock_acompletion.call_count == 1

    def test_sync_output_model_falls_back_to_reasoning(self, client):
        """Sync path: when content is empty but reasoning has JSON, parse reasoning."""
        mock_response = make_mock_response(content="", reasoning='{"value": "negative"}')

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = mock_response

            response = client.call(
                [{"role": "user", "content": "classify"}],
                output_model=SentimentResponse,
            )

            assert isinstance(response.content, SentimentResponse)
            assert response.content.value == "negative"
            assert response.reasoning is None

    @pytest.mark.asyncio
    async def test_output_model_prefers_content_over_reasoning(self, client):
        """When both content and reasoning have JSON, use content."""
        mock_response = make_mock_response(
            content='{"value": "from_content"}',
            reasoning='{"value": "from_reasoning"}',
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            response = await client.acall(
                [{"role": "user", "content": "classify"}],
                output_model=SentimentResponse,
            )

            assert response.content.value == "from_content"
            # Reasoning is preserved when content was used
            assert response.reasoning == '{"value": "from_reasoning"}'

    @pytest.mark.asyncio
    async def test_output_model_both_empty_raises(self, client):
        """When both content and reasoning are empty, raise JSONDecodeError."""
        import json

        mock_response = make_mock_response(content="", reasoning="")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            with pytest.raises(json.JSONDecodeError):
                await client.acall(
                    [{"role": "user", "content": "classify"}],
                    output_model=SentimentResponse,
                )

    @pytest.mark.asyncio
    async def test_output_model_reasoning_with_whitespace_json(self, client):
        """Reasoning with formatted JSON (newlines, spaces) should parse."""
        mock_response = make_mock_response(
            content="",
            reasoning='{\n  "value": "neutral"\n}',
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            response = await client.acall(
                [{"role": "user", "content": "classify"}],
                output_model=SentimentResponse,
            )

            assert response.content.value == "neutral"

    @pytest.mark.asyncio
    async def test_non_output_model_does_not_use_reasoning(self, client):
        """Without output_model, empty content stays empty (no reasoning fallback)."""
        mock_response = make_mock_response(content="", reasoning="some reasoning text")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            response = await client.acall(
                [{"role": "user", "content": "classify"}],
            )

            assert response.content == ""
            assert response.reasoning == "some reasoning text"
