# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that ResponsesClient applies the unified retry wrapper (issue 252).

ResponsesClient previously called litellm.responses()/aresponses() directly, so
transient 502 Bad Gateway errors bubbled out and terminated agent runs even though
RetryConfig lists 502 as retryable. These tests verify parity with CompletionClient.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from nooa.unifiedllm import ResponsesClient, RetryConfig


def make_mock_responses_response(content: str = "ok"):
    """Create a minimal litellm.ResponsesAPIResponse-like object for testing."""
    resp = MagicMock()
    resp.output = [MagicMock(type="message", content=[MagicMock(type="output_text", text=content)])]
    resp.output_text = content
    resp.usage = None
    return resp


def make_bad_gateway() -> litellm.BadGatewayError:
    """A transient 502 error as raised by litellm on /v1/responses gateway failures."""
    return litellm.BadGatewayError(
        message="502 Bad Gateway", model="test-model", llm_provider="openai"
    )


def make_bad_request() -> litellm.BadRequestError:
    """A non-retryable 400 error."""
    return litellm.BadRequestError(
        message="400 Bad Request", model="test-model", llm_provider="openai"
    )


FAST_RETRY = RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.02, jitter_factor=0.0)
NO_RETRY = RetryConfig(max_retries=0, rate_limit_extra_retries=0)


class TestResponsesClientInit:
    """ResponsesClient ctor mirrors CompletionClient and keeps args out of self.config."""

    def test_retry_config_stored_not_leaked(self):
        """retry_config / http_config / cache_control are not forwarded to litellm."""
        client = ResponsesClient(model="test-model", retry_config=FAST_RETRY)
        assert client.retry_config is FAST_RETRY
        # Must NOT leak into self.config (would be passed to litellm.responses()).
        assert "retry_config" not in client.config
        assert "http_config" not in client.config
        assert "cache_control_injection_points" not in client.config

    def test_retry_config_defaults_enabled(self):
        """Endpoint retries are enabled by default."""
        client = ResponsesClient(model="test-model")
        assert isinstance(client.retry_config, RetryConfig)
        assert client.retry_config.max_retries == RetryConfig().max_retries


class TestResponsesClientSyncRetry:
    """Sync call() retry behaviour."""

    def test_retries_on_bad_gateway_then_succeeds(self):
        """A transient 502 is retried and the subsequent success is returned."""
        client = ResponsesClient(model="test-model", retry_config=FAST_RETRY)
        with patch(
            "litellm.responses",
            side_effect=[make_bad_gateway(), make_mock_responses_response("hello")],
        ) as mock_responses:
            resp = client.call(messages=[{"role": "user", "content": "hi"}])

        assert mock_responses.call_count == 2
        assert resp.content == "hello"

    def test_zero_retry_config_disables_retries(self):
        """RetryConfig(max_retries=0, rate_limit_extra_retries=0) disables endpoint retries."""
        client = ResponsesClient(model="test-model", retry_config=NO_RETRY)
        with patch("litellm.responses", side_effect=make_bad_gateway()) as mock_responses:
            with pytest.raises(litellm.BadGatewayError):
                client.call(messages=[{"role": "user", "content": "hi"}])
        assert mock_responses.call_count == 1

    def test_retries_exhausted_raises(self):
        """A persistent 502 is retried up to max_retries, then the error is raised."""
        client = ResponsesClient(model="test-model", retry_config=FAST_RETRY)
        with patch("litellm.responses", side_effect=make_bad_gateway()) as mock_responses:
            with pytest.raises(litellm.BadGatewayError):
                client.call(messages=[{"role": "user", "content": "hi"}])
        # initial attempt + max_retries
        assert mock_responses.call_count == FAST_RETRY.max_retries + 1

    def test_non_retryable_not_retried(self):
        """A non-retryable 400 BadRequestError is not retried even with a retry_config."""
        client = ResponsesClient(model="test-model", retry_config=FAST_RETRY)
        with patch("litellm.responses", side_effect=make_bad_request()) as mock_responses:
            with pytest.raises(litellm.BadRequestError):
                client.call(messages=[{"role": "user", "content": "hi"}])
        assert mock_responses.call_count == 1


class TestResponsesClientAsyncRetry:
    """Async acall() retry behaviour."""

    @pytest.mark.asyncio
    async def test_retries_on_bad_gateway_then_succeeds(self):
        """A transient 502 is retried and the subsequent success is returned."""
        client = ResponsesClient(model="test-model", retry_config=FAST_RETRY)
        mock = AsyncMock(side_effect=[make_bad_gateway(), make_mock_responses_response("hello")])
        with patch("litellm.aresponses", mock):
            resp = await client.acall(messages=[{"role": "user", "content": "hi"}])

        assert mock.call_count == 2
        assert resp.content == "hello"

    @pytest.mark.asyncio
    async def test_zero_retry_config_disables_retries(self):
        """RetryConfig(max_retries=0, rate_limit_extra_retries=0) disables endpoint retries."""
        client = ResponsesClient(model="test-model", retry_config=NO_RETRY)
        mock = AsyncMock(side_effect=make_bad_gateway())
        with patch("litellm.aresponses", mock):
            with pytest.raises(litellm.BadGatewayError):
                await client.acall(messages=[{"role": "user", "content": "hi"}])
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises(self):
        """A persistent 502 is retried up to max_retries, then the error is raised."""
        client = ResponsesClient(model="test-model", retry_config=FAST_RETRY)
        mock = AsyncMock(side_effect=make_bad_gateway())
        with patch("litellm.aresponses", mock):
            with pytest.raises(litellm.BadGatewayError):
                await client.acall(messages=[{"role": "user", "content": "hi"}])
        assert mock.call_count == FAST_RETRY.max_retries + 1

    @pytest.mark.asyncio
    async def test_non_retryable_not_retried(self):
        """A non-retryable 400 BadRequestError is not retried even with a retry_config."""
        client = ResponsesClient(model="test-model", retry_config=FAST_RETRY)
        mock = AsyncMock(side_effect=make_bad_request())
        with patch("litellm.aresponses", mock):
            with pytest.raises(litellm.BadRequestError):
                await client.acall(messages=[{"role": "user", "content": "hi"}])
        assert mock.call_count == 1
