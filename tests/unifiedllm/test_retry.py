# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for retry logic."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nooa.unifiedllm.retry import (
    RetryConfig,
    RetryingWrapper,
    _calculate_delay,
    _is_retryable_error,
    with_retry,
)


class TestCalculateDelay:
    """Tests for delay calculation."""

    def test_exponential_backoff(self):
        """Test that delay increases exponentially."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter_factor=0.0)

        delay_0 = _calculate_delay(0, config)
        delay_1 = _calculate_delay(1, config)
        delay_2 = _calculate_delay(2, config)

        assert delay_0 == 1.0  # 1 * 2^0
        assert delay_1 == 2.0  # 1 * 2^1
        assert delay_2 == 4.0  # 1 * 2^2

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=5.0,
            jitter_factor=0.0,
        )

        # 1 * 2^10 = 1024, but should be capped at 5
        delay = _calculate_delay(10, config)
        assert delay == 5.0

    def test_rate_limit_uses_longer_delay(self):
        """Test that rate limits use longer base delay."""
        config = RetryConfig(
            base_delay=1.0,
            rate_limit_base_delay=3.0,
            jitter_factor=0.0,
        )

        normal_delay = _calculate_delay(0, config, is_rate_limit=False)
        rate_limit_delay = _calculate_delay(0, config, is_rate_limit=True)

        assert rate_limit_delay > normal_delay
        assert rate_limit_delay == 3.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delay."""
        config = RetryConfig(base_delay=1.0, jitter_factor=0.5)

        delays = [_calculate_delay(0, config) for _ in range(100)]

        # All delays should be >= base (1.0) and < base + jitter (1.5)
        assert all(1.0 <= d <= 1.5 for d in delays)

        # There should be variation (not all the same)
        assert len(set(delays)) > 1


class TestIsRetryableError:
    """Tests for error classification."""

    def test_rate_limit_429(self):
        """Test that 429 status is detected as rate limit."""
        config = RetryConfig()
        error = Exception("API call failed with status 429: rate limit exceeded")

        is_retryable, is_rate_limit = _is_retryable_error(error, config)

        assert is_retryable is True
        assert is_rate_limit is True

    def test_server_error_500(self):
        """Test that 500 status is retryable but not rate limit."""
        config = RetryConfig()
        error = Exception("API call failed with status 500: internal server error")

        is_retryable, is_rate_limit = _is_retryable_error(error, config)

        assert is_retryable is True
        assert is_rate_limit is False

    def test_timeout_error(self):
        """Test that timeout errors are retryable."""
        config = RetryConfig()
        error = TimeoutError()

        is_retryable, is_rate_limit = _is_retryable_error(error, config)

        assert is_retryable is True
        assert is_rate_limit is False

    def test_timeout_in_message(self):
        """Test that timeout in error message is retryable."""
        config = RetryConfig()
        error = Exception("Request timed out after 30s")

        is_retryable, is_rate_limit = _is_retryable_error(error, config)

        assert is_retryable is True
        assert is_rate_limit is False

    @pytest.mark.parametrize(
        "error",
        [
            httpx.ConnectError("failed to connect"),
            httpx.ConnectTimeout("connection timed out"),
            httpx.RemoteProtocolError("server disconnected without sending a response"),
            httpx.NetworkError("network is unreachable"),
        ],
    )
    def test_httpx_endpoint_errors_are_retryable(self, error):
        """httpx transport failures are retryable endpoint errors."""
        config = RetryConfig()

        is_retryable, is_rate_limit = _is_retryable_error(error, config)

        assert is_retryable is True
        assert is_rate_limit is False

    def test_api_connection_class_name_is_retryable(self):
        """Provider APIConnectionError-style exceptions are retryable."""

        class APIConnectionError(Exception):
            pass

        config = RetryConfig()
        error = APIConnectionError("endpoint not reachable")

        is_retryable, is_rate_limit = _is_retryable_error(error, config)

        assert is_retryable is True
        assert is_rate_limit is False

    def test_non_retryable_error(self):
        """Test that non-retryable errors are not retried."""
        config = RetryConfig()
        error = Exception("Invalid API key")

        is_retryable, is_rate_limit = _is_retryable_error(error, config)

        assert is_retryable is False
        assert is_rate_limit is False

    def test_400_not_retryable(self):
        """Test that 400 status is not retryable."""
        config = RetryConfig()
        error = Exception("API call failed with status 400: bad request")

        is_retryable, is_rate_limit = _is_retryable_error(error, config)

        assert is_retryable is False

    def test_status_code_attribute_502(self):
        """LiteLLM exceptions expose .status_code; 502 is retryable, not rate limit."""
        import litellm

        config = RetryConfig()
        error = litellm.BadGatewayError(message="502 Bad Gateway", model="m", llm_provider="openai")

        is_retryable, is_rate_limit = _is_retryable_error(error, config)

        assert is_retryable is True
        assert is_rate_limit is False

    def test_internal_server_error_with_server_disconnected_message(self):
        """LiteLLM InternalServerError with server-disconnected text is retryable."""
        import litellm

        config = RetryConfig()
        error = litellm.InternalServerError(
            message=(
                "InternalServerError: OpenAIException - Server disconnected\n"
                "No fallback model group found for original model_group=nvidia/qwen/qwen3.5-35b-a3b."
            ),
            model="m",
            llm_provider="openai",
        )

        assert _is_retryable_error(error, config) == (True, False)

    def test_status_code_attribute_400_not_retryable(self):
        """A 400 .status_code is not in retryable set."""
        import litellm

        config = RetryConfig()
        error = litellm.BadRequestError(message="bad", model="m", llm_provider="openai")

        is_retryable, _ = _is_retryable_error(error, config)

        assert is_retryable is False

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 418])
    def test_structured_non_retryable_status_stays_terminal(self, status_code):
        """Structured non-retryable HTTP statuses do not fall through to endpoint phrases."""

        class StatusError(Exception):
            def __init__(self, code: int):
                self.status_code = code
                super().__init__("endpoint not reachable; request timed out")

        config = RetryConfig()

        assert _is_retryable_error(StatusError(status_code), config) == (False, False)

    def test_bad_gateway_string_without_status_token(self):
        """Raw 'BadGatewayError: 502 Bad Gateway' string (no 'status 502') is retryable."""
        config = RetryConfig()
        error = Exception("litellm.BadGatewayError: OpenAIException - 502 Bad Gateway")

        is_retryable, is_rate_limit = _is_retryable_error(error, config)

        assert is_retryable is True
        assert is_rate_limit is False

    def test_service_unavailable_string(self):
        """'503 Service Unavailable' string is retryable via phrase fallback."""
        config = RetryConfig()
        error = Exception("Server returned 503 Service Unavailable")

        is_retryable, _ = _is_retryable_error(error, config)

        assert is_retryable is True

    def test_gateway_timeout_string(self):
        """'504 Gateway Timeout' string is retryable via phrase fallback."""
        config = RetryConfig()
        error = Exception("Upstream 504 Gateway Timeout")

        is_retryable, _ = _is_retryable_error(error, config)

        assert is_retryable is True

    def test_phrase_disabled_when_code_excluded(self):
        """A 502 string is NOT retryable when 502 is excluded from the config."""
        config = RetryConfig(retryable_status_codes=frozenset({429, 500}))
        error = Exception("OpenAIException - 502 Bad Gateway")

        is_retryable, _ = _is_retryable_error(error, config)

        assert is_retryable is False

    def test_429_disabled_when_code_excluded(self):
        """429 is NOT retried when 429 is excluded from retryable_status_codes."""
        import litellm

        config = RetryConfig(retryable_status_codes=frozenset({500, 502, 503, 504}))
        # Both the structured-attribute path and the string path must respect config.
        attr_error = litellm.RateLimitError(message="rate limit", model="m", llm_provider="openai")
        str_error = Exception("API call failed with status 429: rate limit exceeded")

        assert _is_retryable_error(attr_error, config) == (False, False)
        assert _is_retryable_error(str_error, config) == (False, False)

    def test_429_rate_limit_when_code_included(self):
        """429 stays retryable as a rate limit under the default config."""
        config = RetryConfig()
        error = Exception("API call failed with status 429: rate limit exceeded")

        assert _is_retryable_error(error, config) == (True, True)


class TestWithRetry:
    """Tests for the with_retry function."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test that successful calls don't retry."""
        mock_func = AsyncMock(return_value="success")

        result = await with_retry(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_retry_on_failure_then_success(self):
        """Test retry on transient failure followed by success."""
        mock_func = AsyncMock(
            side_effect=[
                Exception("API call failed with status 500"),
                "success",
            ]
        )
        config = RetryConfig(base_delay=0.01)  # Fast for testing

        result = await with_retry(mock_func, config=config)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_endpoint_disconnect_then_success(self):
        """Endpoint disconnects use the same retry loop as 5xx errors."""
        mock_func = AsyncMock(
            side_effect=[
                Exception("RemoteDisconnected: remote end closed connection without response"),
                "success",
            ]
        )
        config = RetryConfig(base_delay=0.01, jitter_factor=0.0)

        result = await with_retry(mock_func, config=config)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test that exception is raised when max retries exceeded."""
        mock_func = AsyncMock(side_effect=Exception("API call failed with status 500"))
        config = RetryConfig(max_retries=2, base_delay=0.01)

        with pytest.raises(Exception, match="status 500"):
            await with_retry(mock_func, config=config)

        # Initial attempt + 2 retries = 3 calls
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_endpoint_retry_limit_exhausted(self):
        """Endpoint errors stop after the configured retry limit."""
        mock_func = AsyncMock(side_effect=Exception("endpoint not reachable"))
        config = RetryConfig(max_retries=2, base_delay=0.01, jitter_factor=0.0)

        with pytest.raises(Exception, match="endpoint not reachable"):
            await with_retry(mock_func, config=config)

        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_without_retry(self):
        """Task cancellation must not be converted into retry/TimeoutError."""
        mock_func = AsyncMock(side_effect=asyncio.CancelledError())
        config = RetryConfig(max_retries=2, base_delay=0.01, jitter_factor=0.0)

        with pytest.raises(asyncio.CancelledError):
            await with_retry(mock_func, config=config)

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_error_not_retried(self):
        """Test that non-retryable errors are raised immediately."""
        mock_func = AsyncMock(side_effect=Exception("Invalid API key"))
        config = RetryConfig(max_retries=3)

        with pytest.raises(Exception, match="Invalid API key"):
            await with_retry(mock_func, config=config)

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_extra_retries(self):
        """Test that rate limits get extra retries."""
        mock_func = AsyncMock(
            side_effect=[
                Exception("API call failed with status 429"),
                Exception("API call failed with status 429"),
                Exception("API call failed with status 429"),
                Exception("API call failed with status 429"),
                "success",
            ]
        )
        config = RetryConfig(
            max_retries=1,  # Would normally only retry once
            rate_limit_extra_retries=3,  # But rate limits get 3 extra
            base_delay=0.01,
            rate_limit_base_delay=0.01,
        )

        result = await with_retry(mock_func, config=config)

        assert result == "success"
        assert mock_func.call_count == 5

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test that on_retry callback is called."""
        mock_func = AsyncMock(
            side_effect=[
                Exception("API call failed with status 500"),
                "success",
            ]
        )
        callback = MagicMock()
        config = RetryConfig(base_delay=0.01, on_retry=callback)

        await with_retry(mock_func, config=config)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == 1  # attempt number
        assert isinstance(args[1], Exception)  # error
        assert args[2] > 0  # delay


class TestRetryingWrapper:
    """Tests for the RetryingWrapper class."""

    @pytest.mark.asyncio
    async def test_wrapper_calls_function(self):
        """Test that wrapper calls underlying function."""
        mock_func = AsyncMock(return_value="result")
        wrapper = RetryingWrapper(mock_func)

        result = await wrapper("arg1", kwarg1="value1")

        assert result == "result"
        mock_func.assert_called_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_wrapper_uses_config(self):
        """Test that wrapper uses provided config."""
        mock_func = AsyncMock(
            side_effect=[
                Exception("API call failed with status 500"),
                "success",
            ]
        )
        config = RetryConfig(max_retries=1, base_delay=0.01)
        wrapper = RetryingWrapper(mock_func, config=config)

        result = await wrapper()

        assert result == "success"
        assert mock_func.call_count == 2
