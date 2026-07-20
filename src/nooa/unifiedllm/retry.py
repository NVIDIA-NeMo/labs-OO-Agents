# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Retry logic with exponential backoff for LLM calls.

This module provides configurable retry behavior for handling transient
failures in LLM API calls (rate limits, timeouts, server errors).
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from nooa.unifiedllm.retry_config import RetryConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Provider/LiteLLM error substrings to fall back on when an error only carries a
# stringified form (e.g. raw gateway HTML) without a usable ``.status_code`` or a
# ``status {code}`` token. Keyed by HTTP status code so a phrase only triggers a
# retry when that code is present in ``RetryConfig.retryable_status_codes``.
# ``error_str`` is lowercased before matching, so class-name forms (e.g.
# ``BadGatewayError``) match too.
_RETRYABLE_STATUS_PHRASES: dict[int, tuple[str, ...]] = {
    500: ("internal server error", "internalservererror"),
    502: ("bad gateway", "badgatewayerror"),
    503: ("service unavailable", "serviceunavailableerror", "server unavailable"),
    504: ("gateway timeout", "gateway time-out"),
}

# Endpoint failures often arrive without a useful HTTP status. Match exception
# class names through the MRO so optional provider/httpx/aiohttp exception types
# do not become import-time dependencies; the names below cover transport,
# timeout, disconnected-server, and provider APIConnectionError-style failures.
_RETRYABLE_ENDPOINT_CLASS_NAMES = {
    "apiconnectionerror",
    "apitimeouterror",
    "connecterror",
    "connecttimeout",
    "connectionerror",
    "connectiontimeouterror",
    "networkerror",
    "readtimeout",
    "remoteprotocolerror",
    "serverdisconnectederror",
    "timeout",
    "timeouterror",
    "transporterror",
    "writeerror",
    "writetimeout",
}

# When an exception is just a generic wrapper with text, fall back to phrases
# that indicate the endpoint/transport failed before a usable model response was
# produced. Structured non-retryable HTTP statuses return before this list is used.
_RETRYABLE_ENDPOINT_PHRASES = (
    "api connection error",
    "cannot connect",
    "can't connect",
    "connection aborted",
    "connection closed",
    "connection reset",
    "connection refused",
    "connection terminated",
    "connection timed out",
    "endpoint not reachable",
    "failed to connect",
    "max retries exceeded",
    "name resolution",
    "network is unreachable",
    "not reachable",
    "nodename nor servname",
    "remote disconnected",
    "remote end closed",
    "server disconnected",
    "temporary failure in name resolution",
    "timed out",
    "timeout",
    "transport error",
)


def _is_retryable_endpoint_error(error: Exception, error_str: str) -> bool:
    """Return True for transient network/transport failures without HTTP status."""
    for cls in type(error).__mro__:
        if cls.__name__.lower() in _RETRYABLE_ENDPOINT_CLASS_NAMES:
            return True

    if any(phrase in error_str for phrase in _RETRYABLE_ENDPOINT_PHRASES):
        return True

    return "connection" in error_str and any(
        phrase in error_str
        for phrase in (
            "abort",
            "closed",
            "disconnect",
            "failed",
            "lost",
            "reset",
            "refused",
            "terminated",
            "timed out",
            "unreachable",
        )
    )


class EmptyContentError(Exception):
    """Raised when LLM returns empty content but has reasoning.

    Some reasoning models (e.g., NVIDIA NIM nemotron, gpt-oss-20b) may return
    reasoning_content but null/empty content. This exception enables retry
    logic to handle these cases.
    """

    def __init__(self, reasoning: str | None = None):
        self.reasoning = reasoning
        super().__init__(
            f"Empty content with reasoning: {reasoning[:100]}..." if reasoning else "Empty content"
        )


def _calculate_delay(
    attempt: int,
    config: RetryConfig,
    is_rate_limit: bool = False,
) -> float:
    """Calculate delay with exponential backoff and jitter."""
    if is_rate_limit:
        base = config.rate_limit_base_delay
        exp_base = config.rate_limit_backoff_base
    else:
        base = config.base_delay
        exp_base = config.exponential_base

    # Exponential backoff
    delay = base * (exp_base**attempt)

    # Cap at max delay
    delay = min(delay, config.max_delay)

    # Add jitter
    jitter = delay * config.jitter_factor * random.random()
    delay += jitter

    return delay


def _is_retryable_error(error: Exception, config: RetryConfig) -> tuple[bool, bool]:
    """
    Check if an error is retryable.

    Returns:
        Tuple of (is_retryable, is_rate_limit)
    """
    # Check for empty content error (if enabled)
    if isinstance(error, EmptyContentError) and config.retry_on_empty_content:
        return True, False

    # Prefer the structured status code when the exception exposes one (LiteLLM /
    # OpenAI APIStatusError subclasses carry ``.status_code`` — e.g. BadGatewayError
    # has 502). A present but non-retryable status is terminal; do not let broad
    # endpoint text matching retry permanent 4xx/auth/model errors.
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        if status_code == 429:
            return (True, True) if 429 in config.retryable_status_codes else (False, False)
        if status_code in config.retryable_status_codes:
            return True, False
        return False, False

    error_str = str(error).lower()

    # Check for rate limit (429) — only retryable if 429 is in the configured set.
    if "status 429" in error_str or "rate limit" in error_str:
        return (True, True) if 429 in config.retryable_status_codes else (False, False)

    # Check for retryable status codes
    for code in config.retryable_status_codes:
        if f"status {code}" in error_str:
            return True, False
        # Fall back to provider/LiteLLM phrasing (e.g. "502 Bad Gateway",
        # "BadGatewayError") when the "status {code}" token is absent.
        if any(phrase in error_str for phrase in _RETRYABLE_STATUS_PHRASES.get(code, ())):
            return True, False

    # Check for timeout/connection exceptions configured by the caller.
    if isinstance(error, config.retryable_exceptions):
        return True, False

    # Check for endpoint-level transient failures that may not expose a status
    # code, including LiteLLM APIConnectionError and httpx transport errors.
    if _is_retryable_endpoint_error(error, error_str):
        return True, False

    return False, False


async def with_retry(
    func: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> Any:
    """
    Execute an async function with retry logic.

    Args:
        func: Async function to call
        *args: Positional arguments for func
        config: Retry configuration (uses defaults if None)
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        The last exception if all retries are exhausted
    """
    config = config or RetryConfig()
    last_error: Exception | None = None
    attempt = 0

    # Calculate max attempts: normal retries + potential rate limit extra retries
    max_total_attempts = config.max_retries + 1 + config.rate_limit_extra_retries

    while attempt < max_total_attempts:
        try:
            return await func(*args, **kwargs)

        except asyncio.CancelledError:
            raise

        except Exception as e:
            last_error = e
            is_retryable, is_rate_limit = _is_retryable_error(e, config)

            # Check if we should retry
            if not is_retryable:
                raise

            # For rate limits, we allow extra retries
            if is_rate_limit:
                # Rate limits can retry up to max_retries + rate_limit_extra_retries
                total_rate_limit_allowed = config.max_retries + config.rate_limit_extra_retries
                if attempt >= total_rate_limit_allowed:
                    raise
            else:
                # Non-rate-limit errors only get normal retries
                if attempt >= config.max_retries:
                    raise

            # Calculate delay
            delay = _calculate_delay(attempt, config, is_rate_limit)

            # Determine max retries for logging
            max_for_logging = (
                config.max_retries + config.rate_limit_extra_retries
                if is_rate_limit
                else config.max_retries
            )

            # Log retry
            logger.warning(
                f"Retry {attempt + 1}/{max_for_logging + 1}: {type(e).__name__}: {str(e)[:100]}... Waiting {delay:.1f}s"
            )

            # Call retry callback if provided
            if config.on_retry:
                config.on_retry(attempt + 1, e, delay)

            await asyncio.sleep(delay)
            attempt += 1

    # Should not reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Retry loop completed without result or error")


def sync_retry(
    func: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> Any:
    """
    Execute a sync function with retry logic.

    Args:
        func: Sync function to call
        *args: Positional arguments for func
        config: Retry configuration (uses defaults if None)
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        The last exception if all retries are exhausted
    """
    config = config or RetryConfig()
    last_error: Exception | None = None
    attempt = 0

    # Calculate max attempts: normal retries + potential rate limit extra retries
    max_total_attempts = config.max_retries + 1 + config.rate_limit_extra_retries

    while attempt < max_total_attempts:
        try:
            return func(*args, **kwargs)

        except Exception as e:
            last_error = e
            is_retryable, is_rate_limit = _is_retryable_error(e, config)

            # Check if we should retry
            if not is_retryable:
                raise

            # For rate limits, we allow extra retries
            if is_rate_limit:
                total_rate_limit_allowed = config.max_retries + config.rate_limit_extra_retries
                if attempt >= total_rate_limit_allowed:
                    raise
            else:
                # Non-rate-limit errors only get normal retries
                if attempt >= config.max_retries:
                    raise

            # Calculate delay
            delay = _calculate_delay(attempt, config, is_rate_limit)

            # Determine max retries for logging
            max_for_logging = (
                config.max_retries + config.rate_limit_extra_retries
                if is_rate_limit
                else config.max_retries
            )

            # Log retry
            logger.warning(
                f"Retry {attempt + 1}/{max_for_logging + 1}: {type(e).__name__}: {str(e)[:100]}... Waiting {delay:.1f}s"
            )

            # Call retry callback if provided
            if config.on_retry:
                config.on_retry(attempt + 1, e, delay)

            time.sleep(delay)
            attempt += 1

    # Should not reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Retry loop completed without result or error")


class RetryingWrapper:
    """
    Wrapper that adds retry logic to any async callable.

    Example:
        client = CompletionClient(model="gpt-4o-mini")
        retrying_client = RetryingWrapper(client.acall, RetryConfig(max_retries=5))
        response = await retrying_client(messages=[...])
    """

    def __init__(
        self,
        func: Callable[..., Any],
        config: RetryConfig | None = None,
    ):
        self.func = func
        self.config = config or RetryConfig()

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await with_retry(self.func, *args, config=self.config, **kwargs)
