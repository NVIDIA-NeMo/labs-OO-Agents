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

from nooa.unifiedllm.adapters._legacy_error_text import legacy_retryable
from nooa.unifiedllm.admission import AdmissionError
from nooa.unifiedllm.errors import EmptyContentError, FailureKind, LLMFailure, _ReplyParseError
from nooa.unifiedllm.retry_config import RetryConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    # Typed kind is authoritative; settings may narrow, never expand it.
    if isinstance(error, LLMFailure):
        eligible = error.kind in {
            FailureKind.RATE_LIMIT,
            FailureKind.TRANSIENT,
            FailureKind.TIMEOUT,
        }
        if not eligible:
            return False, False
        if error.status_code is not None and error.status_code not in config.retryable_status_codes:
            return False, False
        return True, error.kind == FailureKind.RATE_LIMIT

    # Admission failures happen before the provider. Retrying a timeout would
    # immediately rejoin the same queue, a call-cap rejection cannot succeed,
    # and a controller outage should not be mistaken for a provider outage.
    # A higher-level workflow may still retry later if appropriate.
    if isinstance(error, AdmissionError):
        return False, False

    # The reply arrived; parsing it failed. Re-sending would bill the same
    # request again, and the error text quotes the reply, so never match it.
    if isinstance(error, _ReplyParseError):
        return False, False

    # Check for empty content error (if enabled)
    if isinstance(error, EmptyContentError) and config.retry_on_empty_content:
        return True, False

    return legacy_retryable(error, config)


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
            if isinstance(e, LLMFailure) and e.retry_after is not None:
                delay = max(delay, e.retry_after)

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
            if isinstance(e, LLMFailure) and e.retry_after is not None:
                delay = max(delay, e.retry_after)

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
