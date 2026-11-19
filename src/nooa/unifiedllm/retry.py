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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from nooa.unifiedllm._metrics import _record_llm_metric
from nooa.unifiedllm.admission import AdmissionError
from nooa.unifiedllm.errors import EmptyContentError, LLMDeadlineExceeded
from nooa.unifiedllm.media_inputs import unsupported_attachment_format
from nooa.unifiedllm.retry_config import RetryConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Absolute ``time.monotonic()`` by which the current logical LLM call must end.
# Set by ``llm_deadline`` (a harness's task budget) and, inside one call, by
# ``with_retry`` from ``RetryConfig.total_timeout``; the earlier deadline wins.
_llm_call_deadline: ContextVar[float | None] = ContextVar("nooa_llm_call_deadline", default=None)


@contextmanager
def llm_deadline(seconds: float | None) -> Iterator[None]:
    """Make every LLM call in this context end within ``seconds`` from now.

    A harness with a task or turn budget wraps the run in it, e.g.
    ``with llm_deadline(budget - margin): await agent.solve(task)``. Each
    provider attempt is clipped to the remaining time, no retry starts that
    cannot finish, and an expired call raises ``LLMDeadlineExceeded``. Nested
    deadlines keep the earliest. ``None`` changes nothing. Tasks created inside
    the context inherit it (ContextVar copy semantics).
    """
    if seconds is None:
        yield
        return
    deadline = time.monotonic() + max(0.0, float(seconds))
    current = _llm_call_deadline.get()
    token = _llm_call_deadline.set(deadline if current is None else min(current, deadline))
    try:
        yield
    finally:
        _llm_call_deadline.reset(token)


def set_llm_deadline_at(deadline: float | None) -> None:
    """Set the absolute ``time.monotonic()`` call deadline in the current context.

    For code that starts work in a fresh ``contextvars.Context`` (Atom's
    session turn loops) and so cannot use the ``llm_deadline`` context
    manager around it: run this in that context first. An earlier deadline
    already set there wins.
    """
    if deadline is None:
        return
    current = _llm_call_deadline.get()
    _llm_call_deadline.set(deadline if current is None else min(current, deadline))


def llm_time_remaining() -> float | None:
    """Seconds left before the current LLM call deadline, or None without one."""
    deadline = _llm_call_deadline.get()
    return None if deadline is None else deadline - time.monotonic()


def clip_attempt_timeout(request_timeout: float | None) -> tuple[float | None, bool]:
    """Clip one attempt's timeout to the call deadline: ``(timeout, clipped)``.

    Raises ``LLMDeadlineExceeded`` when the deadline has already passed, so no
    attempt is dispatched without time to run.
    """
    remaining = llm_time_remaining()
    if remaining is None:
        return request_timeout, False
    if remaining <= 0:
        raise LLMDeadlineExceeded(
            "LLM call deadline reached before the provider attempt could start"
        )
    if request_timeout is None or remaining < request_timeout:
        return remaining, True
    return request_timeout, False


def _error_status(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int):
        status_code = getattr(getattr(error, "response", None), "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _record_event(name: str, detail: dict[str, Any]) -> None:
    """Record a retry-loop event as a harness metric and on the active LLM span."""
    _record_llm_metric(name, detail)
    try:
        from opentelemetry import trace as otel_trace

        span = otel_trace.get_current_span()
        if span and span.is_recording():
            span.add_event(
                name.replace("_", ".", 1),
                attributes={k: v for k, v in detail.items() if v is not None},
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not record %s event: %s", name, e)


def _deadline_error(
    error: Exception, attempt: int, total_timeout: float | None, delay: float
) -> LLMDeadlineExceeded:
    remaining = llm_time_remaining() or 0.0
    budget = f"total_timeout={total_timeout:g}s" if total_timeout is not None else "llm_deadline"
    exc = LLMDeadlineExceeded(
        f"LLM call deadline ({budget}) leaves {max(0.0, remaining):.1f}s after attempt "
        f"{attempt}: not enough for another attempt after a {delay:.1f}s backoff. "
        f"Last error: {type(error).__name__}: {str(error)[:200]}",
        deadline_s=total_timeout,
        attempts=attempt,
    )
    _record_event(
        "llm_deadline_exceeded",
        {
            "attempts": attempt,
            "error_type": type(error).__name__,
            "status_code": _error_status(error),
            "remaining_s": round(max(0.0, remaining), 3),
        },
    )
    return exc


def _call_deadline_for(config: RetryConfig) -> float | None:
    """The deadline one ``with_retry`` call runs under: caller's or total_timeout, earliest."""
    current = _llm_call_deadline.get()
    if config.total_timeout is None:
        return current
    own = time.monotonic() + max(0.0, config.total_timeout)
    return own if current is None else min(current, own)


def _note_retry(
    attempt: int, max_attempts: int, error: Exception, delay: float, config: RetryConfig
) -> None:
    """Log, trace and count one retry; call ``config.on_retry``."""
    status = _error_status(error)
    remaining = llm_time_remaining()
    logger.warning(
        "LLM retry: attempt %d/%d failed (%s%s: %s); retrying in %.1fs%s",
        attempt,
        max_attempts,
        f"status {status}, " if status is not None else "",
        type(error).__name__,
        str(error)[:200],
        delay,
        f", {remaining:.1f}s left before the call deadline" if remaining is not None else "",
    )
    _record_event(
        "llm_retry",
        {
            "attempt": attempt,
            "max_attempts": max_attempts,
            "error_type": type(error).__name__,
            "status_code": status,
            "error": str(error)[:200],
            "delay_s": round(delay, 3),
            "remaining_s": round(remaining, 3) if remaining is not None else None,
        },
    )
    if config.on_retry:
        config.on_retry(attempt, error, delay)


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
    # Admission failures happen before the provider. Retrying a timeout would
    # immediately rejoin the same queue, a call-cap rejection cannot succeed,
    # and a controller outage should not be mistaken for a provider outage.
    # A higher-level workflow may still retry later if appropriate.
    if isinstance(error, AdmissionError):
        return False, False

    # Check for empty content error (if enabled)
    if isinstance(error, EmptyContentError) and config.retry_on_empty_content:
        return True, False

    # The call's total deadline is spent: another attempt cannot finish in time.
    if isinstance(error, LLMDeadlineExceeded):
        return False, False

    # An HTTP 500 that says an attachment's format is unsupported (the NVIDIA
    # Inference Hub sends "Unsupported image format: svg+xml" as 500) fails the
    # same way every time: the client's media recovery replaces the attachment
    # with a note instead.
    if unsupported_attachment_format(error):
        return False, False

    # Inspect structured status data before falling back to message heuristics.
    # LiteLLM and OpenAI APIStatusError subclasses expose ``.status_code``
    # directly, while HTTPX stores it on ``HTTPStatusError.response.status_code``.
    # A known non-retryable status must remain terminal: if broad text matching
    # ran first, an endpoint or provider message could accidentally make a
    # permanent 4xx authentication/model error look transient and retry it.
    status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int):
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
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
        The last exception if all retries are exhausted, or
        ``LLMDeadlineExceeded`` when the call deadline (``config.total_timeout``
        or an enclosing ``llm_deadline``) leaves no room for another attempt.
    """
    config = config or RetryConfig()
    deadline = _call_deadline_for(config)
    token = _llm_call_deadline.set(deadline)
    try:
        return await _with_retry_loop(func, args, kwargs, config)
    finally:
        _llm_call_deadline.reset(token)


def _retry_delay_or_raise(error: Exception, attempt: int, config: RetryConfig) -> tuple[float, int]:
    """After attempt ``attempt`` (0-based) failed: ``(delay, max_attempts)``, or raise.

    Raises ``error`` when it is terminal or retries are spent, and
    ``LLMDeadlineExceeded`` when the call deadline leaves no room for the
    backoff plus ``config.min_attempt_seconds``.
    """
    is_retryable, is_rate_limit = _is_retryable_error(error, config)
    if not is_retryable:
        raise error
    # Rate limits can retry up to max_retries + rate_limit_extra_retries;
    # other errors only get normal retries.
    allowed = config.max_retries + (config.rate_limit_extra_retries if is_rate_limit else 0)
    if attempt >= allowed:
        raise error
    delay = _calculate_delay(attempt, config, is_rate_limit)
    remaining = llm_time_remaining()
    if remaining is not None and remaining < delay + config.min_attempt_seconds:
        raise _deadline_error(error, attempt + 1, config.total_timeout, delay) from error
    return delay, allowed + 1


async def _with_retry_loop(
    func: Callable[..., Any], args: tuple, kwargs: dict, config: RetryConfig
) -> Any:
    attempt = 0
    while True:
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            delay, max_attempts = _retry_delay_or_raise(e, attempt, config)
            _note_retry(attempt + 1, max_attempts, e, delay, config)
            await asyncio.sleep(delay)
            attempt += 1


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
        The last exception if all retries are exhausted, or
        ``LLMDeadlineExceeded`` when the call deadline (``config.total_timeout``
        or an enclosing ``llm_deadline``) leaves no room for another attempt.
    """
    config = config or RetryConfig()
    deadline = _call_deadline_for(config)
    token = _llm_call_deadline.set(deadline)
    try:
        attempt = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                delay, max_attempts = _retry_delay_or_raise(e, attempt, config)
                _note_retry(attempt + 1, max_attempts, e, delay, config)
                time.sleep(delay)
                attempt += 1
    finally:
        _llm_call_deadline.reset(token)


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
