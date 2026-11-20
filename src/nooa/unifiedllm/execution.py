# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared attempt execution, independent of provider and instrumentation code.

This is the attempt retry owner; logical.py supplies outer media recovery and
rollback through invocation-local policies. Neither executor stores execution
state on an adapter or client. Instrumentation remains injectable.
"""

from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

from nooa.llm_types import LLMResponse, LLMUsage

from ._accounting import _current_account
from ._route_calibration import _CalibrationAdapter
from .adapters._numeric_usage import _numeric_usage
from .adapters.base import Adapter
from .admission import AdmissionError
from .errors import (
    EmptyContentError,
    FailureKind,
    LLMFailure,
    _aunwrap_parse_errors,
    _ReplyParseError,
    _unwrap_parse_errors,
)
from .request import LLMRequest
from .retry import sync_retry, with_retry
from .retry_config import RetryConfig


class SyncRetry(Protocol):
    """Only the retry invocation the executor needs (also a legacy patch hook)."""

    def __call__(self, call: Callable[[], LLMResponse], *, config: RetryConfig) -> LLMResponse: ...


class AsyncRetry(Protocol):
    """Async counterpart of SyncRetry."""

    def __call__(
        self, call: Callable[[], Awaitable[LLMResponse]], *, config: RetryConfig
    ) -> Awaitable[LLMResponse]: ...


def check_empty_response(response: LLMResponse, enabled: bool) -> None:
    """Honor the opt-in reasoning-only retry policy without provider knowledge."""
    if enabled and not response.content and response.reasoning:
        raise EmptyContentError(response.reasoning)


def _check_response(
    response: LLMResponse,
    retry_config: RetryConfig | None,
    check_empty: Callable[[LLMResponse, bool], None] | None,
) -> LLMResponse:
    if check_empty is not None:
        check_empty(response, retry_config.retry_on_empty_content if retry_config else False)
    return response


def _record_usage(
    response: LLMResponse, record_usage: Callable[[LLMUsage], None] | None
) -> LLMResponse:
    # Public usage stays the accepted LAST response, never paid subtotals.
    account = _current_account.get()
    if account is not None:
        account.accepted_response = response
    if response.usage and record_usage is not None:
        record_usage(response.usage)
    return response


def _public_attempt_error(error: Exception) -> Exception:
    if isinstance(error, (LLMFailure, AdmissionError, EmptyContentError)):
        return error
    if isinstance(error, _ReplyParseError):
        return LLMFailure(FailureKind.BAD_REQUEST, str(error.error), cause=error.error)
    kind = FailureKind.TIMEOUT if isinstance(error, TimeoutError) else FailureKind.CONFIG
    return LLMFailure(kind, str(error), cause=error)


def _public_terminal(error: Exception) -> Exception:
    if isinstance(error, EmptyContentError):
        return LLMFailure(FailureKind.OUTPUT_LIMIT, str(error), cause=error)
    if isinstance(error, AdmissionError):
        return LLMFailure(FailureKind.CONFIG, str(error), cause=error)
    return _public_attempt_error(error)


def execute_sync(
    adapter: Adapter,
    request: LLMRequest,
    retry_config: RetryConfig | None,
    *,
    tracking: AbstractContextManager[object] | None = None,
    check_empty: Callable[[LLMResponse, bool], None] | None = None,
    record_usage: Callable[[LLMUsage], None] | None = None,
    retry: SyncRetry = sync_retry,
    _public_failures: bool = False,
) -> LLMResponse:
    """Send exactly once per attempt; unwrap terminal parse errors after retry.

    Optional policies preserve legacy hooks without importing their runtime.
    Passing no RetryConfig disables retries, including empty-response retries.
    """

    def attempt() -> LLMResponse:
        account = _current_account.get()
        if account is not None:
            account.describe_request(messages=request.messages)
        indexed = account.attempt() if account is not None else None
        try:
            if indexed is not None and isinstance(adapter, _CalibrationAdapter):
                response = adapter._send_sync_owned(request, indexed[1])
            else:
                response = adapter.send_sync(request)
                if indexed is not None:
                    indexed[1].writer.delivered(_numeric_usage(response))
            return _check_response(response, retry_config, check_empty)
        except Exception as error:
            if indexed is not None:
                indexed[1].writer.failed(_public_terminal(error))
            if not _public_failures:
                raise
            normalized = _public_attempt_error(error)
            if normalized is error:
                raise
            raise normalized from normalized.__cause__
        finally:
            if indexed is not None:
                account.fold(*indexed)

    with tracking if tracking is not None else nullcontext():
        try:

            def call():
                return retry(attempt, config=retry_config) if retry_config else attempt()

            response = call() if _public_failures else _unwrap_parse_errors(call)
        except Exception as error:
            if not _public_failures:
                raise
            normalized = _public_terminal(error)
            if normalized is error:
                raise
            raise normalized from normalized.__cause__
    return _record_usage(response, record_usage)


async def execute_async(
    adapter: Adapter,
    request: LLMRequest,
    retry_config: RetryConfig | None,
    *,
    tracking: AbstractContextManager[object] | None = None,
    check_empty: Callable[[LLMResponse, bool], None] | None = None,
    record_usage: Callable[[LLMUsage], None] | None = None,
    retry: AsyncRetry = with_retry,
    _public_failures: bool = False,
) -> LLMResponse:
    """Async execute_sync; cancellation propagates through retry and tracking."""

    async def attempt() -> LLMResponse:
        account = _current_account.get()
        if account is not None:
            account.describe_request(messages=request.messages)
        indexed = account.attempt() if account is not None else None
        try:
            if indexed is not None and isinstance(adapter, _CalibrationAdapter):
                response = await adapter._send_owned(request, indexed[1])
            else:
                response = await adapter.send(request)
                if indexed is not None:
                    indexed[1].writer.delivered(_numeric_usage(response))
            return _check_response(response, retry_config, check_empty)
        except Exception as error:
            if indexed is not None:
                indexed[1].writer.failed(_public_terminal(error))
            if not _public_failures:
                raise
            normalized = _public_attempt_error(error)
            if normalized is error:
                raise
            raise normalized from normalized.__cause__
        finally:
            if indexed is not None:
                account.fold(*indexed)

    with tracking if tracking is not None else nullcontext():
        try:

            def call():
                return retry(attempt, config=retry_config) if retry_config else attempt()

            response = await call() if _public_failures else await _aunwrap_parse_errors(call)
        except Exception as error:
            if not _public_failures:
                raise
            normalized = _public_terminal(error)
            if normalized is error:
                raise
            raise normalized from normalized.__cause__
    return _record_usage(response, record_usage)
