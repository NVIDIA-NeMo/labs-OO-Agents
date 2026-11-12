# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared attempt execution, independent of provider and instrumentation code.

This is the retry owner, not yet the outer logical-call facade. Legacy clients
still prepare requests and recover media outside it. Policies are call-local;
no execution state is stored on an adapter or client.
"""

from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

from nooa.llm_types import LLMResponse, LLMUsage

from .adapters.base import Adapter
from .errors import _aunwrap_parse_errors, _unwrap_parse_errors
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
    # Keep the legacy success-only metric, after tracking exits. The outer
    # logical-call accounting policy will be integrated in a later chunk.
    if response.usage and record_usage is not None:
        record_usage(response.usage)
    return response


def execute_sync(
    adapter: Adapter,
    request: LLMRequest,
    retry_config: RetryConfig | None,
    *,
    tracking: AbstractContextManager[object] | None = None,
    check_empty: Callable[[LLMResponse, bool], None] | None = None,
    record_usage: Callable[[LLMUsage], None] | None = None,
    retry: SyncRetry = sync_retry,
) -> LLMResponse:
    """Send exactly once per attempt; unwrap terminal parse errors after retry.

    Optional policies preserve legacy hooks without importing their runtime.
    Passing no RetryConfig disables retries, including empty-response retries.
    """

    def attempt() -> LLMResponse:
        return _check_response(adapter.send_sync(request), retry_config, check_empty)

    with tracking if tracking is not None else nullcontext():
        response = _unwrap_parse_errors(
            lambda: retry(attempt, config=retry_config) if retry_config else attempt()
        )
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
) -> LLMResponse:
    """Async execute_sync; cancellation propagates through retry and tracking."""

    async def attempt() -> LLMResponse:
        return _check_response(await adapter.send(request), retry_config, check_empty)

    with tracking if tracking is not None else nullcontext():
        response = await _aunwrap_parse_errors(
            lambda: retry(attempt, config=retry_config) if retry_config else attempt()
        )
    return _record_usage(response, record_usage)
