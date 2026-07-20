# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime async safety patches for agent-generated code.

Prevents deadlocks when code runs inside an async context by patching
concurrent.futures blocking calls to raise errors when called from
the event loop thread.

The patches are scoped via context manager to avoid breaking test infrastructure.

Note: AST-based validation is now in code_validator.py (BlockingCallValidator).
This module only provides runtime patches for cross-block patterns that AST
analysis cannot catch (e.g., future assigned in one code block, .result()
called in another).
"""

import asyncio
import concurrent.futures
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import current_thread
from typing import Any

# =============================================================================
# Runtime patches for concurrent.futures blocking calls (scoped via context)
# =============================================================================

# Context variable to track if we're inside agent code execution
_in_agent_context: ContextVar[bool] = ContextVar("in_agent_context", default=False)

# Store original methods for restoration
_original_future_result = concurrent.futures.Future.result
_original_future_exception = concurrent.futures.Future.exception
_original_wait = concurrent.futures.wait
_original_as_completed = concurrent.futures.as_completed


def _is_event_loop_thread() -> bool:
    """Check if current thread is running the event loop."""
    try:
        loop = asyncio.get_running_loop()
        # Compare thread IDs - _thread_id is set when loop starts
        if hasattr(loop, "_thread_id"):
            return bool(getattr(loop, "_thread_id") == current_thread().ident)  # noqa: B009
        return True  # Assume yes if we can't check
    except RuntimeError:
        return False  # No running loop


def _safe_future_result(self: concurrent.futures.Future[Any], timeout: float | None = None) -> Any:
    """Wrapped Future.result() that blocks deadlocks in agent context."""
    if _in_agent_context.get() and _is_event_loop_thread() and not self.done():
        raise RuntimeError(
            "Can't call Future.result() from async context - "
            "this would deadlock. Use 'await asyncio.wrap_future(future)' instead."
        )
    return _original_future_result(self, timeout)


def _safe_future_exception(
    self: concurrent.futures.Future[Any], timeout: float | None = None
) -> BaseException | None:
    """Wrapped Future.exception() that blocks deadlocks in agent context."""
    if _in_agent_context.get() and _is_event_loop_thread() and not self.done():
        raise RuntimeError(
            "Can't call Future.exception() from async context - "
            "this would deadlock. Use 'await asyncio.wrap_future(future)' instead."
        )
    return _original_future_exception(self, timeout)


def _safe_wait(
    fs: Iterable[concurrent.futures.Future[Any]],
    timeout: float | None = None,
    return_when: str = concurrent.futures.ALL_COMPLETED,
) -> tuple[set[concurrent.futures.Future[Any]], set[concurrent.futures.Future[Any]]]:
    """Wrapped wait() that blocks deadlocks in agent context."""
    if _in_agent_context.get() and _is_event_loop_thread():
        raise RuntimeError(
            "Can't call concurrent.futures.wait() from async context - "
            "this would deadlock. Use 'await asyncio.gather(...)' instead."
        )
    return _original_wait(fs, timeout, return_when)


def _safe_as_completed(
    fs: Iterable[concurrent.futures.Future[Any]], timeout: float | None = None
) -> Iterator[concurrent.futures.Future[Any]]:
    """Wrapped as_completed() that blocks deadlocks in agent context."""
    if _in_agent_context.get() and _is_event_loop_thread():
        raise RuntimeError(
            "Can't call concurrent.futures.as_completed() from async context - "
            "this would deadlock. Use 'async for' with asyncio instead."
        )
    return _original_as_completed(fs, timeout)


# Apply patches at import time - they only activate when _in_agent_context is True
concurrent.futures.Future.result = _safe_future_result  # type: ignore[method-assign]
concurrent.futures.Future.exception = _safe_future_exception  # type: ignore[method-assign]
concurrent.futures.wait = _safe_wait  # type: ignore[assignment]
concurrent.futures.as_completed = _safe_as_completed  # type: ignore[assignment]


@contextmanager
def agent_async_safety_context() -> Generator[None, None, None]:
    """Context manager that enables runtime safety checks for agent code.

    Use this around agent code execution to catch cross-block Future.result() calls:

        async with agent_async_safety_context():
            result = await agent.runtime.execute_code(code)

    Outside this context, the runtime patches are inactive and won't affect
    test infrastructure or other code.
    """
    token = _in_agent_context.set(True)
    try:
        yield
    finally:
        _in_agent_context.reset(token)
