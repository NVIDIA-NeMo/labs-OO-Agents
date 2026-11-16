# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UnifiedLLM failures callers may handle independently of the transport.

Malformed replay state is terminal: retrying must not silently drop it.
EmptyContentError participates in the existing empty-output retry policy.
"""

from collections.abc import Awaitable, Callable


class ReasoningReplayError(RuntimeError):
    """A provider response or replay input violates the assistant-turn contract."""


class EmptyContentError(Exception):
    """A model returned reasoning without the requested final content."""

    def __init__(self, reasoning: str | None = None):
        self.reasoning = reasoning
        super().__init__(
            f"Empty content with reasoning: {reasoning[:100]}..." if reasoning else "Empty content"
        )


class _ReplyParseError(Exception):
    """A reply that arrived could not be parsed: never retried, then unwrapped.

    Adapters parse inside the transport retry. A parse error can quote the
    model's reply (e.g. a JSONDecodeError), and that text may contain "rate
    limit" or "status 503", which the retry's message matching would take for
    a transient failure and re-send. Adapters raise this wrapper instead; the
    retry re-raises it unchanged and the client unwraps it outside the retry,
    so callers see the original exception.
    """

    def __init__(self, error: Exception):
        super().__init__(error)
        self.error = error


def _parse_reply[T](parse: Callable[[], T]) -> T:
    """Run an adapter's parse step; wrap its failure as ``_ReplyParseError``."""
    try:
        return parse()
    except Exception as exc:
        raise _ReplyParseError(exc) from exc


def _unwrap_parse_errors[T](call: Callable[[], T]) -> T:
    """Run ``call`` (retries included); re-raise a wrapped parse error as itself."""
    try:
        return call()
    except _ReplyParseError as wrapped:
        error = wrapped.error
    raise error  # outside the handler: the error keeps its own context


async def _aunwrap_parse_errors[T](call: Callable[[], Awaitable[T]]) -> T:
    """Async ``_unwrap_parse_errors``."""
    try:
        return await call()
    except _ReplyParseError as wrapped:
        error = wrapped.error
    raise error
