# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UnifiedLLM failures callers may handle independently of the transport.

Malformed replay state is terminal: retrying must not silently drop it.
EmptyContentError participates in the existing empty-output retry policy.
"""

import math
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Literal


class FailureKind(StrEnum):
    """The public, transport-independent failure taxonomy."""

    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    CONTEXT_WINDOW = "context_window"
    MEDIA_LIMIT = "media_limit"
    MEDIA_REJECTED = "media_rejected"
    OUTPUT_LIMIT = "output_limit"
    BAD_REQUEST = "bad_request"
    AUTH = "auth"
    CONFIG = "config"


class LLMFailure(Exception):
    """A typed failure with optional normalized recovery facts.

    Adapters supply facts, not provider error text: token/count fields are
    non-negative integers, context_window is a positive integer, status_code
    is an HTTP status (100–599), and retry_after is finite seconds >= 0.
    Missing facts stay None; zero is not treated as missing. No recovery or
    retry policy is implied here. The exception remains mutable so Python can
    attach traceback, context and cause normally.
    """

    def __init__(
        self,
        kind: FailureKind | str,
        message: str,
        *,
        prompt_tokens: int | None = None,
        context_window: int | None = None,
        media_kind: Literal["image", "pdf", "audio", "video"] | None = None,
        media_count: int | None = None,
        media_count_limit: int | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
        cause: BaseException | None = None,
    ) -> None:
        if not isinstance(kind, (str, FailureKind)):
            raise ValueError("kind must be a FailureKind or its string value")
        self.kind = FailureKind(kind)
        if not isinstance(message, str):
            raise ValueError("message must be a string")
        for name, value, minimum, maximum in (
            ("prompt_tokens", prompt_tokens, 0, None),
            ("context_window", context_window, 1, None),
            ("media_count", media_count, 0, None),
            ("media_count_limit", media_count_limit, 0, None),
            ("status_code", status_code, 100, 599),
        ):
            if value is not None and (
                type(value) is not int
                or value < minimum
                or (maximum is not None and value > maximum)
            ):
                raise ValueError(
                    f"{name} must be an integer >= {minimum}"
                    + (f" and <= {maximum}" if maximum is not None else "")
                )
        if media_kind is not None and (
            not isinstance(media_kind, str) or media_kind not in ("image", "pdf", "audio", "video")
        ):
            raise ValueError("media_kind must be image, pdf, audio or video")
        if retry_after is not None:
            if type(retry_after) not in (int, float):
                raise ValueError("retry_after must be finite non-negative seconds")
            try:
                retry_after = float(retry_after)
            except OverflowError:
                raise ValueError("retry_after must be finite non-negative seconds") from None
            if not math.isfinite(retry_after) or retry_after < 0:
                raise ValueError("retry_after must be finite non-negative seconds")
        if cause is not None and not isinstance(cause, BaseException):
            raise ValueError("cause must be an exception")
        self.prompt_tokens = prompt_tokens
        self.context_window = context_window
        self.media_kind = media_kind
        self.media_count = media_count
        self.media_count_limit = media_count_limit
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


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
