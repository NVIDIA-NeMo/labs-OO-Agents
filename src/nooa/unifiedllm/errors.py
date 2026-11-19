# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UnifiedLLM failures callers may handle independently of the transport.

Malformed replay state is terminal: retrying must not silently drop it.
EmptyContentError participates in the existing empty-output retry policy.
"""


class ReasoningReplayError(RuntimeError):
    """A provider response or replay input violates the assistant-turn contract."""


class LLMDeadlineExceeded(TimeoutError):
    """One logical LLM call ran out of its total deadline (all attempts together).

    Raised instead of starting a retry that cannot finish in time, or when an
    attempt clipped to the remaining time expires. It is never retried.
    """

    def __init__(self, message: str, *, deadline_s: float | None = None, attempts: int = 0):
        self.deadline_s = deadline_s
        self.attempts = attempts
        super().__init__(message)


class EmptyContentError(Exception):
    """A model returned reasoning without the requested final content."""

    def __init__(self, reasoning: str | None = None):
        self.reasoning = reasoning
        super().__init__(
            f"Empty content with reasoning: {reasoning[:100]}..." if reasoning else "Empty content"
        )
