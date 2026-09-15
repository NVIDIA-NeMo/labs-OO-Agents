# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UnifiedLLM failures callers may handle independently of the transport.

Malformed replay state is terminal: retrying must not silently drop it.
EmptyContentError participates in the existing empty-output retry policy.
"""


class ReasoningReplayError(RuntimeError):
    """A provider response or replay input violates the assistant-turn contract."""


class UnsupportedStopReasonError(RuntimeError):
    """A native stop needs continuation semantics the client cannot represent."""

    def __init__(self, stop_reason):
        self.stop_reason = stop_reason
        super().__init__(
            f"Unsupported Anthropic stop_reason {stop_reason!r}; "
            "server-managed continuation is not supported by the direct transport"
        )


class EmptyContentError(Exception):
    """A model returned reasoning without the requested final content."""

    def __init__(self, reasoning: str | None = None):
        self.reasoning = reasoning
        super().__init__(
            f"Empty content with reasoning: {reasoning[:100]}..." if reasoning else "Empty content"
        )
