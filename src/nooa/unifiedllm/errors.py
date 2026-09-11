# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UnifiedLLM failures callers may handle independently of the transport.

Malformed replay state is terminal: retrying must not silently drop it.
EmptyContentError participates in the existing empty-output retry policy.
"""


class ReasoningReplayError(RuntimeError):
    """A provider response or replay input violates the assistant-turn contract."""


class EmptyContentError(Exception):
    """A model returned reasoning without the requested final content."""

    def __init__(self, reasoning: str | None = None):
        self.reasoning = reasoning
        super().__init__(
            f"Empty content with reasoning: {reasoning[:100]}..." if reasoning else "Empty content"
        )
