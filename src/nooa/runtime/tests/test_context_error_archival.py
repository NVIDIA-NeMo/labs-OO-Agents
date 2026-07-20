# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for context-error event archival sizing."""

from nooa.context_blocks.models import ContextWindowStats
from nooa.events import Feedback
from nooa.runtime.actor import ActorRuntime
from nooa.runtime.event_manager import EventManager


class _FakeAgent:
    def __init__(self) -> None:
        self.event_manager = EventManager()


def test_context_error_archival_does_not_collapse_all_zero_token_events():
    """Structured tool events can have chars/tokens attributed as zero.

    In that case the context-error fallback should still shed history, but it
    must not treat every requested token as one event and collapse the whole
    active history in a single pass.
    """
    agent = _FakeAgent()
    runtime = ActorRuntime(agent)
    for i in range(100):
        agent.event_manager.add(Feedback(content=f"event {i}"))
    runtime._last_context_stats = ContextWindowStats(
        context_blocks_count=1,
        events_count=100,
        prompt_tokens=10_000,
        context_blocks_chars=1_000,
        events_chars=0,
        model_context_window=10_000,
    )

    runtime._archive_on_context_error(ctx_window=10_000)

    active_tags = agent.event_manager.keys()
    assert active_tags[0] == "1..58"
    assert len(active_tags) == 43
    assert active_tags[-1] == "100"
