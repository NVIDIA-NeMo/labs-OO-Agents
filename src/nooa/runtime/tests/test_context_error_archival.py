# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for deterministic context-error event archival."""

from nooa.events import Feedback
from nooa.runtime.actor import ActorRuntime
from nooa.runtime.event_manager import EventManager


class _FakeAgent:
    def __init__(self) -> None:
        from nooa.config.truncation_config import TruncationConfig

        self.event_manager = EventManager()
        self._truncation = TruncationConfig(context_error_event_batch=10)


def test_context_error_archival_collapses_fixed_oldest_batch():
    agent = _FakeAgent()
    runtime = ActorRuntime(agent)
    for i in range(100):
        agent.event_manager.add(Feedback(content=f"event {i}"))

    assert runtime._archive_on_context_error() == 10

    active_tags = agent.event_manager.keys()
    assert active_tags[0] == "1..10"
    assert len(active_tags) == 91
    assert active_tags[-1] == "100"
