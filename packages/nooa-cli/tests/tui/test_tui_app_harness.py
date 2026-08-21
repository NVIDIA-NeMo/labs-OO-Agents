# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canary tests for ``TUIHarness`` itself.

If these fail, nothing in ``test_tui_app_behavior.py`` can be trusted —
the scaffolding is broken. Keep this file tiny and focused on the
harness mechanics (start, type, press, teardown), not on TUI behavior.
"""

from __future__ import annotations

import pytest

from .tui_app_harness import FakeAgent, TUIHarness


@pytest.mark.asyncio
async def test_harness_starts_and_stops_cleanly():
    async with TUIHarness() as h:
        assert h.app is not None
        assert h.app.is_running


@pytest.mark.asyncio
async def test_harness_types_into_input_buffer():
    async with TUIHarness() as h:
        await h.type_keys("hello")
        await h.wait_input_equals("hello")


@pytest.mark.asyncio
async def test_harness_delivers_named_key():
    async with TUIHarness() as h:
        await h.type_keys("abc")
        await h.press("backspace")
        await h.wait_input_equals("ab")


@pytest.mark.asyncio
async def test_fake_agent_receives_notification_and_runs_script():
    """FakeAgent ``respond(notification)`` records the message
    and invokes any scripted step on the item."""
    agent = FakeAgent()
    received: list[str] = []
    agent.queue(lambda _self, msg: received.append(msg))
    result = await agent.handle({"user_messages": ["hi"]})
    assert received == ["hi"]
    assert agent.messages_received == ["hi"]
    assert result.kind == "NEED_INPUT"
