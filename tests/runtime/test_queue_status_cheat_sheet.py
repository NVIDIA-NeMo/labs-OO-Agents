# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for queue_status cheat sheet in QueueManager.status()."""

from __future__ import annotations

import asyncio

import pytest

from nooa.runtime.channels import QueueManager


@pytest.mark.parametrize(
    "names", [("user_messages",), ("ci_monitor",), ("user_messages", "ci_monitor")]
)
def test_no_cheat_sheet_with_empty_channels(names):
    """Empty registered queues have the same status as no channels."""
    qm = QueueManager()
    for name in names:
        qm.queue(name)
    status = qm.status()
    assert status == "(nothing queued)"
    assert "queue_manager" not in status.lower()
    assert "remove_channel" not in status
    assert "shutdown" not in status


def test_status_never_names_the_hidden_queue_manager():
    """Pending items on extra channels are shown, but no queue_manager commands."""
    qm = QueueManager()
    qm.queue("user_messages")
    ch = qm.queue("ci_monitor")
    ch.put("some status line")
    status = qm.status()
    assert "ci_monitor" in status
    assert "queue_manager" not in status
    assert "remove_channel" not in status
    assert "shutdown" not in status


def test_cheat_sheet_lists_extra_channel_names():
    """Cheat sheet includes the actual extra channel names."""
    qm = QueueManager()
    qm.queue("user_messages")
    ch1 = qm.queue("pipeline_a")
    ch1.put("line")
    ch2 = qm.queue("pipeline_b")
    ch2.put("line")
    status = qm.status()
    assert "pipeline_a" in status
    assert "pipeline_b" in status


def test_no_cheat_sheet_when_no_channels():
    """No cheat sheet when no channels at all."""
    qm = QueueManager()
    status = qm.status()
    assert status == "(nothing queued)"
    assert "remove_channel" not in status


def test_status_populated_formatting_unchanged_then_drained():
    qm = QueueManager()
    messages = qm.queue("user_messages")
    monitor = qm.queue("ci_monitor")
    qm.queue("empty")
    messages.put("first")
    messages.put("second")
    monitor.put("ok")

    assert qm.status() == "Unread:\n  • user_messages: 2\n  • ci_monitor: 1"
    assert messages.drain() == ["first", "second"]
    assert monitor.drain() == ["ok"]
    assert qm.status() == "(nothing queued)"
    assert messages.status() == monitor.status() == ""


async def _dummy_gen():
    """Async generator that never finishes."""
    yield "started"
    await asyncio.sleep(9999)


@pytest.mark.asyncio
async def test_active_spawns_shown_when_queues_empty():
    """Active spawn jobs appear in status even when no items pending."""
    qm = QueueManager()
    qm.queue("user_messages")
    qm.queue("ci_monitor")
    qm.spawn(_dummy_gen(), channel="ci_monitor", buffer=5)

    # Let the task start
    await asyncio.sleep(0.05)

    status = qm.status()
    assert "\nRunning:\n  • " in status  # after "Unread:" (the item it yielded)
    assert "ci_monitor" in status
    assert "running" not in status  # the heading says it

    # Cleanup
    await qm.shutdown()


@pytest.mark.asyncio
async def test_undescribed_spawn_shows_default_stop_line():
    """A running job nobody described gets one line naming the call that stops it."""
    qm = QueueManager()
    qm.queue("ci_monitor")
    qm.spawn(_dummy_gen(), channel="ci_monitor")
    qm.spawn(_dummy_gen(), channel="ci_monitor")
    await asyncio.sleep(0.05)
    status = qm.status()
    stop = "Stop: await self.queue_manager.cancel('ci_monitor')"
    assert status.endswith("\n" + stop)
    assert status.count(stop) == 1
    await qm.shutdown()


@pytest.mark.asyncio
async def test_described_or_hinted_spawn_shows_only_the_owner_text():
    """An owner's describe() or set_cancel_hint replaces the default line."""
    qm = QueueManager()
    qm.queue("described")
    qm.queue("hinted")
    h = qm.spawn(_dummy_gen(), channel="described")
    h.describe = lambda: "watching CI for PR 7"
    qm.set_cancel_hint("hinted", "Stop a monitor: await self.ci.stop('<label>')")
    qm.spawn(_dummy_gen(), channel="hinted")
    await asyncio.sleep(0.05)
    status = qm.status()
    assert "  • watching CI for PR 7" in status
    assert status.endswith("\nStop a monitor: await self.ci.stop('<label>')")
    assert "queue_manager" not in status
    await qm.shutdown()


@pytest.mark.asyncio
async def test_default_stop_line_can_be_turned_off():
    qm = QueueManager()
    qm.queue("ci_monitor")
    qm.set_default_cancel_hint(None)
    qm.spawn(_dummy_gen(), channel="ci_monitor")
    await asyncio.sleep(0.05)
    assert "Stop:" not in qm.status()
    await qm.shutdown()


def test_no_active_spawns_shown_when_none_running():
    """No spawn section when all handles are done."""
    qm = QueueManager()
    qm.queue("user_messages")
    status = qm.status()
    assert "Running:" not in status
