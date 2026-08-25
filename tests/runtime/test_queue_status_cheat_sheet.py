# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for queue_status cheat sheet in QueueManager.status()."""

from __future__ import annotations

import asyncio

import pytest

from nooa.runtime.channels import QueueManager


def test_no_cheat_sheet_with_only_empty_user_messages():
    """No cheat sheet when the sole user_messages queue is empty."""
    qm = QueueManager()
    qm.queue("user_messages")
    status = qm.status()
    assert "queue_manager" not in status.lower()
    assert "remove_channel" not in status
    assert "shutdown" not in status


def test_pending_queue_shows_public_reader_hint():
    """Pending counts explain how the agent can consume the queued item."""
    qm = QueueManager()
    channel = qm.queue("user_messages")
    channel.put("follow-up")

    status = qm.status()

    assert "user_messages: 1 pending" in status
    assert "💡 dequeue: await self.user_messages.get()" in status


def test_cheat_sheet_appears_with_extra_channels():
    """Cheat sheet appears when channels beyond user_messages exist."""
    qm = QueueManager()
    qm.queue("user_messages")
    ch = qm.queue("ci_monitor")
    ch.put("some status line")
    status = qm.status()
    assert "remove_channel" in status
    assert "ci_monitor" in status


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


def test_cheat_sheet_mentions_shutdown():
    """Cheat sheet includes shutdown as the nuclear option."""
    qm = QueueManager()
    qm.queue("user_messages")
    ch = qm.queue("ci")
    ch.put("line")
    status = qm.status()
    assert "shutdown" in status


def test_no_cheat_sheet_when_no_channels():
    """No cheat sheet when no channels at all."""
    qm = QueueManager()
    status = qm.status()
    assert status == ""
    assert "remove_channel" not in status


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
    assert "active background job" in status
    assert "ci_monitor" in status
    assert "running" in status
    assert "Output arrives through channels" in status
    assert "do not poll job handles" in status

    # Cleanup
    await qm.shutdown()


@pytest.mark.asyncio
async def test_active_spawn_description_is_normalized_and_bounded():
    """Model-facing descriptions are concise even when callers pass raw prose."""
    qm = QueueManager()
    qm.queue("monitor")
    handle = qm.spawn(
        _dummy_gen(),
        channel="monitor",
        label="persistent monitor",
        description="  Persistent   infrastructure producer.\n" + "x" * 300,
    )

    assert handle.description.startswith("Persistent infrastructure producer. ")
    assert len(handle.description) == 240
    assert handle.description.endswith("…")
    status = qm.status()
    assert f"[{handle.job_id}] persistent monitor → monitor (running)" in status
    assert f"    {handle.description}" in status

    await qm.shutdown()


@pytest.mark.asyncio
async def test_active_spawn_status_bounds_number_of_rendered_jobs():
    """Many long-lived producers cannot grow the model context without bound."""
    qm = QueueManager()
    qm.queue("monitor")
    handles = [
        qm.spawn(
            _dummy_gen(),
            channel="monitor",
            label=f"monitor {index}",
            description=f"description {index}",
        )
        for index in range(10)
    ]

    status = qm.status()
    assert "⚡ 10 active background job(s):" in status
    assert "… 2 more active job(s) omitted; inspect " in status
    assert "self.queue_manager.running_handles() for their IDs." in status
    assert qm.running_handles() == handles
    for handle in handles[:8]:
        assert handle.job_id in status
        assert f".cancel_job('{handle.job_id}')" in status
    for handle in handles[8:]:
        assert handle.job_id not in status

    await qm.shutdown()


def test_no_active_spawns_shown_when_none_running():
    """No spawn section when all handles are done."""
    qm = QueueManager()
    qm.queue("user_messages")
    status = qm.status()
    assert "active background job" not in status
