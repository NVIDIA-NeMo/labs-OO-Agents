# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SQLiteStorageManager latest-snapshot convenience methods."""

from __future__ import annotations

from nooa import Agent
from nooa.storage import SQLiteStorageManager
from nooa.unifiedllm import CompletionClient


def _make_storage() -> SQLiteStorageManager:
    return SQLiteStorageManager(":memory:")


class _SimpleAgent(Agent, llm=CompletionClient(model="openai/gpt-4o-mini", api_key="test")):
    value: int = 0


def test_get_latest_snapshot_id_empty():
    storage = _make_storage()
    assert storage.get_latest_snapshot_id() is None


def test_restore_latest_snapshot_empty_returns_false():
    storage = _make_storage()
    agent = _SimpleAgent(storage=storage)
    result = storage.restore_latest_snapshot(agent)
    assert result is False


def test_restore_latest_snapshot_returns_true_after_save():
    storage = _make_storage()
    agent = _SimpleAgent(storage=storage)
    agent.value = 42
    storage.save_snapshot(agent)

    # Fresh agent — value starts at 0
    agent2 = _SimpleAgent(storage=storage)
    assert agent2.value == 0

    result = storage.restore_latest_snapshot(agent2)
    assert result is True
    assert agent2.value == 42


def test_restore_latest_returns_most_recent():
    storage = _make_storage()
    agent = _SimpleAgent(storage=storage)

    agent.value = 1
    storage.save_snapshot(agent)

    agent.value = 99
    storage.save_snapshot(agent)

    agent2 = _SimpleAgent(storage=storage)
    storage.restore_latest_snapshot(agent2)
    assert agent2.value == 99
