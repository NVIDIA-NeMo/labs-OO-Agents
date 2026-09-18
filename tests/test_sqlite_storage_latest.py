# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SQLiteStorageManager latest-snapshot convenience methods."""

from __future__ import annotations

import pytest

from nooa import Agent, Context
from nooa.storage import SQLiteStorageManager
from nooa.unifiedllm import CompletionClient


def _make_storage() -> SQLiteStorageManager:
    return SQLiteStorageManager(":memory:")


class _SimpleAgent(Agent, llm=CompletionClient(model="openai/gpt-4o-mini", api_key="test")):
    value: int = 0


class _PrefixResumeAgent(Agent, llm=CompletionClient(model="openai/gpt-4o-mini", api_key="test")):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context["python_cell_tools"] = Context(expr="'TOOLS'", prefix=True)

    async def work(self) -> str:
        """Exercise context rendering after snapshot resume."""
        ...


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


@pytest.mark.asyncio
async def test_restart_resume_keeps_expression_block_in_cached_prefix(tmp_path):
    db_path = tmp_path / "resume.db"
    storage = SQLiteStorageManager(db_path)
    agent = _PrefixResumeAgent(storage=storage)
    storage.save_snapshot(agent)
    storage.close()

    restored_storage = SQLiteStorageManager(db_path)
    try:
        restored = _PrefixResumeAgent(storage=restored_storage)
        assert restored_storage.restore_latest_snapshot(restored) is True

        messages = await restored.runtime._build_messages(type(restored).work)
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        trailing_context = "\n".join(
            m["content"] for m in messages if m["role"] == "user" and "<context>" in m["content"]
        )

        assert "<python_cell_tools " in system_content
        assert "TOOLS" in system_content
        assert "python_cell_tools" not in trailing_context
    finally:
        restored_storage.close()
