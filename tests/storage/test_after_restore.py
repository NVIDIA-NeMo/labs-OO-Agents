# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every snapshot restore runs the agent's ``after_restore()`` once (``AgentSnapshot.restore``)."""

from __future__ import annotations

from nooa import Agent
from nooa.context_blocks import Context
from nooa.interactive import _QUEUES_EXPR, InteractiveAgent
from nooa.storage import SQLiteStorageManager, snapshot_from_json, snapshot_to_json
from nooa.storage.snapshot import AgentSnapshot
from nooa.unifiedllm import FakeLLMClient

OLD_QUEUES_EXPR = "self.queue_manager.status()"


class _Counting(InteractiveAgent, llm=FakeLLMClient()):
    def after_restore(self) -> None:
        super().after_restore()
        self.restores = getattr(self, "restores", 0) + 1


class _Plain(Agent, llm=FakeLLMClient()):
    value: int = 0


def _old_agent(storage=None) -> _Counting:
    """An agent whose ``queues`` block still names the old expression, as an old snapshot does."""
    agent = _Counting(llm=FakeLLMClient(), storage=storage)
    agent.context["queues"] = Context(expr=OLD_QUEUES_EXPR)
    return agent


def _queues_expr(agent: Agent) -> str:
    [block] = [b for b in AgentSnapshot.from_agent(agent).context if b.key == "queues"]
    return block.expr


def test_sqlite_restore_snapshot_runs_after_restore_once():
    storage = SQLiteStorageManager(":memory:")
    try:
        snapshot_id = storage.save_snapshot(_old_agent(storage))
        restored = _Counting(llm=FakeLLMClient(), storage=storage)
        storage.restore_snapshot(snapshot_id, restored)
        assert restored.restores == 1
        assert _queues_expr(restored) == _QUEUES_EXPR
    finally:
        storage.close()


def test_sqlite_restore_latest_snapshot_runs_after_restore_once():
    storage = SQLiteStorageManager(":memory:")
    try:
        storage.save_snapshot(_old_agent(storage))
        restored = _Counting(llm=FakeLLMClient(), storage=storage)
        assert storage.restore_latest_snapshot(restored) is True
        assert restored.restores == 1
        assert _queues_expr(restored) == _QUEUES_EXPR
    finally:
        storage.close()


def test_snapshot_from_json_runs_after_restore_once():
    data = snapshot_to_json(_old_agent())
    restored = _Counting(llm=FakeLLMClient())
    snapshot_from_json(data, restored)
    assert restored.restores == 1
    assert _queues_expr(restored) == _QUEUES_EXPR


def test_an_agent_without_after_restore_restores_as_before():
    agent = _Plain(llm=FakeLLMClient())
    agent.value = 7
    restored = _Plain(llm=FakeLLMClient())
    snapshot_from_json(snapshot_to_json(agent), restored)
    assert restored.value == 7
