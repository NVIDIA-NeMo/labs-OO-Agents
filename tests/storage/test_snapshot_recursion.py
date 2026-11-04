# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recursion failures follow existing snapshot skip-or-reject policies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from nooa import Agent
from nooa.errors.storage import SerializationError
from nooa.storage import SQLiteStorageManager
from nooa.storage.markers import nosnapshot, snapshotable
from nooa.storage.serialization import deserialize, serialize
from nooa.storage.snapshot import AgentSnapshot
from nooa.unifiedllm import FakeLLMClient


class _SnapshotAgent(Agent, llm=FakeLLMClient()):
    """Agent with independent durable state and an explicit persistence opt-out."""

    checkpoint: int = 0
    excluded: Annotated[Any, nosnapshot] = None


class _RecursiveModel(BaseModel):
    """Supported model type whose stored fields can form a recursive graph."""

    parent: _RecursiveModel | None = None


@dataclass
class _RecursiveDataclass:
    """Supported dataclass type with an optional back-reference."""

    parent: Any = None


@snapshotable
class _RecursiveObject:
    """Opted-in object whose attributes can refer back to itself."""

    def __init__(self):
        """Initialize the object before a test introduces its back-reference."""
        self.parent = None


def _cycle(shape):
    """Build a fresh recursive graph for each supported serialization branch."""
    if shape == "list":
        value = []
        value.append(value)
    elif shape == "dict":
        value = {}
        value["self"] = value
    elif shape == "two-node":
        value = {"children": []}
        value["children"].append({"parent": value})
    elif shape == "tuple-list":
        child = []
        value = (child,)
        child.append(value)
    elif shape == "pydantic":
        value = _RecursiveModel()
        value.parent = value
    elif shape == "dataclass":
        value = _RecursiveDataclass()
        value.parent = value
    elif shape == "snapshotable":
        value = _RecursiveObject()
        value.parent = value
    else:
        raise AssertionError(shape)
    return value


_SHAPES = ["list", "dict", "two-node", "tuple-list", "pydantic", "dataclass", "snapshotable"]


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
def test_cyclic_attribute_does_not_prevent_new_checkpoint(tmp_path, caplog, shape, journal_mode):
    """Save and reopen SQLite while preserving independent new and old state."""
    path = tmp_path / "checkpoint.sqlite"
    with SQLiteStorageManager(path, journal_mode=journal_mode) as storage:
        agent = _SnapshotAgent(storage=storage)
        agent.checkpoint = 1
        previous_id = storage.save_snapshot(agent)
        agent.checkpoint = 2
        agent.live_graph = _cycle(shape)
        with caplog.at_level(logging.WARNING, logger="nooa.storage.snapshot"):
            latest_id = storage.save_snapshot(agent)
        assert latest_id != previous_id
        assert storage.get_latest_snapshot_id() == latest_id
        assert any("live_graph" in record.message for record in caplog.records)

    with SQLiteStorageManager(path, must_exist=True, journal_mode=journal_mode) as storage:
        restored = _SnapshotAgent(storage=storage)
        assert storage.restore_latest_snapshot(restored)
        assert restored.checkpoint == 2
        assert "live_graph" not in restored.__dict__
        storage.restore_snapshot(previous_id, restored)
        assert restored.checkpoint == 1


@pytest.mark.parametrize("shape", _SHAPES)
def test_cyclic_context_still_rejects_save(tmp_path, shape):
    """Required context must fail with the documented error, not disappear."""
    agent = _SnapshotAgent()
    agent.context_manager["required_graph"] = _cycle(shape)
    with SQLiteStorageManager(tmp_path / "context.sqlite") as storage:
        with pytest.raises(SerializationError, match="required_graph"):
            storage.save_snapshot(agent)
        assert storage.get_latest_snapshot_id() is None


@pytest.mark.parametrize("shape", _SHAPES)
def test_recursive_codec_value_raises_serialization_error(shape):
    """Cycles are detected explicitly, before any RecursionError can occur."""
    with pytest.raises(SerializationError, match="contains a cycle") as caught:
        serialize(_cycle(shape))
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("shape", _SHAPES)
def test_explicit_nosnapshot_marker_still_excludes_cycles(shape, caplog):
    """Intentional opt-out bypasses serialization without introducing warnings."""
    agent = _SnapshotAgent()
    agent.checkpoint = 3
    agent.excluded = _cycle(shape)
    snapshot = AgentSnapshot.from_agent(agent)
    assert snapshot.attributes["checkpoint"] == 3
    assert "excluded" not in snapshot.attributes
    assert not caplog.records


def test_shared_acyclic_containers_still_roundtrip():
    """Repeated references are supported data, not recursive cycles."""
    shared = {"labels": ["same", "object"]}
    value = {"left": shared, "right": shared, "pair": (1, {"nested": [2, 3]})}
    blob, allowlist = serialize(value)
    assert deserialize(blob, allowlist) == value


def test_excessive_acyclic_depth_does_not_abort_checkpoint(tmp_path, caplog):
    """Depth overflow uses the same boundary without claiming a cycle was detected."""
    value = []
    for _ in range(2000):
        value = [value]
    agent = _SnapshotAgent()
    agent.checkpoint = 4
    agent.deep_graph = value
    with SQLiteStorageManager(tmp_path / "deep.sqlite") as storage:
        storage.save_snapshot(agent)
        restored = _SnapshotAgent()
        storage.restore_latest_snapshot(restored)
        assert restored.checkpoint == 4
        assert "deep_graph" not in restored.__dict__
    assert any("deep_graph" in record.message for record in caplog.records)
