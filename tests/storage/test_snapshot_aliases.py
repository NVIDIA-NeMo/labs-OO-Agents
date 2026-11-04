# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input aliases must not lose or exchange saved fields across SQLite resume."""

import pytest
from pydantic import BaseModel, Field

from nooa import Agent
from nooa.storage import SQLiteStorageManager
from nooa.storage.snapshot_vars import SnapshotVars
from nooa.unifiedllm import FakeLLMClient


class _RequiredResult(BaseModel):
    """Required aliased fields must not prevent resume."""

    value: int = Field(alias="apiValue")


class _DefaultedResult(BaseModel):
    """A saved value must not silently revert to its default."""

    value: int = Field(default=0, alias="apiValue")


class _CrossedResult(BaseModel):
    """External aliases must not exchange two canonical fields on resume."""

    first: int = Field(alias="second")
    second: int = Field(alias="first")


class _AliasAgent(Agent, llm=FakeLLMClient()):
    """Minimal agent exercising the supported persistence entry point."""


@pytest.mark.parametrize(
    "value",
    [
        _RequiredResult(apiValue=7),
        _DefaultedResult(apiValue=7),
        _CrossedResult(second=11, first=22),
    ],
    ids=lambda value: type(value).__name__,
)
@pytest.mark.parametrize("location", ["attribute", "context", "vars"])
@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
def test_sqlite_reopen_preserves_aliased_field_values(tmp_path, value, location, journal_mode):
    """Save, close, reopen and restore aliased values through all durable routes."""
    path = tmp_path / "aliases.sqlite"
    agent = _AliasAgent()
    if location == "attribute":
        agent.saved = value
    elif location == "context":
        agent.context_manager["saved"] = value
    else:
        agent.saved = SnapshotVars({"result": value})
    with SQLiteStorageManager(path, journal_mode=journal_mode) as storage:
        snapshot_id = storage.save_snapshot(agent)
    with SQLiteStorageManager(path, must_exist=True, journal_mode=journal_mode) as storage:
        restored = _AliasAgent()
        storage.restore_snapshot(snapshot_id, restored)
    if location == "context":
        result = restored.context_manager["saved"]
    elif location == "vars":
        result = restored.saved["result"]
    else:
        result = restored.saved
    assert type(result) is type(value)
    assert result == value
