# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore real pre-description TodoManager serializer output without losing work."""

import json
from pathlib import Path

import pytest

from nooa.storage.serialization import deserialize, serialize


@pytest.mark.parametrize("status", ["open", "blocked", "done"])
@pytest.mark.parametrize("canonical_description", [None, "Updated requirements"])
def test_legacy_todo_snapshot_preserves_notes_and_migrates_status(status, canonical_description):
    # Captured with serialize(TodoManager) at cf28719f, not TodoManager.to_dict().
    fixture = Path(__file__).parent / "fixtures/todo_manager_before_description.json"
    raw = json.loads(fixture.read_text())
    fields = raw["blob"]["data"]["_todos"]["86456f8e"]["data"]
    fields["status"] = status
    if canonical_description is not None:
        fields["description"] = canonical_description
    manager = deserialize(raw["blob"], set(raw["allowlist"]))
    todo = manager.get("86456f8e")
    assert todo.description == (canonical_description or "Do not lose the requirements")
    assert todo.status == ("open" if status == "blocked" else status)
    blob, allowlist = serialize(manager)
    restored = deserialize(blob, allowlist).get(todo.id)
    assert restored.description == todo.description
    assert restored.status == todo.status
