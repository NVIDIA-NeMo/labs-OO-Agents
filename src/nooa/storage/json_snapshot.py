# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON serialization for AgentSnapshot.

Thin wrappers around Pydantic's built-in serialization.
The model types and extraction/restoration logic live in ``snapshot``.
"""

from typing import Any

from nooa.storage.snapshot import AgentSnapshot


def snapshot_to_dict(snapshot: AgentSnapshot) -> dict[str, Any]:
    """Convert an AgentSnapshot to a JSON-serializable dict."""
    return snapshot.model_dump()


def snapshot_from_dict(data: dict[str, Any]) -> AgentSnapshot:
    """Construct an AgentSnapshot from a JSON-deserialized dict."""
    return AgentSnapshot.model_validate(data)


# -- Convenience functions (agent ↔ JSON dict in one step) --
def snapshot_to_json(agent: Any) -> dict[str, Any]:
    """Serialize agent state to a JSON-serializable dict.

    Shorthand for ``AgentSnapshot.from_agent(agent).model_dump()``.

    Args:
        agent: An Agent instance.

    Returns:
        A dict suitable for ``json.dumps``.

    Raises:
        SerializationError: If a user attribute is not JSON-serializable.
    """
    return AgentSnapshot.from_agent(agent).model_dump()


def snapshot_from_json(snapshot: dict[str, Any], agent: Any) -> None:
    """Restore agent state from a JSON snapshot dict.

    Shorthand for ``AgentSnapshot.model_validate(snapshot).restore(agent)``.

    Args:
        snapshot: A dict previously produced by ``snapshot_to_json``.
        agent: An Agent instance to restore into.

    Raises:
        SerializationError: If the snapshot version doesn't match.
    """
    AgentSnapshot.model_validate(snapshot).restore(agent)
