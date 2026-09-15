# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent snapshot — intermediate representation of serializable agent state.

``AgentSnapshot`` captures everything needed to save/restore an agent.
Pydantic models provide validation and JSON serialization out of the box.
"""

import logging
from collections.abc import Mapping
from typing import Any, Final

from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator

from nooa.context_blocks import (
    ContextBlock,
    ExpressionContextBlock,
    LiteralContextBlock,
)
from nooa.errors.storage import SerializationError
from nooa.storage.markers import is_nosnapshot_field, is_nosnapshot_value
from nooa.storage.serialization import SKIP, deserialize, serialize

SNAPSHOT_VERSION: Final = 3
LEGACY_SNAPSHOT_VERSION: Final = 2

logger = logging.getLogger(__name__)


class AgentSnapshot(BaseModel):
    """Intermediate representation of serializable agent state.

    Captures everything needed to restore an agent to a prior state.
    Uses Pydantic for validation and JSON serialization.
    """

    version: int = SNAPSHOT_VERSION
    context: list[ContextBlock] = Field(default_factory=list)
    disabled_context: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    type_allowlist: set[str] = Field(default_factory=set)

    @model_validator(mode="before")
    @classmethod
    def _migrate_v2(cls, value: Any) -> Any:
        """Upgrade legacy v2 block tags to the canonical v3 schema."""
        if not isinstance(value, Mapping) or value.get("version") != LEGACY_SNAPSHOT_VERSION:
            return value

        migrated = dict(value)
        blocks: list[Any] = []
        for raw_block in value.get("context", []):
            if isinstance(raw_block, BaseModel):
                raw_block = raw_block.model_dump()
            if not isinstance(raw_block, Mapping):
                blocks.append(raw_block)
                continue

            block = dict(raw_block)
            block_type = block.get("type")
            if block_type == "static":
                block["type"] = "literal"
            elif block_type == "dynamic":
                block["type"] = "expression"

            block.setdefault("prefix", False)
            if block.get("type") == "expression":
                block.setdefault("display_expr", None)
            blocks.append(block)

        migrated["context"] = blocks
        migrated["version"] = SNAPSHOT_VERSION
        return migrated

    @field_serializer("type_allowlist")
    @classmethod
    def _serialize_allowlist(cls, v: set[str]) -> list[str]:
        return sorted(v)

    @field_validator("type_allowlist", mode="before")
    @classmethod
    def _validate_allowlist(cls, v: Any) -> set[str]:
        if isinstance(v, (list, tuple)):
            return set(v)
        return v

    @staticmethod
    def from_agent(agent: Any) -> "AgentSnapshot":
        """Extract serializable state from an agent.

        Args:
            agent: An Agent instance.

        Returns:
            An AgentSnapshot capturing the agent's current state.

        Raises:
            SerializationError: If a context block value or user attribute is
                not JSON-serializable.
        """
        all_allowlist: set[str] = set()

        context_blocks: list[ContextBlock] = []
        protected = agent.context_manager.protected_keys
        for key, block in agent.context_manager._raw_items():
            if key in protected:
                continue  # Framework blocks are recreated by __init__
            if isinstance(block, LiteralContextBlock):
                try:
                    serialized, allowlist = serialize(block.value)
                    all_allowlist |= allowlist
                except SerializationError as exc:
                    raise SerializationError(
                        f"Context block {key!r} is not serializable: {exc}"
                    ) from exc
                context_blocks.append(block.model_copy(update={"value": serialized}))
            else:
                context_blocks.append(block)

        attributes: dict[str, Any] = {}
        agent_cls = type(agent)
        for attr_name, attr_value in agent.__dict__.items():
            if attr_name.startswith("_agentdoc_"):
                continue
            if is_nosnapshot_field(agent_cls, attr_name):
                continue
            if is_nosnapshot_value(attr_value):
                continue
            if callable(attr_value):
                continue
            try:
                serialized, allowlist = serialize(attr_value)
            except SerializationError as exc:
                # A single non-serializable attribute must not abort the whole
                # snapshot — that silently loses ALL durable state (vars, todos,
                # ...). Skip it and warn so the failure is visible but recoverable.
                logger.warning(
                    "Snapshot: skipping non-serializable attribute %r (%s): %s",
                    attr_name,
                    type(attr_value).__name__,
                    exc,
                )
                continue
            all_allowlist |= allowlist
            if serialized is SKIP:
                continue
            attributes[attr_name] = serialized

        return AgentSnapshot(
            version=SNAPSHOT_VERSION,
            context=context_blocks,
            disabled_context=sorted(agent.context_manager.disabled()),
            attributes=attributes,
            type_allowlist=all_allowlist,
        )

    def restore(self, agent: Any) -> None:
        """Restore this snapshot's state into an agent, mutating it in place.

        Note: this performs additive restoration — it does not clear
        pre-existing context blocks or attributes on the target agent.
        The expected usage is with a freshly constructed agent (via
        ``StorageManager.restore_snapshot()``), not an agent with
        in-progress state.

        Args:
            agent: A freshly constructed Agent instance to restore into.

        Raises:
            SerializationError: If the snapshot version doesn't match.
        """
        if self.version != SNAPSHOT_VERSION:
            raise SerializationError(
                f"Snapshot version mismatch: expected {SNAPSHOT_VERSION}, got {self.version}"
            )

        for block in self.context:
            if isinstance(block, LiteralContextBlock):
                block = block.model_copy(
                    update={"value": deserialize(block.value, self.type_allowlist)}
                )
            else:
                assert isinstance(block, ExpressionContextBlock)
            agent.context_manager.restore_block(block)

        if self.disabled_context:
            agent.context_manager.disable(*self.disabled_context)

        for attr_name, attr_value in self.attributes.items():
            setattr(agent, attr_name, deserialize(attr_value, self.type_allowlist))
