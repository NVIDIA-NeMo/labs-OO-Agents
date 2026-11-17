# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent snapshot — intermediate representation of serializable agent state.

``AgentSnapshot`` captures everything needed to save/restore an agent.
Pydantic models provide validation and JSON serialization out of the box.
"""

import logging
from typing import Any, Final, Literal

from pydantic import BaseModel, field_serializer, field_validator

from nooa.context_blocks import DynamicContext
from nooa.errors.storage import SerializationError
from nooa.storage.markers import is_nosnapshot_field, is_nosnapshot_value
from nooa.storage.serialization import SKIP, deserialize, serialize

SNAPSHOT_VERSION: Final = 2

logger = logging.getLogger(__name__)


class StaticContextBlock(BaseModel):
    """A static context block with a JSON-serializable value."""

    key: str
    type: Literal["static"] = "static"
    value: Any = None


class DynamicContextBlock(BaseModel):
    """A dynamic context block with a Python expression string."""

    key: str
    type: Literal["dynamic"] = "dynamic"
    expr: str


class AgentSnapshot(BaseModel):
    """Intermediate representation of serializable agent state.

    Captures everything needed to restore an agent to a prior state.
    Uses Pydantic for validation and JSON serialization.
    """

    version: int = SNAPSHOT_VERSION
    context: list[StaticContextBlock | DynamicContextBlock] = []
    disabled_context: list[str] = []
    attributes: dict[str, Any] = {}
    type_allowlist: set[str] = set()

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

        context_blocks: list[StaticContextBlock | DynamicContextBlock] = []
        for key, value in agent.context_manager.declarations():
            if isinstance(value, DynamicContext):
                context_blocks.append(DynamicContextBlock(key=key, expr=value.expr))
            else:
                try:
                    serialized, allowlist = serialize(value)
                    all_allowlist |= allowlist
                except SerializationError as exc:
                    raise SerializationError(
                        f"Context block {key!r} is not serializable: {exc}"
                    ) from exc
                context_blocks.append(StaticContextBlock(key=key, value=serialized))

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

    def restore(self, agent: Any) -> dict[str, str]:
        """Restore this snapshot's state into an agent, mutating it in place.

        Note: this performs additive restoration — it does not clear
        pre-existing context blocks or attributes on the target agent.
        The expected usage is with a freshly constructed agent (via
        ``StorageManager.restore_snapshot()``), not an agent with
        in-progress state.

        Args:
            agent: A freshly constructed Agent instance to restore into.

        An attribute that fails to deserialize is skipped with a warning; the
        other attributes are still restored. Last, if the agent has a callable
        ``after_restore`` (``InteractiveAgent`` and its subclasses), it runs
        once, so every restore path repairs what did not survive the restart;
        hosts do not call it again.

        Returns:
            The skipped attributes: name mapped to a short reason
            (``"<ExceptionType>: <message>"``). Empty when everything restored.

        Raises:
            SerializationError: If the snapshot version doesn't match.
            DeserializationError: If a static context block can't be restored.
        """
        if self.version != SNAPSHOT_VERSION:
            raise SerializationError(
                f"Snapshot version mismatch: expected {SNAPSHOT_VERSION}, got {self.version}"
            )

        for block in self.context:
            if isinstance(block, DynamicContextBlock):
                agent.context_manager.set_dynamic(block.key, block.expr)
            else:
                agent.context_manager[block.key] = deserialize(block.value, self.type_allowlist)

        if self.disabled_context:
            agent.context_manager.disable(*self.disabled_context)

        skipped: dict[str, str] = {}
        for attr_name, attr_value in self.attributes.items():
            try:
                value = deserialize(attr_value, self.type_allowlist)
            except Exception as exc:
                # Mirror ``from_agent``: one unrestorable attribute (e.g. nested
                # deeper than ``MAX_SNAPSHOT_DEPTH`` in an older checkpoint, a
                # schema-drifted model, or a failing ``__restore_snapshot__``)
                # must not abort the whole restore. Skip it and warn.
                logger.warning(
                    "Snapshot: skipping non-restorable attribute %r: %s",
                    attr_name,
                    exc,
                    exc_info=True,
                )
                skipped[attr_name] = f"{type(exc).__name__}: {exc}"
                continue
            setattr(agent, attr_name, value)
        after_restore = getattr(agent, "after_restore", None)
        if callable(after_restore):
            after_restore()
        return skipped
