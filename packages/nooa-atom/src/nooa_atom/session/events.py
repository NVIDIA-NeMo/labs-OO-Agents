# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable metadata for interactive agent sessions."""

from typing import ClassVar

from pydantic import Field

from nooa.context_blocks import EventBase, Metadata
from nooa.context_blocks.roles import Role
from nooa_atom.session.items import TurnCancelled, Usage


class SessionStarted(Metadata):
    """Identity, environment and tree position recorded once when a session is created.

    Records written before the session tree carry ``origin`` and
    ``working_directory`` instead of ``host`` and ``workspace``; the store
    reads either (``Metadata`` keeps unknown fields).
    """

    _role: ClassVar[Role] = Role.METADATA

    host: str = ""
    model: str = ""
    agent: str = ""
    workspace: str = ""
    parent_id: str | None = None
    depth: int = 0
    name: str | None = None
    retained: bool = False
    turn_method: str = "handle"
    mode: str = "auto"
    utc_offset: float | None = None
    """Seconds east of UTC of the writer's local time at creation.

    Event timestamps are naive local times (``EventBase`` uses
    ``datetime.now()``); the store reads them with this offset, so every
    reader gets the same epoch whatever its own time zone.
    """


class SessionModeChanged(Metadata):
    """The session's permission mode changed (``set_mode``); a load restores it."""

    _role: ClassVar[Role] = Role.METADATA

    mode: str = ""


class SessionModelChanged(Metadata):
    """The session's model alias changed (``set_model``); a load restores it."""

    _role: ClassVar[Role] = Role.METADATA

    model: str = ""


class SessionReasoningChanged(Metadata):
    """The session's reasoning level changed (``set_reasoning``); a load restores it.

    A ``SessionModelChanged`` after it resets the level: a new model starts
    from its own default.
    """

    _role: ClassVar[Role] = Role.METADATA

    level: str | None = None


class SessionTitleUpdated(Metadata):
    """The latest human- or agent-selected session title."""

    _role: ClassVar[Role] = Role.METADATA

    title: str = ""
    user_set: bool = False


class ItemAdmitted(Metadata):
    """An item accepted onto one of the agent's channels, recorded before it is queued.

    ``item_id`` identifies the item for its whole life (receipt, consumed,
    withdrawn, re-queued). ``item_type`` is ``module:qualname`` of the
    item's class so a re-queued pydantic item comes back typed. A steer is
    admitted on channel ``"steer"``; if no model call sees it, it is
    admitted again on ``user_messages`` with the same ``item_id``.
    """

    _role: ClassVar[Role] = Role.METADATA

    channel: str = ""
    item_id: str = ""
    item_json: str = ""
    item_type: str = ""
    source: str = ""


class ItemConsumed(Metadata):
    """An admitted item was taken off its channel (by the turn loop or by agent code)."""

    _role: ClassVar[Role] = Role.METADATA

    item_id: str = ""


class ItemWithdrawn(Metadata):
    """An admitted item was withdrawn by its sender before anything consumed it."""

    _role: ClassVar[Role] = Role.METADATA

    item_id: str = ""


class ItemDiscarded(Metadata):
    """An admitted item left its channel unconsumed and not withdrawn.

    Code flushed, cleared or removed the channel. A later load does not
    re-queue it. ``reason`` says why, when the writer gave one (a recovered
    copy of a session discards the items its original never read).
    """

    _role: ClassVar[Role] = Role.METADATA

    item_id: str = ""
    reason: str = ""


class ItemRequeued(Metadata):
    """An item admitted but never consumed was put back on its channel after a load."""

    _role: ClassVar[Role] = Role.METADATA

    item_id: str = ""


class TurnStarted(Metadata):
    """A turn started; ``item_ids`` are the admitted items its notification carries."""

    _role: ClassVar[Role] = Role.METADATA

    item_ids: list[str] = Field(default_factory=list)
    item_preview: str = ""


class TurnEnded(Metadata):
    """A turn ended.

    ``outcome_kind`` is ``done``, ``need_input``, ``need_input_form``, ``waiting``,
    ``cancelled`` or ``error``. ``result_json`` is the outcome as JSON
    data. ``usage`` is this turn's own token and cost delta.
    """

    _role: ClassVar[Role] = Role.METADATA

    outcome_kind: str = ""
    explanation: str = ""
    result_json: str = ""
    usage: Usage = Field(default_factory=Usage)


class ChildDeleted(Metadata):
    """Tombstone: a child session of this one was deleted."""

    _role: ClassVar[Role] = Role.METADATA

    child_id: str = ""
    name: str | None = None


class UsageAttributed(Metadata):
    """A child's own usage from one of its turns, added to this session's attributed totals.

    Recorded in every ancestor, so a load rebuilds the attributed totals.
    """

    _role: ClassVar[Role] = Role.METADATA

    child_id: str = ""
    usage: Usage = Field(default_factory=Usage)


class SnapshotRestoreFailed(Metadata):
    """The saved agent state could not be restored when the session was loaded."""

    _role: ClassVar[Role] = Role.METADATA

    error: str = ""


class SessionRecovered(EventBase):
    """This session is a copy of ``forked_from``, made by ``SessionStore.fork``.

    The model sees it on the next turn: ``note`` says what was not carried
    over (items the original never read, its subagent sessions, events a
    damaged file lost). ``copied_by_event`` is set when the file could not
    be copied whole and was copied event by event; ``skipped_events`` then
    counts the events that could not be read, and ``end_unreadable`` says
    the end of the file could not be read, so the newest events may be
    missing without being counted. ``original_title`` is the original's
    title as the copy read it.
    """

    _role: ClassVar[Role] = Role.USER

    forked_from: str
    original_title: str | None = None
    note: str = ""
    copied_by_event: bool = False
    skipped_events: int = 0
    end_unreadable: bool = False


SESSION_EVENT_TYPES: tuple[type[EventBase], ...] = (
    SessionStarted,
    SessionTitleUpdated,
    SessionModeChanged,
    SessionModelChanged,
    SessionReasoningChanged,
    ItemAdmitted,
    ItemConsumed,
    ItemWithdrawn,
    ItemDiscarded,
    ItemRequeued,
    TurnStarted,
    TurnEnded,
    ChildDeleted,
    TurnCancelled,
    SnapshotRestoreFailed,
    UsageAttributed,
    SessionRecovered,
)
"""Event types registered on every session's storage backend.

Registering them per backend (not only through the global registry) keeps
them loading as their own classes even when another package defines a
class with the same name; an unknown type would load back as ``Metadata``
and a model-visible event like ``TurnCancelled`` would drop out of the
prompt.
"""
