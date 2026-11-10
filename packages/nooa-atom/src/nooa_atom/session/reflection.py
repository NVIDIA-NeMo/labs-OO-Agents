# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflection: now and then, a root session asks its agent to tidy up.

When a root session's turn ends with ``Done`` or ``Waiting``, the session
counts it (``ReflectionTrigger``). When a reflection is due, it puts the
configured instructions on the agent's ``system_messages`` queue as an
ordinary item, and the agent handles them in a normal turn.

Settings live in the workspace's ``atom.reflection`` settings
(``ReflectionSettings``); ``SessionOptions.reflection`` overrides ``enabled``
for one run (a benchmark arm). The default is on for ACP hosts only.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REFLECTION_SOURCE = "reflection"
"""The ``source`` of a reflection item."""

REFLECTION_CHANNEL = "system_messages"
"""The queue the instructions go on; without it a session does not reflect."""

INTERACTIVE_HOSTS = frozenset({"acp"})
"""Hosts where a person is in the loop: reflection is on there unless disabled."""

Reason = Literal["turns", "minutes", "context"]

DEFAULT_INSTRUCTIONS = """\
[reflection] A housekeeping message from the session, not a request from the user. \
Tidy up so you can keep working well:
1. Rename this session so its title says what you are working on now: \
`await self.rename_session("<title>")`.
2. Compact history with `self.events.collapse(start_tag, end_tag, summary_text=...)` for \
ranges you no longer need in full. Every summary must keep the user's standing rules, open \
promises and decisions, and the IDs, paths and commands needed to continue.
3. Delete Python variables you no longer need with `del`. Move anything you will need after \
a restart into `self.v`.
4. Tidy `self.todo`: complete finished items, delete stale ones, note what is blocked.
5. For each running job you still need, make sure `self.v` says how to relaunch it after a \
restart. Cancel jobs nobody is waiting for.
6. Delete `self.v` keys for finished work.
Then finish with a short Done."""


class ReflectionSettings(BaseModel):
    """The ``atom.reflection`` settings: when to send the reflection message, and its text.

    ``enabled``: ``None`` (the default) turns it on for ACP hosts
    (``INTERACTIVE_HOSTS``) and off elsewhere; ``True`` or ``False`` decides
    for every host. Only root sessions reflect. A reflection is due when a
    turn ends with ``Done`` or ``Waiting``, at least ``min_gap_turns`` turns
    passed since the last one, and ``every_turns`` turns or ``every_minutes``
    minutes passed or the context window is ``context_fraction`` full.
    ``None`` turns a trigger off. ``instructions`` is the message sent.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    every_turns: int | None = Field(default=20, ge=1)
    every_minutes: float | None = Field(default=60.0, gt=0)
    min_gap_turns: int = Field(default=5, ge=0)
    context_fraction: float | None = Field(default=0.5, gt=0)
    instructions: str = Field(default=DEFAULT_INSTRUCTIONS, min_length=1)


def reflection_enabled(
    settings: ReflectionSettings | None,
    *,
    override: bool | None,
    host: str,
    is_root: bool,
) -> bool:
    """Whether a session reflects: never a child; else an explicit choice wins, else the host.

    ``override`` is ``SessionOptions.reflection``; ``settings`` is ``None`` for
    an agent without Atom settings, which reflects only when told to.
    """
    if not is_root:
        return False
    if override is not None:
        return override
    if settings is None:
        return False
    if settings.enabled is not None:
        return settings.enabled
    return host in INTERACTIVE_HOSTS


def has_queue(agent: object, name: str) -> bool:
    """Whether ``agent`` has a queue-mode channel called ``name``."""
    manager = getattr(agent, "queue_manager", None)
    channel = manager.channels().get(name) if manager is not None else None
    return channel is not None and channel.mode == "queue"


class ReflectionTrigger:
    """Counts turns and time since the last reflection and says when the next is due.

    In memory only: a restarted session starts a fresh window. ``clock``
    (monotonic seconds) can be injected for tests.
    """

    def __init__(
        self, settings: ReflectionSettings, *, clock: Callable[[], float] | None = None
    ) -> None:
        """Start a fresh window: no turns counted, anchored now."""
        self.settings = settings
        self._clock = clock or time.monotonic
        self.turns = 0
        self.anchor = self._clock()

    def minutes(self) -> float:
        """Minutes since the window's anchor."""
        return max(0.0, self._clock() - self.anchor) / 60

    def turn_ended(self, kind: str, *, context_fraction: float | None) -> Reason | None:
        """Count a ``Done`` or ``Waiting`` turn; return why a reflection is due now, else ``None``."""
        if kind not in ("done", "waiting"):
            return None
        self.turns += 1
        settings = self.settings
        if self.turns < settings.min_gap_turns:
            return None
        if settings.every_turns is not None and self.turns >= settings.every_turns:
            return "turns"
        if settings.every_minutes is not None and self.minutes() >= settings.every_minutes:
            return "minutes"
        if (
            settings.context_fraction is not None
            and context_fraction is not None
            and context_fraction >= settings.context_fraction
        ):
            return "context"
        return None

    def reflected(self) -> None:
        """A reflection was sent: start the next window."""
        self.turns = 0
        self.anchor = self._clock()


def context_fraction(agent: object) -> float | None:
    """The used share of the usable context window, as the agent's context_usage line shows it."""
    stats = getattr(agent, "context_stats", None)
    value = getattr(stats, "overall_utilization", None)
    return value if isinstance(value, int | float) else None
