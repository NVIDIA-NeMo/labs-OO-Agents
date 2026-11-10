# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflection: now and then, a root session asks its agent to tidy up.

Two parts. ``ReflectionBullets`` is the registry of bullets the message is
made of; every Atom agent owns one from construction as ``agent.reflection``.
``Reflection`` is the mechanics: a root session with a ``system_messages``
queue owns one; when a turn ends with ``Done`` or ``Waiting``, the session
tells it (``Reflection.turn_ended``), and when a reflection is due it posts
``agent.reflection.message()`` on ``system_messages`` as an ordinary item.
The agent handles it in a normal turn.

The message is a fixed header, the registered bullets numbered in
registration order, and a fixed last line. The registry starts with
``DEFAULT_BULLETS``; ``create_session_agent()`` applies the workspace's
``atom.reflection.bullets`` (add, replace, or remove with an empty string)
before any skill attaches, and a skill can ``add`` its own bullet in
``attach()`` and ``remove`` it in ``detach()``. A session that does not
reflect still has ``agent.reflection``; it just never posts.

Settings live in the workspace's ``atom.reflection`` settings
(``ReflectionSettings``); ``SessionOptions.reflection`` overrides ``enabled``
for one run (a benchmark arm). The default is on for ACP hosts only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

REFLECTION_SOURCE = "reflection"
"""The ``source`` of a reflection item."""

REFLECTION_CHANNEL = "system_messages"
"""The queue the message goes on; without it a session does not reflect."""

INTERACTIVE_HOSTS = frozenset({"acp"})
"""Hosts where a person is in the loop: reflection is on there unless disabled."""

Reason = Literal["turns", "minutes", "context"]

HEADER = (
    "[reflection] A housekeeping message from the session, not a request from the user. "
    "Tidy up so you can keep working well:"
)
FOOTER = "Then finish with a short Done."

DEFAULT_BULLETS: dict[str, str] = {
    "rename": (
        "Rename this session so its title says what you are working on now: "
        '`await self.rename_session("<title>")`.'
    ),
    "compact": (
        "Compact history with `self.events.collapse(start_tag, end_tag, summary_text=...)` for "
        "ranges you no longer need in full. Every summary must keep the user's standing rules, "
        "open promises and decisions, and the IDs, paths and commands needed to continue."
    ),
    "variables": (
        "Delete Python variables you no longer need with `del`. Move anything you will need "
        "after a restart into `self.v`."
    ),
    "todo": (
        "Tidy `self.todo`: complete finished items, delete stale ones, add new ones for work "
        "you've taken on, note what is blocked."
    ),
    "vars": "Delete `self.v` keys for finished work.",
}
"""The bullets core registers, by name, in message order."""


class ReflectionSettings(BaseModel):
    """The ``atom.reflection`` settings: when to send the reflection message, and its bullets.

    ``enabled``: ``None`` (the default) turns it on for ACP hosts
    (``INTERACTIVE_HOSTS``) and off elsewhere; ``True`` or ``False`` decides
    for every host. Only root sessions reflect. A reflection is due when a
    turn ends with ``Done`` or ``Waiting``, at least ``min_gap_turns`` turns
    passed since the last one, and ``every_turns`` turns or ``every_minutes``
    minutes passed or the context window is ``context_fraction`` full.
    ``None`` turns a trigger off. ``bullets`` adds or replaces bullets by
    name (a new name goes last); an empty string removes one.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    every_turns: int | None = Field(default=20, ge=1)
    every_minutes: float | None = Field(default=60.0, gt=0)
    min_gap_turns: int = Field(default=5, ge=0)
    context_fraction: float | None = Field(default=0.5, gt=0)
    bullets: dict[str, str] = Field(default_factory=dict)


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


def context_fraction(agent: object) -> float | None:
    """The used share of the usable context window, as the agent's context_usage line shows it."""
    stats = getattr(agent, "context_stats", None)
    value = getattr(stats, "overall_utilization", None)
    return value if isinstance(value, int | float) else None


class ReflectionBullets:
    """The bullets of the reflection message, by name: an agent's ``agent.reflection``.

    It starts with ``DEFAULT_BULLETS``; ``overrides`` (the settings'
    ``bullets``) then adds or replaces bullets, and an empty string removes one.
    """

    def __init__(self, overrides: Mapping[str, str] | None = None) -> None:
        """Register the default bullets, then apply ``overrides``."""
        self._bullets: dict[str, str] = dict(DEFAULT_BULLETS)
        self.apply(overrides or {})

    def apply(self, overrides: Mapping[str, str]) -> None:
        """Add or replace each named bullet; an empty string removes it."""
        for name, text in overrides.items():
            if text:
                self.add(name, text)
            else:
                self.remove(name)

    def add(self, name: str, text: str) -> None:
        """Register bullet ``name``; a new name goes last, a known one keeps its place."""
        if not text.strip():
            raise ValueError("a reflection bullet needs text")
        self._bullets[name] = text.strip()

    def remove(self, name: str) -> None:
        """Unregister bullet ``name``; an unknown name is ignored."""
        self._bullets.pop(name, None)

    @property
    def bullets(self) -> dict[str, str]:
        """The registered bullets by name, in message order (a copy)."""
        return dict(self._bullets)

    def message(self) -> str:
        """The header, the bullets numbered in registration order, then the last line."""
        lines = [HEADER]
        lines += [f"{i}. {text}" for i, text in enumerate(self._bullets.values(), 1)]
        lines.append(FOOTER)
        return "\n".join(lines)


class Reflection:
    """One root session's reflection mechanics: when it is due, and posting the message.

    ``bullets`` is the agent's ``ReflectionBullets`` (read when a reflection
    is due, so later changes count); ``post`` puts the message on the agent's
    ``system_messages`` queue; ``context`` returns the used share of the
    context window (or ``None``). The counters are in memory only: a
    restarted session starts a fresh window. ``clock`` (monotonic seconds)
    can be injected for tests.
    """

    def __init__(
        self,
        settings: ReflectionSettings,
        *,
        bullets: ReflectionBullets,
        post: Callable[[str], object],
        context: Callable[[], float | None] = lambda: None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Start a fresh window."""
        self.settings = settings
        self.bullets = bullets
        self._post = post
        self._context = context
        self._clock = clock or time.monotonic
        self.turns = 0
        self.anchor = self._clock()

    def minutes(self) -> float:
        """Minutes since the window's anchor."""
        return max(0.0, self._clock() - self.anchor) / 60

    def due(self, kind: str, *, context_fraction: float | None) -> Reason | None:
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

    def turn_ended(self, kind: str) -> Reason | None:
        """After a turn: when a reflection is due, post the message and start the next window.

        Return why it was posted, else ``None``. A failure is logged and
        ignored (``None``): it must not affect the turn that just ended.
        """
        try:
            reason = self.due(kind, context_fraction=self._context())
            if reason is not None:
                self._post(self.bullets.message())
                self.turns = 0
                self.anchor = self._clock()
            return reason
        except Exception:
            logger.exception("The reflection check failed")
            return None
