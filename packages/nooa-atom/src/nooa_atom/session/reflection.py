# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflection: a quiet housekeeping turn the session queues now and then.

When a root session's turn ends with ``Done`` or ``Waiting``, the session asks
its ``ReflectionTrigger`` whether to queue a reflection: the configured
instructions, delivered as a turn on the agent's ``slash_commands`` channel
(as ``/compact`` is), with source ``"reflection"``. It runs only when no other
input is waiting, at most one is ever pending, and hosts render it as one
status card (``TurnStartedUpdate.reflection``/``TurnEndedUpdate.reflection``).

Settings live in the workspace's ``atom.reflection`` settings
(``ReflectionSettings``); ``SessionOptions.reflection`` overrides ``enabled``
for one run (a benchmark arm). The default is on for interactive hosts only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

REFLECTION_SOURCE = "reflection"
"""The ``source`` of a reflection item on the ``slash_commands`` channel."""

REFLECTION_CHANNEL = "slash_commands"

INTERACTIVE_HOSTS = frozenset({"acp"})
"""Hosts where a person is in the loop: reflection is on there unless disabled."""

SPAN_NAME = "atom.reflection"
REASON_ATTR = "nooa.atom.reflection.reason"
TURNS_ATTR = "nooa.atom.reflection.turns_since_last"
MINUTES_ATTR = "nooa.atom.reflection.minutes_since_last"
CONTEXT_ATTR = "nooa.atom.reflection.context_fraction"

Reason = Literal["turns", "minutes", "context"]

DEFAULT_INSTRUCTIONS = """\
[reflection] Housekeeping turn, not a request from the user. Do not message the user.
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
7. End with `Done(explanation="<one line: what you did>")` and no message. If you were \
Waiting on work, end with the same Waiting instead, its explanation saying what you did."""


class ReflectionSettings(BaseModel):
    """The ``atom.reflection`` settings: when to queue a reflection turn, and its text.

    ``enabled``: ``None`` (the default) turns it on for interactive hosts
    (``INTERACTIVE_HOSTS``: ACP clients such as Pool and Zed)
    and off for headless runs (benchmarks, Gym, ``nooa_bench``); ``True`` or
    ``False`` decides for every host. Only root sessions reflect: a child
    session never does, whatever these settings say. A reflection is due when a turn ends with ``Done``
    or ``Waiting`` and at least ``min_gap_turns`` turns passed since the last
    one, and ``every_turns`` turns or ``every_minutes`` minutes passed, or the
    context window is ``context_fraction`` full. ``None`` turns a trigger off.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    every_turns: int | None = Field(default=20, ge=1)
    every_minutes: float | None = Field(default=60.0, gt=0)
    min_gap_turns: int = Field(default=5, ge=0)
    context_fraction: float | None = Field(default=0.5, gt=0)
    instructions: str = Field(default=DEFAULT_INSTRUCTIONS, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _drop_removed_children(cls, values: Any) -> Any:
        """Ignore the removed ``children`` key, with a warning, so old settings still load."""
        if isinstance(values, dict) and "children" in values:
            values = {key: value for key, value in values.items() if key != "children"}
            logger.warning(
                "atom.reflection.children is no longer supported and is ignored: "
                "only root sessions reflect; remove it from your settings"
            )
        return values


def reflection_enabled(
    settings: ReflectionSettings | None,
    *,
    override: bool | None,
    host: str,
    is_root: bool,
) -> bool:
    """Whether a session reflects: never a child; else an explicit choice wins, else the host.

    ``override`` is ``SessionOptions.reflection``; ``settings`` is ``None`` for
    an agent without Atom settings, which reflects only when told to. A child
    session (``is_root`` false) never reflects, whatever the override or settings.
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
        self.pending = False

    def minutes(self) -> float:
        """Minutes since the window's anchor."""
        return max(0.0, self._clock() - self.anchor) / 60

    def turn_ended(self, kind: str, *, context_fraction: float | None) -> Reason | None:
        """Count a (non-reflection) turn; return why a reflection is due now, else ``None``.

        Only ``Done`` and ``Waiting`` turns can start one; while one is
        pending nothing new is due. The caller sets ``pending`` when it queues.
        """
        if kind in ("done", "waiting", "need_input"):
            self.turns += 1
        if kind not in ("done", "waiting") or self.pending:
            return None
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
        """A reflection turn ended: start the next window."""
        self.turns = 0
        self.anchor = self._clock()
        self.pending = False


def context_fraction(agent: object) -> float | None:
    """The used share of the usable context window, as the agent's context_usage line shows it."""
    stats = getattr(agent, "context_stats", None)
    value = getattr(stats, "overall_utilization", None)
    return value if isinstance(value, int | float) else None


def emit_reflection_span(
    reason: str, *, turns: int, minutes: float, fraction: float | None, session_id: str
) -> None:
    """One ``atom.reflection`` span for a queued reflection; nothing without OpenTelemetry."""
    try:
        from opentelemetry import trace
    except ImportError:
        return
    from nooa_atom.session.tracing import SESSION_ID_ATTR

    attributes: dict[str, str | int | float] = {
        REASON_ATTR: reason,
        TURNS_ATTR: turns,
        MINUTES_ATTR: round(minutes, 3),
        SESSION_ID_ATTR: session_id,
    }
    if fraction is not None:
        attributes[CONTEXT_ATTR] = round(fraction, 4)
    span = trace.get_tracer(__name__).start_span(SPAN_NAME, attributes=attributes)
    span.end()
