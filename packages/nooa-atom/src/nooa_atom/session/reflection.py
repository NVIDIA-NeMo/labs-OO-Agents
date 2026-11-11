# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflection: now and then, a root session asks its agent to tidy up.

Two parts. ``ReflectionBlocks`` is the registry of blocks the message is
made of; every Atom agent owns one from construction as ``agent.reflection``.
A block is a name and either text or a snippet: a Python expression, like a
dynamic context block's ``expr``. A bare registry is empty: the Atom factory
(``create_session_agent()``) registers ``CORE_BLOCKS``, switched on by
default. The triggers belong to the session.

``Reflection`` is the mechanics: a root session with a ``system_messages``
queue owns one. When a turn ends with ``Done`` or ``Waiting`` the session
tells it (``turn_ended``); when a reflection is due and on, the session then
calls ``send()``, which evaluates the snippets of the switched-on blocks with
the evaluator context blocks use and posts the text on ``system_messages``.
The agent handles it in a normal turn.

Settings live in the workspace's ``atom.reflection`` (``ReflectionSettings``),
which ``/reflection`` edits: ``enabled`` and which ``blocks`` are on. A block
starts on if it was registered with ``on=True``, else off. ``SessionOptions.reflection`` overrides
``enabled`` for one run (a benchmark arm). The default is on for ACP hosts.
"""

from __future__ import annotations

import ast
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

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

CORE_BLOCKS: dict[str, str] = {
    "rename": (
        "Rename this session so its title says what you are working on now: "
        '`await self.rename_session("<title>")`.'
    ),
    "compact": (
        "If `context_usage` shows events taking most of the context, compact history with "
        "`self.events.collapse(start_tag, end_tag, summary_text=...)` for ranges you no longer "
        "need in full. Every summary must keep the user's standing rules, open promises and "
        "decisions, and the IDs, paths and commands needed to continue."
    ),
    "variables": (
        "Look at `python_cell_state`: delete Python variables you no longer need with `del`. "
        "Move anything you will need after a restart into `self.v`."
    ),
    "todo": (
        "Tidy `self.todo` (see `todo_status`): complete finished items, delete stale ones, add "
        "new ones for work you've taken on, note what is blocked."
    ),
    "vars": "Delete `self.v` keys for finished work.",
}
"""The core blocks, by name, in message order.

The Atom factory (``create_session_agent()``) registers them, switched on by
default (``on=True``).
"""


class ReflectionSettings(BaseModel):
    """The ``atom.reflection`` settings: whether and when to reflect, and which blocks are on.

    ``enabled``: ``None`` (the default) turns it on for ACP hosts
    (``INTERACTIVE_HOSTS``) and off elsewhere; ``True`` or ``False`` decides
    for every host. Only root sessions reflect. A reflection is due when a
    turn ends with ``Done`` or ``Waiting``, at least ``min_gap_turns`` turns
    passed since the last one, and ``every_turns`` turns or ``every_minutes``
    minutes passed or the context window is ``context_fraction`` full.
    ``None`` turns a trigger off. ``blocks`` switches registered blocks on or
    off by name; a block it does not name is on if it was registered with
    ``on=True``, else off.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    every_turns: int | None = Field(default=20, ge=1)
    every_minutes: float | None = Field(default=60.0, gt=0)
    min_gap_turns: int = Field(default=5, ge=0)
    context_fraction: float | None = Field(default=0.5, gt=0)
    blocks: dict[str, bool] = Field(default_factory=dict)

    def block_on(self, name: str, default: bool) -> bool:
        """Whether block ``name`` is switched on: the user's choice, else its ``default``."""
        return self.blocks.get(name, default)


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


@dataclass(frozen=True)
class ReflectionBlock:
    """One block of the reflection message: fixed ``text``, or a snippet (``expr``).

    ``on`` is its default state, for settings that do not name it.
    """

    name: str
    text: str | None = None
    expr: str | None = None
    on: bool = False


Evaluate = Callable[[str], Awaitable[Any]]
"""Evaluates a snippet (``ActorRuntime.evaluate_expression`` with ``error_mode="raise"``)."""


class ReflectionBlocks:
    """The blocks of the reflection message, by name: an agent's ``agent.reflection``.

    It starts empty; the Atom factory registers ``CORE_BLOCKS``, switched on.
    ``register(name, text=...)`` or ``register(name, expr="...")`` adds a
    block (a known name is replaced in place); ``unregister(name)`` removes one.
    """

    def __init__(self) -> None:
        """Start empty."""
        self._blocks: dict[str, ReflectionBlock] = {}

    def register(
        self, name: str, *, text: str | None = None, expr: str | None = None, on: bool = False
    ) -> None:
        """Register block ``name`` with either ``text`` or a snippet ``expr`` (checked to parse).

        ``on`` is its default state: whether it is sent while the settings do
        not name it.
        """
        if (text is None) == (expr is None):
            raise ValueError("a reflection block needs either text or expr")
        if text is not None and not text.strip():
            raise ValueError("a reflection block needs text")
        if expr is not None:
            try:
                ast.parse(expr.strip(), mode="eval")
            except SyntaxError as exc:
                raise ValueError(f"reflection block {name!r}: not an expression: {expr!r}") from exc
        self._blocks[name] = ReflectionBlock(
            name, text.strip() if text else None, expr.strip() if expr else None, on
        )

    def unregister(self, name: str) -> None:
        """Remove block ``name``; an unknown name is ignored."""
        self._blocks.pop(name, None)

    @property
    def blocks(self) -> dict[str, ReflectionBlock]:
        """The registered blocks by name, in message order (a copy)."""
        return dict(self._blocks)

    async def message(self, settings: ReflectionSettings, evaluate: Evaluate) -> str | None:
        """The message: the header, the switched-on blocks numbered, then the last line.

        A snippet's value is its text; one that raises shows as
        ``ExceptionType: message``, and ``None`` or ``""`` leaves the block out.
        ``None`` when no block has anything to say.
        """
        texts = []
        for block in self._blocks.values():
            if not settings.block_on(block.name, block.on):
                continue
            text = block.text if block.expr is None else await _evaluate(block, evaluate)
            if text:
                texts.append(text)
        if not texts:
            return None
        lines = [HEADER, *(f"{i}. {text}" for i, text in enumerate(texts, 1)), FOOTER]
        return "\n".join(lines)


async def _evaluate(block: ReflectionBlock, evaluate: Evaluate) -> str | None:
    """A snippet's text, as a dynamic context block renders its value (errors inline)."""
    try:
        value = await evaluate(block.expr or "")
    except Exception as exc:
        logger.warning("Reflection block %r failed: %s: %s", block.name, type(exc).__name__, exc)
        return f"{type(exc).__name__}: {exc}"
    if value is None or isinstance(value, str):
        return value
    from nooa.agentdoc import pformat

    return pformat(value, unquote_strings=True)


def runtime_evaluator(agent: Any) -> Evaluate:
    """The evaluator dynamic context blocks use: the agent runtime's ``evaluate_expression``."""

    async def evaluate(expr: str) -> Any:
        return await agent.runtime.evaluate_expression(expr, error_mode="raise")

    return evaluate


class Reflection:
    """One root session's reflection mechanics: when it is due, and sending the message.

    ``settings`` is the agent's ``ReflectionSettings``, read at every turn so
    ``/reflection`` changes count at once; ``enabled`` says whether reflection
    is on now; ``blocks`` is the agent's ``ReflectionBlocks``; ``evaluate``
    evaluates a snippet; ``post`` puts the text on ``system_messages``;
    ``context`` returns the used share of the context window (or ``None``).
    The counters are in memory only: a restarted session starts a fresh
    window. While reflection is off the window stays fresh. ``clock``
    (monotonic seconds) can be injected for tests.
    """

    def __init__(
        self,
        settings: ReflectionSettings,
        *,
        blocks: ReflectionBlocks,
        post: Callable[[str], object],
        evaluate: Evaluate,
        enabled: Callable[[], bool] = lambda: True,
        context: Callable[[], float | None] = lambda: None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Start a fresh window."""
        self.settings = settings
        self.blocks = blocks
        self.enabled = enabled
        self._post = post
        self._evaluate = evaluate
        self._context = context
        self._clock = clock or time.monotonic
        self.pending: Reason | None = None
        self.turns = 0
        self.anchor = self._clock()

    def minutes(self) -> float:
        """Minutes since the window's anchor."""
        return max(0.0, self._clock() - self.anchor) / 60

    def restart(self) -> None:
        """Start a fresh window."""
        self.turns = 0
        self.anchor = self._clock()

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
        """After a turn: when a reflection is due, mark it ``pending`` and start the next window.

        Return why it is due, else ``None``; ``send()`` posts it. While
        reflection is off nothing counts. A failure is logged and ignored
        (``None``): it must not affect the turn that just ended.
        """
        try:
            if not self.enabled():
                self.restart()
                return None
            reason = self.due(kind, context_fraction=self._context())
            if reason is not None:
                self.pending = reason
                self.restart()
            return reason
        except Exception:
            logger.exception("The reflection check failed")
            return None

    async def send(self) -> str | None:
        """Post the pending reflection, its snippets evaluated now; return the text posted.

        Nothing is posted when none is pending or no block has anything to
        say. A failure is logged and ignored.
        """
        if self.pending is None:
            return None
        self.pending = None
        try:
            text = await self.blocks.message(self.settings, self._evaluate)
            if text is not None:
                self._post(text)
            return text
        except Exception:
            logger.exception("Sending the reflection failed")
            return None

    def describe(self) -> str:
        """When the next reflection is due, in words."""
        settings = self.settings
        parts = []
        if settings.every_turns is not None:
            parts.append(f"in {max(settings.every_turns - self.turns, 1)} turns")
        if settings.every_minutes is not None:
            left = max(settings.every_minutes - self.minutes(), 0)
            parts.append(f"in {left:.0f} minutes")
        if settings.context_fraction is not None:
            parts.append(f"at {settings.context_fraction:.0%} context")
        if not parts:
            return "never: every trigger is off"
        text = ", or ".join(parts)
        gap = settings.min_gap_turns - self.turns
        return text + (f" (not before {gap} more turns)" if gap > 1 else "")
