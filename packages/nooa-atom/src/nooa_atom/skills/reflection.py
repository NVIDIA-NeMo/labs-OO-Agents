# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflection: now and then, a root session asks its agent to tidy up.

``Reflection`` is a skill the Atom factory (``create_session_agent()``)
registers as ``nemo.reflection`` with ``CORE_BLOCKS``. After each settled
turn the session calls ``after_turn()``; when a reflection is due it posts
the message on ``system_messages``, an ordinary item the agent handles in a
normal turn. Settings: ``atom.reflection`` (``ReflectionConfig``), which
``/reflection`` edits.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nooa.agentdoc import hidden
from nooa.skill import Skill, slash_command

logger = logging.getLogger(__name__)

INTERACTIVE_HOSTS = frozenset({"acp"})
"""Hosts where a person is in the loop: reflection is on there unless disabled."""

HEADER = (
    "[reflection] A housekeeping message from the session, not a request from the user. "
    "Tidy up so you can keep working well:"
)
FOOTER = "Then finish with a short Done."
USAGE = "Usage: /reflection [on|off|config [BLOCK on|off]]"

Block = str | Callable[[Any], str | None]
"""Fixed text, or a function of the agent; ``None`` or ``""`` leaves the block out."""


class ReflectionConfig(BaseModel):
    """The ``atom.reflection`` settings.

    ``enabled``: ``None`` is on for ACP hosts, off elsewhere. A reflection is
    due after a ``Done`` or ``Waiting`` turn when ``min_gap_turns`` turns
    passed since the last one and ``every_turns`` turns or ``every_minutes``
    minutes passed or the context window is ``context_fraction`` full
    (``None`` turns a trigger off). ``blocks`` switches blocks off (or on) by
    name; a block it does not name is on.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    every_turns: int | None = Field(default=20, ge=1)
    every_minutes: float | None = Field(default=60.0, gt=0)
    min_gap_turns: int = Field(default=5, ge=0)
    context_fraction: float | None = Field(default=0.5, gt=0)
    blocks: dict[str, bool] = Field(default_factory=dict)


def compact_block(agent: Any) -> str:
    """The context fill and its split, then how to compact."""
    action = (
        "Compact ranges you no longer need with "
        "`self.events.collapse(start_tag, end_tag, summary_text=...)`; every summary keeps the "
        "user's standing rules, open promises, decisions, and the IDs, paths and commands "
        "needed to continue."
    )
    stats = getattr(agent, "context_stats", None)
    if stats is None or stats.prompt_tokens is None:
        return action
    fill = stats.overall_utilization
    used = f"{stats.prompt_tokens:,}" + (
        f" / {stats.effective_window:,} usable tokens" if stats.effective_window else " tokens"
    )
    return (
        (f"Context is {fill:.0%} full ({used}): " if fill is not None else f"Context: {used}: ")
        + f"events ~{stats.events_tokens:,} tokens in {stats.events_count} events, "
        f"context blocks ~{stats.context_blocks_tokens:,} in {stats.context_blocks_count} "
        f"blocks. {action}"
    )


def variables_block(agent: Any, few: int = 10) -> str | None:
    """How many Python variables the cells keep (none when fewer than ``few``)."""
    # Private: the session namespace python_cell_state lists; there is no public accessor.
    namespace = getattr(agent, "_handle_namespace", None)
    if namespace is None:
        return None
    count = sum(name in namespace.globals for name in namespace.user_names)
    if count < few:
        return None
    return (
        f"{count} Python variables are retained (see `python_cell_state`). `del` the ones you "
        "no longer need; move what must survive a restart into `self.v`."
    )


def todo_block(agent: Any) -> str | None:
    """How many todos are open and the age of the oldest (none when none are)."""
    todos = [t for t in agent.todo.list_todos() if t.status != "done"]
    if not todos:
        return None
    oldest = min(t.created_at for t in todos)  # "%Y-%m-%d %H:%M" sorts as text
    days = (datetime.now() - datetime.strptime(oldest, "%Y-%m-%d %H:%M")).days
    age = f"{days} days ago" if days > 1 else "1 day ago" if days == 1 else "today"
    return (
        f"{len(todos)} open todos in `self.todo`, the oldest opened {oldest} ({age}). Complete "
        "finished items, delete stale ones, add new ones for work you've taken on, note what "
        "is blocked."
    )


def vars_block(agent: Any) -> str | None:
    """How many ``self.v`` keys there are (none when empty)."""
    count = len(agent.vars)
    return f"`self.v` holds {count} keys: delete those for finished work." if count else None


CORE_BLOCKS: dict[str, Block] = {
    "rename": (
        "Rename this session so its title says what you are working on now: "
        '`await self.rename_session("<title>")`.'
    ),
    "compact": compact_block,
    "variables": variables_block,
    "todo": todo_block,
    "vars": vars_block,
}
"""The core blocks, in message order; the Atom factory registers them."""


class Reflection(Skill):
    """Periodic housekeeping messages for this session; the user controls them with /reflection."""

    def __init__(
        self,
        config: ReflectionConfig | None = None,
        *,
        workspace: str | Path = ".",
        host: str = "",
        override: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``override`` is ``SessionOptions.reflection``; ``clock`` is for tests."""
        self.config = config or ReflectionConfig()
        self.blocks: dict[str, Block] = {}
        self.workspace = Path(workspace)
        self.host = host
        self.override = override
        self._clock = clock
        self._answering = False  # posted one: the next turn answers it
        self._restart()

    @hidden
    def enabled(self) -> bool:
        """The session override, else ``enabled``, else on for ACP hosts."""
        for choice in (self.override, self.config.enabled):
            if choice is not None:
                return choice
        return self.host in INTERACTIVE_HOSTS

    def _restart(self) -> None:
        self.turns = 0
        self.anchor = self._clock()

    def _due(self) -> bool:
        c = self.config
        if self.turns < c.min_gap_turns:
            return False
        stats = getattr(self._agent, "context_stats", None)
        fill = getattr(stats, "overall_utilization", None)
        return (
            (c.every_turns is not None and self.turns >= c.every_turns)
            or (c.every_minutes is not None and self._minutes() >= c.every_minutes)
            or (c.context_fraction is not None and fill is not None and fill >= c.context_fraction)
        )

    def _minutes(self) -> float:
        return max(0.0, self._clock() - self.anchor) / 60

    @hidden
    def after_turn(self, kind: str, *, waiting: bool, post: Callable[[str], object]) -> None:
        """Count a settled turn; ``post`` the message when a reflection is due.

        The turn that answers a reflection does not count. While a prompt is
        ``waiting`` a due reflection waits too. Errors are logged, never raised.
        """
        try:
            if self._answering:
                self._answering = False
                return
            if not self.enabled():
                self._restart()
                return
            if kind not in ("done", "waiting"):
                return
            self.turns += 1
            if waiting or not self._due():
                return
            self._restart()
            text = self.message()
            if text:
                post(text)
                self._answering = True
        except Exception:
            logger.exception("Reflection failed after a turn")

    @hidden
    def message(self) -> str | None:
        """The header, the switched-on blocks numbered, then the footer; ``None`` if empty."""
        texts = []
        for name, block in self.blocks.items():
            if not self.config.blocks.get(name, True):
                continue
            try:
                text = block(self._agent) if callable(block) else block
                if inspect.iscoroutine(text):
                    text.close()
                    raise TypeError("blocks must be sync, got a coroutine")
                if text is not None and not isinstance(text, str):
                    raise TypeError(f"returned {type(text).__name__}, not str")
            except Exception as exc:
                logger.exception("Reflection block %r failed", name)
                text = f"(Block {name!r} failed: {type(exc).__name__}: {exc})"
            if text:
                texts.append(text)
        if not texts:
            return None
        return "\n".join([HEADER, *(f"{i}. {t}" for i, t in enumerate(texts, 1)), FOOTER])

    @slash_command(
        "reflection",
        argument_hint="[on|off|config [BLOCK on|off]]",
        completions=("on", "off", "config"),
        output_to_agent=False,
    )
    async def reflection_command(self, args: str) -> str:
        """Show reflection, turn it on or off, or switch its blocks."""
        words = args.split()
        if not words:
            return self._status()
        if words == ["config"]:
            return self._config()
        if len(words) == 1 and words[0] in ("on", "off"):
            on = words[0] == "on"
            saved = self._save(("enabled",), on)
            if not saved.startswith("saved"):
                return saved
            self.config.enabled = on
            note = ""
            if self.override is not None:
                note = f" This session's override (reflection={self.override}) is cleared."
                self.override = None
            return f"Reflection is {words[0]} ({saved}).{note}"
        if len(words) == 3 and words[0] == "config" and words[2] in ("on", "off"):
            name, on = words[1], words[2] == "on"
            if name not in self.blocks:
                return f"No reflection block {name!r}. Blocks: {', '.join(self.blocks)}"
            saved = self._save(("blocks", name), on)
            if not saved.startswith("saved"):
                return saved
            self.config.blocks[name] = on
            return f"Reflection block {name} is {words[2]} ({saved})."
        return USAGE

    def _save(self, key: tuple[str, ...], value: object) -> str:
        """Persist first: ``saved to PATH``, or why not (then nothing changes)."""
        from nooa_atom.workspace.settings import write_settings_updates

        try:
            path, _ = write_settings_updates(
                {("atom", "reflection", *key): value}, workspace=self.workspace
            )
        except Exception as exc:
            logger.exception("Saving the reflection settings failed")
            return f"Could not save the setting ({type(exc).__name__}: {exc}); nothing changed."
        return f"saved to {path}"

    def _status(self) -> str:
        c = self.config
        on = [name for name in self.blocks if c.blocks.get(name, True)]
        triggers = [
            f"every {c.every_turns} turns" if c.every_turns else "",
            f"every {c.every_minutes:g} minutes" if c.every_minutes else "",
            f"at {c.context_fraction:.0%} context" if c.context_fraction else "",
        ]
        return "\n".join(
            [
                f"Reflection: {'on' if self.enabled() else 'off'}",
                f"Blocks on: {', '.join(on) or 'none'}",
                f"Triggers: {', '.join(t for t in triggers if t) or 'none'}; "
                f"at least {c.min_gap_turns} turns apart",
                f"Since the last: {self.turns} turns, {self._minutes():.0f} minutes",
                "Change blocks with /reflection config; triggers in atom.reflection.",
            ]
        )

    def _config(self) -> str:
        width = max((len(name) for name in self.blocks), default=0)
        lines = ["Reflection blocks (/reflection config BLOCK on|off):"]
        for name, block in self.blocks.items():
            state = "on " if self.config.blocks.get(name, True) else "off"
            what = block if isinstance(block, str) else f"dynamic: {block.__doc__ or ''}"
            what = what.strip().split("\n")[0]
            lines.append(
                f"  [{state}] {name:<{width}}  {what[:57] + '...' if len(what) > 60 else what}"
            )
        return "```text\n" + "\n".join(lines) + "\n```"
