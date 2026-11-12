# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fix hints for model-facing errors, registered per agent class.

A validator that blocks something says what to use instead, but only the
agent (or a tool it holds) knows that alternative. Whoever provides it
registers the hint for the agent classes that have it::

    register_hint(RUN_COMMAND, "use `await self.tool.run(cmd)`", owner=MyAgent)

``lookup_hint(kind, agent)`` walks the agent class's MRO: the most derived
registered owner wins, so a subclass can replace its base's hint. An agent
whose classes registered nothing gets no hint. A hint is a string, or a
callable ``(agent) -> str | None`` for text that depends on the instance.

Kinds:

- :data:`RUN_COMMAND`: how to run a shell command; shown for a blocked
  ``subprocess``/``os.system``/``os.popen`` and as the ``curl`` fallback of a
  blocked network module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["RUN_COMMAND", "Hint", "lookup_hint", "register_hint"]

RUN_COMMAND = "run_command"

Hint = str | Callable[[Any], str | None]

# (kind, owner class) -> hint
_HINTS: dict[tuple[str, type], Hint] = {}


def register_hint(kind: str, hint: Hint, *, owner: type) -> None:
    """Register ``hint`` for ``kind`` on agent class ``owner`` and its subclasses.

    Registering the same ``(kind, owner)`` again replaces the hint.
    """
    _HINTS[(kind, owner)] = hint


def has_own_hint(kind: str, agent: Any) -> bool:
    """True when the agent's class (or a base) has a ``kind`` hint registered."""
    return _find(kind, agent) is not None


def lookup_hint(kind: str, agent: Any) -> str | None:
    """The ``kind`` hint for ``agent`` (an instance or a class), or None."""
    hint = _find(kind, agent)
    if hint is None or isinstance(hint, str):
        return hint
    try:
        return hint(agent)
    except Exception:  # noqa: BLE001 - a hint must never break the error it decorates
        return None


def _find(kind: str, agent: Any) -> Hint | None:
    if agent is None:
        return None
    agent_type = agent if isinstance(agent, type) else type(agent)
    for owner in agent_type.__mro__:
        hint = _HINTS.get((kind, owner))
        if hint is not None:
            return hint
    return None
