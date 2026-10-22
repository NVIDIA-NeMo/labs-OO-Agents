# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An agent's live child sessions, for the per-turn "Background work" list and ``Waiting``."""

from typing import Any

from pydantic import BaseModel

from nooa_atom.session.items import ChildQuestion

# One line under "Background work" while a child is listed, like the jobs cancel line.
CLOSE_HINT = "Close a child: await self.child('<name>').close()"

# Shown for a child per status; ``working``: its last turn ended Waiting on its own job.
_SHOWN = {"running": "running", "working": "waiting on its own job", "retained": "retained"}


class LiveChild(BaseModel):
    """One of this session's loaded, open children that can still deliver or answer."""

    id: str
    name: str
    state: str  # what the status list shows: running, retained, has a question for you, ...
    active: bool  # running, working, or has input queued: something will arrive from it


def live_children(agent: Any) -> list[LiveChild]:
    """This session's loaded children that are not closed: running, working, retained, or queued.

    Reads only the sessions in memory (no store listing), so it is cheap
    enough for the per-turn status block. Idle non-kept children with
    nothing queued are left out.
    """
    port = getattr(agent, "session", None)
    if port is None:
        return []
    registry = port._registry
    asked = _asked(agent)
    found = []
    for session in list(registry.sessions.values()):
        if session.parent_id != port.id or session.closing:
            continue
        status = registry.live_info(session).status
        queued = session.has_queued_input()
        active = status in ("running", "working") or queued
        if status == "closed" or not (active or status == "retained" or session.id in asked):
            continue
        if session.id in asked:
            state = "has a question for you"
        elif status in _SHOWN:
            state = _SHOWN[status]
        else:
            state = "input queued"
        found.append(
            LiveChild(id=session.id, name=session.info.name or "", state=state, active=active)
        )
    return found


def child_status_lines(agent: Any) -> list[str]:
    """``<name> · child · <state>`` for each live child, for the "Background work" list."""
    return [f"{child.name or child.id} · child · {child.state}" for child in live_children(agent)]


def _asked(agent: Any) -> set[str]:
    """Ids of children with an unread ``ChildQuestion`` buffered on ``delegates``."""
    channel = agent.queue_manager.channels().get("delegates")
    if channel is None or channel.mode != "queue":
        return set()
    return {item.child.id for item in channel.snapshot() if isinstance(item, ChildQuestion)}
