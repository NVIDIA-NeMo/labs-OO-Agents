# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An agent's live child sessions, for the per-turn ``queues`` block and ``Waiting``."""

import time
from typing import Any

from pydantic import BaseModel

from nooa_atom.session.items import ChildQuestion

# One line after the status sections while a child is listed, like the jobs cancel line.
CLOSE_HINT = "Close a child: await self.child('<name>').close()"

# Shown for a child per status; ``working``: its last turn ended Waiting on its own job.
_SHOWN = {"working": "waiting on its own job", "retained": "retained"}
# How to read each channel listed under "Unread:" (the queues block's "Read now:" line).
READ_HINTS = {
    "user_messages": "await self.user_messages.get()",
    "delegates": "await self.delegates.get()",
    "jobs": "job updates come in next turn's notification",
}
# Its line for a question that was read (taken off ``delegates``) but not answered yet.
UNANSWERED = "question not answered yet"


class LiveChild(BaseModel):
    """One of this session's loaded, open children that can still deliver or answer."""

    id: str
    name: str
    path: str
    # What the status list shows after "child": the turn's age (running; "" when
    # unknown), "waiting on its own job", "retained", "input queued", "question
    # not answered yet", or "asks" (an unread ChildQuestion: listed by that item).
    state: str
    active: bool  # running, working, or has input queued: something will arrive from it
    section: str  # "unread", "running" or "idle": where the queues block lists it


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
        unanswered = session.id not in asked and _unanswered(port, session.id)
        if status == "closed" or not (
            active or status == "retained" or session.id in asked or unanswered
        ):
            continue
        if session.id in asked:
            state, section = "asks", "unread"
        elif unanswered and not active:
            state, section = UNANSWERED, "unread"
        elif status == "running":
            state, section = _turn_age(session), "running"
        elif status in _SHOWN:
            state = _SHOWN[status]
            section = "idle" if status == "retained" else "running"
        else:
            state, section = "input queued", "running"
        found.append(
            LiveChild(
                id=session.id,
                name=session.name or "",
                path=registry.path(session.id),
                state=state,
                active=active,
                section=section,
            )
        )
    return found


def child_sections(agent: Any) -> dict[str, list[str]]:
    """The live children's ``queues`` lines by section: ``unread``, ``running``, ``idle``.

    Each line is ``<path> · child · <state>`` (a running child's age is left
    out when its turn start is unknown). A child with an unread question has
    no line: it is counted, by path, in the ``delegates`` line under
    "Unread:"; ``asking`` holds those paths. One whose question was read but
    not answered is under ``unread``.
    """
    sections: dict[str, list[str]] = {"unread": [], "running": [], "idle": [], "asking": []}
    for child in live_children(agent):
        if child.state == "asks":
            sections["asking"].append(child.path)
            continue
        parts = (child.path, "child", child.state)
        sections[child.section].append(" · ".join(part for part in parts if part))
    return sections


def _turn_age(session: Any) -> str:
    """How long the child's running turn has run (``4s``, ``3m``, ``1h05m``); "" if unknown."""
    started = getattr(session, "turn_started_at", None)
    if started is None:
        return ""
    seconds = max(0, int(time.monotonic() - started))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _unanswered(port: Any, child_id: str) -> bool:
    """Whether the child has a question its parent read but did not answer yet."""
    pending = getattr(port, "pending_question", None)
    return callable(pending) and pending(child_id) is not None


def _asked(agent: Any) -> set[str]:
    """Ids of children with an unread ``ChildQuestion`` buffered on ``delegates``."""
    channel = agent.queue_manager.channels().get("delegates")
    if channel is None or channel.mode != "queue":
        return set()
    return {item.child.id for item in channel.snapshot() if isinstance(item, ChildQuestion)}
