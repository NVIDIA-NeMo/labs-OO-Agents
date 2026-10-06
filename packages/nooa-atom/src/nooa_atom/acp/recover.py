# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``/recover``: list the sessions marked in use, or copy one into a new session.

A session whose owner crashed stays marked in use: it is left out of the
session list and cannot be opened. ``/recover`` forks it
(``SessionStore.fork``); the original file is never written.
"""

from __future__ import annotations

import shlex
from collections.abc import Collection
from datetime import datetime
from pathlib import Path

from nooa_atom.session.items import CommandInfo, InUseSession
from nooa_atom.session.store import SessionStore

COMMAND = CommandInfo(
    name="recover",
    description=(
        "Continue a session that is marked in use (for example after a crash) in a new "
        "copy; the original is not changed. Without an argument, list those sessions."
    ),
    input_hint="[session id, id prefix or title]",
)

NONE_IN_USE = "No session in this workspace is marked in use."


def recover(
    store: SessionStore, workspace: str | Path | None, raw_args: str, *, open_here: Collection[str]
) -> str:
    """The reply to ``/recover raw_args`` for the sessions of ``workspace`` in ``store``.

    ``open_here`` are the ids of the sessions this process runs; they are
    not in use by anyone else and are never listed or forked.
    """
    held = [session for session in store.in_use(workspace=workspace) if session.id not in open_here]
    try:
        words = shlex.split(raw_args)
    except ValueError:
        words = raw_args.split()
    argument = " ".join(words)
    if not argument:
        return _listing(held)

    in_use = {session.id: _title(session.title, session.id) for session in held}
    free = {info.id: _title(info.title, info.id) for info in store.list(workspace=workspace)}
    candidates = {**free, **in_use}
    matches = [argument] if argument in candidates else []
    matches = matches or [
        session_id
        for session_id, title in candidates.items()
        if session_id.startswith(argument) or title == argument
    ]
    if not matches:
        return f'No session in this workspace matches "{argument}".'
    if len(matches) > 1:
        lines = [f'"{argument}" matches {len(matches)} sessions. Give more of the id:']
        lines += [f"- {candidates[session_id]} ({session_id})" for session_id in sorted(matches)]
        return "\n".join(lines)
    [session_id] = matches
    title = candidates[session_id]
    if session_id in open_here:
        return f'"{title}" is open here; it does not need recovering.'
    if session_id not in in_use:
        return f'"{title}" is not in use. Open /resume to continue it.'
    fork = store.fork(session_id)
    [(_, recovered)] = store.load_rows(fork.id, frozenset({"SessionRecovered"}))
    # The title as the copy read it: a damaged file lists with no title.
    title = _title(_optional(recovered.get("original_title")), session_id)
    reply = f'Recovered "{title}" as "{fork.title}". Open /resume to continue it.'
    if recovered.get("copied_by_event"):
        reply += " " + _damage(
            int(recovered.get("skipped_events") or 0), bool(recovered.get("end_unreadable"))
        )
    return reply


def _listing(held: list[InUseSession]) -> str:
    if not held:
        return NONE_IN_USE
    short = _short_ids([session.id for session in held])
    rows = [
        [
            _title(session.title, session.id),
            short[session.id],
            session.owner,
            datetime.fromtimestamp(session.last_write).strftime("%Y-%m-%d %H:%M"),
        ]
        for session in held
    ]
    header = ["Title", "Id", "Owner", "Last write"]
    widths = [max(len(row[column]) for row in [header, *rows]) for column in range(len(header))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in [header, *rows]
    ]
    table = "\n".join(lines)
    return (
        "Sessions marked in use in this workspace:\n"
        f"```text\n{table}\n```\n"
        "Run /recover with an id or a title to continue one of them in a new copy. "
        "The original is not changed."
    )


def _short_ids(ids: list[str]) -> dict[str, str]:
    """The first 8 characters of each id, or more where that is not unique."""
    short = {}
    for session_id in ids:
        length = 8
        while any(
            other != session_id and other.startswith(session_id[:length]) for other in ids
        ) and length < len(session_id):
            length += 1
        short[session_id] = session_id[:length]
    return short


def _optional(value: object) -> str | None:
    return str(value) if value else None


def _title(title: str | None, session_id: str) -> str:
    return title or f"Untitled session [{session_id[:8]}]"


def _damage(skipped: int, end_unreadable: bool) -> str:
    parts = []
    if skipped:
        parts.append(f"{skipped} events could not be read and were left out")
    if end_unreadable:
        parts.append("the end of the file could not be read, so the newest events may be missing")
    detail = "; ".join(parts) if parts else "no events were lost"
    return (
        f"The original file could not be copied whole, so it was copied event by event: {detail}."
    )


__all__ = ["COMMAND", "NONE_IN_USE", "recover"]
