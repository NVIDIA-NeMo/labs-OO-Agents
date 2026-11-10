# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``session/list``: a read-only scan of the session stores.

Imports only the ``acp`` library and the session store, so the router can
answer ``session/list`` without loading the adapter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from acp import RequestError
from acp.schema import ListSessionsResponse
from acp.schema import SessionInfo as ACPSessionInfo

from nooa_atom.session.store import SessionStore

SESSION_PAGE_SIZE = 50


def validate_workspace(cwd: str, additional_directories: list[str] | None) -> Path:
    """The resolved workspace directory ``cwd`` names; ``invalid_params`` if it is not one."""
    if additional_directories:
        raise RequestError.invalid_params({"reason": "Additional directories are not supported"})
    root = Path(cwd).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise RequestError.invalid_params(
            {"cwd": cwd, "reason": "cwd must be an existing absolute directory"}
        )
    return root.resolve()


async def list_sessions(
    store_for: Callable[[Path], SessionStore],
    *,
    cwd: str | None = None,
    cursor: str | None = None,
    live: Callable[[str], tuple[str, str | None] | None] = lambda _session_id: None,
    known: Iterable[SessionStore] = (),
) -> ListSessionsResponse:
    """Root sessions with at least one message, most recent first.

    Listed: root sessions with a message the agent answered, or a title a
    person set. ``cwd`` lists that workspace's sessions from ``store_for(cwd)``.
    Without it, every session in the ``known`` stores is listed: sessions
    are stored per workspace and there is no index of every workspace, so
    the caller passes the stores of the workspaces it has seen. Before any
    workspace is named, the server's own working directory is the workspace.
    ``live(session_id)`` returns ``(status,
    title)`` for a session that runs in this process (or, for the router,
    in one of its workers), else ``None``. Sessions held by another
    process are left out: opening them would fail. ``_meta["dev.nooa/status"]``
    is the live status (``running``, ``idle`` or ``retained``) or ``on_disk``.

    Read-only: the store is scanned without claiming any session, so the
    router can answer ``session/list`` without a worker.
    """
    root = validate_workspace(cwd, None) if cwd is not None else None
    try:
        offset = int(cursor) if cursor is not None else 0
    except ValueError:
        raise RequestError.invalid_params({"cursor": cursor, "reason": "Invalid cursor"}) from None
    if offset < 0:
        raise RequestError.invalid_params({"cursor": cursor, "reason": "Invalid cursor"})

    if root is not None:
        stores = [store_for(root)]
    else:
        # Workspaces sharing one directory give the same store more than once.
        stores = list({store.root.resolve(): store for store in known}.values())
        if not stores:
            # Pool 1.0.16 lists without ``cwd`` before it opens any session;
            # the client starts the server in the directory it works in.
            stores = [store_for(Path.cwd())]

    def scan() -> list[tuple[Any, SessionStore, bool]]:
        # Pure filesystem work, one lock probe per session: off the loop.
        # A session the agent never answered and nobody named is noise from
        # a failed first turn; it is not listed.
        infos = [
            (info, store)
            for store in stores
            for info in store.list(workspace=root, roots_only=True)
            if info.turn_count > 0 and (info.reply_count > 0 or info.title_is_user_set)
        ]
        infos.sort(key=lambda pair: pair[0].last_active, reverse=True)
        return [(info, store, store.is_active(info.id)) for info, store in infos]

    found: list[tuple[Any, str]] = []
    for info, store, active in await asyncio.to_thread(scan):
        here = live(info.id)
        if here is not None:
            status, title = here
            info = info.model_copy(update={"title": title or info.title})
        elif active:
            continue
        else:
            status = "on_disk"
        # ACP requires an absolute cwd for every entry. The old TUI recorded
        # the directory as typed ("../"), so fall back to the request's cwd,
        # then to the directory the store belongs to.
        workspace = info.workspace if Path(info.workspace).is_absolute() else None
        workspace = workspace or (str(root) if root is not None else None)
        workspace = workspace or (str(store.workspace) if store.workspace is not None else None)
        if workspace is None:
            continue
        found.append((info.model_copy(update={"workspace": workspace}), status))
    page = found[offset : offset + SESSION_PAGE_SIZE]
    sessions = [
        ACPSessionInfo(
            session_id=info.id,
            cwd=info.workspace,
            title=info.title or f"Untitled session [{info.id[:8]}]",
            updated_at=datetime.fromtimestamp(info.last_active, UTC).isoformat(),
            field_meta={"dev.nooa/status": status},
        )
        for info, status in page
    ]
    next_cursor = str(offset + len(page)) if len(found) > offset + len(page) else None
    return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)


__all__ = ["SESSION_PAGE_SIZE", "list_sessions", "validate_workspace"]
