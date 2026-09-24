# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SessionRegistry: the live sessions of one process, as a tree.

A flat map from id to Session; each session carries its parent id, so
roots and children are queries over the map, and the store gives the same
view of sessions that are not live. Children are created by the registry,
never by an agent; an agent reaches the registry only through its port.
"""

import asyncio
import contextvars
import logging
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nooa.events import TuiSessionResumed
from nooa.storage.sqlite import SessionAlreadyActiveError
from nooa_coder.session.events import ChildDeleted, ItemRequeued
from nooa_coder.session.items import ChildCreatedUpdate, SessionEvent, SessionInfo, SessionStatus
from nooa_coder.session.loader import AgentFactory, default_agent_factory, load_typed
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.session import Session
from nooa_coder.session.store import SessionHandle, SessionStore

logger = logging.getLogger(__name__)


_REQUEUED_CHANNELS = ("user_messages", "delegates")


class ChildActiveElsewhereError(SessionAlreadyActiveError):
    """A child of the session being loaded is live in another registry or process.

    A parent and its running children always share one registry, so the
    parent cannot be loaded until that child is closed.
    """


class DepthLimitError(ValueError):
    """Creating the session would exceed the tree's depth cap (``options.max_depth``)."""


class SessionRegistry:
    """Creates, finds, lists and closes the sessions of one process."""

    def __init__(self, store: SessionStore, *, agent_factory: AgentFactory | None = None) -> None:
        self.store = store
        self.sessions: dict[str, Session] = {}
        self._reserved: dict[str, asyncio.Future[Session | None]] = {}
        self._agent_factory: AgentFactory = agent_factory or default_agent_factory

    # ---- create ------------------------------------------------------

    async def create(
        self,
        options: SessionOptions,
        *,
        parent_id: str | None = None,
        initial_items: Iterable[tuple[str, Any]] = (),
        initial_source: str = "user",
    ) -> Session:
        """Create a root session, or a child of a live ``parent_id``.

        Two phases: the id is reserved (not visible), the file, agent and
        Session are built and ``initial_items`` admitted, then the session
        is published and started. A failure before publishing closes and
        deletes what was built and drops the reservation.
        """
        parent = None
        if parent_id is not None:
            parent = self.sessions.get(parent_id)
            if parent is None:
                raise KeyError(f"Parent session {parent_id!r} is not live here")
        depth = parent.depth + 1 if parent is not None else 0
        if depth > options.max_depth:
            raise DepthLimitError(
                f"A session at depth {depth} exceeds the depth cap {options.max_depth}"
            )
        session_id = str(uuid.uuid4())
        reservation = self._reserve(session_id)
        handle: SessionHandle | None = None
        try:
            handle = self.store.create(
                model=options.model or "",
                agent=options.agent_spec,
                workspace=str(options.workspace),
                host=options.host,
                parent_id=parent_id,
                depth=depth,
                name=options.name,
                retained=options.retain,
                session_id=session_id,
            )
            session = self._build(options, handle)
            for channel, item in initial_items:
                session._admit(item, channel=channel, source=initial_source)
            session.start()
        except BaseException:
            self._discard(session_id, handle, reservation)
            raise
        self._publish(session, reservation)
        if parent is not None:
            parent._emit(
                ChildCreatedUpdate(
                    session_id=parent.id,
                    child_id=session.id,
                    name=session.name,
                    depth=session.depth,
                    retained=options.retain,
                )
            )
        return session

    def _reserve(self, session_id: str) -> asyncio.Future[Session | None]:
        reservation: asyncio.Future[Session | None] = asyncio.get_running_loop().create_future()
        self._reserved[session_id] = reservation
        return reservation

    def _build(self, options: SessionOptions, handle: SessionHandle) -> Session:
        # Build the agent outside the caller's context: a child is created
        # from inside its parent's cell, and the agent must not inherit the
        # parent's call stack or LLM inheritance.
        agent = contextvars.Context().run(self._agent_factory, options, handle.storage)
        session = Session(options=options, agent=agent, handle=handle)
        session._before_close = lambda: self._close_children(session.id)
        return session

    def _discard(
        self,
        session_id: str,
        handle: SessionHandle | None,
        reservation: asyncio.Future[Session | None],
    ) -> None:
        self._reserved.pop(session_id, None)
        if not reservation.done():
            reservation.set_result(None)
        if handle is None:
            return
        handle.close()
        try:
            self.store.delete(session_id)
        except Exception:
            logger.warning("Could not delete the file of failed session %s", session_id)

    def _publish(self, session: Session, reservation: asyncio.Future[Session | None]) -> None:
        self.sessions[session.id] = session
        session.subscribe(lambda update: self._on_update(session, update))
        self._reserved.pop(session.id, None)
        if not reservation.done():
            reservation.set_result(session)

    def _on_update(self, session: Session, update: SessionEvent) -> None:
        if update.kind == "closed" and self.sessions.get(session.id) is session:
            del self.sessions[session.id]

    # ---- load --------------------------------------------------------

    async def load(self, session_id: str, options: SessionOptions) -> Session:
        """Attach to a live session, or open one from disk and resume it.

        A live id returns the same Session (the caller subscribes and reads
        ``transcript()``). Otherwise the file is opened (claim-checked by
        the store: ``SessionAlreadyActiveError`` if another owner has it),
        the agent is built and its latest snapshot restored,
        ``TuiSessionResumed`` is emitted, items admitted but never consumed
        or withdrawn are re-queued (``ItemRequeued``), and the session is
        published and started. Loading a child whose parent is not live is
        allowed (a detached child: its results stay in its own transcript).
        Loading a parent whose child is live elsewhere is refused with
        ``ChildActiveElsewhereError``. Concurrent loads of one id share it.
        """
        while True:
            live = self.sessions.get(session_id)
            if live is not None:
                return live
            pending = self._reserved.get(session_id)
            if pending is None:
                break
            loaded = await asyncio.shield(pending)
            if loaded is not None:
                return loaded
        reservation = self._reserve(session_id)
        handle: SessionHandle | None = None
        try:
            self._refuse_if_children_active_elsewhere(session_id)
            handle = self.store.open(session_id)
            session = self._build(options, handle)
            restored = handle.storage.restore_latest_snapshot(session.agent)
            session.agent.event_manager.add(
                TuiSessionResumed(session_id=session_id, restored=restored)
            )
            self._requeue(session)
            session.start()
        except BaseException:
            self._reserved.pop(session_id, None)
            if not reservation.done():
                reservation.set_result(None)
            if handle is not None:
                handle.close()
            raise
        self._publish(session, reservation)
        return session

    def _refuse_if_children_active_elsewhere(self, session_id: str) -> None:
        on_disk = self.store.list(roots_only=False)
        pending = [session_id]
        while pending:
            parent = pending.pop()
            for info in on_disk:
                if info.parent_id != parent:
                    continue
                if info.id not in self.sessions and self.store.is_active(info.id):
                    raise ChildActiveElsewhereError(
                        f"Session {session_id!r} has a child {info.id!r} "
                        f"({info.name or 'unnamed'}) that is active elsewhere"
                    )
                pending.append(info.id)

    def _requeue(self, session: Session) -> None:
        """Put back items admitted before the last close that no turn consumed."""
        rows = self.store.load_rows(
            session.id, frozenset(("ItemAdmitted", "ItemConsumed", "ItemWithdrawn"))
        )
        admitted: dict[str, dict[str, object]] = {}
        finished: set[str] = set()
        for event_type, raw in rows:
            item_id = str(raw.get("item_id", ""))
            if event_type == "ItemAdmitted":
                if raw.get("channel") in _REQUEUED_CHANNELS:
                    admitted[item_id] = raw
            else:
                finished.add(item_id)
        channels = session.agent.queue_manager.channels()
        for item_id, raw in admitted.items():
            if item_id in finished:
                continue
            channel = str(raw.get("channel"))
            if channel not in channels:
                logger.warning(
                    "Session %s: cannot re-queue item %s, no channel %r",
                    session.id,
                    item_id,
                    channel,
                )
                continue
            item = load_typed(str(raw.get("item_type", "")), str(raw.get("item_json", "null")))
            session.handle.events.add(ItemRequeued(item_id=item_id))
            session._admit(
                item,
                channel=channel,
                source=str(raw.get("source", "")),
                item_id=item_id,
                record=False,
            )

    # ---- delete ------------------------------------------------------

    async def delete(self, session_id: str, *, keep_files: bool = True) -> None:
        """Close the session if live, and mark it deleted in its live parent's record.

        Files are kept unless ``keep_files=False``.
        """
        session = self.sessions.get(session_id)
        info = session.info if session is not None else self.store.get(session_id)
        if session is not None:
            await self.close(session_id)
        parent = self.sessions.get(info.parent_id) if info.parent_id else None
        if parent is not None:
            parent.handle.events.add(ChildDeleted(child_id=session_id, name=info.name))
        if not keep_files:
            self.store.delete(session_id)

    # ---- queries -----------------------------------------------------

    def get(self, session_id: str) -> Session | None:
        """The live session with this id, if any."""
        return self.sessions.get(session_id)

    def live_info(self, session: Session) -> SessionInfo:
        """A session's info with its live status (running, idle or retained)."""
        return session.info.model_copy(update={"status": _live_status(session)}, deep=True)

    def children(self, parent_id: str) -> list[SessionInfo]:
        """Children of ``parent_id``: live ones and those only on disk, by id."""
        found = {
            info.id: info
            for info in self.store.list(roots_only=False)
            if info.parent_id == parent_id
        }
        for session in self.sessions.values():
            if session.parent_id == parent_id:
                found[session.id] = self.live_info(session)
        return sorted(found.values(), key=lambda info: info.created_at)

    def list(
        self, *, workspace: str | Path | None = None, roots_only: bool = True
    ) -> list[SessionInfo]:
        """Sessions on disk, most recent first, with live status for live ones."""
        return [
            self.live_info(self.sessions[info.id]) if info.id in self.sessions else info
            for info in self.store.list(workspace=workspace, roots_only=roots_only)
        ]

    # ---- close -------------------------------------------------------

    async def close(self, session_id: str) -> None:
        """Close a live session; its children close first."""
        session = self.sessions.get(session_id)
        if session is not None:
            await session.close()

    async def _close_children(self, parent_id: str) -> None:
        for child in [s for s in self.sessions.values() if s.parent_id == parent_id]:
            await self.close(child.id)

    async def close_all(self) -> None:
        """Close every live session, deepest first."""
        for session in sorted(self.sessions.values(), key=lambda s: s.depth, reverse=True):
            await session.close()


def _live_status(session: Session) -> SessionStatus:
    if session.info.status == "running":
        return "running"
    if session.info.status == "closed":
        return "closed"
    if session.parent_id is not None and session.options.retain:
        return "retained"
    return "idle"
