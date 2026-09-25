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
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

from nooa.events import TuiSessionResumed
from nooa.interactive import Done
from nooa.storage.sqlite import SessionAlreadyActiveError
from nooa_coder.session.events import ChildDeleted, SnapshotRestoreFailed
from nooa_coder.session.items import (
    ChildCreatedUpdate,
    ChildFailed,
    ChildFailedError,
    ChildQuestion,
    ChildRef,
    ChildResult,
    Receipt,
    SessionEvent,
    SessionInfo,
    SessionStatus,
    TurnEndedUpdate,
)
from nooa_coder.session.loader import AgentFactory, load_typed
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.port import install_port
from nooa_coder.session.session import Session, SessionClosedError
from nooa_coder.session.store import (
    InvalidSessionIdError,
    SessionHandle,
    SessionNotFoundError,
    SessionStore,
)

logger = logging.getLogger(__name__)

Prepare = Callable[[Session], Awaitable[None]]
"""Host hook run on a built session before it starts and is published."""

LLMFactory = Callable[[str | None, Path], Any]
"""Builds a model client from a model-registry alias and the session's workspace."""


_REQUEUED_CHANNELS = ("user_messages", "delegates")


class ChildActiveElsewhereError(SessionAlreadyActiveError):
    """Part of the loaded session's tree is live in another registry or process.

    A parent and its running children always share one registry, so a
    parent cannot be loaded while one of its descendants is live
    elsewhere, and a child cannot be loaded while one of its ancestors is.
    """


class DepthLimitError(ValueError):
    """Creating the session would exceed the tree's depth cap (``options.max_depth``)."""


class SessionRegistry:
    """Creates, finds, lists and closes the sessions of one process.

    A registry serves one store: one workspace's ``.nooa/sessions``
    directory, or the one directory ``NOOA_SESSIONS_DIR`` names for every
    workspace (see ``sessions_root``). Children go in their root's store,
    so a whole tree sits in one directory. A host serving several
    workspaces keeps one registry per store.
    """

    def __init__(
        self,
        store: SessionStore,
        *,
        agent_factory: AgentFactory | None = None,
        llm_factory: LLMFactory | None = None,
    ) -> None:
        """``agent_factory(options, storage)`` builds each agent (default: the
        coding agent's ``create_session_agent``, which loads
        ``options.agent_spec`` and gives a coding agent its workspace
        settings). ``llm_factory(model_alias, workspace)`` builds the model
        client for every session whose options carry no ``llm``;
        ``model_alias`` is ``options.model``, or ``None`` for the factory's
        default. The session owns that client and closes it. When the client
        has a non-empty ``alias`` attribute, it is recorded as ``info.model``.
        """
        self.store = store
        self.llm_factory = llm_factory
        self.sessions: dict[str, Session] = {}
        self._reserved: dict[str, asyncio.Future[Session | None]] = {}
        if agent_factory is None:
            from nooa_coder.coding.factory import create_session_agent

            agent_factory = create_session_agent
        self._agent_factory: AgentFactory = agent_factory
        # Parent-side delivery state, by (parent id, child id): only the
        # child's own parent can wait for or take its results.
        self._waiters: dict[tuple[str, str], asyncio.Future[Done]] = {}
        self._queued: dict[tuple[str, str], list[tuple[Receipt, ChildResult | ChildFailed]]] = {}
        self._background: set[asyncio.Task[None]] = set()

    # ---- create ------------------------------------------------------

    async def create(
        self,
        options: SessionOptions,
        *,
        parent_id: str | None = None,
        initial_items: Iterable[tuple[str, Any]] = (),
        initial_source: str = "user",
        prepare: Prepare | None = None,
    ) -> Session:
        """Create a root session, or a child of a live ``parent_id``.

        Two phases: the id is reserved (not visible), the file, agent and
        Session are built and ``initial_items`` admitted, then the session
        is published and started. ``prepare(session)`` runs right after
        the build, before the initial items are admitted and before the
        session starts or is visible, so a host can attach listeners (which
        then see those items' ``item_admitted`` updates) and tools before
        any turn runs. A failure before
        publishing (including in ``prepare``) closes and deletes what was
        built and drops the reservation.
        """
        parent = None
        if parent_id is not None:
            parent = self.sessions.get(parent_id)
            if parent is None:
                raise KeyError(f"Parent session {parent_id!r} is not live here")
            if parent.closing:
                raise SessionClosedError(f"Parent session {parent_id!r} is closing")
        depth = parent.depth + 1 if parent is not None else 0
        if depth > options.max_depth:
            raise DepthLimitError(
                f"A session at depth {depth} exceeds the depth cap {options.max_depth}"
            )
        session_id = str(uuid.uuid4())
        reservation = self._reserve(session_id)
        handle: SessionHandle | None = None
        session: Session | None = None
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
                turn_method=options.turn_method,
                mode=options.permission_mode,
                session_id=session_id,
            )
            session = await self._build(options, handle)
            if prepare is not None:
                await prepare(session)
            for channel, item in initial_items:
                session.admit(item, channel=channel, source=initial_source)
            session.start()
        except BaseException:
            await self._close_half_built(session)
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

    async def _close_half_built(self, session: Session | None) -> None:
        """Close a session that was built but never published (agent, client, handle)."""
        if session is None:
            return
        try:
            await asyncio.shield(session.close())
        except Exception:
            logger.exception("Closing the half-built session %s failed", session.id)

    async def _build(
        self, options: SessionOptions, handle: SessionHandle, *, restore: bool = False
    ) -> Session:
        # Build the agent outside the caller's context: a child is created
        # from inside its parent's cell, and the agent must not inherit the
        # parent's call stack or LLM inheritance.
        owned_llm = None
        build_options = options
        if options.llm is None and self.llm_factory is not None:
            # A None model asks the factory for its default; the session owns
            # the client either way and closes it.
            owned_llm = self.llm_factory(options.model, options.workspace)
            build_options = options.model_copy(update={"llm": owned_llm})
        try:
            agent = contextvars.Context().run(self._agent_factory, build_options, handle.storage)
            if restore:
                agent = await self._restore(agent, build_options, handle)
        except BaseException:
            if owned_llm is not None and hasattr(owned_llm, "aclose"):
                await asyncio.shield(owned_llm.aclose())
            raise
        # The Session keeps the options with the client, so a same-model
        # child (options.inherit) shares it instead of building another.
        session = Session(
            options=build_options,
            agent=agent,
            handle=handle,
            owned_llm=owned_llm,
            llm_factory=self.llm_factory,
            before_close=lambda: self._close_children(handle.id),
            llm_in_use=lambda llm: any(
                other.id != handle.id and other.options.llm is llm
                for other in self.sessions.values()
            ),
        )
        resolved = getattr(owned_llm, "alias", None)
        if isinstance(resolved, str) and resolved:
            session.info.model = resolved
        install_port(agent, session, self)
        return session

    async def _restore(self, agent: Any, options: SessionOptions, handle: SessionHandle) -> Any:
        """Restore the latest snapshot into ``agent``; on failure, a fresh agent.

        A snapshot that cannot be restored (from an older agent class, or
        damaged) does not stop the load: a warning is logged, a
        ``SnapshotRestoreFailed`` note goes in the transcript, and the
        session goes on with a newly built agent, since the failed restore
        may have changed part of the first one.
        """
        try:
            restored = handle.storage.restore_latest_snapshot(agent)
            if restored:
                after_restore = getattr(agent, "after_restore", None)
                if callable(after_restore):
                    after_restore()
        except Exception as exc:
            logger.warning(
                "Session %s: could not restore its saved state; going on from an empty state",
                handle.id,
                exc_info=True,
            )
            handle.events.add(SnapshotRestoreFailed(error=f"{type(exc).__name__}: {exc}"))
            await agent.queue_manager.shutdown()
            await agent.aclose()
            agent = contextvars.Context().run(self._agent_factory, options, handle.storage)
            restored = False
        agent.event_manager.add(TuiSessionResumed(session_id=handle.id, restored=restored))
        return agent

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
        if session.parent_id is not None:
            session.subscribe(lambda update: self._deliver(session, update))
        self._reserved.pop(session.id, None)
        if not reservation.done():
            reservation.set_result(session)

    def _on_update(self, session: Session, update: SessionEvent) -> None:
        if update.kind != "closed":
            return
        if self.sessions.get(session.id) is session:
            del self.sessions[session.id]
        if session.parent_id is None:
            return
        waiter = self._waiters.pop((session.parent_id, session.id), None)
        if waiter is not None and not waiter.done():
            waiter.set_exception(ChildFailedError(f"Child {session.name!r} was closed"))

    # ---- child results -----------------------------------------------

    def child_ref(self, child: Session) -> ChildRef:
        """A data handle on a live child."""
        return ChildRef(
            id=child.id,
            name=child.name or "",
            depth=child.depth,
            status=_live_status(child),
            parent_id=child.parent_id,
        )

    def check_owner(self, parent: Session, child_id: str) -> None:
        """Raise ``ChildFailedError`` unless ``child_id`` is a child of ``parent``.

        The child's recorded parent decides, for a live child and one on
        disk alike: holding a child's id is not enough to act on it.
        """
        live = self.sessions.get(child_id)
        if live is not None:
            owner = live.parent_id
        else:
            try:
                owner = self.store.get(child_id).parent_id
            except (SessionNotFoundError, InvalidSessionIdError) as exc:
                raise ChildFailedError(f"Child {child_id!r} does not exist") from exc
        if owner != parent.id:
            raise ChildFailedError(f"Session {child_id!r} is not a child of {parent.id!r}")

    def _deliver(self, child: Session, update: SessionEvent) -> None:
        """Route a child's turn result to its live parent (a detached child keeps its own)."""
        if not isinstance(update, TurnEndedUpdate):
            return
        ancestor = self.sessions.get(child.parent_id) if child.parent_id else None
        parent = ancestor
        while ancestor is not None:
            ancestor.add_attributed_usage(update.usage, child_id=child.id)
            ancestor = self.sessions.get(ancestor.parent_id) if ancestor.parent_id else None
        if parent is None or parent.closing:
            return
        kind = update.outcome_kind
        if kind not in ("done", "need_input", "error", "cancelled"):
            return
        ref = self.child_ref(child)
        source = f"child:{child.name or child.id}"
        waiter = self._waiters.pop((parent.id, child.id), None)
        if waiter is not None and waiter.done():
            waiter = None
        delivered = False
        try:
            if kind == "done":
                # Built for both paths: ChildResult rebuilds a TaskResult sent as data.
                result = ChildResult(child=ref, done=_rebuild_done(update))
                if waiter is not None:
                    waiter.set_result(result.done)
                else:
                    self._put(parent, child.id, result, source)
            elif kind in ("error", "cancelled"):
                # A cancelled turn ends without a result: to the parent it is a failure.
                error = (
                    f"cancelled by {update.outcome.get('by', 'unknown')}"
                    if kind == "cancelled"
                    else str(update.outcome.get("error", "the child's turn failed"))
                )
                if waiter is not None:
                    waiter.set_exception(ChildFailedError(error))
                else:
                    self._put(parent, child.id, ChildFailed(child=ref, error=error), source)
            else:
                question = ChildQuestion(
                    child=ref,
                    question=str(update.outcome.get("question", "")),
                    options=update.outcome.get("options"),
                    answer_schema=update.outcome.get("answer_schema"),
                )
                if waiter is not None:
                    waiter.set_exception(
                        ChildFailedError(
                            f"Child {child.name!r} asked a question; answer the ChildQuestion "
                            "that arrives on the delegates channel"
                        )
                    )
                _admit_delegate(parent, question, source)
            delivered = True
        finally:
            # Keep a child's own durable result available if parent admission fails.
            if delivered and kind != "need_input" and not child.options.retain:
                # Not from inside this callback: closing awaits the child's own loop.
                task = asyncio.get_running_loop().create_task(self.close(child.id))
                self._background.add(task)
                task.add_done_callback(self._background.discard)

    def _put(
        self, parent: Session, child_id: str, item: ChildResult | ChildFailed, source: str
    ) -> None:
        receipt = _admit_delegate(parent, item, source)
        self._queued.setdefault((parent.id, child_id), []).append((receipt, item))

    def take_queued_result(self, parent: Session, child_id: str) -> Done | None:
        """Withdraw a result of this child still queued for the parent, and return it.

        Raises ``ChildFailedError`` for a queued failure, or when ``child_id``
        is not a child of ``parent``.
        """
        self.check_owner(parent, child_id)
        key = (parent.id, child_id)
        queued = self._queued.get(key, [])
        while queued:
            receipt, item = queued.pop(0)
            if parent.withdraw(receipt):
                if isinstance(item, ChildFailed):
                    raise ChildFailedError(item.error)
                return item.done
        return None

    def waiter(self, parent: Session, child_id: str) -> asyncio.Future[Done]:
        """The future the next ``Done`` of this child of ``parent`` resolves.

        It takes the place of a delegates item.

        Raises ``ChildFailedError`` when ``child_id`` is not a child of ``parent``.
        """
        self.check_owner(parent, child_id)
        key = (parent.id, child_id)
        waiter = self._waiters.get(key)
        if waiter is None or waiter.done():
            waiter = asyncio.get_running_loop().create_future()
            self._waiters[key] = waiter
        return waiter

    def drop_waiter(self, parent: Session, child_id: str, waiter: asyncio.Future[Done]) -> None:
        """Forget a ``wait()`` that was cancelled; a result it already holds goes to delegates."""
        key = (parent.id, child_id)
        if self._waiters.get(key) is waiter:
            del self._waiters[key]
        if not waiter.done() or waiter.cancelled() or parent._closed:
            return
        child = self.sessions.get(child_id)
        ref = self.child_ref(child) if child is not None else self._ref_from_disk(child_id)
        error = waiter.exception()
        item: ChildResult | ChildFailed = (
            ChildResult(child=ref, done=waiter.result())
            if error is None
            else ChildFailed(child=ref, error=str(error))
        )
        with suppress(SessionClosedError):
            self._put(parent, child_id, item, f"child:{ref.name or child_id}")

    def _ref_from_disk(self, child_id: str) -> ChildRef:
        info = self.store.get(child_id)
        return ChildRef(
            id=info.id,
            name=info.name or "",
            depth=info.depth,
            status="on_disk",
            parent_id=info.parent_id,
        )

    async def open_child(self, parent: Session, child_id: str) -> Session:
        """A child of ``parent``: the live one, or loaded from disk with inherited options.

        Raises ``ChildFailedError`` when ``child_id`` is not a child of ``parent``.
        """
        self.check_owner(parent, child_id)
        live = self.sessions.get(child_id)
        if live is not None:
            return live
        info = self.store.get(child_id)
        # The record gives the child's own options; the parent passes on
        # what it does not record (the depth cap) and a client it shares.
        shared_llm = parent.options.llm if (info.model or None) == parent.options.model else None
        return await self.load(
            child_id,
            max_depth=parent.options.max_depth,
            llm=shared_llm,
        )

    def info(self, session_id: str) -> SessionInfo:
        """Metadata of a session, live or on disk."""
        live = self.sessions.get(session_id)
        return self.live_info(live) if live is not None else self.store.get(session_id)

    # ---- load --------------------------------------------------------

    async def load(
        self,
        session_id: str,
        *,
        prepare: Prepare | None = None,
        **overrides: Any,
    ) -> Session:
        """Attach to a live session, or open one from disk and resume it.

        The session's options come from its record (agent spec, turn
        method, model, permission mode, workspace, name, retain, host; the
        latest ``set_model``/``set_mode`` win); keyword
        ``overrides`` (any ``SessionOptions`` field, e.g. ``host="acp"``
        or ``llm=...``) replace those fields and nothing else does.

        A live id returns the same Session (the caller subscribes and reads
        ``transcript()``). Otherwise the file is opened (claim-checked by
        the store: ``SessionAlreadyActiveError`` if another owner has it),
        the agent is built and its latest snapshot restored (a snapshot
        that fails to restore leaves a fresh agent and a note in the
        transcript, see ``_restore``; after a restore the agent's
        ``after_restore()`` runs, if it has one), ``TuiSessionResumed`` is emitted, items admitted but never consumed
        or withdrawn are re-queued (``ItemRequeued``), and the session is
        published and started. Loading a child whose parent is not live is
        allowed (a detached child: its results stay in its own transcript).
        Loading a parent whose descendant is live elsewhere, or a child
        whose ancestor is, is refused with ``ChildActiveElsewhereError``. Concurrent loads of one id share it.
        ``prepare(session)`` runs before the re-queue and before the session
        starts or is visible (not when attaching to a live session).
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
        session: Session | None = None
        try:
            handle = self.store.open(session_id)
            # Checked with this session's file lock held: another registry
            # that opens a relative checks after taking its own lock too, so
            # of two overlapping loads at least one sees the other.
            self._refuse_if_tree_active_elsewhere(session_id, handle.info.parent_id)
            session = await self._build(
                self._stored_options(handle.info, overrides), handle, restore=True
            )
            if prepare is not None:
                await prepare(session)
            self._requeue(session)
            session.start()
        except BaseException:
            await self._close_half_built(session)
            self._reserved.pop(session_id, None)
            if not reservation.done():
                reservation.set_result(None)
            if handle is not None:
                handle.close()
            raise
        self._publish(session, reservation)
        return session

    def _stored_options(self, info: SessionInfo, overrides: dict[str, Any]) -> SessionOptions:
        values: dict[str, Any] = {
            "workspace": info.workspace or ".",
            "agent_spec": info.agent,
            "model": info.model or None,
            "turn_method": info.turn_method,
            "name": info.name,
            "retain": info.retained,
            "sessions_dir": self.store.root,
        }
        if info.host:
            values["host"] = info.host
        if info.mode in ("auto", "ask"):
            values["permission_mode"] = info.mode
        values.update(overrides)
        return SessionOptions.model_validate(values)

    def _refuse_if_tree_active_elsewhere(self, session_id: str, parent_id: str | None) -> None:
        """Raise ``ChildActiveElsewhereError`` if an ancestor or a descendant is live elsewhere."""

        def elsewhere(other_id: str) -> bool:
            local = other_id in self.sessions or other_id in self._reserved
            return not local and self.store.is_active(other_id)

        seen = {session_id}
        while parent_id is not None and parent_id not in seen:
            seen.add(parent_id)
            if elsewhere(parent_id):
                raise ChildActiveElsewhereError(
                    f"Session {session_id!r} has an ancestor {parent_id!r} that is active elsewhere"
                )
            try:
                parent_id = self.store.get(parent_id).parent_id
            except SessionNotFoundError:
                break
        on_disk = self.store.list(roots_only=False)
        pending = [session_id]
        while pending:
            parent = pending.pop()
            for info in on_disk:
                if info.parent_id != parent:
                    continue
                if elsewhere(info.id):
                    raise ChildActiveElsewhereError(
                        f"Session {session_id!r} has a child {info.id!r} "
                        f"({info.name or 'unnamed'}) that is active elsewhere"
                    )
                pending.append(info.id)

    def _requeue(self, session: Session) -> None:
        """Put back items admitted before the last close that no turn consumed.

        That includes steers no model call saw (on ``user_messages``); a
        steer a model call saw was marked consumed.
        """
        rows = self.store.load_rows(
            session.id,
            frozenset(("ItemAdmitted", "ItemConsumed", "ItemWithdrawn", "ItemDiscarded")),
        )
        admitted: dict[str, dict[str, object]] = {}
        finished: set[str] = set()
        for event_type, raw in rows:
            item_id = str(raw.get("item_id", ""))
            if event_type == "ItemAdmitted":
                if raw.get("channel") == "steer":
                    # A steer still buffered when the process stopped: no model
                    # call saw it, so it comes back as the message it is.
                    admitted[item_id] = {**raw, "channel": "user_messages"}
                elif raw.get("channel") in _REQUEUED_CHANNELS:
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
            session.requeue(
                item,
                channel=channel,
                source=str(raw.get("source", "")),
                item_id=item_id,
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

    async def close_child(self, parent: Session, child_id: str) -> None:
        """Close ``parent``'s child; ``ChildFailedError`` if it is not ``parent``'s."""
        self.check_owner(parent, child_id)
        await self.close(child_id)

    async def _close_children(self, parent_id: str) -> None:
        for child in [s for s in self.sessions.values() if s.parent_id == parent_id]:
            await self.close(child.id)

    async def close_all(self) -> None:
        """Close every live session, deepest first."""
        for session in sorted(self.sessions.values(), key=lambda s: s.depth, reverse=True):
            try:
                await session.close()
            except Exception:
                logger.exception("Closing session %s failed", session.id)


def _admit_delegate(parent: Session, item: Any, source: str) -> Receipt:
    """Admit a child's item on the parent's ``delegates`` channel.

    The channel is re-created if the parent's agent removed it, so a
    child's result is never lost to a ``remove_channel("delegates")``.
    """
    queues = parent.agent.queue_manager
    if "delegates" not in queues.channels():
        queues.queue("delegates")
    return parent.admit(item, channel="delegates", source=source)


def _live_status(session: Session) -> SessionStatus:
    if session.info.status == "running":
        return "running"
    if session.info.status == "closed":
        return "closed"
    if session.parent_id is not None and session.options.retain:
        return "retained"
    return "idle"


def _rebuild_done(update: TurnEndedUpdate) -> Done:
    """The child's ``Done`` rebuilt from data; a pydantic result comes back as its class."""
    done = Done.model_validate(update.outcome)
    if update.result_type is not None and done.result is not None:
        done.result = load_typed(update.result_type, done.result)
    return done
