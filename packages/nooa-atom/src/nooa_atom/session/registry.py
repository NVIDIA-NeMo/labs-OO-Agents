# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SessionRegistry: the live sessions of one process, as a tree.

A flat map from id to Session; each session carries its parent id, so
roots and children are queries over the map, and the store gives the same
view of sessions that are not live. Children are created by the registry,
never by an agent; an agent reaches the registry only through its port.
"""

import asyncio
import contextlib
import contextvars
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

from nooa.events import TuiSessionResumed
from nooa.interactive import AgentMessage, Done
from nooa.storage.sqlite import SessionAlreadyActiveError
from nooa_atom.session.events import (
    ChildClosed,
    ChildDeleted,
    ChildNamed,
    SnapshotRestoreFailed,
)
from nooa_atom.session.items import (
    CHILD_MESSAGE_CHARS,
    CHILD_MESSAGE_CUT,
    AgentEventUpdate,
    ChildCreatedUpdate,
    ChildFailed,
    ChildFailedError,
    ChildMessage,
    ChildQuestion,
    ChildRef,
    ChildResult,
    Receipt,
    SessionEvent,
    SessionInfo,
    SessionStatus,
    TurnEndedUpdate,
    TurnStartedUpdate,
    question_id,
)
from nooa_atom.session.loader import AgentFactory, load_typed
from nooa_atom.session.names import ROOT_PATH, child_path, is_valid_name
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.port import install_port
from nooa_atom.session.session import Session, SessionClosedError
from nooa_atom.session.store import (
    InvalidSessionIdError,
    SessionHandle,
    SessionNotFoundError,
    SessionStore,
)
from nooa_atom.session.tracing import trace_in_tree, trace_under_caller

logger = logging.getLogger(__name__)

ChildOutcome = ChildResult | ChildQuestion | ChildFailed
"""What a child's turn end puts on its parent's delegates channel (or hands to ``wait()``)."""


class ChildRoute(Protocol):
    """Where a routed child's deliveries go instead of its parent's ``delegates`` channel.

    ``message(text)`` gets each ``self.message()`` note (already cut to
    ``CHILD_MESSAGE_CHARS``), ``outcome(item)`` each turn outcome nobody
    ``wait()``s for (a ``ChildResult``, ``ChildQuestion`` or ``ChildFailed``),
    and ``closed()`` is called once when the child session closes. Each is
    called from the registry's listeners and must not block or raise.

    A route may also define ``turn_started(item_ids)``: called as each of the
    child's turns starts, with the ids of the admitted items it carries.
    """

    def message(self, text: str) -> None: ...

    def outcome(self, item: ChildOutcome) -> None: ...

    def closed(self) -> None: ...


Prepare = Callable[[Session], Awaitable[None]]
"""Host hook run on a built session before it starts and is published."""

LLMFactory = Callable[[str | None, Path], Any]
"""Builds a model client from a model-registry alias and the session's workspace."""


_REQUEUED_CHANNELS = ("user_messages", "delegates")
# Rows the store counts, with TurnEnded, as a session's replies (``info.reply_count``).
_REPLY_EVENT_TYPES = frozenset(("AgentMessage", "TUIAgentMessage"))


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
        agent_factory: AgentFactory,
        llm_factory: LLMFactory | None = None,
    ) -> None:
        """``agent_factory(options, storage)`` builds each agent; the host
        chooses it (usually the agent layer's ``create_session_agent``,
        which loads ``options.agent_spec`` and gives an Atom agent its
        workspace settings). ``llm_factory(model_alias, workspace)`` builds the model
        client for every session whose options carry no ``llm``;
        ``model_alias`` is ``options.model``, or ``None`` for the factory's
        default. The session owns that client and closes it. When the client
        has a non-empty ``alias`` attribute, it is recorded as ``info.model``.
        """
        self.store = store
        self.llm_factory = llm_factory
        self.sessions: dict[str, Session] = {}
        self._reserved: dict[str, asyncio.Future[Session | None]] = {}
        self._agent_factory: AgentFactory = agent_factory
        # Parent-side delivery state, by (parent id, child id): only the
        # child's own parent can wait for or take its results.
        self._waiters: dict[tuple[str, str], asyncio.Future[ChildResult | ChildQuestion]] = {}
        # The question each live child asked and its parent has not answered
        # yet; popped once answered, ended or closed. A child loaded from disk
        # is seeded from its record (``load``); one not loaded is read from
        # its record (``pending_question``), so a question survives a reload.
        self._questions: dict[tuple[str, str], ChildQuestion] = {}
        # Children each parent closed for good, read once per parent from its
        # tombstones (``ChildClosed``) and kept current by ``_mark_closed``.
        # Dropped when the parent closes: the tombstones are re-read on demand.
        self._closed_children: dict[str, set[str]] = {}
        self._queued: dict[tuple[str, str], list[tuple[Receipt, ChildOutcome]]] = {}
        self._background: set[asyncio.Task[None]] = set()
        # Ancestors further up watching a session's next outcome (a deeper
        # ``ChildRef.wait()``), by the watched session's id. They get a copy;
        # the outcome still goes to the session's parent.
        self._watchers: dict[str, list[asyncio.Future[ChildResult | ChildQuestion]]] = {}
        # Names derived for children whose record has no valid name (older
        # records, named by their description), and every path computed so
        # far: a session's name and parent never change.
        self._derived_names: dict[str, str] = {}
        # Derived names each parent has recorded (``ChildNamed``), by parent id
        # and child id: read once per parent, kept current as names are recorded.
        self._recorded_names: dict[str, dict[str, str]] = {}
        self._paths: dict[str, str] = {}
        # Children whose deliveries go to their own route instead of the
        # parent's delegates channel (``route_child``), by child id.
        self._routes: dict[str, ChildRoute] = {}

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
                retained=options.kept,
                turn_method=options.turn_method,
                mode=options.permission_mode,
                session_id=session_id,
                todo_id=options.todo_id,
            )
            session = await self._build(options, handle)
            if parent is not None:
                trace_under_caller(session)
                await self._inherit_tools(session, parent)
            if prepare is not None:
                await prepare(session)
            for channel, item in initial_items:
                session.admit(item, channel=channel, source=initial_source)
            session.start()
        except BaseException:
            try:
                await self._close_half_built(session)
            finally:
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
                    retained=options.kept,
                )
            )
        return session

    @staticmethod
    async def _inherit_tools(child: Session, parent: Session) -> None:
        """Give a child its parent's connected tools (MCP servers), shared, not reconnected."""
        for warning in await child.inherit_tools(parent):
            logger.warning("Session %s: %s", child.id, warning)

    def _reserve(self, session_id: str) -> asyncio.Future[Session | None]:
        reservation: asyncio.Future[Session | None] = asyncio.get_running_loop().create_future()
        self._reserved[session_id] = reservation
        return reservation

    async def _close_half_built(self, session: Session | None) -> None:
        """Close a session that was built but never published (agent, client, handle)."""
        if session is None:
            return
        # Shielding alone leaves an orphan when this caller is cancelled: the
        # unpublished session is absent from close_all(). Keep a strong join and
        # defer single/repeated caller cancellation until owned teardown ends.
        closing = asyncio.create_task(session.close())
        cancelled = None
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError as exc:
                cancelled = exc
            except Exception:
                break  # retrieve/log the actual close failure below
        try:
            closing.result()
        except Exception:
            logger.exception("Closing the half-built session %s failed", session.id)
        if cancelled is not None:
            raise cancelled

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
                "Session %s: could not restore its latest saved state", handle.id, exc_info=True
            )
            agent, restored, note = await self._restore_older(agent, options, handle, exc)
            handle.events.add(SnapshotRestoreFailed(error=note))
            if restored and callable(after_restore := getattr(agent, "after_restore", None)):
                after_restore()
        agent.event_manager.add(TuiSessionResumed(session_id=handle.id, restored=restored))
        return agent

    async def _restore_older(
        self, agent: Any, options: SessionOptions, handle: SessionHandle, latest_error: Exception
    ) -> tuple[Any, bool, str]:
        """Try older snapshots, newest first, on a fresh agent each time.

        A damaged file usually loses its newest pages first, so the snapshot
        before the latest is often intact. Returns the agent to use, whether
        anything was restored, and the note for the transcript.
        """
        failed = f"{type(latest_error).__name__}: {latest_error}"
        latest = None
        with contextlib.suppress(Exception):
            latest = handle.storage.get_latest_snapshot_id()
        for snapshot_id in self.store.snapshot_ids(handle.id):
            if snapshot_id == latest:
                continue
            await agent.queue_manager.shutdown()
            await agent.aclose()
            agent = contextvars.Context().run(self._agent_factory, options, handle.storage)
            try:
                handle.storage.restore_snapshot(snapshot_id, agent)
            except Exception:
                logger.warning(
                    "Session %s: snapshot %s could not be restored either",
                    handle.id,
                    snapshot_id,
                    exc_info=True,
                )
                continue
            logger.warning("Session %s: restored an older snapshot %s", handle.id, snapshot_id)
            return agent, True, f"{failed}; restored the older snapshot {snapshot_id} instead"
        await agent.queue_manager.shutdown()
        await agent.aclose()
        agent = contextvars.Context().run(self._agent_factory, options, handle.storage)
        return agent, False, failed

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
        self._forget_parent(session.id)
        route = self._routes.pop(session.id, None)
        if route is not None:
            route.closed()
        if session.parent_id is None:
            return
        self._questions.pop((session.parent_id, session.id), None)
        waiter = self._waiters.pop((session.parent_id, session.id), None)
        if waiter is not None and not waiter.done():
            waiter.set_exception(ChildFailedError(f"Child {session.name!r} was closed"))
        for watcher in self._watchers.pop(session.id, []):
            if not watcher.done():
                watcher.set_exception(ChildFailedError(f"{self.path(session.id)} was closed"))

    # ---- child results -----------------------------------------------

    def child_ref(self, child: Session) -> ChildRef:
        """A data handle on a live child."""
        return ChildRef(
            id=child.id,
            name=child.name if is_valid_name(child.name) else self.child_name(child.info),
            path=self.path(child.id),
            depth=child.depth,
            status=_live_status(child),
            parent_id=child.parent_id,
            todo_id=child.options.todo_id,
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

    # ---- names and paths ---------------------------------------------

    def child_name(self, info: SessionInfo) -> str:
        """A child's name: the stored one, or for an older record without a valid one, ``child_<n>``.

        ``n`` is the child's place among its parent's children by creation,
        bumped past every name a sibling already has (a stored one, or one
        derived before). The names of all such siblings are derived together,
        so they never collide, and each is recorded in the parent
        (``ChildNamed``) when first derived, so it stays the same in a later
        process even if siblings are deleted. A root's name is its own.
        """
        if info.parent_id is None or is_valid_name(info.name):
            return info.name or ""
        derived = self._derived_names.get(info.id)
        if derived is None:
            self._derive_names(info.parent_id, extra=info)
            derived = self._derived_names[info.id]
        return derived

    def _derive_names(self, parent_id: str, *, extra: SessionInfo | None = None) -> None:
        """Derive (and record) a ``child_<n>`` name for every child of ``parent_id`` lacking one.

        ``extra`` is a child not (yet) listed by the store; it comes last.
        A name is recorded only while the parent is live (its record is
        written through its own handle); one derived while it is not is
        recorded on a later call once it is.
        """
        siblings = sorted(
            (other for other in self.store.list(roots_only=False) if other.parent_id == parent_id),
            key=lambda other: other.created_at,
        )
        if extra is not None and all(other.id != extra.id for other in siblings):
            siblings.append(extra)
        legacy = [other for other in siblings if not is_valid_name(other.name)]
        recorded = self._recorded_of(parent_id)
        for other in legacy:
            name = recorded.get(other.id)
            if name is not None and other.id not in self._derived_names:
                self._derived_names[other.id] = name
        taken = {other.name for other in siblings if is_valid_name(other.name)}
        taken.update(self._derived_names[o.id] for o in legacy if o.id in self._derived_names)
        legacy_ids = {other.id for other in legacy}
        for place, other in enumerate(siblings, 1):
            if other.id in legacy_ids and other.id not in self._derived_names:
                n = place
                while f"child_{n}" in taken:
                    n += 1
                self._derived_names[other.id] = f"child_{n}"
                taken.add(f"child_{n}")
        parent = self.sessions.get(parent_id)
        if parent is None or parent.closing:
            return
        for other in legacy:
            name = self._derived_names[other.id]
            if recorded.get(other.id) != name:
                with suppress(Exception):
                    parent.handle.events.add(ChildNamed(child_id=other.id, name=name))
                    recorded[other.id] = name

    def _recorded_of(self, parent_id: str) -> dict[str, str]:
        """The ``child_<n>`` names ``parent_id`` recorded (``ChildNamed``), by child id; read once."""
        recorded = self._recorded_names.get(parent_id)
        if recorded is None:
            recorded = {}
            with suppress(Exception):
                for _kind, raw in self.store.load_rows(parent_id, frozenset({"ChildNamed"})):
                    child_id, name = str(raw.get("child_id") or ""), raw.get("name")
                    if child_id and isinstance(name, str) and is_valid_name(name):
                        recorded.setdefault(child_id, name)  # the first record wins
            self._recorded_names[parent_id] = recorded
        return recorded

    def path(self, session_id: str) -> str:
        """A session's path: ``/root`` for a root, else its parent's path, ``/`` and its name."""
        known = self._paths.get(session_id)
        if known is not None:
            return known
        live = self.sessions.get(session_id)
        try:
            info = live.info if live is not None else self.store.get(session_id)
        except (SessionNotFoundError, InvalidSessionIdError):
            return session_id
        if info.parent_id is None:
            found = ROOT_PATH
        else:
            name = live.name if live is not None and is_valid_name(live.name) else None
            found = child_path(self.path(info.parent_id), name or self.child_name(info))
        self._paths[session_id] = found
        return found

    def is_ancestor(self, ancestor_id: str, session_id: str) -> bool:
        """Whether ``ancestor_id`` is ``session_id``'s parent, grandparent, and so on."""
        current = session_id
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            live = self.sessions.get(current)
            try:
                parent_id = (
                    live.parent_id if live is not None else self.store.get(current).parent_id
                )
            except (SessionNotFoundError, InvalidSessionIdError):
                return False
            if parent_id is None:
                return False
            if parent_id == ancestor_id:
                return True
            current = parent_id
        return False

    def watcher(self, session_id: str) -> asyncio.Future[ChildResult | ChildQuestion]:
        """A future for a deeper session's next outcome, for an ancestor further up.

        A copy: the outcome still goes to the session's own parent.
        """
        future: asyncio.Future[ChildResult | ChildQuestion] = (
            asyncio.get_running_loop().create_future()
        )
        self._watchers.setdefault(session_id, []).append(future)
        return future

    def _notify_watchers(
        self, session_id: str, outcome: "ChildResult | ChildQuestion | ChildFailedError | None"
    ) -> None:
        if outcome is None:
            return
        for future in self._watchers.pop(session_id, []):
            if future.done():
                continue
            if isinstance(outcome, ChildFailedError):
                future.set_exception(ChildFailedError(str(outcome)))
            else:
                future.set_result(outcome)

    def _deliver(self, child: Session, update: SessionEvent) -> None:
        """Route a child's messages and turn result to its live parent (a detached child keeps its own)."""
        if isinstance(update, AgentEventUpdate) and isinstance(update.event, AgentMessage):
            self._deliver_message(child, update.event)
            return
        if isinstance(update, TurnStartedUpdate):
            # A new turn, by any route (the parent's send, a host
            # submit), ends the child's pending question.
            if child.parent_id is not None:
                self._questions.pop((child.parent_id, child.id), None)
            started = getattr(self._routes.get(child.id), "turn_started", None)
            if started is not None:
                started(list(update.item_ids))
            return
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
        key = (parent.id, child.id)
        waiter = self._waiters.pop(key, None)
        if waiter is not None and waiter.done():
            waiter = None
        # One place only: a waiting wait() gets the outcome, else delegates does.
        delivered = False
        watched: ChildResult | ChildQuestion | ChildFailedError | None = None
        try:
            if kind == "done":
                self._questions.pop(key, None)
                # Built for both paths: ChildResult rebuilds a TaskResult sent as data.
                result = watched = ChildResult(child=ref, done=_rebuild_done(update))
                if waiter is not None:
                    waiter.set_result(result)
                else:
                    self._put(parent, child.id, result, source)
            elif kind in ("error", "cancelled") or not child.options.kept:
                self._questions.pop(key, None)
                # A cancelled turn ends without a result: to the parent it is a
                # failure. So is a one-shot worker's question, which nobody
                # would answer (its turn prompt forbids one; this is the
                # backstop for an agent class without that check).
                if kind == "cancelled":
                    error = f"cancelled by {update.outcome.get('by', 'unknown')}"
                elif kind == "error":
                    error = str(update.outcome.get("error", "the child's turn failed"))
                else:
                    error = (
                        "a one-shot worker cannot ask questions, but it asked: "
                        f"{update.outcome.get('question', '')!r}"
                    )
                watched = ChildFailedError(error)
                if waiter is not None:
                    waiter.set_exception(ChildFailedError(error))
                else:
                    self._put(parent, child.id, ChildFailed(child=ref, error=error), source)
            else:
                question = watched = _question_from_outcome(ref, update.outcome)
                # Pending until answered; it does not block the parent's Done.
                self._questions[key] = question
                if waiter is not None:
                    waiter.set_result(question)
                else:
                    self._put(parent, child.id, question, source)
            delivered = True
            self._notify_watchers(child.id, watched)
        finally:
            # Keep a child's own durable result available if parent admission fails.
            if delivered and not child.options.kept:
                # Not from inside this callback: closing awaits the child's own loop.
                self._mark_closed(parent, child.id)
                task = asyncio.get_running_loop().create_task(self.close(child.id))
                self._background.add(task)
                task.add_done_callback(self._background.discard)

    def _deliver_message(self, child: Session, event: AgentMessage) -> None:
        """Send a child's ``self.message()`` text to its live parent as a ``ChildMessage``.

        Listeners run in the order the child emits, so its messages reach the
        parent before the result of the same turn. The message stays in the
        child's own events too. A ``Done``'s message (marked
        ``is_result_text``) is not sent: it arrives with the ``ChildResult``.
        Text over ``CHILD_MESSAGE_CHARS`` is cut, with a note saying so.
        A closing or closed parent drops it.
        """
        if getattr(event, "is_result_text", False):
            return
        parent = self.sessions.get(child.parent_id) if child.parent_id else None
        if parent is None or parent.closing:
            return
        text = str(event.content)
        if len(text) > CHILD_MESSAGE_CHARS:
            text = text[:CHILD_MESSAGE_CHARS] + CHILD_MESSAGE_CUT
        route = self._routes.get(child.id)
        if route is not None:
            route.message(text)
            return
        item = ChildMessage(child=self.child_ref(child), text=text)
        with suppress(SessionClosedError):
            _admit_delegate(parent, item, f"child:{child.name or child.id}")

    def route_child(self, child_id: str, route: ChildRoute) -> None:
        """Send this child's messages and unawaited outcomes to ``route``, not to ``delegates``.

        For a work adapter that owns the child: install it before the child's
        first turn (``SessionPort.delegate(route=...)`` does, in ``prepare``)
        so nothing is delivered elsewhere first. It holds until the child
        closes (``route.closed()``).
        """
        self._routes[child_id] = route

    def _put(self, parent: Session, child_id: str, item: ChildOutcome, source: str) -> None:
        route = self._routes.get(child_id)
        if route is not None:
            route.outcome(item)  # routed: never also on delegates
            return
        receipt = _admit_delegate(parent, item, source)
        self._queued.setdefault((parent.id, child_id), []).append((receipt, item))

    def take_queued_result(
        self, parent: Session, child_id: str
    ) -> ChildResult | ChildQuestion | None:
        """Withdraw this child's oldest outcome still queued for the parent, and return it.

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
                return item
        return None

    def pending_question(self, parent: Session, child_id: str) -> ChildQuestion | None:
        """The child's question ``parent`` has not answered yet, if any.

        Kept in memory for a live child (seeded from its record when it was
        loaded); for a child not loaded here, read from its record: its last
        turn ended with a question and nothing was sent to it since. A child
        the parent closed has none.
        """
        self.check_owner(parent, child_id)
        if self.is_closed(parent.id, child_id):
            return None
        question = self._questions.get((parent.id, child_id))
        if question is not None or child_id in self.sessions:
            return question
        return self._question_from_record(parent.id, child_id)

    def _question_from_record(
        self, parent_id: str, child_id: str, child: Session | None = None
    ) -> ChildQuestion | None:
        """The pending question in a child's record (``child``: the session being loaded)."""
        if self.is_closed(parent_id, child_id):
            return None  # closed means closed: its last question died with it
        live = child or self.sessions.get(child_id)
        if live is not None and live.info.status == "running":
            return None
        try:
            rows = self.store.load_rows(
                child_id,
                frozenset({"TurnStarted", "TurnEnded", "ItemAdmitted", *_REPLY_EVENT_TYPES}),
            )
        except Exception:
            return None
        asked: dict[str, Any] | None = None
        replies = 0  # counted as the store seeds info.reply_count, for older records
        for event_type, raw in rows:
            if event_type in _REPLY_EVENT_TYPES:
                replies += 1
            elif event_type == "TurnEnded":
                replies += 1
                asked = None
                if raw.get("outcome_kind") == "need_input":
                    with suppress(ValueError):
                        asked = json.loads(str(raw.get("result_json") or "{}"))
                        asked.setdefault("question_id", question_id(child_id, replies))
            elif event_type == "TurnStarted":
                asked = None
            elif raw.get("channel") in ("user_messages", "steer"):
                asked = None  # something was sent to it after it asked
        if asked is None:
            return None
        kept = live.options.kept if live is not None else self.store.get(child_id).retained
        if not kept:
            return None  # a one-shot worker's question was delivered as a failure
        ref = self.child_ref(live) if live is not None else self._ref_from_disk(child_id)
        return _question_from_outcome(ref, asked)

    def clear_question(self, parent: Session, child_id: str) -> None:
        """The parent sent the child a message (its reply): no question is pending."""
        self._questions.pop((parent.id, child_id), None)

    def _forget_parent(self, parent_id: str) -> None:
        """Drop a closed or deleted parent's per-child state.

        Safe: its children close before it, and what must survive (which
        children it closed, a kept child's question) is in the record and is
        read again on demand (``_closed_of``, ``load``, ``pending_question``).
        """
        self._closed_children.pop(parent_id, None)
        self._recorded_names.pop(parent_id, None)
        for key in [key for key in self._questions if key[0] == parent_id]:
            del self._questions[key]

    def is_idle(self, child: Session) -> bool:
        """Live, not running, not ``working`` (its last turn did not end ``Waiting``), nothing queued."""
        return (
            child.info.status != "running"
            and child.last_outcome_kind != "waiting"
            and not child.has_queued_input()
        )

    def waiter(self, parent: Session, child_id: str) -> asyncio.Future[ChildResult | ChildQuestion]:
        """The future the next outcome (result or question) of this child of ``parent`` resolves.

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

    def drop_waiter(
        self,
        parent: Session,
        child_id: str,
        waiter: asyncio.Future[ChildResult | ChildQuestion],
    ) -> None:
        """Forget a ``wait()`` that was cancelled; an outcome it already holds goes to delegates."""
        key = (parent.id, child_id)
        if self._waiters.get(key) is waiter:
            del self._waiters[key]
        if not waiter.done() or waiter.cancelled() or parent._closed:
            return
        child = self.sessions.get(child_id)
        ref = self.child_ref(child) if child is not None else self._ref_from_disk(child_id)
        error = waiter.exception()
        item: ChildOutcome = (
            waiter.result() if error is None else ChildFailed(child=ref, error=str(error))
        )
        with suppress(SessionClosedError):
            self._put(parent, child_id, item, f"child:{ref.name or child_id}")

    def _ref_from_disk(self, child_id: str) -> ChildRef:
        info = self.store.get(child_id)
        closed = info.parent_id is not None and self.is_closed(info.parent_id, child_id)
        return ChildRef(
            id=info.id,
            name=self.child_name(info),
            path=self.path(info.id),
            depth=info.depth,
            status="closed" if closed else "on_disk",
            parent_id=info.parent_id,
            todo_id=info.todo_id,
        )

    def _closed_of(self, parent_id: str) -> set[str]:
        """The children ``parent_id`` closed for good: its tombstones, read once."""
        closed = self._closed_children.get(parent_id)
        if closed is None:
            closed = set()
            with suppress(Exception):
                rows = self.store.load_rows(parent_id, frozenset({"ChildClosed"}))
                closed.update(str(raw.get("child_id")) for _kind, raw in rows)
            self._closed_children[parent_id] = closed
        return closed

    def _mark_closed(self, parent: Session, child_id: str) -> None:
        """Record that ``parent`` closed this child for good (a durable tombstone)."""
        closed = self._closed_of(parent.id)
        if child_id in closed:
            return
        closed.add(child_id)
        self._questions.pop((parent.id, child_id), None)
        with suppress(Exception):
            parent.handle.events.add(ChildClosed(child_id=child_id))

    def is_closed(self, parent_id: str, child_id: str) -> bool:
        """Whether the parent closed this child (``close()``, or a one-shot worker after its result)."""
        return child_id in self._closed_of(parent_id)

    def child_status(self, parent: Session, child_id: str) -> SessionStatus:
        """A child's live status: ``closed`` once its parent closed it, ``on_disk`` if not loaded."""
        self.check_owner(parent, child_id)
        if self.is_closed(parent.id, child_id):
            return "closed"
        live = self.sessions.get(child_id)
        return _live_status(live) if live is not None else "on_disk"

    async def open_child(self, parent: Session, child_id: str) -> Session:
        """A child of ``parent``: the live one, or loaded from disk with inherited options.

        Raises ``ChildFailedError`` when ``child_id`` is not a child of
        ``parent``, or when the parent closed it: closed means closed.
        """
        self.check_owner(parent, child_id)
        live = self.sessions.get(child_id)
        closed = self.is_closed(parent.id, child_id)
        if live is not None and not live.closing and not closed:
            return live
        if live is not None or closed:
            name = live.name if live is not None else self.store.get(child_id).name
            raise ChildFailedError(
                f"Child {name or child_id!r} is closed; spawn a new child for more work"
            )
        info = self.store.get(child_id)
        # The record gives the child's own options; the parent passes on
        # what it does not record (the depth cap) and a client it shares.
        shared_llm = parent.options.llm if (info.model or None) == parent.options.model else None

        async def trace(child: Session) -> None:
            trace_under_caller(child)

        return await self.load(
            child_id,
            prepare=trace,
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
        method, model, permission mode, workspace, name, kept, host; the
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
        A child loaded with no ``prepare`` while its parent is live is traced
        in its tree's trace session (the root's id), as its own top-level
        trace (``trace_in_tree``); ``open_child`` passes a ``prepare`` that
        nests it under the reopening call instead.
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
            info = handle.info
            if info.parent_id is not None and not is_valid_name(info.name):
                # An older record, named by its description: it gets child_<n>.
                info.name = self.child_name(info)
            session = await self._build(self._stored_options(info, overrides), handle, restore=True)
            session.name = session.info.name = info.name
            live_parent = self.sessions.get(session.parent_id) if session.parent_id else None
            if live_parent is not None:
                await self._inherit_tools(session, live_parent)
            if prepare is not None:
                await prepare(session)
            elif session.parent_id in self.sessions:
                # A direct load of a child whose parent is live: no calling
                # cell to nest under, so trace it in its tree's trace session.
                trace_in_tree(session, self._tree_root_id(session.parent_id))
            last_kind = self._last_outcome_kind(session_id)
            # A turn that ended Waiting waited on jobs that died with the old
            # process: nothing can deliver, so the child reads idle, not working.
            session.last_outcome_kind = None if last_kind == "waiting" else last_kind
            if session.parent_id is not None:
                question = self._question_from_record(session.parent_id, session_id, session)
                if question is not None:
                    self._questions[(session.parent_id, session_id)] = question
            self._requeue(session)
            session.start()
        except BaseException:
            try:
                await self._close_half_built(session)
            finally:
                self._reserved.pop(session_id, None)
                if not reservation.done():
                    reservation.set_result(None)
                if handle is not None:
                    handle.close()
            raise
        self._publish(session, reservation)
        return session

    def _tree_root_id(self, session_id: str) -> str:
        """The id of the topmost live session of ``session_id``'s ancestry (itself included)."""
        root_id = session_id
        live = self.sessions.get(session_id)
        while live is not None and live.parent_id is not None:
            parent = self.sessions.get(live.parent_id)
            if parent is None:
                break
            root_id, live = parent.id, parent
        return root_id

    def _stored_options(self, info: SessionInfo, overrides: dict[str, Any]) -> SessionOptions:
        values: dict[str, Any] = {
            "workspace": info.workspace or ".",
            "agent_spec": info.agent,
            "model": info.model or None,
            "turn_method": info.turn_method,
            "name": info.name,
            "kept": info.retained,
            "todo_id": info.todo_id,
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

    def _last_outcome_kind(self, session_id: str) -> str | None:
        """How the session's last recorded turn ended, as recorded (``load`` drops ``waiting``)."""
        kind: str | None = None
        with suppress(Exception):
            for _type, raw in self.store.load_rows(session_id, frozenset({"TurnEnded"})):
                kind = str(raw.get("outcome_kind") or "") or None
        return kind

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
        channels = session._agent.queue_manager.channels()
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
            receipt = session.requeue(
                item,
                channel=channel,
                source=str(raw.get("source", "")),
                item_id=item_id,
            )
            if channel == "delegates" and isinstance(item, ChildOutcome):
                # A child's outcome still queued: wait() can withdraw it again.
                child_id = item.child.id
                self._queued.setdefault((session.id, child_id), []).append((receipt, item))

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
        if info.parent_id is not None:
            self._questions.pop((info.parent_id, session_id), None)
        self._forget_parent(session_id)
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
        """Children of ``parent_id``: live ones and those only on disk, by id.

        A child its parent closed reads ``closed``; one merely not loaded, ``on_disk``.
        """
        found = {
            info.id: info.model_copy(
                update={
                    "name": self.child_name(info),
                    **({"status": "closed"} if self.is_closed(parent_id, info.id) else {}),
                }
            )
            for info in self.store.list(roots_only=False)
            if info.parent_id == parent_id
        }
        for session in self.sessions.values():
            if session.parent_id == parent_id:
                info = self.live_info(session)
                info.name = self.child_name(info)
                found[session.id] = info
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
        """Close ``parent``'s child for good; ``ChildFailedError`` if it is not ``parent``'s."""
        self.check_owner(parent, child_id)
        self._mark_closed(parent, child_id)
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
    queues = parent._agent.queue_manager
    if "delegates" not in queues.channels():
        queues.queue("delegates")
    return parent.admit(item, channel="delegates", source=source)


def _live_status(session: Session) -> SessionStatus:
    if session.info.status == "running":
        return "running"
    if session.info.status == "closed":
        return "closed"
    if session.last_outcome_kind == "waiting":
        return "working"
    if session.parent_id is not None and session.options.kept:
        return "retained"
    return "idle"


def _rebuild_done(update: TurnEndedUpdate) -> Done:
    """The child's ``Done`` rebuilt from data; a pydantic result comes back as its class."""
    done = Done.model_validate(update.outcome)
    if update.result_type is not None and done.result is not None:
        done.result = load_typed(update.result_type, done.result)
    return done


def _question_from_outcome(ref: ChildRef, outcome: dict[str, Any]) -> ChildQuestion:
    """A child's ``need_input`` outcome as a ``ChildQuestion``.

    The id is the one the child's session stored in the outcome
    (``question_id``), so the question read back from the record after a
    reload has the same id.
    """
    return ChildQuestion(
        id=str(outcome.get("question_id") or question_id(ref.id, 0)),
        child=ref,
        question=str(outcome.get("question", "")),
        reason=outcome.get("reason"),
    )
