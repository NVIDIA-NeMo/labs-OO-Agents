# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session: owns one agent, its turn loop and its durable record.

Items go in on the agent's named queue channels (``submit``); each is
recorded (``ItemAdmitted``) before it is queued, so nothing admitted is
lost. The loop races the channels while idle and runs one turn when
something arrives, calling the agent's turn method named in the options.
Output leaves as data: session updates to subscribers and the transcript.
"""

import asyncio
import contextvars
import hashlib
import inspect
import json
import logging
import sqlite3
import uuid
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from nooa.context_blocks.roles import Role
from nooa.events import Notification, PythonOutput, ResultStatus
from nooa.interactive import (
    AgentMessage,
    Done,
    InteractiveAgent,
    NeedInput,
    Waiting,
    apply_model_limits,
)
from nooa.llm_types import LLMResponse
from nooa.storage.json_snapshot import snapshot_to_json
from nooa_coder.session.events import (
    ItemAdmitted,
    ItemConsumed,
    ItemDiscarded,
    ItemWithdrawn,
    TurnEnded,
    TurnStarted,
    UsageAttributed,
)
from nooa_coder.session.items import (
    USAGE_FIELDS,
    AgentEventUpdate,
    CancelledUpdate,
    ClosedUpdate,
    CommandInfo,
    CommandResult,
    ItemAdmittedUpdate,
    ModeChangedUpdate,
    Receipt,
    SessionEvent,
    SessionInfo,
    TitleChangedUpdate,
    TranscriptEntry,
    TurnCancelled,
    TurnCancelledOutcome,
    TurnEndedUpdate,
    TurnStartedUpdate,
    Usage,
    UsageChangedUpdate,
)
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.store import SessionHandle

logger = logging.getLogger(__name__)

Outcome = Done | NeedInput | Waiting | TurnCancelledOutcome
"""What ``prompt()`` returns."""

_FINISHED_KEPT = 256
"""How many finished items' outcomes ``outcome()`` still answers."""

OutcomeKind = Literal["done", "need_input", "waiting", "cancelled", "error"]
_MODES = ("auto", "ask")


class TurnFailedError(RuntimeError):
    """The turn that consumed a prompted item failed with an error.

    ``error`` is the original exception (also chained as ``__cause__``),
    or ``None`` when there was none (a turn that returned no turn result).
    """

    def __init__(self, message: str, error: BaseException | None = None) -> None:
        super().__init__(message)
        self.error = error
        self.__cause__ = error


class ItemWithdrawnError(RuntimeError):
    """The prompted item was withdrawn before any turn consumed it."""


class ItemDiscardedError(RuntimeError):
    """The item left its channel before any turn consumed it, without a withdraw.

    Agent or host code flushed, cleared or removed the channel.
    """


class SessionClosedError(RuntimeError):
    """The session is closed."""


def type_name(value: Any) -> str:
    """``module:qualname`` of a value's class, as recorded for typed re-loading."""
    cls = type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def item_to_json(item: Any) -> str:
    """Serialise an item for the durable record; items must be JSON data or pydantic."""
    if isinstance(item, BaseModel):
        return item.model_dump_json()
    try:
        return json.dumps(item)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Session items must be JSON data or pydantic models, got {type(item).__name__}"
        ) from exc


def as_data(item: Any) -> Any:
    """A copy of ``item`` that shares nothing with the sender: data only crosses sessions.

    A pydantic model is rebuilt as the same class from its JSON dump;
    anything else must be JSON data and is copied through JSON.
    """
    if isinstance(item, BaseModel):
        return type(item).model_validate(item.model_dump(mode="json"))
    return json.loads(item_to_json(item))


async def _aclose(client: Any) -> None:
    """Close a model client if it has ``aclose()``."""
    aclose = getattr(client, "aclose", None)
    if aclose is not None:
        await aclose()


def _write_snapshot(path: Path, blob: str) -> None:
    """Insert one serialised snapshot into the session file's ``snapshots`` table.

    Runs in a worker thread on its own connection. It cannot go through
    the session's ``SQLiteStorageManager``: that one's connection belongs
    to the event loop thread, and a second manager would try to take the
    session's file lock, which this process already holds. The row
    matches what ``SQLiteStorageManager.save_snapshot`` writes, so
    ``restore_latest_snapshot`` reads it back. The caller holds the
    storage manager's lock (see ``Session._checkpoint``).
    """
    uri = f"{path.resolve().as_uri()}?mode=rw"
    connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
    try:
        connection.execute("PRAGMA busy_timeout=5000")
        with connection:
            connection.execute(
                "INSERT INTO snapshots (snapshot_id, created_at, data) VALUES (?, ?, ?)",
                (str(uuid.uuid4()), datetime.now(UTC).isoformat(), blob),
            )
    finally:
        connection.close()


def _preview(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Session:
    """One agent, its turn loop and its durable record.

    Built by the registry, which calls :meth:`start` after publishing it.
    Nobody else holds the agent.
    """

    def __init__(
        self,
        *,
        options: SessionOptions,
        agent: InteractiveAgent,
        handle: SessionHandle,
        owned_llm: Any = None,
        llm_factory: Callable[[str | None, Path], Any] | None = None,
    ) -> None:
        info = handle.info
        self.id: str = info.id
        self.parent_id: str | None = info.parent_id
        self.depth: int = info.depth
        self.name: str | None = info.name
        self.options = options
        self.agent = agent
        self.handle = handle
        # A deep copy: the Session owns its usage totals and pushes them to
        # the handle, whose metadata other threads read under its lock.
        self.info: SessionInfo = info.model_copy(
            update={"status": "idle", "mode": options.permission_mode}, deep=True
        )
        self._owned_llm = owned_llm
        # Clients this session built and swapped out while another live
        # session (a child) still used them; closed when this one closes.
        self._retired_llms: list[Any] = []
        # Whether another live session uses a client; the registry sets it.
        self.llm_in_use: Callable[[Any], bool] = lambda _llm: False
        self._llm_factory = llm_factory
        self._pending_model: tuple[str, Any] | None = None  # (alias, built client)
        self._listeners: list[Callable[[SessionEvent], None]] = []
        # Per channel, (item, item_id) in put order: channels hold raw
        # objects, so this is how an item keeps its identity until consumed.
        self._ids: dict[str, deque[tuple[Any, str]]] = {}
        self._futures: dict[str, asyncio.Future[Outcome]] = {}
        self._finished: OrderedDict[str, Any] = OrderedDict()
        self._consumed: list[str] = []  # consumed since the last turn settled
        self._waiting: list[str] = []  # items whose prompt stays open over a Waiting
        self._loop_task: asyncio.Task[None] | None = None
        self._turn_task: asyncio.Task[Any] | None = None
        self._cancel_by: str | None = None
        self._interrupted: PythonOutput | None = None
        # Agent message texts sent in the running turn (see _send_result_message).
        self._turn_messages: set[str] = set()
        self._settled = asyncio.Event()
        self._settled.set()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False
        self._before_close: Callable[[], Awaitable[None]] | None = None
        self.port: Any = None  # the agent's SessionPort, set by install_port()
        self._loop_context_hooks: list[Callable[[], object]] = []
        self._pending_steers: list[tuple[str, str, str]] = []  # (item_id, text, source)
        self._snapshot_digest: str | None = None
        self._checkpoint_task: asyncio.Task[None] | None = None
        self._unsubscribe_agent = agent.event_manager.on("*", self._on_agent_event)
        self._unsubscribe_steers = agent.event_manager.on("BeforeTurn", self._flush_steers)
        # The agent's queue channels publish every item they hand to a consumer
        # (the loop's race and drain, agent get()) and every item they drop
        # unconsumed (flush, clear, channel removed).
        self._unsubscribe_items = (
            agent.event_manager.on(
                "ChannelItemConsumed", lambda e: self._on_consumed(e.channel, e.item)
            ),
            agent.event_manager.on(
                "ChannelItemsDiscarded", lambda e: self._on_discarded(e.channel, e.items)
            ),
        )

    # ---- lifecycle ---------------------------------------------------

    def add_loop_context_hook(self, hook: Callable[[], object]) -> None:
        """Run ``hook`` once at the top of the loop, inside the loop's own context.

        Context variables set there are seen by every turn (each turn task
        copies the loop's context), e.g. the port ``ChildRef`` resolves.
        """
        self._loop_context_hooks.append(hook)

    def start(self) -> None:
        """Start the turn loop.

        The loop task runs in a fresh ``contextvars.Context``: a session
        created from inside another agent's cell must not inherit that
        agent's call stack, generation state or scoped blocks.
        """
        if self._loop_task is not None:
            return
        self._ensure_open()
        self._loop_task = asyncio.get_running_loop().create_task(
            self._loop(), name=f"session-loop:{self.id}", context=contextvars.Context()
        )

    async def close(self) -> None:
        """Stop the loop and release everything the session owns. Idempotent.

        Order: children first (the registry's hook), cancel a running turn,
        stop the loop, shut down the agent's background jobs, close the
        agent, close a model client the session created, close the store
        handle.
        """
        if self._closed:
            return
        if self._close_task is None:
            self._close_task = asyncio.ensure_future(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        # Every step has its own guard: a failure (or a CancelledError out
        # of a step) is logged and the close goes on, so the model client is
        # closed, the file lock released and ClosedUpdate emitted whatever
        # failed before.
        self._closing = True
        await self._close_step("closing its children", self._before_close)
        await self._close_step("stopping the turn loop", self._stop_loop)
        self._resolve_all(TurnCancelledOutcome(by="host"))
        pending, self._pending_model = self._pending_model, None
        if pending is not None:
            await self._close_step("closing the pending model client", lambda: _aclose(pending[1]))
        await self._close_step("waiting for the checkpoint", self.wait_for_checkpoint)
        await self._close_step("unsubscribing from agent events", self._unsubscribe_agent)
        await self._close_step("unsubscribing the steer flush", self._unsubscribe_steers)
        await self._close_step("stopping the agent's jobs", self.agent.queue_manager.shutdown)
        await self._close_step("closing the agent", self.agent.aclose)
        for unsubscribe in self._unsubscribe_items:
            await self._close_step("unsubscribing from its queues", unsubscribe)
        await self._close_step("closing its model client", self._close_owned_llm)
        await self._close_step("closing its record", self.handle.close)
        self._closed = True
        self.info.status = "closed"
        self._emit(ClosedUpdate(session_id=self.id))

    async def _close_step(self, what: str, step: Callable[[], object] | None) -> None:
        """Run one close step (sync or async); log a failure instead of raising it."""
        if step is None:
            return
        try:
            result = step()
            if inspect.isawaitable(result):
                await result
        except (Exception, asyncio.CancelledError):
            logger.exception("Session %s: %s failed", self.id, what)

    async def _stop_loop(self) -> None:
        await self._stop_turn(by="host")
        if self._loop_task is not None:
            self._loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._loop_task

    async def cancel(self, *, by: str = "user") -> bool:
        """Stop the running turn; return whether one was running.

        Returns only after the turn has settled: the interrupted cell's
        cancelled output and a ``TurnCancelled`` event are in the agent's
        events, so the model sees at its next turn that it was stopped.
        Queued items are kept and the loop goes on. With no turn running,
        prompts left open by a ``Waiting`` are closed with
        ``TurnCancelledOutcome`` and no event is written.
        """
        task = self._turn_task
        if task is None or task.done():
            waiting, self._waiting = self._waiting, []
            for item_id in waiting:
                self._resolve(item_id, TurnCancelledOutcome(by=by))
            return False
        await self._stop_turn(by=by)
        return True

    async def _stop_turn(self, *, by: str) -> None:
        """Cancel a running turn and wait until the loop has settled it."""
        task = self._turn_task
        if task is None or task.done():
            return
        self._cancel_by = by
        task.cancel()
        await self._settled.wait()

    async def _close_owned_llm(self) -> None:
        llm, self._owned_llm = self._owned_llm, None
        retired, self._retired_llms = self._retired_llms, []
        for client in (*retired, llm):
            await self._close_step("closing a model client", lambda c=client: _aclose(c))

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise SessionClosedError(f"Session {self.id!r} is closed")

    # ---- input -------------------------------------------------------

    async def submit(
        self, item: Any, *, channel: str = "user_messages", source: str = "user"
    ) -> Receipt:
        """Admit ``item`` on ``channel``: recorded first, then queued for a turn."""
        return self._admit(item, channel=channel, source=source)

    async def steer(self, text: str, *, source: str = "user") -> Receipt:
        """Give the running turn extra text; while idle this is ``submit(text)``.

        During a turn the text waits in a buffer that is flushed into a
        ``Notification`` whose source names the sender right before the
        turn's next model call, so the model reads it in order with its own
        cell output. If no model call comes (the turn was already
        finishing), the text is admitted on ``user_messages`` when the turn
        settles, with the same ``item_id``, and the next turn handles it.
        A steer is never lost.
        """
        self._ensure_open()
        task = self._turn_task
        if task is None or task.done():
            return self._admit(text, channel="user_messages", source=source)
        event = ItemAdmitted(
            channel="steer", item_json=item_to_json(text), item_type=type_name(text), source=source
        )
        event.item_id = str(event.id)
        self.handle.events.add(event)
        self._pending_steers.append((event.item_id, text, source))
        self._emit(
            ItemAdmittedUpdate(
                session_id=self.id,
                channel="steer",
                item_id=event.item_id,
                source=source,
                preview=_preview(text),
                text=text,
            )
        )
        return Receipt(
            session_id=self.id, channel="steer", item_id=event.item_id, delivered="steered"
        )

    def _flush_steers(self, _event: Any) -> None:
        """``BeforeTurn`` handler: hand buffered steers to the coming model call."""
        if not self._pending_steers or self._turn_task is None:
            return
        steers, self._pending_steers = self._pending_steers, []
        for item_id, text, source in steers:
            self.agent.event_manager.add(
                Notification(source=_steer_source(source), description=text)
            )
            self.handle.events.add(ItemConsumed(item_id=item_id))
            self._consumed.append(item_id)

    def _admit_leftover_steers(self) -> None:
        """Steers no model call saw become ordinary messages for the next turn."""
        steers, self._pending_steers = self._pending_steers, []
        for item_id, text, source in steers:
            self._admit(
                text, channel="user_messages", source=source, item_id=item_id, internal=True
            )

    def withdraw(self, receipt: Receipt) -> bool:
        """Take back an item nothing has consumed yet; return whether it was withdrawn.

        Works for queued items and for steers still in the buffer. Writes
        ``ItemWithdrawn`` so a later load does not re-queue it; a
        ``prompt()`` waiting on it raises ``ItemWithdrawnError``.
        """
        if self._closed:
            return False
        item_id = receipt.item_id
        steer = next((s for s in self._pending_steers if s[0] == item_id), None)
        # A steer no model call saw was admitted again on user_messages
        # with the same id: look for it there too.
        channels = [receipt.channel] + (["user_messages"] if receipt.channel == "steer" else [])
        if steer is not None:
            self._pending_steers.remove(steer)
        elif not any(self._remove_queued(name, item_id) for name in channels):
            return False
        self.handle.events.add(ItemWithdrawn(item_id=item_id))
        self._resolve(item_id, ItemWithdrawnError(f"Item {item_id!r} was withdrawn"))
        return True

    def _remove_queued(self, channel_name: str, item_id: str) -> bool:
        entries = self._ids.get(channel_name)
        channel = self.agent.queue_manager.channels().get(channel_name)
        if not entries or channel is None:
            return False
        index = next((i for i, (_, known) in enumerate(entries) if known == item_id), None)
        if index is None:
            return False
        obj = entries[index][0]
        # Equal-identity items put earlier are ahead of this one in the
        # channel (FIFO), so skip that many matches.
        skip = sum(1 for other, _ in list(entries)[:index] if other is obj)
        # Channel exposes no remove-by-item; edit its deque directly.
        pending = channel._items
        for position, queued in enumerate(pending):
            if queued is obj:
                if skip == 0:
                    del pending[position]
                    del entries[index]
                    return True
                skip -= 1
        return False

    async def prompt(self, text: str, *, source: str = "user") -> Outcome:
        """Submit ``text`` and wait for the outcome of the turn that consumes it.

        A ``Waiting`` outcome keeps the wait open; the next turn's outcome
        resolves it. A cancelled turn resolves it with
        ``TurnCancelledOutcome``; a failed turn raises ``TurnFailedError``.
        """
        receipt = self._admit(text, channel="user_messages", source=source)
        return await self.outcome(receipt.item_id)

    def outcome(self, item_id: str) -> Awaitable[Outcome]:
        """Wait for the outcome of the turn that consumes an admitted item.

        Works for items from ``submit()`` and for steers: a steer a model
        call saw resolves with that turn; one admitted again as a message
        resolves with the turn that consumed it. ``Waiting`` outcomes keep
        it open; a cancelled turn resolves it with ``TurnCancelledOutcome``;
        a failed turn raises ``TurnFailedError``. The outcomes of the last
        256 finished items are kept, so a recently finished item still
        answers; ``KeyError`` for an unknown item or an older one.
        """
        future = self._futures.get(item_id)
        if future is None:
            if item_id in self._finished and not self._is_pending(item_id):
                done: asyncio.Future[Outcome] = asyncio.get_running_loop().create_future()
                finished = self._finished[item_id]
                if isinstance(finished, BaseException):
                    done.set_exception(finished)
                else:
                    done.set_result(finished)
                return done
            if not self._is_pending(item_id):
                raise KeyError(item_id)
            future = asyncio.get_running_loop().create_future()
            self._futures[item_id] = future
        return asyncio.shield(future)

    def _is_pending(self, item_id: str) -> bool:
        return (
            item_id in self._consumed
            or item_id in self._waiting
            or any(known == item_id for known, _, _ in self._pending_steers)
            or any(known == item_id for entries in self._ids.values() for _, known in entries)
        )

    def _admit(
        self,
        item: Any,
        *,
        channel: str,
        source: str,
        item_id: str | None = None,
        internal: bool = False,
        record: bool = True,
    ) -> Receipt:
        """Record the item, then put it. Synchronous so sync listeners can admit.

        ``internal`` admissions (steer leftovers) are allowed while the
        session is closing, so they are recorded and re-queued on a later load.
        ``record=False`` puts an item that is already recorded (a re-queue).
        """
        if self._closed or (self._closing and not internal):
            raise SessionClosedError(f"Session {self.id!r} is closed")
        target = self.agent.queue_manager.channels().get(channel)
        if target is None or target.mode != "queue":
            raise ValueError(f"Session {self.id!r} has no queue channel {channel!r}")
        event = ItemAdmitted(
            channel=channel,
            item_json=item_to_json(item),
            item_type=type_name(item),
            source=source,
        )
        event.item_id = item_id or str(event.id)
        if record:
            self.handle.events.add(event)
            if channel == "user_messages":
                self.info.turn_count += 1  # as the store counts it
        self._ids.setdefault(channel, deque()).append((item, event.item_id))
        target.put(item)
        self._emit(
            ItemAdmittedUpdate(
                session_id=self.id,
                channel=channel,
                item_id=event.item_id,
                source=source,
                preview=_preview(item),
                text=item if isinstance(item, str) else event.item_json,
            )
        )
        return Receipt(
            session_id=self.id, channel=channel, item_id=event.item_id, delivered="queued"
        )

    def _on_consumed(self, channel: str, item: Any) -> None:
        entries = self._ids.get(channel)
        if not entries:
            return  # an item another producer put; it has no identity here
        index = next((i for i, (obj, _) in enumerate(entries) if obj is item), None)
        if index is None:
            return
        item_id = entries[index][1]
        del entries[index]
        if not self.handle._closed:
            self.handle.events.add(ItemConsumed(item_id=item_id))
        self._consumed.append(item_id)

    def _on_discarded(self, channel: str, items: list[Any]) -> None:
        """Items left ``channel`` unconsumed: record it and fail their outcomes."""
        entries = self._ids.get(channel)
        for item in items:
            if not entries:
                return
            index = next((i for i, (obj, _) in enumerate(entries) if obj is item), None)
            if index is None:
                continue  # an item another producer put; it has no identity here
            item_id = entries[index][1]
            del entries[index]
            if not self.handle._closed:
                self.handle.events.add(ItemDiscarded(item_id=item_id))
            self._resolve(
                item_id,
                ItemDiscardedError(
                    f"Item {item_id!r} was dropped from channel {channel!r} before any turn "
                    "consumed it"
                ),
            )

    # ---- the loop ----------------------------------------------------

    async def _loop(self) -> None:
        for hook in self._loop_context_hooks:
            hook()
        queues = self.agent.queue_manager
        while not self._closing:
            try:
                wins = await queues.race()
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise  # the loop itself is being cancelled (close)
                # A raced channel was flushed or removed, which cancels its
                # waiters: race again over the channels that are left.
                continue
            except Exception as exc:
                # No channel left to wait on (race() raises ValueError):
                # nothing can reach this session any more.
                logger.exception("Session %s: the turn loop cannot wait for input", self.id)
                self._end_loop(exc)
                return
            notification: dict[str, list[Any]] = {}
            for name, item in wins:
                notification.setdefault(name, []).append(item)
            for name, channel in queues.channels().items():
                if drained := channel.drain():
                    notification.setdefault(name, []).extend(drained)
            if not notification and not any(
                channel.mode == "event" for channel in queues.channels().values()
            ):
                continue  # a wake with nothing to hand over
            try:
                await self._run_turn(notification)
            except Exception as exc:
                # The loop must outlive any turn: fail what the turn owed and go on.
                logger.exception("Session %s: turn bookkeeping failed", self.id)
                self._fail_turn(exc)

    def _end_loop(self, exc: Exception) -> None:
        """The loop cannot go on: fail every open outcome and close the session."""
        error = TurnFailedError(
            f"The session's turn loop stopped: {type(exc).__name__}: {exc}", exc
        )
        self._waiting = []
        for item_id in list(self._futures):
            self._resolve(item_id, error)
        if self._close_task is None:
            self._close_task = asyncio.get_running_loop().create_task(
                self._close(), name=f"session-close:{self.id}"
            )

    def _fail_turn(self, exc: Exception) -> None:
        """Settle a turn whose own settling failed: its prompts get ``TurnFailedError``."""
        self._turn_task = None
        owed, self._waiting, self._consumed = self._waiting + self._consumed, [], []
        error = TurnFailedError(f"{type(exc).__name__}: {exc}", exc)
        for item_id in owed:
            self._resolve(item_id, error)
        self.info.status = "idle"
        with suppress(Exception):
            self._emit(
                TurnEndedUpdate(
                    session_id=self.id, outcome_kind="error", outcome={"error": str(error)}
                )
            )
        self._settled.set()

    async def _run_turn(self, notification: dict[str, list[Any]]) -> None:
        if self._pending_model is not None:
            await self._apply_pending_model()
        item_ids = list(self._consumed)
        self.handle.events.add(TurnStarted(item_ids=item_ids, item_preview=_preview(notification)))
        self._emit(TurnStartedUpdate(session_id=self.id, item_ids=item_ids))
        self._settled.clear()
        self._cancel_by = None
        self._interrupted = None
        self.info.status = "running"
        usage_before = self.info.usage.model_copy()
        self._turn_messages = set()
        method = getattr(self.agent, self.options.turn_method)
        self._turn_task = asyncio.create_task(method(notification), name=f"session-turn:{self.id}")
        outcome: Any
        kind: OutcomeKind
        try:
            outcome, kind = _classify(await self._turn_task)
            if self._send_result_message(outcome):
                # Listeners hear of agent events one loop step later; let
                # this message's update go out before turn_ended.
                await asyncio.sleep(0)
        except asyncio.CancelledError as exc:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                # The loop itself is being cancelled (close): not a turn outcome.
                self._turn_task = None
                self._settled.set()
                raise
            if self._cancel_by is not None:
                outcome, kind = TurnCancelledOutcome(by=self._cancel_by), "cancelled"
            else:
                # Something inside the turn cancelled it, not cancel().
                outcome = TurnFailedError("turn was cancelled from inside", exc)
                kind = "error"
        except Exception as exc:
            logger.exception("Turn failed in session %s", self.id)
            outcome, kind = TurnFailedError(f"{type(exc).__name__}: {exc}", exc), "error"
        finally:
            self._turn_task = None
        self._settle(outcome, kind, usage_before)

    def _send_result_message(self, outcome: Any) -> bool:
        """Show a ``Done``/``Waiting`` message as an agent message, inside the turn.

        It goes through ``agent.message()`` like any reply, before the turn
        is recorded as ended; text the turn already sent is not sent again.
        Returns whether it sent one.
        """
        text = getattr(outcome, "message", None) if isinstance(outcome, Done | Waiting) else None
        if not text or text in self._turn_messages:
            return False
        send = getattr(self.agent, "message", None)
        if callable(send):
            send(text)
        else:
            self.agent.event_manager.add(AgentMessage(content=text))
        return True

    def _settle(self, outcome: Any, kind: OutcomeKind, usage_before: Usage) -> None:
        # Items stay in self._consumed / self._waiting until the end, so a
        # failure part way leaves them for _fail_turn to resolve.
        consumed = list(self._consumed)
        if kind == "cancelled":
            self._record_cancel(outcome.by)
        self._admit_leftover_steers()
        usage = _usage_delta(usage_before, self.info.usage)
        data, result_type = _outcome_data(outcome, kind)
        explanation = _explanation(outcome, kind)
        self.handle.events.add(
            TurnEnded(
                outcome_kind=kind,
                explanation=explanation,
                result_json=json.dumps(data),
                usage=usage,
            )
        )
        waiting = self._waiting + consumed
        if kind != "waiting":
            for item_id in waiting:
                self._resolve(item_id, outcome)
            waiting = []
        self._waiting = waiting
        self._consumed = [i for i in self._consumed if i not in consumed]
        if kind != "cancelled":
            self._checkpoint()
        self.info.status = "idle"
        self._emit(
            TurnEndedUpdate(
                session_id=self.id,
                outcome_kind=kind,
                outcome=data,
                result_type=result_type,
                usage=usage,
            )
        )
        self._settled.set()

    def _checkpoint(self) -> None:
        """Save the agent's state if it changed since the last checkpoint.

        The snapshot is serialised and hashed here, on the loop, where the
        agent is not changing; only the write happens in a thread. Writes
        run one after another. A failure is logged and the turn is not
        affected; the next settled turn tries again.
        """
        try:
            blob = json.dumps(snapshot_to_json(self.agent), sort_keys=True)
        except Exception:
            logger.warning(
                "Session %s: checkpoint could not serialise the agent", self.id, exc_info=True
            )
            return
        digest = hashlib.sha256(blob.encode()).hexdigest()
        if digest == self._snapshot_digest:
            return
        self._snapshot_digest = digest
        previous = self._checkpoint_task
        path = self.handle.path
        # The storage manager's lock serialises every write to this file in
        # this process (the loop's event writes all take it). Holding it
        # here means the two connections never contend for SQLite's write
        # lock: the loop never waits in the busy handler or gets "database
        # is locked"; at worst a loop-side write waits for this one insert.
        db_lock = self.handle.storage._db_lock

        def write_locked() -> None:
            with db_lock:
                _write_snapshot(path, blob)

        async def write() -> None:
            if previous is not None:
                with suppress(Exception):
                    await previous
            try:
                await asyncio.to_thread(write_locked)
            except Exception:
                self._snapshot_digest = None
                logger.warning("Session %s: checkpoint write failed", self.id, exc_info=True)

        self._checkpoint_task = asyncio.get_running_loop().create_task(
            write(), name=f"session-checkpoint:{self.id}"
        )

    async def wait_for_checkpoint(self) -> None:
        """Wait until the checkpoint writes started so far have finished."""
        task = self._checkpoint_task
        if task is not None:
            await asyncio.shield(task)

    def _record_cancel(self, by: str) -> None:
        """Append ``TurnCancelled`` after the interrupted cell's output, and tell listeners."""
        interrupted = self._interrupted.tag if self._interrupted is not None else None
        self._interrupted = None
        self.agent.event_manager.add(TurnCancelled(by=by, interrupted=interrupted))
        self._emit(CancelledUpdate(session_id=self.id, by=by, interrupted=interrupted))

    def _record_finished(self, item_id: str, outcome: Any) -> None:
        """Keep a recent item's outcome so ``outcome()`` can still answer it."""
        self._finished[item_id] = outcome
        self._finished.move_to_end(item_id)
        while len(self._finished) > _FINISHED_KEPT:
            self._finished.popitem(last=False)

    def _resolve(self, item_id: str, outcome: Any) -> None:
        self._record_finished(item_id, outcome)
        future = self._futures.pop(item_id, None)
        if future is None or future.done():
            return
        if isinstance(outcome, BaseException):
            future.set_exception(outcome)
        else:
            future.set_result(outcome)

    def _resolve_all(self, outcome: Outcome) -> None:
        self._waiting = []
        for item_id in list(self._futures):
            self._resolve(item_id, outcome)

    # ---- slash commands ----------------------------------------------

    def commands(self) -> list[CommandInfo]:
        """Slash commands of the agent's ``slash_commands`` registry, if it has one.

        The registry has the coding agent's shape (``CodingSlashCommandRegistry``):
        ``commands()`` returns objects with ``name``, ``description`` and
        ``argument_hint``, and ``invoke(name, raw_args)`` runs one.
        """
        registry = getattr(self.agent, "slash_commands", None)
        if registry is None:
            return []
        return [
            CommandInfo(
                name=str(command.name),
                description=str(command.description or ""),
                input_hint=command.argument_hint,
            )
            for command in registry.commands()
        ]

    async def invoke_command(self, name: str, raw_args: str) -> CommandResult:
        """Run a slash command through the agent's registry; ``KeyError`` if unknown."""
        registry = getattr(self.agent, "slash_commands", None)
        if registry is None:
            raise KeyError(name)
        result = await registry.invoke(name, raw_args)
        value = getattr(result, "value", None)
        if isinstance(value, BaseModel):
            data: dict[str, Any] | None = value.model_dump(mode="json")
        elif isinstance(value, dict):
            data = json.loads(item_to_json(value))
        else:
            data = None
        text = getattr(result, "text", None)
        return CommandResult(
            text=str(text) if text is not None else str(result),
            output_to_agent=bool(getattr(result, "output_to_agent", False)),
            data=data,
        )

    # ---- title, mode, usage -----------------------------------------

    async def set_title(self, title: str, *, user_set: bool = False) -> None:
        """Set the title. Once a person has set one, titles from the agent are ignored."""
        if not user_set and self.info.title_is_user_set:
            return
        self.handle.set_title(title, user_set=user_set)
        self.info.title = title
        self.info.title_is_user_set = self.info.title_is_user_set or user_set
        self._emit(TitleChangedUpdate(session_id=self.id, title=title, user_set=user_set))

    async def set_model(self, alias: str) -> None:
        """Switch the model from the next turn on.

        The client is built now with the registry's ``llm_factory``, so a
        bad alias fails here. The loop swaps it in right before the next
        turn and closes the old client if this session created it; a
        running turn keeps its model. A second call before that turn
        replaces (and closes) the first pending client. The alias is
        recorded at once, so a load resumes on it.
        """
        if self._llm_factory is None:
            raise RuntimeError("set_model() needs the registry's llm_factory to build clients")
        self._ensure_open()
        client = self._llm_factory(alias, self.options.workspace)
        # Recorded now: a load before the next turn resumes on this model.
        self.handle.set_model(alias)
        previous, self._pending_model = self._pending_model, (alias, client)
        if previous is not None:
            await _aclose(previous[1])

    async def _apply_pending_model(self) -> None:
        pending, self._pending_model = self._pending_model, None
        if pending is None:
            return
        alias, client = pending
        self.agent.set_llm(client)
        apply_model_limits(self.agent)
        old, self._owned_llm = self._owned_llm, client
        if old is not None:
            if self.llm_in_use(old):
                self._retired_llms.append(old)  # a child shares it: close it with this session
            else:
                await _aclose(old)
        # New same-model children share the new client.
        self.options = self.options.model_copy(update={"model": alias, "llm": client})
        self.info.model = alias

    async def set_mode(self, mode: str) -> None:
        """Record the permission mode (``auto`` or ``ask``); nothing enforces it yet.

        It is persisted, so a load restores it, and children created after
        this call inherit it.
        """
        if mode not in _MODES:
            raise ValueError(f"Unknown permission mode {mode!r}; expected one of {_MODES}")
        self.handle.set_mode(mode)
        self.info.mode = mode
        # Children created from now on inherit it.
        self.options = self.options.model_copy(update={"permission_mode": mode})
        self._emit(ModeChangedUpdate(session_id=self.id, mode=mode))

    def add_attributed_usage(self, usage: Usage, *, child_id: str = "") -> None:
        """Add a child's own usage to this session's attributed totals, and tell listeners.

        The addition is recorded (``UsageAttributed``) so a load rebuilds it.
        """
        own = usage.own()
        if not any(getattr(own, name) for name in USAGE_FIELDS):
            return
        totals = self.info.usage
        for name in USAGE_FIELDS:
            attributed = f"attributed_{name}"
            setattr(totals, attributed, getattr(totals, attributed) + getattr(own, name))
        try:
            self.handle.events.add(UsageAttributed(child_id=child_id, usage=own))
        except Exception:
            # A closed handle must not break the child's delivery.
            logger.warning("Session %s: could not record a child's usage", self.id, exc_info=True)
        self.handle.update_usage(totals)
        self._emit(UsageChangedUpdate(session_id=self.id, usage=totals.model_copy()))

    def _count_usage(self, response: LLMResponse) -> None:
        usage = response.usage
        if usage is None:
            return
        totals = self.info.usage
        for name in USAGE_FIELDS:
            setattr(totals, name, getattr(totals, name) + (getattr(usage, name, 0) or 0))
        self.handle.update_usage(totals)

    # ---- output ------------------------------------------------------

    def subscribe(self, listener: Callable[[SessionEvent], None]) -> Callable[[], None]:
        """Receive session updates (data only); returns an unsubscribe function."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._listeners.remove(listener)

        return unsubscribe

    def _emit(self, update: SessionEvent) -> None:
        for listener in list(self._listeners):
            try:
                listener(update)
            except Exception:
                logger.warning("Session listener %r raised", listener, exc_info=True)

    def _on_agent_event(self, event: Any) -> None:
        if event._role is Role.RUNTIME_EVENT:
            return
        if isinstance(event, LLMResponse):
            self._count_usage(event)
        if isinstance(event, AgentMessage) and self._turn_task is not None:
            self._turn_messages.add(event.content)
        if (
            isinstance(event, PythonOutput)
            and event.execution_status is ResultStatus.CANCELLED
            and self._turn_task is not None
            and self._interrupted is None
        ):
            # The first cancelled cell output of this turn. Handlers run
            # before the event gets its tag, so keep the event and read the
            # tag when the turn settles.
            self._interrupted = event
        update = AgentEventUpdate(
            session_id=self.id, event_id=str(event.id), event_type=event.event_type
        )
        # Handlers run before the event manager stores the event; tell
        # listeners on the next loop step, when event_manager.get(event_id)
        # finds it.
        try:
            asyncio.get_running_loop().call_soon(self._emit, update)
        except RuntimeError:  # no running loop: nothing to defer to
            self._emit(update)

    def transcript(self, *, limit: int | None = None) -> list[TranscriptEntry]:
        """The session's transcript as a person would see it; the last ``limit`` entries."""
        entries = self.handle.transcript()
        return entries if limit is None else entries[-limit:]


def _steer_source(source: str) -> str:
    """The ``Notification.source`` sentence for a steer: who sent it."""
    if source == "user":
        return "New message from the user while you were working."
    if source.startswith("parent:"):
        name = source.removeprefix("parent:")
        return f"New message from your parent agent {name} while you were working."
    return f"New message from {source} while you were working."


def _classify(result: Any) -> tuple[Any, OutcomeKind]:
    """Map a turn method's return value to an outcome."""
    if isinstance(result, Done):
        return result, "done"
    if isinstance(result, NeedInput):
        return result, "need_input"
    if isinstance(result, Waiting):
        return result, "waiting"
    return TurnFailedError(f"turn returned {type(result).__name__}, not a turn result"), "error"


def _outcome_data(outcome: Any, kind: OutcomeKind) -> tuple[dict[str, Any], str | None]:
    """The outcome as JSON data, and the type of a pydantic ``Done.result``."""
    if kind == "done":
        result = outcome.result
        result_type = type_name(result) if isinstance(result, BaseModel) else None
        try:
            return outcome.model_dump(mode="json"), result_type
        except Exception:  # a result that is not data
            return {"explanation": outcome.explanation, "result": repr(result)}, None
    if kind == "need_input":
        schema = (
            outcome.answer_type.model_json_schema() if outcome.answer_type is not None else None
        )
        return {
            "question": outcome.question,
            "options": outcome.options,
            "reason": getattr(outcome, "reason", None),
            "answer_schema": schema,
        }, None
    if kind == "waiting":
        return outcome.model_dump(mode="json"), None
    if kind == "cancelled":
        return {"by": outcome.by}, None
    return {"error": str(outcome)}, None


def _explanation(outcome: Any, kind: OutcomeKind) -> str:
    if kind in ("done", "waiting"):
        return outcome.explanation
    if kind == "need_input":
        return outcome.question
    if kind == "cancelled":
        return f"cancelled by {outcome.by}"
    return str(outcome)


def _usage_delta(before: Usage, after: Usage) -> Usage:
    return Usage(**{name: getattr(after, name) - getattr(before, name) for name in USAGE_FIELDS})
