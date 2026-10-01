# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable admission, prompt outcomes and resource ownership around core TurnLoop.

The cancellable prepare hook applies model changes while inputs remain queued.
The synchronous commit hook records ItemConsumed + TurnStarted atomically before
ownership transfer. Channel events account for mid-turn get/drain separately.
See the package README for failure recovery and crash replay limitations.
"""

import asyncio
import contextvars
import hashlib
import inspect
import json
import logging
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from nooa.context_blocks.roles import Role
from nooa.events import Notification
from nooa.interactive import (
    AgentMessage,
    Done,
    InteractiveAgent,
    NeedInput,
    Waiting,
    apply_model_limits,
)
from nooa.llm_types import LLMResponse
from nooa.runtime.turn_loop import TurnLoopEnded, TurnSettled
from nooa.storage.json_snapshot import snapshot_to_json
from nooa_coder.session.events import (
    ItemAdmitted,
    ItemConsumed,
    ItemDiscarded,
    ItemRequeued,
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
    CommandsChangedUpdate,
    ItemAdmittedUpdate,
    ItemConsumedUpdate,
    ModeChangedUpdate,
    ModelChangedUpdate,
    ModelInfo,
    PlanEntry,
    ReasoningChangedUpdate,
    Receipt,
    SessionEvent,
    SessionInfo,
    TitleChangedUpdate,
    TranscriptEntry,
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


def _preview(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


_closing_sessions: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "closing_sessions", default=()
)


class Session:
    """One agent, its turn loop and its durable record.

    Built by the registry, which calls :meth:`start` after publishing it.
    Nobody else holds the agent: it is private (``_agent``), read only by
    the registry in this package. Hosts use the methods and the update
    stream (``subscribe``), which carries the agent's events as
    ``AgentEventUpdate``.
    """

    def __init__(
        self,
        *,
        options: SessionOptions,
        agent: InteractiveAgent,
        handle: SessionHandle,
        owned_llm: Any = None,
        llm_factory: Callable[[str | None, Path], Any] | None = None,
        before_close: Callable[[], Awaitable[None]] | None = None,
        llm_in_use: Callable[[Any], bool] | None = None,
    ) -> None:
        info = handle.info
        self.id: str = info.id
        self.parent_id: str | None = info.parent_id
        self.depth: int = info.depth
        self.name: str | None = info.name
        self.options = options
        self._agent = agent
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
        self.llm_in_use: Callable[[Any], bool] = llm_in_use or (lambda _llm: False)
        self._llm_factory = llm_factory
        self._pending_model: tuple[str, Any] | None = None  # (alias, built client)
        self._listeners: list[Callable[[SessionEvent], None]] = []
        # Per channel, (item, item_id) in put order: channels hold raw
        # objects, so this is how an item keeps its identity until consumed.
        self._ids: dict[str, deque[tuple[Any, str]]] = {}
        # Consumed raw inputs whose ledger write failed: not queued identities.
        # Retain ownership/error details for diagnosis and never-consumed crash replay.
        self._orphans: dict[str, tuple[str, Any, BaseException]] = {}
        self._futures: dict[str, asyncio.Future[Outcome]] = {}
        self._finished: OrderedDict[str, Any] = OrderedDict()
        self._consumed: list[str] = []  # consumed since the last turn settled
        self._waiting: list[str] = []  # items whose prompt stays open over a Waiting
        self._recording_error: BaseException | None = None
        self._started = False
        self._usage_before: Usage = self.info.usage.model_copy()
        self._turn_messages: set[str] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False
        self._before_close = before_close
        self._loop_context_hooks: list[Callable[[], object]] = []
        self._pending_steers: list[tuple[str, str, str]] = []  # (item_id, text, source)
        self._snapshot_digest: str | None = None
        self._checkpoint_task: asyncio.Task[None] | None = None
        self._unsubscribe_agent = agent.event_manager.on("*", self._on_agent_event)
        self._unsubscribe_steers = agent.event_manager.on("BeforeTurn", self._flush_steers)
        # The agent's queue channels publish every item they hand to a consumer
        # (the loop's race and drain, agent get()) and every item they drop
        # unconsumed (flush, clear, channel removed).
        # The agent's TurnLoop publishes each turn's start and end, and its own end.
        events = agent.event_manager
        self._unsubscribe_items = (
            events.on(
                "ChannelItemConsumed",
                lambda e: None if e.batch else self._on_consumed(e.channel, e.item),
            ),
            events.on("ChannelItemsDiscarded", lambda e: self._on_discarded(e.channel, e.items)),
            events.on("TurnBegan", self._on_turn_began),
            events.on("TurnSettled", self._on_turn_settled),
            events.on("TurnLoopEnded", self._on_loop_ended),
        )
        if self.info.reasoning:
            self._restore_reasoning(self.info.reasoning)
        # The one listener of the agent's command registry: hosts see
        # changes as CommandsChangedUpdate.
        set_on_change = getattr(getattr(agent, "slash_commands", None), "set_on_change", None)
        if callable(set_on_change):
            set_on_change(self._on_commands_changed)

    # ---- lifecycle ---------------------------------------------------

    def add_loop_context_hook(self, hook: Callable[[], object]) -> None:
        """Run ``hook`` once in the loop's own context before the loop starts.

        Context variables set there are seen by every turn (each turn task
        copies the loop's context), e.g. the port ``ChildRef`` resolves.
        """
        self._loop_context_hooks.append(hook)

    def start(self) -> None:
        """Start the agent's turn loop with the options' turn method.

        The loop runs in a fresh ``contextvars.Context`` (a session created
        from inside another agent's cell must not inherit that agent's call
        stack, generation state or scoped blocks), after the loop context
        hooks have run in it.
        """
        if self._started:
            return
        self._ensure_open()
        self._started = True
        context = contextvars.Context()
        for hook in self._loop_context_hooks:
            context.run(hook)
        self._agent.turns.start(
            turn_method=self.options.turn_method,
            context=context,
            prepare=self._prepare_turn,
            commit=self._commit_turn,
        )

    @property
    def closing(self) -> bool:
        """Closing or closed: new external admissions are forbidden."""
        return self._closing or self._closed

    async def close(self) -> None:
        """Stop the loop and release everything the session owns. Idempotent.

        Order: children first (the registry's hook), cancel a running turn,
        stop the loop, shut down the agent's background jobs, close the
        agent, close a model client the session created, close the store
        handle.
        """
        if self.id in _closing_sessions.get():
            return
        if self._agent.turns.in_turn:
            raise RuntimeError("an executing turn cannot close its own session")
        if self._closed:
            return
        if self._close_task is None:
            self._closing = True
            self._agent.turns.pause()
            self._close_task = asyncio.ensure_future(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        token = _closing_sessions.set((*_closing_sessions.get(), self.id))
        try:
            await self._close_body()
        finally:
            _closing_sessions.reset(token)

    async def _close_body(self) -> None:
        # Every step has its own guard: a failure (or a CancelledError out
        # of a step) is logged and the close goes on, so the model client is
        # closed, the file lock released and ClosedUpdate emitted whatever
        # failed before.
        self._closing = True
        # No new turn from here on (a running one goes on while children
        # close): an item a doomed turn took would be recorded as consumed
        # and never come back on load.
        self._agent.turns.stop_starting()
        await self._close_step("closing its children", self._before_close)
        await self._close_step("stopping the turn loop", self._agent.turns.stop)
        self._resolve_all(TurnCancelledOutcome(by="host"))
        pending, self._pending_model = self._pending_model, None
        if pending is not None:
            await self._close_step("closing the pending model client", lambda: _aclose(pending[1]))
        await self._close_step("waiting for the checkpoint", self.wait_for_checkpoint)
        await self._close_step("unsubscribing from agent events", self._unsubscribe_agent)
        await self._close_step("unsubscribing the steer flush", self._unsubscribe_steers)
        await self._close_step("stopping the agent's jobs", self._agent.queue_manager.shutdown)
        await self._close_step("closing the agent", self._agent.aclose)
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

    async def cancel(self, *, by: str = "user") -> bool:
        """Stop the running turn; return whether one was running.

        Returns only after the turn has settled: the interrupted cell's
        cancelled output and a ``TurnCancelled`` event are in the agent's
        events, so the model sees at its next turn that it was stopped.
        Queued items are kept and the loop goes on. With no turn running,
        prompts left open by a ``Waiting`` are closed with
        ``TurnCancelledOutcome`` and no event is written.
        """
        if await self._agent.turns.cancel(by=by):
            return True
        waiting, self._waiting = self._waiting, []
        for item_id in waiting:
            self._resolve(item_id, TurnCancelledOutcome(by=by))
        return False

    async def _close_owned_llm(self) -> None:
        llm, self._owned_llm = self._owned_llm, None
        retired, self._retired_llms = self._retired_llms, []
        for client in (*retired, llm):
            await self._close_step("closing a model client", lambda c=client: _aclose(c))

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise SessionClosedError(f"Session {self.id!r} is closed")

    # ---- input -------------------------------------------------------

    def admit(self, item: Any, *, channel: str = "user_messages", source: str = "user") -> Receipt:
        """Synchronous record-before-queue admission for internal event delivery."""
        self._ensure_open()
        return self._admit(item, channel=channel, source=source)

    def _ensure_external_admission(self) -> None:
        """Reject new external work while allowing durable internal deliveries."""
        self._ensure_open()
        if self._agent.turns.dispatch_error is not None:
            raise TurnFailedError(
                "dispatch is blocked; repair and resume_dispatch()", self._agent.turns.dispatch_error
            )

    def requeue(self, item: Any, *, channel: str, source: str, item_id: str) -> Receipt:
        """Own the replay record and synchronous queue admission together."""
        self._ensure_open()
        target = self._agent.queue_manager.channels().get(channel)
        if target is None or target.mode != "queue":
            raise ValueError(f"no queue channel {channel!r}")
        self.handle.events.add(ItemRequeued(item_id=item_id))
        return self._admit(item, channel=channel, source=source, item_id=item_id, record=False)

    def resume_dispatch(self) -> None:
        """Explicit retry after fixing a preparation/ledger failure; inputs stayed queued.

        Previously failed prompt outcomes remain failed; use a new prompt or
        observe updates for retries. Crash replay remains never-consumed only.
        """
        self._ensure_open()
        self._admit_leftover_steers()  # repaired ledger: explicit buffered-steer retry
        self._recording_error = None
        self._agent.turns.resume()

    async def submit(
        self, item: Any, *, channel: str = "user_messages", source: str = "user"
    ) -> Receipt:
        """Admit ``item`` on ``channel``: recorded first, then queued for a turn."""
        self._ensure_external_admission()
        return self.admit(item, channel=channel, source=source)

    async def steer(self, text: str, *, source: str = "user") -> Receipt:
        """Give the running turn extra text; while idle this is ``submit(text)``.

        During a turn the text waits in a buffer that is flushed into a
        ``Notification`` right before the turn's next model call, so the
        model reads it in order with its own cell output. The event has the
        shape of a turn's input: its ``value`` is ``{"user_messages":
        [text]}``, its ``source`` names the channel and the sender, and its
        ``description`` says how to reach the items. If no model call comes (the turn was already
        finishing), the text is admitted on ``user_messages`` when the turn
        settles, with the same ``item_id``, and the next turn handles it.
        A steer is never lost.
        """
        self._ensure_external_admission()
        if not self._agent.turns.running or self.info.status != "running":
            return self.admit(text, channel="user_messages", source=source)
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
        if not self._pending_steers or not self._agent.turns.running:
            return
        while self._pending_steers:
            item_id, text, source = self._pending_steers[0]
            try:
                self.handle.events.add(ItemConsumed(item_id=item_id))
                self._consumed.append(item_id)
                self._pending_steers.pop(0)
                self._agent.event_manager.add(
                    Notification(source=_steer_source(source), description=_STEER_HINT, value={"user_messages": [text]})
                )
                self._emit(ItemConsumedUpdate(session_id=self.id, channel="steer", item_id=item_id))
            except (Exception, asyncio.CancelledError) as exc:
                self._record_failure(exc, [item_id])
                return

    def _admit_leftover_steers(self) -> None:
        """Transfer one at a time; a write failure retains remaining ownership."""
        while self._pending_steers:
            item_id, text, source = self._pending_steers[0]
            self._admit(
                text, channel="user_messages", source=source, item_id=item_id, internal=True
            )
            self._pending_steers.pop(0)

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
        channel = self._agent.queue_manager.channels().get(channel_name)
        if not entries or channel is None:
            return False
        index = next((i for i, (_, known) in enumerate(entries) if known == item_id), None)
        if index is None:
            return False
        # Channels match by identity; equal-identity entries are interchangeable
        # in the channel, so removing the first one keeps both sides in step.
        if not channel.remove(entries[index][0]):
            return False
        del entries[index]
        return True

    async def prompt(self, text: str, *, source: str = "user") -> Outcome:
        """Submit ``text`` and wait for the outcome of the turn that consumes it.

        A ``Waiting`` outcome keeps the wait open; the next turn's outcome
        resolves it. A cancelled turn resolves it with
        ``TurnCancelledOutcome``; a failed turn raises ``TurnFailedError``.
        """
        receipt = await self.submit(text, channel="user_messages", source=source)
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
            if item_id in self._finished:
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
        target = self._agent.queue_manager.channels().get(channel)
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

    def _record_failure(self, exc: BaseException, item_ids: list[str]) -> None:
        self._recording_error = exc
        self._agent.turns.dispatch_error = exc
        self._agent.turns.pause()
        error = TurnFailedError(f"consumption recording failed: {type(exc).__name__}: {exc}", exc)
        for item_id in [*item_ids, *(identity for identity, _, _ in self._pending_steers)]:
            self._resolve(item_id, error)

    def _on_consumed(self, channel: str, item: Any) -> None:
        entries = self._ids.get(channel)
        if not entries:
            return
        index = next((i for i, (obj, _) in enumerate(entries) if obj is item), None)
        if index is None:
            return
        item_id = entries[index][1]
        try:
            if self.handle.closed:
                raise SessionClosedError("consumption on a closed handle")
            self.handle.events.add(ItemConsumed(item_id=item_id))
        except (Exception, asyncio.CancelledError) as exc:
            # get() already transferred the raw object. It no longer owns a
            # queue slot, even if the same object is submitted again on resume.
            del entries[index]
            self._orphans[item_id] = (channel, item, exc)
            self._record_failure(exc, [item_id])
            return
        del entries[index]
        self._consumed.append(item_id)
        self._emit(ItemConsumedUpdate(session_id=self.id, channel=channel, item_id=item_id))

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
            if not self.handle.closed:
                self.handle.events.add(ItemDiscarded(item_id=item_id))
            self._resolve(
                item_id,
                ItemDiscardedError(
                    f"Item {item_id!r} was dropped from channel {channel!r} before any turn "
                    "consumed it"
                ),
            )

    # ---- turns: the agent's TurnLoop runs them, the Session records them ----

    async def _prepare_turn(self) -> None:
        self._usage_before = self.info.usage.model_copy()
        await self._apply_pending_model()

    def _commit_turn(self, notification: dict[str, list[Any]]) -> None:
        """No await/observers: atomic ledger first, identity ownership second."""
        self._ensure_open()
        if self.handle.closed:
            raise SessionClosedError("turn commit on a closed handle")
        selected: list[tuple[str, Any, str]] = []
        for channel, items in notification.items():
            entries = list(self._ids.get(channel, []))
            for item in items:
                index = next((i for i, (obj, _) in enumerate(entries) if obj is item), None)
                if index is not None:
                    obj, item_id = entries.pop(index)
                    selected.append((channel, obj, item_id))
        item_ids = [item_id for _, _, item_id in selected]
        self.handle.events.record_batch(
            [
                *(ItemConsumed(item_id=item_id) for item_id in item_ids),
                TurnStarted(item_ids=item_ids, item_preview=_preview(notification)),
            ]
        )
        for channel, obj, item_id in selected:
            entries = self._ids[channel]
            index = next(
                i
                for i, (known, identity) in enumerate(entries)
                if known is obj and identity == item_id
            )
            del entries[index]
        self._batch_consumed = [(channel, item_id) for channel, _, item_id in selected]
        self._consumed.extend(item_ids)
        self.info.status = "running"

    def _on_turn_began(self, _event: Any) -> None:
        self._turn_messages.clear()
        for channel, item_id in self._batch_consumed:
            self._emit(ItemConsumedUpdate(session_id=self.id, channel=channel, item_id=item_id))
        self._emit(TurnStartedUpdate(session_id=self.id, item_ids=list(self._consumed)))

    def _on_turn_settled(self, event: TurnSettled) -> None:
        """``TurnSettled``: record the outcome; if recording fails, fail the turn's prompts."""
        if not event.committed:
            if event.kind in ("error", "cancelled"):
                error = (
                    TurnFailedError(event.message, event.error)
                    if event.kind == "error"
                    else TurnCancelledOutcome(by=event.cancelled_by or "host")
                )
                self._resolve_all(error)  # includes late queries for Waiting/steers/queued input
            return  # No durable turn began; queued input retains replay eligibility.
        if self._recording_error is not None:
            event = TurnSettled(
                kind="error",
                error=self._recording_error,
                message=f"consumption recording failed: {self._recording_error}",
            )
        try:
            self._settle(event)
        except (Exception, asyncio.CancelledError) as exc:
            logger.exception("Session %s: turn bookkeeping failed", self.id)
            self._fail_turn(exc)

    def _on_loop_ended(self, event: TurnLoopEnded) -> None:
        """The loop cannot wait for input: fail every open outcome and close the session."""
        error = TurnFailedError(f"The session's turn loop stopped: {event.message}", event.error)
        self._resolve_all(error)
        if self._close_task is None:
            self._close_task = asyncio.get_running_loop().create_task(
                self._close(), name=f"session-close:{self.id}"
            )

    def _fail_turn(self, exc: BaseException) -> None:
        """Settle a turn whose own settling failed: its prompts get ``TurnFailedError``."""
        owed = self._waiting + self._consumed + [item_id for item_id, _, _ in self._pending_steers]
        self._waiting, self._consumed = [], []
        if self._pending_steers:
            self._record_failure(exc, owed)
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

    def _send_result_message(self, outcome: Any) -> bool:
        """Show a ``Done``/``Waiting`` message as an agent message, inside the turn.

        It goes through ``agent.message()`` like any reply, before the turn
        is recorded as ended; text the turn already sent is not sent again.
        Returns whether it sent one.
        """
        text = getattr(outcome, "message", None) if isinstance(outcome, Done | Waiting) else None
        if not text or text in self._turn_messages:
            return False
        send = getattr(self._agent, "message", None)
        if callable(send):
            send(text)
        else:
            self._agent.event_manager.add(AgentMessage(content=text))
        return True

    def _settle(self, event: TurnSettled) -> None:
        # Items stay in self._consumed / self._waiting until the end, so a
        # failure part way leaves them for _fail_turn to resolve.
        kind: OutcomeKind = event.kind
        outcome: Any
        if kind == "cancelled":
            outcome = TurnCancelledOutcome(by=event.cancelled_by or "host")
            self._emit(
                CancelledUpdate(session_id=self.id, by=outcome.by, interrupted=event.interrupted)
            )
        elif kind == "error":
            outcome = TurnFailedError(event.message, event.error)
        else:
            outcome = event.result
        self._send_result_message(outcome)
        consumed = list(self._consumed)
        if self._recording_error is None:
            self._admit_leftover_steers()
        # A failed steer delivery retains its buffer/ledger ownership for recovery.
        usage = _usage_delta(self._usage_before, self.info.usage)
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
        self.info.reply_count += 1  # as the store counts it
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

    def _checkpoint(self) -> None:
        """Save the agent's state if it changed since the last checkpoint.

        The snapshot is serialised and hashed here, on the loop, where the
        agent is not changing; only the write happens in a thread. Writes
        run one after another. A failure is logged and the turn is not
        affected; the next settled turn tries again.
        """
        try:
            blob = json.dumps(snapshot_to_json(self._agent), sort_keys=True)
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
        storage = self.handle.storage

        async def write() -> None:
            if previous is not None:
                with suppress(Exception):
                    await previous
            try:
                # Takes the storage manager's lock, like every event write.
                await asyncio.to_thread(storage.save_snapshot_json, blob)
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

    def _record_finished(self, item_id: str, outcome: Any) -> None:
        """Keep a recent item's outcome so ``outcome()`` can still answer it."""
        if item_id in self._finished:
            return  # A retry must not rewrite an already reported terminal outcome.
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
        pending = set(self._futures) | set(self._waiting) | set(self._consumed)
        pending.update(item_id for entries in self._ids.values() for _, item_id in entries)
        pending.update(item_id for item_id, _, _ in self._pending_steers)
        self._waiting = []
        for item_id in pending:
            self._resolve(item_id, outcome)

    # ---- tools -------------------------------------------------------

    async def prepare_tools(self) -> list[str]:
        """Run the agent's own tool set-up; return warnings for the user.

        Awaits the agent's ``prepare_tools()`` hook if it has one (the
        coding agent connects the MCP servers its workspace remembers).
        A host calls this once, in the registry's ``prepare`` step, before
        the first turn.
        """
        hook = getattr(self._agent, "prepare_tools", None)
        if not callable(hook):
            return []
        return [str(warning) for warning in await hook()]

    def register_tools(self, tools: Mapping[str, Any]) -> dict[str, str]:
        """Register and activate each tool as an agent skill under its name.

        Returns the tools that were not registered, name to reason (the
        agent has no skills, or the name collides with one the agent
        already provides); the others are registered. Call before the
        first turn (the registry's ``prepare`` step).
        """
        skills = getattr(self._agent, "skills", None)
        failed: dict[str, str] = {}
        for name, tool in tools.items():
            if skills is None:
                failed[name] = "the agent has no skills"
                continue
            try:
                skills.register(name, tool)
                skills.activate([name])
            except ValueError as exc:
                failed[name] = str(exc)
        return failed

    # ---- slash commands ----------------------------------------------

    def _on_commands_changed(self, _commands: object) -> None:
        self._emit(CommandsChangedUpdate(session_id=self.id, commands=self.commands()))

    def commands(self) -> list[CommandInfo]:
        """Slash commands of the agent's ``slash_commands`` registry, if it has one.

        The registry has the coding agent's shape (``CodingSlashCommandRegistry``):
        ``commands()`` returns objects with ``name``, ``description`` and
        ``argument_hint``, and ``invoke(name, raw_args)`` runs one.
        """
        registry = getattr(self._agent, "slash_commands", None)
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
        registry = getattr(self._agent, "slash_commands", None)
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
        recorded at once (``info.model``, and the store, so a load resumes
        on it) and a ``ModelChangedUpdate`` is emitted.
        """
        if self._llm_factory is None:
            raise RuntimeError("set_model() needs the registry's llm_factory to build clients")
        self._ensure_open()
        client = self._llm_factory(alias, self.options.workspace)
        # Own immediately: a metadata write failure must not orphan this client.
        self._retired_llms.append(client)
        # Recorded now: a load before the next turn resumes on this model.
        # The new client starts from its own reasoning default.
        self.handle.set_model(alias)
        self.info.model = alias
        self.info.reasoning = None
        previous, self._pending_model = self._pending_model, (alias, client)
        self._retired_llms.remove(client)
        if previous is not None:
            self._retired_llms.append(previous[1])
            await _aclose(previous[1])
            self._retired_llms.remove(previous[1])
        self._emit(ModelChangedUpdate(session_id=self.id, model=alias))

    async def _apply_pending_model(self) -> None:
        pending, self._pending_model = self._pending_model, None
        if pending is None:
            return
        alias, client = pending
        old_agent_llm = self._agent.llm
        try:
            self._agent.set_llm(client)
            apply_model_limits(self._agent)
        except BaseException:
            # Attempt once. Failed clients remain owned for later cleanup.
            self._retired_llms.append(client)
            self._agent.set_llm(old_agent_llm)
            raise
        old, self._owned_llm = self._owned_llm, client
        # New same-model children share the new client.
        self.options = self.options.model_copy(update={"model": alias, "llm": client})
        self.info.model = alias
        if old is not None:
            self._retired_llms.append(old)  # retain ownership across cancelled/hung disposal
            if not self.llm_in_use(old):
                await _aclose(old)
                self._retired_llms.remove(old)

    def channels(self) -> list[str]:
        """Names of the agent's queue channels: where ``submit`` can put an item."""
        return list(self._agent.queue_manager.channels())

    def _next_llm(self) -> Any:
        """The client the next model call uses: one a ``set_model`` left pending, else the agent's."""
        if self._pending_model is not None:
            return self._pending_model[1]
        return getattr(self._agent, "llm", None)

    def model_info(self) -> ModelInfo:
        """The model the next call uses, as data: alias, context window, reasoning levels.

        After a ``set_model`` whose client is not swapped in yet, this
        describes that client: it is the one the next turn uses.
        """
        client = self._next_llm()
        return ModelInfo(
            alias=self.info.model,
            context_window=getattr(client, "context_window", None),
            reasoning_level=getattr(client, "reasoning_level", None),
            reasoning_levels=list(getattr(client, "reasoning_levels", None) or ()),
            reasoning_default=getattr(client, "reasoning_default", None),
        )

    async def set_reasoning(self, level: str | None) -> None:
        """Choose a reasoning level the client declares; it applies from the next model call.

        The level is set on the client the next call uses (see
        ``model_info``), recorded (``info.reasoning``, and the store, so a
        load restores it) and announced with a ``ReasoningChangedUpdate``.
        A later ``set_model`` resets it: the new client starts from its own
        default. A same-model child shares its parent's client, so a level
        set on either applies to both.

        Raises:
            ValueError: If the client does not declare ``level``.
        """
        self._ensure_open()
        client = self._next_llm()
        levels = tuple(getattr(client, "reasoning_levels", None) or ())
        if level is not None and level not in levels:
            allowed = ", ".join(levels) if levels else "none for this model"
            raise ValueError(f"Unknown reasoning level {level!r}; allowed: {allowed}")
        # None clears the choice: the model's own default applies again.
        client.reasoning_level = level
        self.handle.set_reasoning(level)
        self.info.reasoning = level
        self._emit(ReasoningChangedUpdate(session_id=self.id, level=level))

    def _restore_reasoning(self, level: str) -> None:
        """Apply a recorded level on load, if the client still declares it."""
        client = getattr(self._agent, "llm", None)
        if level in tuple(getattr(client, "reasoning_levels", None) or ()):
            client.reasoning_level = level
        else:
            logger.info(
                "Session %s: the recorded reasoning level %r is not offered by the model; "
                "using its default",
                self.id,
                level,
            )

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
        totals.last_input_tokens = usage.input_tokens or 0
        self.handle.update_usage(totals)
        self._emit(UsageChangedUpdate(session_id=self.id, usage=totals.model_copy()))

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
            except (Exception, asyncio.CancelledError):
                logger.warning("Session listener %r raised", listener, exc_info=True)

    def _on_agent_event(self, event: Any) -> None:
        # First, so listeners see the event before anything it causes here
        # (the usage update of a model response).
        if self._listeners:
            self._emit(AgentEventUpdate(session_id=self.id, event=event))
        if event.event_role is Role.RUNTIME_EVENT:
            return
        if isinstance(event, LLMResponse):
            self._count_usage(event)
        if isinstance(event, AgentMessage) and self._agent.turns.in_turn:
            self._turn_messages.add(event.content)

    def plan(self) -> list[PlanEntry]:
        """The agent's plan for a host to show, as ACP plan entries; empty if it has none.

        An agent offers it with a ``plan()`` method returning ``PlanEntry``
        values or dicts with their fields (the coding agent derives them
        from its todos). A failure is logged and reads as no plan.
        """
        hook = getattr(self._agent, "plan", None)
        if not callable(hook):
            return []
        try:
            return [PlanEntry.model_validate(entry) for entry in hook()]
        except Exception:
            logger.warning("Session %s: could not read the agent's plan", self.id, exc_info=True)
            return []

    def transcript(self, *, limit: int | None = None) -> list[TranscriptEntry]:
        """The session's transcript as a person would see it; the last ``limit`` entries."""
        entries = self.handle.transcript()
        return entries if limit is None else entries[-limit:]


_STEER_HINT = (
    "Sent during this turn. The value has the form of the notification argument of "
    'handle(); reach it as self.events["N"].value, where N is the tag of this event.'
)
"""The ``Notification.description`` of a steer: when it came and how to reach it."""

_PERSON_SOURCES = ("user", "acp")
"""Item sources that are the person: ``acp`` is a person typing in an ACP client."""


def _steer_source(source: str) -> str:
    """The ``Notification.source`` sentence for a steer: its channel and who sent it."""
    if source in _PERSON_SOURCES:
        sender = "the user"
    elif source.startswith("parent:"):
        sender = "your parent agent " + source.removeprefix("parent:")
    else:
        sender = source
    return f"New message on user_messages from {sender} while you were working."


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
