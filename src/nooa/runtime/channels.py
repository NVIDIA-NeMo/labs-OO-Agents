# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Channel + QueueManager — unified producer side for agent input.

Replaces ``InputQueue`` / ``OutputQueue`` / ``wait_for_any`` with one
class plus a registry. Two modes:

- **queue**: deque-backed, races for delivery, agent can drain mid-turn.
- **event**: fire-and-forget into the agent's event log; no buffer,
  no get(). Renders inline in the next turn's prompt.

``QueueManager`` owns the channel registry, the composite ``status()``
block, and the ``race()`` primitive. Channels still live as direct
agent attributes for producer/consumer code; the manager is for
registration + aggregation.

"""

import asyncio
import inspect
import logging
from collections import deque
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Annotated, Any, Literal

from pydantic import Field

from nooa.context_blocks import EventBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QueueOutput — event fired by event-mode channels into event_manager.
# Lives here (not in events.py) so the channel package stays additive.
# ---------------------------------------------------------------------------


class QueueOutput(EventBase):
    """Event emitted by an event-mode channel's ``put()``.

    Renders via ``pformat(event)`` per the existing event pipeline:
    ``source`` and ``value_type`` and ``value_preview`` go into the
    prompt; ``value`` is hidden via ``repr=False`` but accessible to
    non-LLM code via ``event_manager.get(tag).value``.
    """

    source: str
    value_type: str
    value_preview: str
    value: Annotated[Any, Field(repr=False)] = None


class StreamEnd(EventBase):
    """Emitted to the agent's event channel when a spawned job finishes.

    Signals that no more values will arrive on ``channel_name``.
    Lives here (not in events.py) per the additive-only constraint.
    """

    channel_name: str


class QueueReadTimeoutError(TimeoutError):
    """Raised when an LLM-facing queue reader times out waiting for an item."""


class JobError(EventBase):
    """Emitted to the agent's event channel when a spawned job fails."""

    channel_name: str
    error_type: str
    error_message: str


# ---------------------------------------------------------------------------
# JobHandle
# ---------------------------------------------------------------------------

JobState = Literal["running", "done", "cancelled", "failed"]


class JobHandle:
    """Handle returned by ``QueueManager.spawn()``.

    Tracks the lifecycle of one spawned coroutine or async generator.
    ``cancel()`` requests cancellation and awaits cleanup so generator
    ``finally`` blocks run before the call returns.

    If ``buffer`` was passed to ``spawn()``, yielded values accumulate
    in ``self.values``:
    - ``buffer=False`` (default): no accumulation, ``.values`` is empty.
    - ``buffer=True``: unbounded list of all yielded values.
    - ``buffer=N`` (int): ring buffer keeping the last N values.
    """

    def __init__(
        self, name: str, task: asyncio.Task[Any], buffer: bool | int = False, label: str = ""
    ) -> None:
        self.name = name
        self.label = label or name
        self.state: JobState = "running"
        self._task = task
        if buffer is True:
            self._values: list[Any] | deque[Any] = []
        elif isinstance(buffer, int) and buffer > 0:
            self._values = deque(maxlen=buffer)
        else:
            self._values = None  # type: ignore[assignment]

    @property
    def values(self) -> list[Any]:
        """All buffered values (or last N if ring buffer). Empty if buffer=False."""
        if self._values is None:
            return []
        return list(self._values)

    async def cancel(self) -> None:
        """Cancel the underlying task and await its unwinding."""
        if self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            # Task.cancel() delegates to the future being awaited, so the
            # job's unwinding and our caller's own cancellation both surface
            # here. cancelling() distinguishes them; the caller's is not ours
            # to swallow.
            current = asyncio.current_task()
            if current is not None and current.cancelling() > 0:
                raise
        except Exception:
            pass
        if self.state == "running":
            self.state = "cancelled"

    def __repr__(self) -> str:
        return f"JobHandle(name={self.name!r}, label={self.label!r}, state={self.state!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


ChannelMode = Literal["queue", "event"]


def _preview_item(item: Any, max_chars: int) -> str:
    """Single-line bounded preview for the queue ``status()`` block.

    Strings render as-is; non-strings go through ``pformat`` so deeply
    nested dicts/lists come out bounded by default. Newlines and tabs
    flatten to single chars so previews stay one line, then the whole
    thing is clipped to ``max_chars`` with an ellipsis when needed.
    """
    from nooa.agentdoc import pformat

    if isinstance(item, str):
        text = item
    else:
        text = pformat(item, max_length=max_chars, max_string=max_chars, max_depth=2)
    text = text.replace("\r\n", "\n").replace("\n", "↵").replace("\t", " ")
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


def _default_event_preview(value: Any) -> str:
    """Default ``value_preview`` for event-mode puts.

    Strings under 200 chars pass through; everything else is
    ``pformat``-rendered with bounded budgets.
    """
    from nooa.agentdoc import pformat

    if isinstance(value, str):
        return value if len(value) <= 200 else value[:199] + "…"
    return pformat(value, max_length=10, max_string=200, max_depth=3)


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------


class Channel[T]:
    """A named channel for agent input.

    Construct via ``QueueManager.queue(...)`` or ``QueueManager.event(...)``.
    Two modes:

    queue mode
        Deque-backed. ``put()`` enqueues; ``get()`` blocks until an
        item arrives. The dispatcher races multiple queue-mode
        channels via ``QueueManager.race()``. Agent code can drain
        mid-turn via ``await self.<channel>.get()``. The ``on_get``
        hook fires once per item at the queued → consumed transition,
        regardless of whether the dispatcher or agent code is the
        consumer.

    event mode
        Fire-and-forget. ``put()`` adds a ``QueueOutput`` event to
        the agent's ``event_manager``; the value renders inline in
        the next turn's prompt. No buffer, no ``get()``. Use for
        notifications and one-shot status the agent should notice
        without actively pulling.
    """

    def __init__(
        self,
        name: str,
        mode: ChannelMode,
        *,
        event_manager: Any = None,
        on_get: Callable[[T], None] | None = None,
        on_put: Callable[[], None] | None = None,
        preview: Callable[[T], str] | None = None,
    ) -> None:
        if mode not in ("queue", "event"):
            raise ValueError(
                f"Channel mode must be 'queue' or 'event', got {mode!r}. "
                "Use QueueManager.queue() / QueueManager.event() factories "
                "rather than constructing Channel directly."
            )
        self.name = name
        self.mode: ChannelMode = mode
        self._event_manager = event_manager
        self._items: deque[T] = deque()
        self._waiters: deque[asyncio.Future[Any]] = deque()
        # Fired once per item at the queued → consumed transition. Both
        # the dispatcher (race winner) and agent mid-turn ``get()`` go
        # through the same firing path so callers can rely on symmetric
        # observation.
        self._on_get: Callable[[T], None] | None = on_get
        # Fired on every put() — used by QueueManager to set its
        # _notify event so race() can wake on event-mode puts.
        self._on_put: Callable[[], None] | None = on_put
        self._preview: Callable[[T], str] = preview or _default_event_preview

    # ---- producer side ---------------------------------------------------

    def put(self, item: T) -> None:
        """Push *item* through the channel.

        queue mode
            Append to the backing deque, then wake one waiter. Consumers pop
            only after their task resumes, matching ``asyncio.Queue``
            cancellation semantics: a cancelled waiter cannot strand a
            delivered-but-unclaimed item in its Future.

        event mode
            Build a ``QueueOutput`` and add it to ``event_manager``.
            No buffer; the value lands in the prompt for the next
            agent turn.

        Both modes fire the ``_on_put`` callback (when set) so the
        owning ``QueueManager`` can wake ``race()`` on event-mode puts
        and on queue-mode buffered puts alike.
        """
        if self.mode == "event":
            if self._event_manager is None:
                raise RuntimeError(f"event-mode channel {self.name!r} has no event_manager")
            self._event_manager.add(
                QueueOutput(
                    source=self.name,
                    value_type=type(item).__name__,
                    value_preview=self._preview(item),
                    value=item,
                )
            )
            if self._on_put is not None:
                self._on_put()
            return

        self._items.append(item)
        self._wake_next_waiter()
        if self._on_put is not None:
            self._on_put()

    def _wake_next_waiter(self) -> None:
        """Wake one live waiter, if any, without transferring an item."""
        while self._waiters:
            waiter = self._waiters.popleft()
            if waiter.done():
                continue
            waiter_loop = waiter.get_loop()
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is waiter_loop:
                if not waiter.done():
                    waiter.set_result(None)
            else:

                def _safe_wake(w=waiter) -> None:
                    if not w.done():
                        w.set_result(None)

                try:
                    waiter_loop.call_soon_threadsafe(_safe_wake)
                except RuntimeError:
                    continue
            return

    def _restore_front(self, item: T) -> None:
        """Restore *item* at the head and wake a waiter to claim it."""
        self._items.appendleft(item)
        self._wake_next_waiter()
        if self._on_put is not None:
            self._on_put()

    # ---- consumer side ---------------------------------------------------

    async def _drain_one(self) -> T:
        """Block until an item arrives; return WITHOUT firing ``on_get``.

        Internal: ``QueueManager.race()`` calls this directly so it can
        fire ``on_get`` exactly once for the race winner even when
        multiple racing tasks complete in the same loop tick. The
        public ``get()`` always fires the hook.

        Cancellation-safe: a cancelled drain removes its waiter from
        the queue without consuming any item. Items stay in ``_items``
        until the consumer task resumes and pops them, so cancellation
        cannot leave a result stranded in a cancelled Future.
        """
        if self.mode != "queue":
            raise NotImplementedError(
                f"_drain_one is queue-mode only; channel {self.name!r} is {self.mode}"
            )
        while True:
            if self._items:
                return self._items.popleft()
            loop = asyncio.get_running_loop()
            waiter: asyncio.Future[Any] = loop.create_future()
            self._waiters.append(waiter)
            try:
                await waiter
            except asyncio.CancelledError:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
                if self._items:
                    self._wake_next_waiter()
                raise

    async def get(self) -> T:
        """Block until an item arrives, then return it.

        Fires ``on_get(item)`` exactly once per returned item, on the
        caller's event loop thread.

        Raises ``NotImplementedError`` for event-mode channels — they
        don't buffer, so there's nothing to ``get()``. Use the agent's
        event log to observe event-channel output.
        """
        if self.mode != "queue":
            raise NotImplementedError(
                f"get() is not valid on {self.mode}-mode channel {self.name!r}; "
                "event channels render values directly into the event log"
            )
        item = await self._drain_one()
        self._fire_on_get(item)
        return item

    def _fire_on_get(self, item: T) -> None:
        """Fire ``on_get`` for *item*, swallowing any exception.

        Catches ``BaseException`` deliberately. ``get()`` has already
        popped the item from the deque by the time this is called; if
        the hook raises out, the caller's ``await`` never binds and
        the item is lost. The hook is fire-and-forget UI bookkeeping
        and must never affect item delivery.
        """
        if self._on_get is None:
            return
        try:
            self._on_get(item)
        except BaseException:
            logger.exception("Channel(%s).on_get raised", self.name)

    def drain(self) -> list[T]:
        """Pop every buffered item now, firing ``on_get`` for each.

        Non-blocking, queue-mode only. Pops items head-to-tail (FIFO),
        firing ``on_get(item)`` once per item at the queued → consumed
        transition — the same firing path as ``get()`` and the race
        winner — then returns them in dequeue order. Returns an empty
        list for event-mode channels (they don't buffer) and for a
        drained queue.

        Encapsulates the ``popleft`` + ``_fire_on_get`` loop the TUI
        dispatcher runs to flush items already buffered when a turn
        wakes, so callers no longer reimplement channel-drain semantics.
        """
        if self.mode != "queue":
            return []
        drained: list[T] = []
        while self._items:
            item = self._items.popleft()
            self._fire_on_get(item)
            drained.append(item)
        return drained

    def set_on_get(self, callback: Callable[[T], None] | None) -> None:
        """Late-bind the ``on_get`` hook (queue-mode only).

        Useful when the channel is constructed before the consumer
        that wants to observe dequeues exists — e.g. the TUI
        ``Session`` wires its user-bar renderer here after the agent
        has already created its ``user_messages`` channel.
        """
        self._on_get = callback

    # ---- introspection ---------------------------------------------------

    def qsize(self) -> int:
        """Number of pending items. Always ``0`` for event mode."""
        return len(self._items)

    def is_empty(self) -> bool:
        return not self._items

    def has_waiters(self) -> bool:
        """True if a consumer is currently blocked on ``get()`` (queue mode)."""
        return any(not w.done() for w in self._waiters)

    def snapshot(self) -> list[T]:
        """Return a copy of pending items (head to tail). Non-consuming."""
        return list(self._items)

    def pop_last(self) -> T | None:
        """Remove and return the most recently put item (queue-mode tail).

        Used by the TUI for "edit what I just queued" UX (Up-arrow):
        the user pulls the queued message back into the input buffer.
        Returns ``None`` if the channel is empty.
        """
        if not self._items:
            return None
        return self._items.pop()

    def clear(self) -> None:
        self._items.clear()

    def flush(self) -> int:
        """Discard all pending items and cancel waiting consumers.

        Returns the number of buffered items that were discarded.
        Pending ``get()`` waiters are cancelled so they raise
        ``asyncio.CancelledError`` rather than hanging forever on a
        channel that has been flushed.
        """
        n = self.qsize()
        self._items.clear()
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.cancel()
        return n

    def status(self, *, max_items: int = 3, max_chars: int = 80) -> str:
        """Pending-count summary + short preview of waiting items.

        Returns an empty string when:
        - the channel is event-mode (no buffer),
        - or the queue is drained.

        Both make it cheap for ``QueueManager.status()`` to compose
        the per-channel parts and skip empty ones.
        """
        if self.mode != "queue" or not self._items:
            return ""
        lines = [f"{self.name}: {len(self._items)} pending"]
        snapshot = list(self._items)
        for i, item in enumerate(snapshot[:max_items], start=1):
            lines.append(f"  {i}. {_preview_item(item, max_chars)}")
        overflow = len(snapshot) - max_items
        if overflow > 0:
            lines.append(f"  … {overflow} more")
        return "\n".join(lines)

    # ---- lifecycle -------------------------------------------------------

    def __repr__(self) -> str:
        if self.mode == "queue":
            return f"Channel(name={self.name!r}, mode='queue', pending={len(self._items)})"
        return f"Channel(name={self.name!r}, mode='event')"

    @property
    def reader(self) -> "_ChannelReader[T]":
        """Read-only facade for LLM exposure.

        Use when attaching to an agent: keep the underlying ``Channel``
        on a ``hidden`` attribute (it has ``put``, ``snapshot``,
        ``pop_last``, etc. — full producer/dispatcher access) and
        expose ``.reader`` as the public attribute the LLM sees in
        ``doc(self)``.

        Cached, so repeated ``self.user_messages`` accesses return the
        same facade.
        """
        cached = getattr(self, "_reader_cache", None)
        if cached is None:
            cached = _ChannelReader(self)
            self._reader_cache = cached
        return cached


class _ChannelReader[T]:
    """LLM-facing read facade for a queue-mode ``Channel``.

    Exposes ``get()`` / ``qsize()`` / ``status()`` / ``name``. ``get()``
    has a short default timeout so generated code cannot accidentally
    block a turn forever while waiting on an empty queue.
    """

    def __init__(self, source: Channel[T]) -> None:
        self._source = source

    @property
    def name(self) -> str:
        return self._source.name

    async def get(self, timeout: float | None = 5.0) -> T:
        """Return the next item, timing out by default for LLM-facing reads.

        Pass ``timeout=None`` only when an indefinite mid-turn wait is
        intentional. Framework/dispatcher code should use ``Channel.get()``
        directly when it owns the wait lifecycle.
        """
        if timeout is None:
            return await self._source.get()
        try:
            async with asyncio.timeout(timeout):
                return await self._source.get()
        except TimeoutError as exc:
            raise QueueReadTimeoutError(
                f"Timed out after {timeout}s waiting for queue {self.name!r}. "
                "Use status() or qsize() to inspect queued items, or return "
                "WAIT/NEED_INPUT instead of blocking mid-cell. Pass "
                "timeout=None only when an indefinite wait is intentional."
            ) from exc

    def qsize(self) -> int:
        """Return the number of currently buffered items."""
        return self._source.qsize()

    def status(self, *, max_items: int = 3, max_chars: int = 80) -> str:
        """Delegate to the underlying ``Channel.status()``.

        Exposed so the LLM can peek at pending items mid-turn —
        ``self.user_messages.status()`` — without reaching into the
        producer-side handle.
        """
        return self._source.status(max_items=max_items, max_chars=max_chars)

    def flush(self) -> int:
        """Discard all pending items and cancel waiting consumers.

        Delegates to ``Channel.flush()``. Returns the number of
        buffered items that were discarded.
        """
        return self._source.flush()

    def __repr__(self) -> str:
        return f"ChannelReader(name={self.name!r})"


# ---------------------------------------------------------------------------
# QueueManager
# ---------------------------------------------------------------------------


class QueueManager:
    """Owns the channel registry for one agent.

    Peer of ``EventManager`` / ``ContextManager``. Hidden from the
    LLM by default (subclasses opt in via ``spec(self, "queue_manager",
    hidden=False)``).

    Construct channels via the factories::

        qm = QueueManager(event_manager=self.event_manager)
        self.user_messages_in = qm.queue("user_messages")
        self.user_messages = self.user_messages_in.reader
        self.completions = qm.event("completions")

    Then race them per turn::

        items = await qm.race()   # list[(name, item)] — winner first
    """

    def __init__(self, *, event_manager: Any = None) -> None:
        self._event_manager = event_manager
        # Insertion order matters — race() picks the winner by this
        # order, matching the FIFO-by-position contract the fast path
        # documents.
        self._channels: dict[str, Channel[Any]] = {}
        self._handles: list[JobHandle] = []
        # Wakeup signal: set by any Channel.put() (event-mode or
        # queue-mode buffer) so race() can wake on puts that don't
        # have a direct _drain_one waiter.  Created lazily on first
        # race() call to avoid requiring a running event loop at
        # construction time. Stored as a (event, loop) tuple for
        # atomic publication to _set_notify from other threads.
        self._notify_pair: tuple[asyncio.Event, asyncio.AbstractEventLoop] | None = None
        # Thread-safe flag: set by _set_notify when it can't deliver a wake
        # (stale/closed loop). race() checks this after recreating the pair.
        self._notify_pending = False
        # Optional host-supplied hook fired after every channel put(), in
        # addition to the internal race() wakeup. Lets a host (e.g. the
        # TUI) react to puts — such as (re)starting its dispatcher — via a
        # supported API instead of monkey-patching _set_notify.
        self._notify_callback: Callable[[], None] | None = None

    def set_notify_callback(self, callback: Callable[[], None] | None) -> None:
        """Register an extra callback fired after each channel ``put()``.

        The callback runs *after* the internal ``race()`` wakeup on every
        put (both queue- and event-mode), including the first put before
        ``race()`` has run (when the internal wakeup is a no-op). It fires
        on whichever thread called ``put()``, so cross-thread hosts must
        marshal back to their own loop themselves. Pass ``None`` to clear.

        Replaces the previous pattern of monkey-patching ``_set_notify``.
        """
        self._notify_callback = callback

    def _set_notify(self) -> None:
        """Callback passed to channels; wakes ``race()`` on every put.

        Fires the internal loop-safe wakeup, then any host callback
        registered via ``set_notify_callback`` — even when the internal
        wakeup no-ops because ``race()`` has not created its notify pair
        yet (e.g. the very first put before the dispatcher starts).
        """
        self._signal_notify()
        if self._notify_callback is not None:
            self._notify_callback()

    def _signal_notify(self) -> None:
        """Set the ``_notify`` event so ``race()`` wakes on a put.

        Channels can be produced from the TUI thread while ``race()`` is
        waiting on the agent-loop thread, so wake the event via its owning
        loop instead of mutating it cross-thread. Reads the (event, loop)
        tuple atomically to avoid init races.
        """
        pair = self._notify_pair
        if pair is None:
            return
        event, loop = pair
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            event.set()
        else:
            try:
                loop.call_soon_threadsafe(event.set)
            except (RuntimeError, AttributeError):
                self._notify_pending = True  # race() will consume on next call

    # ---- factories -------------------------------------------------------

    def queue[T](
        self,
        name: str,
        *,
        on_get: Callable[[T], None] | None = None,
        replace: bool = False,
    ) -> Channel[T]:
        """Register a queue-mode channel.

        The agent's producer side keeps the returned ``Channel``; the
        LLM-facing read facade is ``channel.reader``.

        If *replace* is ``True`` and a channel with *name* already
        exists, it is removed (via ``remove_channel``) before the new
        one is created. Otherwise a duplicate name raises
        ``ValueError``.
        """
        if name in self._channels:
            if not replace:
                raise ValueError(f"channel {name!r} already registered")
            self.remove_channel(name)
        ch: Channel[T] = Channel(name, "queue", on_get=on_get, on_put=self._set_notify)
        self._channels[name] = ch
        return ch

    def event[T](
        self,
        name: str,
        *,
        preview: Callable[[T], str] | None = None,
        replace: bool = False,
    ) -> Channel[T]:
        """Register an event-mode channel.

        Requires ``event_manager`` on the manager. ``put()`` on the
        returned channel emits a ``QueueOutput`` event.

        If *replace* is ``True`` and a channel with *name* already
        exists, it is removed (via ``remove_channel``) before the new
        one is created. Otherwise a duplicate name raises
        ``ValueError``.
        """
        if self._event_manager is None:
            raise RuntimeError("QueueManager has no event_manager; event-mode channels require one")
        if name in self._channels:
            if not replace:
                raise ValueError(f"channel {name!r} already registered")
            self.remove_channel(name)
        ch: Channel[T] = Channel(
            name,
            "event",
            event_manager=self._event_manager,
            on_put=self._set_notify,
            preview=preview,
        )
        self._channels[name] = ch
        return ch

    # ---- registry --------------------------------------------------------

    def get_channel(self, name: str) -> Channel[Any]:
        return self._channels[name]

    def channels(self) -> dict[str, Channel[Any]]:
        """Return a snapshot of the channel registry (name → Channel).

        A shallow copy in registration order, so host code can enumerate
        or look up channels without reaching into ``_channels`` or
        mutating the live registry.
        """
        return dict(self._channels)

    def names(self) -> list[str]:
        return list(self._channels.keys())

    def running_handles(self) -> list[JobHandle]:
        """Return handles for spawned jobs currently in the ``running`` state.

        Public accessor for the subset of ``spawn()`` handles a host
        (e.g. the TUI) shows as active background work, replacing reads of
        the private ``_handles`` list.
        """
        return [h for h in self._handles if h.state == "running"]

    def remove_channel(self, name: str) -> None:
        """Remove a channel by name.

        Clears pending items, cancels any spawned jobs feeding the
        channel (via ``task.cancel()``), and removes it from the
        registry.

        Raises ``KeyError`` if *name* is not registered.
        """
        ch = self._channels.pop(name)  # KeyError if not found
        ch.flush()
        # Cancel handles targeting this channel. task.cancel() is sync
        # (it requests cancellation); the task will finish on its own.
        remaining: list[JobHandle] = []
        for h in self._handles:
            if h.name == name:
                h._task.cancel()
                if h.state == "running":
                    h.state = "cancelled"
            else:
                remaining.append(h)
        self._handles = remaining

    # ---- composite status block -----------------------------------------

    def status(self, *, max_items: int = 3, max_chars: int = 80) -> str:
        """Composite ``status`` across queue-mode channels and active spawns.

        Wired into a dynamic context block (``queues``) so the LLM
        sees pending counts every turn. Event-mode channels skip —
        they don't accumulate.

        Also shows active spawned jobs so the LLM knows background
        producers are still running even when queues are empty.
        """
        parts = [
            ch.status(max_items=max_items, max_chars=max_chars)
            for ch in self._channels.values()
            if ch.mode == "queue"
        ]
        body = "\n".join(p for p in parts if p)

        # Show active spawns — tells the LLM "monitors are running"
        # even when no items are pending (items were already delivered).
        active_spawns = [h for h in self._handles if h.state == "running"]
        if active_spawns:
            spawn_lines = [f"⚡ {len(active_spawns)} active background job(s):"]
            for h in active_spawns:
                spawn_lines.append(f"  • {h.label} → {h.name} (running)")
            spawn_status = "\n".join(spawn_lines)
            body = f"{body}\n{spawn_status}" if body else spawn_status

        # Cheat sheet: show cancel/cleanup commands
        cancel_hints = [f".cancel('{h.name}')" for h in active_spawns]
        extra = [
            n
            for n, ch in self._channels.items()
            if n != "user_messages" and (ch.mode != "queue" or not ch.is_empty())
        ]
        cleanup_hints = [f".remove_channel('{n}')" for n in extra]
        all_hints = cancel_hints + cleanup_hints
        if all_hints:
            hints = " | ".join(all_hints)
            cheat = f"💡 self.queue_manager{hints} | .shutdown()"
            body = f"{body}\n{cheat}" if body else cheat

        return body

    # ---- race ------------------------------------------------------------

    async def race(self) -> list[tuple[str, Any]]:
        """Wait for the first channel to produce an item.

        Returns ``[(name, item)]`` — a length-1 list — for a queue-mode
        winner. Returns ``[]`` for an event-triggered wake (the events
        are already rendered into the prompt by the time ``handle()``
        runs). The list shape is the contract; future ``deliver=``
        modes return more items per call without changing it.

        Race-completion semantics: ``asyncio.wait(FIRST_COMPLETED)``
        may return more than one done task when multiple ``put`` calls
        land on adjacent loop ticks. The first task in queue-list
        order wins; the others are restored to the head of their
        source channels so no item is silently dropped. ``on_get``
        fires only for the winner — that's why this races on the
        internal ``_drain_one`` (which does not fire ``on_get``)
        rather than on ``get()``.

        Wakeup contract: both queue-mode and event-mode puts wake
        ``race()``. Event-mode channels fire ``_on_put`` which sets
        the manager's ``_notify`` event; ``race()`` includes that
        event in its wait and returns ``[]`` so the caller knows no
        queue items were consumed but new events appeared.

        Exception contract:
        - Raises ``ValueError`` if no channels at all (queue or event)
          are registered. Callers (the dispatcher) handle this as
          "exit cleanly".
        - Propagates outer cancellation (``CancelledError``) after
          cancelling and awaiting all in-flight drain tasks.
        - Propagates any non-CancelledError raised by ``_drain_one``
          (currently impossible — ``_drain_one`` only raises
          ``CancelledError``). Future ``_drain_one`` failure modes
          will surface to the caller; loser items will still be
          restored. See ``test_race_propagates_drain_one_failure_*``.
        """
        queue_channels = [ch for ch in self._channels.values() if ch.mode == "queue"]
        if not queue_channels and not self._channels:
            raise ValueError("QueueManager.race() requires at least one channel")

        # Lazy-init: avoid requiring a running loop at construction time.
        # Published as an atomic tuple so _set_notify never sees a
        # half-initialized state from another thread.
        # Recreate if the loop changed (gl-212: TUI agent loop restart).
        current_loop = asyncio.get_running_loop()
        if self._notify_pair is None or self._notify_pair[1] is not current_loop:
            self._notify_pair = (asyncio.Event(), current_loop)

        notify_event = self._notify_pair[0]

        # Clear before checking fast path — any put() that arrives
        # between here and the await will re-set it.
        notify_event.clear()

        # Consume pending wake that _set_notify couldn't deliver (gl-212).
        # Must come AFTER clear() so the event stays set for the slow path.
        if self._notify_pending:
            self._notify_pending = False
            notify_event.set()

        # Fast path: any queue channel already has an item — return
        # without creating tasks. Preserves FIFO by registration order.
        for ch in queue_channels:
            if not ch.is_empty():
                item = await ch.get()
                return [(ch.name, item)]

        # Slow path: race the internal drain tasks (one per queue
        # channel) plus the _notify event (fires on any event-mode
        # put or on a queue-mode buffered put without a waiter).

        async def _wait_notify() -> None:
            """Await the _notify event; used as a sentinel task."""
            await notify_event.wait()

        notify_task = asyncio.create_task(_wait_notify(), name="qm.race[_notify]")

        tasks: dict[asyncio.Task[Any], Channel[Any] | None] = {
            notify_task: None,  # sentinel — no channel
        }
        for ch in queue_channels:
            t = asyncio.create_task(ch._drain_one(), name=f"qm.race[{ch.name}]")
            tasks[t] = ch

        try:
            done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in pending:
                try:
                    await t
                except asyncio.CancelledError:
                    # Same delegation as JobHandle.cancel(): reaping a loser we
                    # just cancelled and being cancelled ourselves both surface
                    # here. Only the loser's unwinding is ours to absorb.
                    current = asyncio.current_task()
                    if current is not None and current.cancelling() > 0:
                        raise
                except BaseException:
                    pass

            # Check if a queue-channel drain won (not the notify sentinel).
            drain_done = [t for t in done if tasks[t] is not None]

            if not drain_done:
                # Only _notify fired — event-mode wakeup.
                notify_event.clear()
                return []

            # Pick winner by registration order (= dict insertion order).
            winner_task = next(t for t in tasks if t in drain_done)
            winner_ch = tasks[winner_task]
            assert winner_ch is not None  # mypy: drain_done excludes None
            winner_item = winner_task.result()
            for t in drain_done:
                if t is winner_task:
                    continue
                ch = tasks[t]
                assert ch is not None
                try:
                    lost_item = t.result()
                except BaseException:
                    continue
                ch._restore_front(lost_item)
            winner_ch._fire_on_get(winner_item)
            notify_event.clear()
            return [(winner_ch.name, winner_item)]
        except BaseException:
            # Outer cancellation: cancel any tasks still in flight,
            # await all of them so each _drain_one's CancelledError
            # handler can restore late-set results, then explicitly
            # restore the result of any task that completed before
            # cancellation propagated.
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t, ch in tasks.items():
                if ch is None:
                    continue  # skip notify sentinel
                try:
                    await t
                except BaseException:
                    pass
                if t.done() and not t.cancelled():
                    try:
                        ch._restore_front(t.result())
                    except BaseException:
                        pass
            raise

    # ---- spawn -----------------------------------------------------------

    def spawn(
        self,
        job: Coroutine[Any, Any, Any] | AsyncGenerator[Any, None],
        *,
        channel: str,
        buffer: bool | int = False,
        label: str = "",
    ) -> JobHandle:
        """Run *job* in the background, routing output to *channel*.

        Coroutine input
            ``await`` once; ``put(result)`` to the named channel on
            completion; emit ``StreamEnd`` to the event channel.

        Async generator input
            Iterate; ``put(value)`` per yield until exhaustion or
            cancellation; emit ``StreamEnd`` on close.

        Errors mid-job emit ``JobError`` on both the event channel
        (for context rendering) and the data channel (so race() wakes
        the agent). The handle's state transitions to ``"failed"``.

        Returns a ``JobHandle`` with ``name``, ``state``, ``cancel()``,
        and ``values`` (buffered items if ``buffer`` is set).

        ``buffer`` controls value accumulation on the handle:
        - ``False`` (default): no buffering.
        - ``True``: unbounded list of all yielded values.
        - ``int > 0``: ring buffer keeping the last N values.
        """
        if channel not in self._channels:
            raise ValueError(
                f"channel {channel!r} not registered; known channels: {list(self._channels)}"
            )
        data_ch = self._channels[channel]

        is_agen = inspect.isasyncgen(job)

        async def _run() -> None:
            handle = _handles_by_task[asyncio.current_task()]  # type: ignore[arg-type]
            try:
                if is_agen:
                    async for value in job:  # type: ignore[union-attr]
                        if handle._values is not None:
                            handle._values.append(value)
                        data_ch.put(value)
                else:
                    result = await job  # type: ignore[misc]
                    if handle._values is not None:
                        handle._values.append(result)
                    data_ch.put(result)
                handle.state = "done"
            except asyncio.CancelledError:
                handle.state = "cancelled"
                raise
            except Exception as exc:
                handle.state = "failed"
                error = JobError(
                    channel_name=channel,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                if self._event_manager is not None:
                    self._event_manager.add(error)
                # Also put on data channel so race() wakes the agent
                data_ch.put(error)
                logger.exception("spawn(%s) failed", channel)
            finally:
                if is_agen:
                    await job.aclose()  # type: ignore[union-attr]
                if self._event_manager is not None:
                    self._event_manager.add(StreamEnd(channel_name=channel))

        # Dict created before create_task so the closure can find the
        # handle even if the task runs immediately (single-threaded
        # guarantee, but clearer ordering).
        _handles_by_task: dict[asyncio.Task[Any], JobHandle] = {}
        task = asyncio.create_task(_run(), name=f"spawn[{channel}]")
        handle = JobHandle(name=channel, task=task, buffer=buffer, label=label)
        _handles_by_task[task] = handle
        self._handles.append(handle)
        return handle

    # ---- shutdown --------------------------------------------------------

    async def shutdown(self) -> None:
        """Cancel and await all outstanding spawned jobs.

        Safe to call multiple times. After shutdown, the handles list
        is cleared.
        """
        handles = list(self._handles)
        self._handles.clear()
        for h in handles:
            await h.cancel()

    # ---- job registry ----------------------------------------------------

    def jobs(self) -> dict[str, JobState]:
        """Return channel_name -> state for the most recent handle per channel."""
        return {h.name: h.state for h in self._handles}

    def job(self, name: str) -> JobHandle | None:
        """Look up the most recent handle for a channel name."""
        for h in reversed(self._handles):
            if h.name == name:
                return h
        return None

    async def cancel(self, channel: str) -> bool:
        """Cancel the job on the named channel and flush pending items."""
        h = self.job(channel)
        if h is None:
            return False
        await h.cancel()
        # Flush pending items and cancel waiting consumers.
        ch = self._channels.get(channel)
        if ch is not None:
            ch.flush()
        return True
