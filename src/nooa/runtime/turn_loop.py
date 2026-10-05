# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-neutral channel dispatch: wait → prepare → gated commit → run.

Readiness never consumes. Preparation is cancellable while inputs stay queued;
commit records an opaque batch synchronously before ownership transfer. Events
observe outcomes and isolate ordinary observer failures, including CancelledError.
See runtime/README.md for recovery, cancellation and concurrency contracts.
"""

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field

from nooa.context_blocks import EventBase
from nooa.context_blocks.roles import Role

logger = logging.getLogger(__name__)

PrepareTurn = Callable[[], Awaitable[None]]
CommitTurn = Callable[[dict[str, list[Any]]], None]

TurnKind = Literal["done", "need_input", "waiting", "cancelled", "error"]


class TurnCancelled(EventBase):
    """A person, a parent or the host stopped the turn before it finished.

    Appended to the agent's events when a cancel takes effect, after the
    interrupted cell's output, so the model sees at its next turn that it
    was stopped rather than that a cell failed. ``by`` says who stopped it
    (``"user"``, ``"parent:<name>"``, ``"host"``); ``interrupted`` is the
    tag of the interrupted cell's ``PythonOutput``, or ``None`` when the
    turn was stopped between cells (for example during a model call).
    """

    _role: ClassVar[Role] = Role.USER

    by: str
    interrupted: str | None = None


class TurnBegan(EventBase):
    """Published right before the loop calls the turn method.

    ``notification`` is what the turn receives (channel name → items).
    Every item in it has already been published as ``ChannelItemConsumed``.
    A runtime event: never recorded, never shown to the model.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    notification: Annotated[dict[str, list[Any]], Field(repr=False)] = Field(default_factory=dict)


class TurnSettled(EventBase):
    """Published once a turn has finished, however it finished.

    - ``done`` / ``need_input`` / ``waiting``: ``result`` is the turn's
      ``Done`` / ``NeedInput`` / ``Waiting``.
    - ``cancelled``: ``cancel(by=...)`` stopped it. ``cancelled_by`` names
      who; ``interrupted`` is the tag of the interrupted cell's output (or
      ``None``). ``TurnCancelled`` is already in the agent's events, unless
      ``ran`` is ``False`` (cancelled during preparation).
    - ``error``: ``message`` says what went wrong and ``error`` is the
      exception (``None`` for a turn that returned something other than a
      turn result).

    ``ran`` is ``False`` when the turn method was never called (preparation
    raised, or a cancel came during it).

    ``committed=False`` means inputs never left their channels and no durable
    turn started. Preparation failure/cancellation pauses dispatch for recovery.
    A runtime event: never recorded, never shown to the model.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    kind: TurnKind
    result: Annotated[Any, Field(repr=False)] = None
    error: Annotated[Any, Field(repr=False)] = None
    message: str = ""
    cancelled_by: str | None = None
    interrupted: str | None = None
    ran: bool = True
    committed: bool = True


class TurnLoopEnded(EventBase):
    """Published when the loop stops because it can no longer wait for input.

    ``QueueManager.wait_ready()`` raised (no channel left to wait on); ``error``
    is that exception. Not published when ``stop()`` ends the loop.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    error: Annotated[Any, Field(repr=False)] = None
    message: str = ""


def _settled_from(result: Any) -> TurnSettled:
    """Map a turn method's return value to a ``TurnSettled``."""
    from nooa.interactive import Done, NeedInput, Waiting

    if isinstance(result, Done):
        return TurnSettled(kind="done", result=result)
    if isinstance(result, NeedInput):
        return TurnSettled(kind="need_input", result=result)
    if isinstance(result, Waiting):
        return TurnSettled(kind="waiting", result=result)
    return TurnSettled(
        kind="error", message=f"turn returned {type(result).__name__}, not a turn result"
    )


class TurnLoop:
    """Serial wait → cancellable preparation → gated commit → invocation.

    ``pause`` gates new dispatch, not an already committed invocation.
    Preparation/commit failure blocks dispatch until explicit ``resume``;
    untouched input is never retried in a busy loop. Cancelling preparation
    also pauses, leaving its input queued. Events are observers, not recorders.
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._turn_method = "handle"
        self._prepare: PrepareTurn | None = None
        self._commit: CommitTurn | None = None
        self._task: asyncio.Task[None] | None = None
        self._turn: asyncio.Task[Any] | None = None
        self._paused = False
        self._stopping = False
        self._gate = asyncio.Event()
        self._gate.set()
        self._cancel_by: str | None = None
        self._interrupted: Any = None
        self._settled = asyncio.Event()
        self._settled.set()
        self._unsubscribe: Any = None
        self._unsubscribe_close: Any = agent.event_manager.on_close(self.stop)
        self.dispatch_error: BaseException | None = None

    @property
    def started(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def running(self) -> bool:
        """Includes preparation and settlement, not merely agent execution."""
        return not self._settled.is_set()

    @property
    def in_turn(self) -> bool:
        """Whether the caller is the turn task (self-close would deadlock)."""
        return asyncio.current_task() is self._turn

    @property
    def paused(self) -> bool:
        return self._paused

    def start(
        self,
        *,
        turn_method: str = "handle",
        context: contextvars.Context | None = None,
        prepare: PrepareTurn | None = None,
        commit: CommitTurn | None = None,
    ) -> None:
        """Start in a fresh context; each turn receives its own context copy.

        ``prepare()`` may await; no channel items have been consumed yet.
        ``commit(batch)`` must be synchronous and failure-propagating, with
        no observer callbacks or channel mutation. It records the selected
        batch before ownership transfer. These replace unpublished before_turn.
        Stopped/ended loops may be restarted, without duplicate subscriptions.
        """
        if self.started:
            return
        self._turn_method, self._prepare, self._commit = turn_method, prepare, commit
        self._stopping = False
        self._paused = False
        self.dispatch_error = None
        self._gate = asyncio.Event()
        self._gate.set()
        self._settled = asyncio.Event()
        self._settled.set()
        if self._unsubscribe_close is None:
            self._unsubscribe_close = self._agent.event_manager.on_close(self.stop)
        self._unsubscribe = self._agent.event_manager.on("PythonOutput", self._on_output)
        self._task = asyncio.get_running_loop().create_task(
            self._run(),
            name=f"turn-loop:{type(self._agent).__name__}",
            context=context if context is not None else contextvars.Context(),
        )

    def pause(self) -> None:
        """Gate new dispatch. A committed turn keeps running."""
        self._paused = True
        self._gate.clear()

    def resume(self) -> None:
        """Explicitly retry queued input after pause/dispatch failure."""
        if self._stopping:
            return
        self.dispatch_error = None
        self._paused = False
        self._gate.set()

    def stop_starting(self) -> None:
        """Compatibility alias for pause; use stop to terminate."""
        self.pause()

    async def cancel(self, *, by: str = "user") -> bool:
        if self._settled.is_set():
            return False
        self._cancel_by = by
        turn = self._turn
        if turn is not None and not turn.done():
            turn.cancel()
        # Reentrant close from the turn cannot await its own settlement.
        if asyncio.current_task() in (self._turn, self._task):
            return True
        await self._settled.wait()
        return True

    async def stop(self, *, by: str = "host") -> None:
        if self.in_turn:
            raise RuntimeError("an executing turn cannot stop its own loop")
        self.pause()
        self._stopping = True
        await self.cancel(by=by)
        task = self._task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._cleanup()

    def _cleanup(self) -> None:
        for name in ("_unsubscribe", "_unsubscribe_close"):
            unsubscribe = getattr(self, name)
            if unsubscribe is not None:
                unsubscribe()
                setattr(self, name, None)

    def _publish(self, event: EventBase) -> None:
        try:
            self._agent.event_manager.add(event)
        except (Exception, asyncio.CancelledError):
            logger.exception("TurnLoop: could not publish %s", event.event_type)

    def _on_output(self, event: Any) -> None:
        from nooa.events import ResultStatus

        if (
            event.execution_status is ResultStatus.CANCELLED
            and self.running
            and self._interrupted is None
        ):
            self._interrupted = event

    async def _run(self) -> None:
        queues = self._agent.queue_manager
        try:
            while not self._stopping:
                await self._gate.wait()
                if self._stopping:
                    return
                try:
                    await queues.wait_ready()
                except Exception as exc:
                    self._publish(TurnLoopEnded(error=exc, message=f"{type(exc).__name__}: {exc}"))
                    return
                if self._paused or self._stopping or not queues.ready():
                    continue
                self._settled.clear()
                self._cancel_by = None
                self._interrupted = None
                settled = None
                try:
                    self._turn = asyncio.create_task(self._turn_outcome(), name="turn")
                    try:
                        settled = await self._turn
                    except asyncio.CancelledError:
                        if asyncio.current_task().cancelling():
                            raise
                        # Cancellation before the new task executes its first line.
                        self.pause()
                        settled = TurnSettled(
                            kind="cancelled",
                            cancelled_by=self._cancel_by or "host",
                            ran=False,
                            committed=False,
                        )
                finally:
                    self._turn = None
                    try:
                        if settled is not None:
                            self._publish(settled)
                    finally:
                        self._interrupted = None
                        self._settled.set()
        finally:
            self._cleanup()
            self._settled.set()

    async def _turn_outcome(self) -> TurnSettled | None:
        committed = ran = False
        try:
            if self._prepare is not None:
                await self._prepare()
            # Cancellation intent survives a prepare hook swallowing CancelledError.
            # Never claim its still-queued inputs after a logical cancel.
            if self._cancel_by is not None:
                self.pause()
                return TurnSettled(
                    kind="cancelled", cancelled_by=self._cancel_by, ran=False, committed=False
                )
            if self._paused or self._stopping or not self._agent.queue_manager.ready():
                return None
            notification = self._agent.queue_manager.claim_batch(self._commit)
            if notification is None:
                return None
            committed = True
            self._publish(TurnBegan(notification=notification))
            method = getattr(self._agent, self._turn_method)
            # Lookup/call and invalid return types are inside the outcome handler.
            result = method(notification)
            ran = True
            return _settled_from(await result)
        except asyncio.CancelledError as exc:
            if self._task is not None and self._task.cancelling():
                raise  # Raw loop teardown is not a logical turn settlement.
            if not committed:
                self.pause()
            if self._cancel_by is not None:
                if ran:
                    return self._cancelled()
                return TurnSettled(
                    kind="cancelled", cancelled_by=self._cancel_by, ran=False, committed=committed
                )
            if not committed:
                self.dispatch_error = exc
            return TurnSettled(
                kind="error",
                error=exc,
                message="turn was cancelled from inside",
                ran=ran,
                committed=committed,
            )
        except Exception as exc:
            logger.exception("TurnLoop: turn failed")
            if not committed:
                self.dispatch_error = exc
                self.pause()
            return TurnSettled(
                kind="error",
                error=exc,
                message=f"{type(exc).__name__}: {exc}",
                ran=ran,
                committed=committed,
            )

    def _cancelled(self) -> TurnSettled:
        by = self._cancel_by or "host"
        interrupted = self._interrupted.tag if self._interrupted is not None else None
        try:
            self._agent.event_manager.add(TurnCancelled(by=by, interrupted=interrupted))
        except (Exception, asyncio.CancelledError) as exc:
            return TurnSettled(
                kind="error",
                error=exc,
                message=f"the turn was cancelled but recording it failed: {type(exc).__name__}: {exc}",
            )
        return TurnSettled(kind="cancelled", cancelled_by=by, interrupted=interrupted)
