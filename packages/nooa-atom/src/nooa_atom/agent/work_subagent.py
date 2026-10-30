# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subagent work for the work manager: a child agent session with its own history.

``Subagent(description, model=...)`` is inert configuration; nothing starts
until it is used, inside a session's turn (it binds to that session then).

- ``await Subagent().run(todo)``: a one-shot worker. It cannot ask
  questions; its ``TaskResult`` is the result, and it closes after it.
- ``rev = manager.launch(Subagent(), name="review")``: a kept child.
  ``rev.send(todo)`` returns a request whose result is the child's
  ``TaskResult``, or a ``Question`` it asks; answer with the next
  ``rev.send(answer)``. Requests run one at a time, in order. Cancelling a
  request stops the child's current turn; cancelling the resource closes
  the child. ``await rev.message(text)`` hands it a note now, even mid-turn,
  with no answer promised: the running turn reads it, or it starts a turn
  whose outcome arrives as an item of ``rev``.

The child's ``self.message()`` notes are the job's items (a request's while
one runs, else the resource's): they go to the work manager like any
other item, never to the legacy ``delegates`` channel.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nooa import hidden
from nooa.tools import Todo as _AnyTodo
from nooa_atom.agent.work_manager import current_name, describe, emitter
from nooa_atom.session.items import ChildFailed, ChildQuestion, ChildResult, TaskResult
from nooa_atom.session.names import is_valid_name
from nooa_atom.tools.todo_list import Todo

__all__ = ["Question", "Subagent"]


@dataclass(frozen=True)
class Question:
    """A kept subagent's question: answer it with the next ``send(answer)``."""

    text: str
    id: str = ""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"Question has no attribute {name!r}: the child asked instead of finishing. "
            "Read its .text and answer with job.send(answer); that request's result is "
            "the next TaskResult or Question."
        )


class _Route:
    """Where the registry sends a routed child's notes and outcomes (a ``ChildRoute``).

    A request is *armed* once the turn carrying its input starts (the first
    request's turn is the child's first). Only an armed request takes a turn's
    outcome; any other turn's outcome (one a note started) is an item.
    """

    def __init__(self, idle_post: Callable[[Any], None], name: str) -> None:
        self.name = name
        self.idle_post = idle_post  # the resource's (or one-shot job's) items
        self.post = idle_post
        self.future: asyncio.Future[Any] | None = None
        self.armed = False
        self.want: str | None = None  # the item id that arms the waiting request
        self.early: list[str] = []  # ids of turns started before ``want`` was known
        self.asking: Question | None = None
        self.drop_failure = False  # the failure of a turn its own request cancelled
        self.closed_reason: str | None = None

    def expect(self, post: Callable[[Any], None], *, armed: bool = True) -> asyncio.Future[Any]:
        """Arm the next outcome before the child can deliver it (``armed=False``: its turn's)."""
        self.future = asyncio.get_running_loop().create_future()
        self.post = post
        self.armed, self.want, self.early = armed, None, []
        self.asking = None
        # A cancelled request's turn failure arrives while its interrupt runs,
        # under the send lock, before the next expect(): a flag left set here
        # would only drop this new request's real failure.
        self.drop_failure = False
        return self.future

    def claim(self, item_id: str) -> None:
        """The waiting request's input was admitted as ``item_id``."""
        self.want = item_id
        self.armed = self.armed or item_id in self.early
        self.early = []

    def turn_started(self, item_ids: list[str]) -> None:
        self.asking = None  # a new turn ends the child's question
        if self.future is None or self.armed:
            return
        if self.want is None:
            self.early.extend(item_ids)  # admitted, its receipt not read yet
        elif self.want in item_ids:
            self.armed = True

    def message(self, text: str) -> None:
        (self.post if self.armed else self.idle_post)(text)

    def disarm(self) -> None:
        self.future, self.post, self.armed, self.want = None, self.idle_post, False, None

    def outcome(self, item: Any) -> None:
        if isinstance(item, ChildQuestion):
            self.asking = Question(str(item.question), str(item.id))
        future = self.future if self.armed else None
        if future is not None:
            self.disarm()
        if future is not None and not future.done():
            future.set_result(item)
        elif isinstance(item, ChildFailed) and self.drop_failure and future is None:
            self.drop_failure = False  # its request was cancelled: nobody waits for it
        else:
            with contextlib.suppress(Exception):  # unasked for (a note's turn, after a cancel)
                self.idle_post(_item(item, self.name))

    def closed(self) -> None:
        self.closed_reason = self.closed_reason or f"subagent {self.name} was closed"
        future = self.future
        self.disarm()
        if future is not None and not future.done():
            future.set_exception(RuntimeError(self.closed_reason))


def _item(item: Any, name: str) -> Any:
    """An unasked-for outcome as a job item: a failure is its text, never raised."""
    if isinstance(item, ChildFailed):
        return f"subagent {name} failed: {item.error}"
    return _value(item, name)


def _value(item: Any, name: str) -> Any:
    """A child outcome as the request's result (a failure raises)."""
    if isinstance(item, ChildResult):
        result = item.done.result
        return result if result is not None else item.done
    if isinstance(item, ChildQuestion):
        return Question(str(item.question), str(item.id))
    if isinstance(item, ChildFailed):
        raise RuntimeError(f"subagent {name} failed: {item.error}")
    return item


def _require_port() -> Any:
    from nooa_atom.session.port import current_port

    port = current_port.get()
    if port is None:
        raise RuntimeError("a Subagent runs only inside a session's turn")
    return port


class Subagent:
    """A child agent session with its own history, working on a brief you hand it.

    ``await Subagent("fix parser").run(todo)`` is a one-shot worker: it
    cannot ask, and its ``TaskResult`` is the result. Launched
    (``rev = self.jobs.launch(Subagent("review"), name="review")``) it is a
    kept child: each ``rev.send(todo or text)`` is a request whose result is
    a ``TaskResult`` or a ``Question``; answer with the next ``send``.
    Awaiting a request blocks until the child ends its turn (with a
    ``TaskResult`` or a ``Question``). Cancelling a request stops the child's turn; cancelling ``rev`` closes
    it. The brief is all it sees of your task, and its ``self.message()``
    notes arrive as job items.
    """

    __nosnapshot__ = True  # bound to a live session once used

    description: str  # Short title for people; the child's name comes from the job's
    model: str | None  # Model alias; the session's own when None

    def __init__(self, description: str = "", *, model: str | None = None) -> None:
        self.description = description
        self.model = model
        self._port: Any = None
        self._route: _Route | None = None
        self._child_id: str | None = None
        self._child_name: str | None = None
        self._lock: asyncio.Lock | None = None
        self._started: asyncio.Event | None = None  # clear while send() starts the child
        self._name: str | None = None
        self._entering = False

    def __repr__(self) -> str:
        return f"Subagent({self.description!r}; {self.summary()})"

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"Subagent has no attribute {name!r}: it is the worker, not its result. "
            "`await Subagent(...).run(todo)` returns a TaskResult (fields: "
            "solution_description, evidence, how_to_verify, report, value)."
        )

    @hidden
    def summary(self) -> str:
        """Its state in one line (never asks the child)."""
        route = self._route
        if self._port is None:
            return "closed" if route is not None else "not started"
        if route is None:
            return "idle · no task yet"
        if route.closed_reason:
            return "closed"
        where = f"child {self._child_name}"
        if route.future is not None:
            return f"working · {where}"
        if route.asking is not None:
            return f"asked a question, awaiting your answer (send it) · {where}"
        return f"idle · {where}"

    async def run(self, task: str | Todo, *, brief: str | None = None) -> TaskResult:
        """Run ``task`` (a Todo or text, plus optional ``brief`` instructions) on a one-shot child; return its ``TaskResult``."""
        port = _require_port()
        name = _child_name(port, current_name())
        describe(f"subagent {name}" + (f": {self.description}" if self.description else ""))
        route = _Route(emitter(), name)
        future = route.expect(emitter())
        ref = await _start(port, self, task, brief, route, name=name, kept=False)
        try:
            item = await future
        except asyncio.CancelledError:
            route.closed_reason = f"subagent {name} was cancelled"
            with contextlib.suppress(Exception):
                await asyncio.shield(port.close_child(ref.id))
            raise
        return _value(item, name)

    # ---- kept child --------------------------------------------------------

    async def __aenter__(self) -> Subagent:
        if self._port is not None or self._entering:
            raise RuntimeError("this Subagent is already running; launch another Subagent()")
        self._entering = True
        try:
            port = _require_port()
        finally:
            self._entering = False
        self._port, self._lock, self._name = port, asyncio.Lock(), current_name()
        self._started = asyncio.Event()
        self._started.set()
        self._idle_post = emitter()
        self._route, self._child_id, self._child_name = None, None, None
        return self

    async def __aexit__(self, *exc: Any) -> None:
        port, child_id, self._port = self._port, self._child_id, None
        route = self._route
        if route is not None:
            route.closed_reason = route.closed_reason or f"subagent {self._child_name} was closed"
        if port is not None and child_id is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(port.close_child(child_id))
        if route is not None:
            route.closed()

    async def send(self, task: str | Todo) -> TaskResult | Question:
        """Give the kept child a task, follow-up or answer; return its result or question."""
        if self._port is None or self._lock is None:
            raise RuntimeError("the subagent is not running; launch it with self.jobs.launch()")
        async with self._lock:
            port = self._port
            route = self._route
            if route is not None and route.closed_reason:
                raise RuntimeError(f"{route.closed_reason}; launch a new Subagent()")
            post = emitter()
            describe(f"task for subagent {self._child_name or self._name or ''}".rstrip())
            receipt = None
            if route is None or self._child_id is None:
                name = _child_name(port, self._name)
                route = self._route = _Route(self._idle_post, name)
                future = route.expect(post)
                self._started.clear()
                try:
                    ref = await _start(port, self, task, None, route, name=name, kept=True)
                except BaseException:
                    self._route = None
                    raise
                finally:
                    self._started.set()
                self._child_id, self._child_name = ref.id, ref.name
            else:
                # Queued, not steered: a note's turn may be running, and this
                # request takes only the outcome of the turn that carries it.
                future = route.expect(post, armed=False)
                text = _todo_text(task) if isinstance(task, _AnyTodo) else task
                try:
                    receipt = await port.send_child(
                        self._child_id, text, channel="user_messages", steer=False
                    )
                except BaseException:
                    route.disarm()
                    raise
                route.claim(receipt.item_id)
            try:
                item = await future
            except asyncio.CancelledError:
                if route.armed or receipt is None or not self._withdraw(route, receipt):
                    await self._interrupt(route)
                raise
            return _value(item, route.name)

    async def message(self, text: str | Todo) -> None:
        """A note the child reads now, even mid-turn; no answer promised."""
        # A running turn reads it before its next model call; else it starts a
        # turn whose outcome is an item of this job. Answer questions with send().
        if self._port is None or self._started is None:
            raise RuntimeError("the subagent is not running; launch it with self.jobs.launch()")
        await self._started.wait()  # the first send() may still be starting the child
        route, child_id = self._route, self._child_id
        if route is None or child_id is None:
            raise RuntimeError("the subagent has no child yet: send() it a task first")
        if route.closed_reason:
            raise RuntimeError(f"{route.closed_reason}; launch a new Subagent()")
        if route.asking is not None:
            raise RuntimeError(
                f"subagent {route.name} asked {route.asking.text!r}: answer it with send()"
            )
        item = _todo_text(text) if isinstance(text, _AnyTodo) else text
        await self._port.send_child(child_id, item, channel="user_messages")

    def _withdraw(self, route: _Route, receipt: Any) -> bool:
        """A cancelled request whose turn has not started: take its input back."""
        try:
            withdrawn = bool(self._port.withdraw_child(self._child_id, receipt))
        except Exception:
            withdrawn = False
        if withdrawn:
            route.disarm()
        return withdrawn

    async def _interrupt(self, route: _Route) -> None:
        """A cancelled request: stop the child's current turn (the child stays open)."""
        route.disarm()
        child = self._port._registry.get(self._child_id) if self._child_id else None
        if child is None:
            return
        route.drop_failure = True
        try:
            stopped = await asyncio.shield(child.cancel(by="parent"))
        except Exception:
            stopped = False
        if not stopped:
            route.drop_failure = False


def _child_name(port: Any, job_name: str | None) -> str:
    """A free child name from the job's (``review#2`` -> ``review``), else ``subagent``."""
    base = "" if (job_name or "").startswith("job#") else (job_name or "").split("#", 1)[0]
    if not is_valid_name(base):
        base = "subagent"
    for n in range(1, 1000):
        candidate = base if n == 1 else f"{base}_{n}"
        try:
            port.check_child_name(candidate)
        except ValueError:
            continue
        return candidate
    raise RuntimeError(f"no free child name like {base!r}")


def _todo_text(todo: Todo) -> str:
    from nooa_atom.agent.base import _todo_prompt

    return _todo_prompt(todo)


async def _start(
    port: Any,
    sub: Subagent,
    task: Any,
    brief: str | None,
    route: _Route,
    *,
    name: str,
    kept: bool,
) -> Any:
    """Create the routed child; the route is installed before its first turn."""
    from nooa_atom.agent.base import _child_prompt
    from nooa_atom.session.names import child_path
    from nooa_atom.session.session import as_data

    handed = as_data(task) if isinstance(task, _AnyTodo) else None
    text = _child_prompt(child_path(port.path, name), port.path, task, handed, brief, kept=kept)
    return await port.delegate(
        sub.description or name,
        text,
        name=name,
        model=sub.model,
        kept=kept,
        todo=handed,
        route=route,
    )
