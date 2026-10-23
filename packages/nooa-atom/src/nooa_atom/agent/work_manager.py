# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic work manager: own, name, cancel and deliver background work; know nothing of its kind.

``WorkManager.launch(work, name=...)`` takes one of three ordinary Python shapes:

- an awaitable (a coroutine, ``asyncio.sleep(600, "check CI")``): its value is the result;
- an async iterable (an async generator): each value it yields is an item;
- a resource, an async context manager with ``send(input) -> awaitable``: it is
  entered at launch and exited when cancelled; ``job.send(x)`` returns a request
  ``Job`` whose result is what ``send`` returned.

Each outcome has one receiver. A pending ``await job`` (or a reader iterating
the job) takes it; otherwise a ``JobEvent`` goes to the outbox (the agent's
``jobs`` channel). Awaiting a job whose event is still queued withdraws it;
awaiting one whose event was already handed to a turn reads the result back.

Work code can call ``emit(item)`` to report progress, ``describe(text)`` to set
the job's one-line summary, and ``job_dir()`` for a scratch directory the
manager owns and removes later. A resource may instead offer ``summary() -> str``
(sync, cheap), read whenever ``job.summary`` is.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import keyword
import os
import shutil
import stat
import tempfile
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

__all__ = ["Job", "JobEvent", "WorkManager", "describe", "emit", "job_dir"]

EventKind = Literal["item", "done", "failed", "cancelled"]
Status = Literal["running", "done", "failed", "cancelled"]


class Outbox(Protocol):
    """Where events go when nobody awaits: a queue-mode ``Channel`` fits."""

    def put(self, item: Any) -> None: ...

    def remove(self, item: Any) -> bool: ...


@dataclass(frozen=True, eq=False)
class JobEvent:
    """What a job delivers when nobody awaits it.

    ``kind`` is ``item`` (a yielded or emitted value), ``done`` (``value`` is
    the result), ``failed`` (``value`` is the error) or ``cancelled``.
    ``dropped`` counts this job's earlier items withdrawn to keep the queue
    bounded. ``id`` is the job's generation: a reused name gets a new id.
    """

    name: str
    id: int
    kind: EventKind
    value: Any = None
    dropped: int = 0


_current: contextvars.ContextVar[Job | None] = contextvars.ContextVar(
    "nooa_work_current_job", default=None
)


def emit(item: Any) -> None:
    """Report a progress item from the running job (a no-op outside a job)."""
    job = _current.get()
    if job is not None:
        job._emit_nowait(item)


def describe(text: str) -> None:
    """Set the running job's one-line summary (a no-op outside a job)."""
    job = _current.get()
    if job is not None:
        job._summary = str(text)


_SUMMARY_CHARS = 160


def _one_line(text: str, limit: int = _SUMMARY_CHARS) -> str:
    """Untrusted text as one bounded, printable line (control characters escaped)."""
    flat = "".join(c if c.isprintable() else repr(c)[1:-1] for c in str(text))
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def job_dir() -> Path | None:
    """A scratch directory owned by the running job (None outside a job).

    It survives the job's result and is removed when the job leaves the
    manager's history or the manager closes.
    """
    job = _current.get()
    return job._dir() if job is not None else None


_END = object()


class _Reader:
    """``async for x in job``: the job's items, then the end (raises if it failed)."""

    def __init__(self, job: Job) -> None:
        self._job = job
        self._buffer: deque[Any] = deque()
        self._wake = asyncio.Event()
        self._space = asyncio.Event()
        self._space.set()
        self._closed = False

    def __aiter__(self) -> _Reader:
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        while not self._buffer:
            self._wake.clear()
            await self._wake.wait()
        value = self._buffer.popleft()
        self._space.set()
        if value is _END:
            self._closed = True
            self._job._reader = None
            self._job._outcome()  # the end is this reader's: raise if failed or cancelled
            raise StopAsyncIteration
        return value

    def _push(self, value: Any) -> None:
        self._buffer.append(value)
        if len(self._buffer) >= self._job._manager.max_items:
            self._space.clear()
        self._wake.set()

    async def aclose(self) -> None:
        self._detach()

    def _detach(self) -> None:
        """The reader left early: what it did not read goes back to the outbox."""
        if self._closed:
            return
        self._closed = True
        job = self._job
        if job._reader is self:
            job._reader = None
        self._space.set()
        for value in self._buffer:
            if value is _END:
                job._queue_final()
            else:
                job._queue_item(value)
        self._buffer.clear()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._detach()


class Job:
    """Background work. ``await job`` -> result; ``async for x in job`` -> items.

    ``job.send(x)`` gives a resource input and returns the request ``Job``
    (await it for that input's result). ``await job.cancel()`` stops the
    work; for a resource it first cancels its requests, then exits it.
    """

    name: str  # Yours, or job#N; a resource's requests are <name>#N
    id: int  # Unique per launch: a reused name gets a new id

    def __init__(self, manager: WorkManager, name: str, id: int, work: Any, kind: str) -> None:
        self.name = name
        self.id = id
        self._manager = manager
        self._work = work
        self._kind = kind  # "await", "iter", "resource" or "request"
        self._status: Status = "running"
        self._result: Any = None
        self._error: BaseException | None = None
        self._cancel_note = ""
        self._summary = ""
        self._silent = False  # cancelled by its owner: no event
        self._final: JobEvent | None = None
        self._queued_final: JobEvent | None = None
        self._waiters: set[asyncio.Future[None]] = set()
        self._handoff: set[asyncio.Future[None]] = set()
        self._taken = False
        self._reader_ref: weakref.ref[_Reader] | None = None  # weak: a dropped reader detaches
        self._items: deque[JobEvent] = deque()  # its item events possibly still in the outbox
        self._dropped = 0
        self._scratch: tuple[Path, int, int] | None = None
        # Resources.
        self._entered: Any = None
        self._ready = asyncio.Event()  # entered, failed to enter, or closing
        self._closing = False
        self._exit = asyncio.Event()
        self._requests: list[Job] = []
        self._sent = 0
        self._task: asyncio.Task[None] | None = None

    # ---- model-facing ---------------------------------------------------

    @property
    def status(self) -> Status:
        return self._status

    @property
    def result(self) -> Any:
        """The result once done (None before)."""
        return self._result

    @property
    def error(self) -> str | None:
        """Why it failed, or the cancellation note."""
        if self._error is not None:
            return f"{type(self._error).__name__}: {self._error}"
        return self._cancel_note or None

    @property
    def summary(self) -> str:
        """One short line from the work itself (a shell's command and cwd), or ""."""
        text = self._summary
        report = getattr(self._entered, "summary", None) if self._kind == "resource" else None
        if callable(report):
            try:
                text = report()
            except Exception as exc:
                text = f"(summary failed: {type(exc).__name__})"
        return _one_line(text) if text else ""

    def __repr__(self) -> str:
        summary = self.summary
        return f"Job({self.name!r}, {self._status}{', ' + summary if summary else ''})"

    @property
    def _reader(self) -> _Reader | None:
        return self._reader_ref() if self._reader_ref is not None else None

    @_reader.setter
    def _reader(self, reader: _Reader | None) -> None:
        self._reader_ref = weakref.ref(reader) if reader is not None else None

    def __await__(self):
        return self._wait().__await__()

    def __aiter__(self) -> _Reader:
        if self._kind == "resource":
            raise TypeError(f"{self.name} is a resource: iterate or await its requests")
        reader = _Reader(self)
        if self._status != "running":
            reader._push(_END)  # finished: nothing more; the end reports the outcome
            if self._queued_final is not None and self._manager._outbox.remove(self._queued_final):
                self._queued_final = None
            return reader
        if self._reader is not None:
            raise RuntimeError(f"{self.name} already has a reader")
        self._reader = reader
        return reader

    def send(self, item: Any) -> Job:
        """Give the resource an input; returns the request Job at once."""
        if self._kind != "resource":
            raise TypeError(f"{self.name} takes no input: only a launched resource has send()")
        if self._closing or self._status != "running":
            raise RuntimeError(f"{self.name} is {self._closing and 'closing' or self._status}")
        active = sum(1 for r in self._requests if r._status == "running")
        if active >= self._manager.max_requests:
            raise RuntimeError(
                f"{self.name} has {active} requests in flight; await some before sending more"
            )
        self._requests = [r for r in self._requests if r._status == "running"]
        self._sent += 1
        request = self._manager._start(f"{self.name}#{self._sent}", self._request(item), "request")
        self._requests.append(request)
        return request

    async def cancel(self) -> None:
        """Stop the work (a resource: its requests first, then its exit)."""
        if self._status != "running" or self._task is None:
            return
        self._silent = True
        if self._kind == "resource":
            self._closing = True
            requests = [r for r in self._requests if r._status == "running"]
            for request in requests:  # before waking them: they must not run on
                request._silent = True
                if request._task is not None:
                    request._task.cancel()
            self._ready.set()
            await asyncio.gather(*(r.cancel() for r in requests))
            if self._entered is not None:
                self._exit.set()
            else:
                self._task.cancel()
        else:
            self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    # ---- running ----------------------------------------------------------

    async def _run(self) -> None:
        _current.set(self)
        try:
            if self._kind == "resource":
                await self._run_resource()
            elif self._kind == "iter":
                await self._run_iter()
            else:
                self._finish("done", await self._work)
        except asyncio.CancelledError as cancelled:
            note = " ".join(str(arg) for arg in cancelled.args)
            self._finish("cancelled", note)
        except Exception as exc:
            self._finish("failed", exc)

    def _task_ended(self, task: asyncio.Task[None]) -> None:
        """A task cancelled before it ran never reached ``_finish``: finish it here."""
        if self._status == "running":
            if inspect.iscoroutine(self._work):
                self._work.close()  # never started: not "never awaited"
            self._finish("cancelled", "")

    async def _run_iter(self) -> None:
        iterator = self._work.__aiter__()
        try:
            async for value in iterator:
                while True:  # backpressure while a reader's buffer is full
                    reader = self._reader
                    if reader is None or reader._space.is_set():
                        break
                    space = reader._space
                    del reader  # hold no reference: a reader that left must be collectable
                    await space.wait()
                if reader is not None:
                    reader._push(value)
                else:
                    self._queue_item(value)
                del reader
                await asyncio.sleep(0)  # an iterator that never awaits must not starve the loop
        finally:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
        self._finish("done", None)

    async def _run_resource(self) -> None:
        work = self._work
        try:
            entered = await work.__aenter__()
        finally:
            self._ready.set()  # requests waiting on entry wake either way
        self._entered = work if entered is None else entered
        if not callable(getattr(self._entered, "send", None)):
            await work.__aexit__(None, None, None)
            raise TypeError(f"{type(self._entered).__name__} has no send(input)")
        try:
            await self._exit.wait()
        finally:
            self._closing = True
            for request in self._requests:
                if request._status == "running" and request._task is not None:
                    request._task.cancel()
            await asyncio.gather(
                *(r._task for r in self._requests if r._task is not None), return_exceptions=True
            )
            await asyncio.shield(work.__aexit__(None, None, None))
        self._finish("done", None)

    async def _request(self, item: Any) -> Any:
        await self._ready.wait()
        if self._entered is None:
            reason = self.error or "it stopped"
            raise RuntimeError(f"{self.name} did not start: {reason}")
        if self._closing:
            raise RuntimeError(f"{self.name} is closing")
        return await self._entered.send(item)

    # ---- delivery ---------------------------------------------------------

    def _emit_nowait(self, value: Any) -> None:
        reader = self._reader
        if reader is None:
            self._queue_item(value)
            return
        if len(reader._buffer) >= self._manager.max_items:
            reader._buffer.popleft()  # a sync emit cannot wait: drop the oldest
        reader._push(value)

    def _queue_item(self, value: Any) -> None:
        outbox = self._manager._outbox
        while len(self._items) >= self._manager.max_items:
            oldest = self._items.popleft()
            if outbox.remove(oldest):
                self._dropped += 1 + oldest.dropped  # its own count carries forward
        event = JobEvent(self.name, self.id, "item", value, dropped=self._dropped)
        self._dropped = 0
        self._items.append(event)
        outbox.put(event)

    def _finish(self, kind: EventKind, value: Any) -> None:
        if self._status != "running":
            return
        if kind == "done":
            self._result = value
        elif kind == "failed":
            self._error = value
            value = f"{type(value).__name__}: {value}"
        else:
            self._cancel_note = value or ""
            value = value or None
        self._status = kind  # type: ignore[assignment]
        self._ready.set()
        self._final = JobEvent(self.name, self.id, kind, value, dropped=self._dropped)
        self._dropped = 0
        waiters = {f for f in self._waiters if not f.done()}
        self._waiters.clear()
        if waiters:
            self._handoff = waiters
            for future in waiters:
                future.set_result(None)
        elif self._reader is not None:
            self._reader._push(_END)
        elif self._kind == "resource" and kind == "done":
            pass  # a resource closed by its owner has no result to report
        else:
            self._queue_final()
        self._manager._finished(self)

    def _queue_final(self) -> None:
        if self._silent or self._final is None or self._queued_final is not None:
            return
        self._queued_final = self._final
        self._manager._outbox.put(self._final)

    async def _wait(self) -> Any:
        if self._kind == "resource":
            raise TypeError(
                f"{self.name} is a resource: await its requests (job.send(...)) or cancel it"
            )
        if self._status == "running":
            future = asyncio.get_running_loop().create_future()
            self._waiters.add(future)
            try:
                await future
            except asyncio.CancelledError:
                self._waiters.discard(future)
                if future.done() and not future.cancelled():
                    # The outcome was handed to this await, which will not see it.
                    self._handoff.discard(future)
                    if not self._handoff and not self._taken:
                        self._queue_final()
                raise
            self._handoff.discard(future)
            self._taken = True
        elif self._queued_final is not None:
            if self._manager._outbox.remove(self._queued_final):
                self._queued_final = None  # consumed here, not in a later turn
        return self._outcome()

    def _outcome(self) -> Any:
        if self._status == "failed" and self._error is not None:
            raise self._error
        if self._status == "cancelled":
            note = f": {self._cancel_note}" if self._cancel_note else ""
            raise RuntimeError(f"job {self.name} was cancelled{note}")
        return self._result

    # ---- scratch -----------------------------------------------------------

    def _dir(self) -> Path:
        if self._scratch is None:
            root = self._manager._root()
            path = root / str(self.id)
            path.mkdir(mode=0o700)
            info = os.lstat(path)
            self._scratch = (path, info.st_dev, info.st_ino)
        return self._scratch[0]


def _valid_name(name: Any) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def _remove_owned(owned: tuple[Path, int, int] | None) -> None:
    """Remove a directory this manager created, only if it is still that directory."""
    if owned is None:
        return
    path, dev, ino = owned
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (dev, ino):
        return  # replaced (a symlink or another directory): not ours any more
    shutil.rmtree(path, ignore_errors=True)  # does not follow symlinks inside


class WorkManager:
    """Owns background jobs: names, cancellation, delivery, scratch cleanup.

    ``outbox`` receives ``JobEvent``s nobody awaited (``put``) and gives
    back ones not yet consumed (``remove``). ``scratch`` is the parent
    directory for this manager's own scratch directory (default: the
    system temp dir). ``keep`` finished jobs are remembered (their scratch
    directories with them); ``max_items`` bounds a job's queued items and a
    reader's buffer; ``max_requests`` bounds a resource's requests in flight.
    """

    keep: int
    max_items: int
    max_requests: int

    def __init__(
        self,
        outbox: Outbox,
        *,
        scratch: str | Path | None = None,
        keep: int = 100,
        max_items: int = 100,
        max_requests: int = 16,
    ) -> None:
        self._outbox = outbox
        self._scratch_parent = Path(scratch) if scratch is not None else None
        self._scratch_root: tuple[Path, int, int] | None = None
        self.keep = keep
        self.max_items = max_items
        self.max_requests = max_requests
        self._next_id = 0
        self._live: dict[int, Job] = {}
        self._history: deque[Job] = deque()
        self._names: dict[str, Job] = {}
        self._closed = False

    def launch(self, work: Any, *, name: str | None = None) -> Job:
        """Start ``work`` (awaitable, async iterable or resource) and return its Job."""
        try:
            if self._closed:
                raise RuntimeError("the work manager is closed")
            if inspect.iscoroutinefunction(work) or inspect.isasyncgenfunction(work):
                raise TypeError(f"pass {work.__name__}(), not {work.__name__}")
            if hasattr(work, "__aenter__") and hasattr(work, "__aexit__"):
                kind = "resource"
            elif hasattr(work, "__aiter__"):
                kind = "iter"
            elif inspect.isawaitable(work):
                kind = "await"
            else:
                raise TypeError(
                    "launch() takes an awaitable, an async iterable, or a resource "
                    f"(an async context manager with send()); got {type(work).__name__}"
                )
            if name is not None:
                if not _valid_name(name):
                    raise ValueError(f"name {name!r} is not a Python identifier")
                current = self._names.get(name)
                if current is not None and current._status == "running":
                    raise ValueError(f"{name!r} is running; cancel it or pick another name")
        except BaseException:
            if inspect.iscoroutine(work):
                work.close()
            raise
        return self._start(name, work, kind)

    def get(self, name: str) -> Job | None:
        """The newest job with this name (a running one, else the latest finished)."""
        return self._names.get(name)

    def list(self) -> list[Job]:
        """Running and remembered finished jobs, oldest first."""
        return sorted({*self._live.values(), *self._history}, key=lambda job: job.id)

    def running(self) -> list[Job]:
        """Jobs still running, resources included."""
        return sorted(self._live.values(), key=lambda job: job.id)

    def records(self) -> list[dict[str, Any]]:
        """Small, snapshot-safe records of running jobs (to report them lost after a restart)."""
        return [{"id": j.id, "name": j.name, "kind": j._kind} for j in self.running()]

    async def aclose(self) -> None:
        """Cancel every running job, wait for them, and remove the scratch directories."""
        self._closed = True
        live = self.running()
        resources = [job for job in live if job._kind == "resource"]
        others = [job for job in live if job._kind != "resource"]
        await asyncio.gather(*(job.cancel() for job in resources))
        await asyncio.gather(*(job.cancel() for job in others))
        for job in [*self._history, *live]:
            _remove_owned(job._scratch)
            job._scratch = None
        _remove_owned(self._scratch_root)
        self._scratch_root = None

    # ---- internals ----------------------------------------------------------

    def _start(self, name: str | None, work: Any, kind: str) -> Job:
        self._next_id += 1
        job_id = self._next_id
        job = Job(self, name or f"job#{job_id}", job_id, work, kind)
        self._live[job_id] = job
        self._names[job.name] = job
        context = contextvars.copy_context()
        context.run(_current.set, None)
        job._task = asyncio.get_running_loop().create_task(
            job._run(), name=f"work[{job.name}]", context=context
        )
        job._task.add_done_callback(job._task_ended)
        return job

    def _finished(self, job: Job) -> None:
        self._live.pop(job.id, None)
        self._history.append(job)
        while len(self._history) > self.keep:
            old = self._history.popleft()
            if self._names.get(old.name) is old:
                del self._names[old.name]
            _remove_owned(old._scratch)
            old._scratch = None

    def _root(self) -> Path:
        if self._scratch_root is None:
            path = Path(tempfile.mkdtemp(prefix="nooa-work-", dir=self._scratch_parent))
            info = os.lstat(path)
            self._scratch_root = (path, info.st_dev, info.st_ino)
        return self._scratch_root[0]
