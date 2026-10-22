# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing JobsApi: run long work in the background, by label.

``JobsApi`` is attached to the Atom agent as ``self.jobs``. It runs
shell commands in their own bash processes (not the agent's persistent
shell session) and arbitrary coroutines or async iterators, and delivers
their output on the agent's ``jobs`` queue channel. The channel registry
(``self.queue_manager``) stays hidden; this is the model-facing view of it,
as ``self.events`` is for ``self.event_manager``.

Every delivered item carries the job's ``label`` and ``path``, so a turn
can wait with ``Waiting(on=["<label>"])`` (or ``on=["jobs"]`` for any job)
and find the outcome in ``notification["jobs"]``. A label is a name in the
session's one namespace of live names, shared with its open children (see
``nooa_atom.session.names``).
"""

import asyncio
import contextlib
import datetime
import inspect
import os
import signal
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from nooa.agentdoc import (
    LineSplitter,
    TruncatingStringIO,
    bounded_lines,
    hidden,
    spec,
)
from nooa.context_blocks import EventBase
from nooa.skill import Skill
from nooa.tools import StreamDone, StreamEvent
from nooa_atom.session.names import ROOT_PATH, check_name, child_path

if TYPE_CHECKING:
    from nooa.runtime.channels import Channel, JobHandle, QueueManager

__all__ = ["JobInfo", "JobUpdate", "JobsApi", "JobsLost", "Throttle"]

JOBS_CHANNEL = "jobs"
# Characters kept of stdout and, separately, of stderr on a run's done
# update (the end of each stream); nothing more is kept anywhere.
_TAIL_CHARS = 20_000
# Longest delivered line (run progress, tail), in characters; the rest of the
# line is replaced by a "[... N chars of this line omitted]" marker.
_LINE_CHARS = 8_000
# Bytes read from a pipe or a tailed file at a time.
_CHUNK_BYTES = 64 * 1024
# A batch is delivered early once it holds this many lines/values or chars of text.
_BATCH_ITEMS = 200
_BATCH_CHARS = 20_000
# Items buffered between a job's reader and its batching loop; a full buffer
# makes the reader wait (backpressure on the pipe or iterator).
_QUEUE_ITEMS = 256
# Finished jobs remembered for list()/get().
_KEEP_FINISHED = 100
# Delivered items remembered per job for get().
_KEEP_ITEMS = 5
# Longest JobInfo.target, in characters.
_TARGET_CHARS = 200
# How often tail() checks its file for new lines.
_TAIL_POLL_SECONDS = 0.5

JobKind = Literal["run", "spawn", "tail", "after", "every"]
JobStateName = Literal["running", "done", "failed", "cancelled"]


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class Throttle(BaseModel):
    """Batching for a streaming job: one delivery per ``items`` values or ``seconds``."""

    items: int | None = None
    seconds: float | None = None


@spec(hide_empty=True)
class JobInfo(BaseModel):
    """A job's status, as returned by ``self.jobs`` methods."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str = Field(description="Name it was started with; items it delivers carry it")
    path: str = Field(default="", description="Its path: your session's path plus /label")
    kind: JobKind = Field(description="The method that started it")
    state: JobStateName = Field(description="running, done, failed or cancelled")
    target: str = Field(default="", description="What runs: command, file, interval or callable")
    description: str = Field(default="", description="Why it runs, as passed with description=")
    payload: Any = Field(
        default=None, description="Value given to after()/every(); delivered as result/values"
    )
    log_path: str | None = Field(
        default=None, description="run with log_path=: the file getting its full output"
    )
    started_at: str = Field(description="Start time (ISO, seconds)")
    finished_at: str | None = Field(default=None, description="End time, None while running")
    throttle: Throttle | None = Field(default=None, description="Current batching, None if off")
    pending: int = Field(
        default=0,
        description=(
            "Values held back by the throttle for the next update (an items= batch waits"
            " for N values or the end; seconds= bounds the wait). 0 once the job ends"
        ),
    )
    last: list["JobUpdate"] = Field(
        default_factory=list,
        description=(
            "Its latest updates (up to 5) as sent to the jobs queue: the newest may not"
            " have reached notification['jobs'] yet"
        ),
    )


@spec(hide_empty=True)
class JobUpdate(BaseModel):
    """What a job delivers to ``notification["jobs"]``: new ``lines``/``values`` while it runs, then one ``done=True`` update.

    A ``run`` ends with ``returncode``, ``timed_out``, ``stdout`` and
    ``stderr``; a coroutine ``spawn`` or ``after`` with ``result``. Every
    job ends with exactly one ``done=True`` update; a cancelled job
    delivers nothing more. Any job's update starts the next turn; check
    ``u.label`` / ``u.done`` and return ``Waiting`` again if yours has not
    finished.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str = Field(description="The job's label")
    path: str = Field(default="", description="The job's path: your session's path plus /label")
    done: bool = Field(default=False, description="True on the job's one final update")
    error: str | None = Field(
        default=None,
        description="Why the job failed ('ExceptionType: message'), on the done update only",
    )
    returncode: int | None = Field(
        default=None, description="run, done update: the exit code (124 on timeout)"
    )
    timed_out: bool = Field(default=False, description="run, done update: hit its timeout")
    stdout: Annotated[str, spec(max_string=2000)] = Field(
        default="", description="run, done update: the last 20,000 chars of stdout"
    )
    stderr: Annotated[str, spec(max_string=2000)] = Field(
        default="", description="run, done update: the last 20,000 chars of stderr"
    )
    log_path: str | None = Field(
        default=None,
        description="run with log_path=, done update: the file with its full output",
    )
    lines: Annotated[list[str], spec(max_length=20, max_string=500)] = Field(
        default_factory=list,
        description=(
            "Output lines since the last update, without the newline: a throttled run's"
            " stdout, or tail"
        ),
    )
    stderr_lines: Annotated[list[str], spec(max_length=20, max_string=500)] = Field(
        default_factory=list,
        description="A throttled run's stderr lines since the last update, without the newline",
    )
    values: Annotated[list[Any], spec(max_length=10, max_string=500)] = Field(
        default_factory=list,
        description="Values since the last update: an async-iterator spawn's yields or every() ticks",
    )
    result: Annotated[Any, spec(max_string=2000)] = Field(
        default=None,
        description="Done update: a coroutine spawn's return value or after()'s payload",
    )

    def status_source(self) -> str:
        """The job's path, for the per-turn ``Unread:`` count (never its content)."""
        return self.path or self.label


_LOST_KEYS = ("label", "kind", "target", "description", "started_at")


class JobsLost(EventBase):
    """Jobs that were running when the previous process ended: they did not survive.

    Added once when a saved session is resumed in a new process. Nothing
    they would have delivered will arrive; start them again if still needed.
    """

    jobs: list[dict[str, str]] = Field(default_factory=list)
    note: str = "These self.jobs jobs were running when the session last stopped; they are gone."


_WAKE = object()  # throttle changed: re-evaluate batching
_END = object()  # source exhausted
_SKIP = object()  # collect(): nothing to deliver for this value


class _StderrLine(str):
    """A run's stderr line in a batch; delivered in ``JobUpdate.stderr_lines``."""


def _strip_newline(line: str) -> str:
    """A delivered line item never ends in a newline (``\\n`` or ``\\r\\n``)."""
    return line.removesuffix("\n").removesuffix("\r")


@hidden
class _Job:
    """Bookkeeping for one job (internal)."""

    def __init__(
        self,
        label: str,
        kind: JobKind,
        throttle: Throttle | None,
        streaming: bool,
        target: str = "",
        description: str = "",
        path: str = "",
    ):
        self.label = label
        self.path = path or label
        self.kind = kind
        self.target = target
        self.description = description
        self.payload: Any = None
        self.log_path: str | None = None  # run with log_path=: the full-output file
        self.id = uuid.uuid4().hex
        self.proc: asyncio.subprocess.Process | None = None  # set while a run command runs
        self.streaming = streaming
        self.throttle = throttle
        self.started_at = _now()
        self.finished_at: str | None = None
        self.handle: JobHandle | None = None
        self.last: deque[Any] = deque(maxlen=_KEEP_ITEMS)
        self.pending = 0
        self.wake: asyncio.Queue[Any] | None = None
        self.rethrottled = False  # set by throttle(); the batching loop flushes and clears it
        self.final_state: JobStateName | None = None

    @property
    def state(self) -> JobStateName:
        if self.final_state is not None:
            return self.final_state
        handle_state = self.handle.state if self.handle is not None else "running"
        return handle_state  # type: ignore[return-value]

    def info(self) -> JobInfo:
        return JobInfo(
            label=self.label,
            path=self.path,
            kind=self.kind,
            state=self.state,
            target=self.target,
            description=self.description,
            payload=self.payload,
            log_path=self.log_path,
            started_at=self.started_at,
            finished_at=self.finished_at,
            throttle=self.throttle,
            pending=self.pending,
            last=list(self.last),
        )

    def record(self) -> dict[str, str]:
        """The small, snapshot-safe record kept while the job runs (see ``JobsLost``)."""
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "target": self.target,
            "description": self.description,
            "started_at": self.started_at,
        }

    def status_line(self) -> str:
        """One short line for the per-turn status.

        ``/root/label · job · 3m · 2 pending · description or target``: the age
        while running (else the end state), the pending count only when above 0.
        """
        state = self.state
        if state == "running":
            state = _age(self.started_at)
        pending = f"{self.pending} pending" if self.pending > 0 else ""
        about = self.description or self.target
        parts = (self.path, "job", state, pending, about)
        return _short(" · ".join(part for part in parts if part), 100)


def _age(started_at: str) -> str:
    """Time since ``started_at`` (ISO) as ``40s``, ``3m`` or ``1h05m``."""
    started = datetime.datetime.fromisoformat(started_at)
    seconds = max(0, int((datetime.datetime.now() - started).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _short(text: str, limit: int = _TARGET_CHARS) -> str:
    """``text`` on one line, cut to ``limit`` characters with an ellipsis."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


_SECONDS_HELP = "seconds must be a number of seconds (e.g. 600 for 10 minutes)"


def _check_seconds(value: Any, *, zero_ok: bool = False) -> float:
    """``value`` as seconds: a non-bool int or float, positive (or 0 if ``zero_ok``)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{_SECONDS_HELP}, got {value!r}")
    if value < 0 or (value == 0 and not zero_ok):
        raise ValueError(f"{_SECONDS_HELP}; it must be {'0 or more' if zero_ok else 'positive'}")
    return value


def _check_description(description: Any) -> None:
    if not isinstance(description, str):
        raise TypeError("description must be a string")


def _throttle(items: int | None, seconds: float | None) -> Throttle | None:
    if items is not None:
        if isinstance(items, bool) or not isinstance(items, int):
            raise TypeError(f"items must be a whole number, got {items!r}")
        if items < 1:
            raise ValueError("items must be at least 1")
    if seconds is not None:
        _check_seconds(seconds)
    if items is None and seconds is None:
        return None
    return Throttle(items=items, seconds=seconds)


async def _stop_pumps(tasks: list[asyncio.Task]) -> None:
    """Cancel a command's pipe readers and wait for them; never blocks on a queue."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _reap(proc: asyncio.subprocess.Process) -> None:
    """Wait for a (killed) command to end, reading and dropping what is left in its pipes.

    Once the readers are stopped nobody reads the pipes, and asyncio's
    ``proc.wait()`` also waits for them to close: a pipe left paused (its
    buffer full) would make it wait forever.
    """

    async def discard(stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        with contextlib.suppress(Exception):
            while await stream.read(_CHUNK_BYTES):
                pass

    drains = [asyncio.create_task(discard(stream)) for stream in (proc.stdout, proc.stderr)]
    try:
        await proc.wait()
    finally:
        for drain in drains:
            drain.cancel()
        await asyncio.gather(*drains, return_exceptions=True)


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


def _text_chars(value: Any) -> int:
    """Characters of text in a delivered line or value (for the batch cap); 0 for non-text values."""
    return len(value) if isinstance(value, str) else 0


async def _tail_lines(path: Path, offset: int) -> AsyncIterator[str]:
    """Yield complete lines appended to ``path`` after byte ``offset``, polling for more.

    The file is opened here, on first iteration, and closed when the
    generator ends: a job cancelled before it ever ran holds no file.
    """
    splitter = LineSplitter(_LINE_CHARS)
    handle = open(path, "rb")  # noqa: SIM115 - closed in the finally below
    handle.seek(offset)
    try:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                with contextlib.suppress(OSError):
                    if os.fstat(handle.fileno()).st_size < handle.tell():
                        handle.seek(0)  # truncated (e.g. rewritten): follow it from the top
                        splitter = LineSplitter(_LINE_CHARS)
                        continue
                await asyncio.sleep(_TAIL_POLL_SECONDS)
                continue
            for line in splitter.feed(chunk):
                yield _strip_newline(line)
    finally:
        handle.close()


async def _ticks(seconds: float, payload: Any = None) -> AsyncIterator[Any]:
    """Yield ``payload`` (if given) or 1, 2, 3, ... every ``seconds``."""
    tick = 0
    while True:
        await asyncio.sleep(seconds)
        tick += 1
        yield tick if payload is None else payload


class JobsApi(Skill):
    """Background jobs: long commands and async work that report back by label.

    Use for anything that may take more than a minute or two (test suites,
    builds, evals) instead of blocking a cell. Each ``run`` command gets its
    own fresh bash process, so ``self.shell`` stays free; it starts in the
    shell's current directory with the shell's exported environment. Output
    arrives as ``JobUpdate`` items in ``notification["jobs"]``, each with ``.label``.

        await self.jobs.run("uv run pytest -q", label="tests", description="full run before pushing")
        return_result(Waiting(explanation="tests running", on=["tests"]))
        # next turn: u = notification["jobs"][0]  -> JobUpdate with done=True
        # u.returncode, u.timed_out; u.stdout / u.stderr: the last 20,000 chars of each

        await self.jobs.run("make build", label="build", items=200, seconds=60)
        # -> JobUpdate batches of output lines while it runs, then the done update

    A ``run``'s output is capped at 20,000 chars per stream: the done
    update keeps the last 20,000 chars of stdout and, separately, of
    stderr. For the full output pass ``log_path=`` to ``run``.

    Any job's update starts the next turn; check ``u.label`` / ``u.done`` and
    return ``Waiting`` again if yours has not finished.

    A label is a Python identifier (``tests``, ``ci_check``), unique among
    your running jobs and open children; the job's path is your session's
    path plus ``/label`` (``/root/tests``).

    ``tail(path)`` follows a file, ``after(seconds)`` is a timer and
    ``every(seconds)`` ticks. ``run`` and ``cancel`` are awaited; the other
    methods return at once. Every start method takes an optional
    ``description=``; ``JobInfo.target`` says what runs. ``replace=True``
    restarts a running job with the same label. ``list()``/``get(label)`` show state; ``await cancel(label)`` stops
    a job (and kills its process).
    """

    # Not in any signature, but it is what arrives in notification["jobs"]:
    # referencing it here lists it under doc(self.jobs)'s Referenced Types.
    JobUpdate: ClassVar[type[JobUpdate]] = JobUpdate

    def __init__(self, agent: Any):
        self._agent = agent
        self._queues: QueueManager = agent.queue_manager
        self._jobs: dict[str, _Job] = {}
        self._finished: deque[_Job] = deque(maxlen=_KEEP_FINISHED)
        self._queues.set_cancel_hint(JOBS_CHANNEL, "Cancel a job: await self.jobs.cancel('<name>')")

    # ---- internals -------------------------------------------------------

    @hidden
    def _channel(self) -> "Channel[Any]":
        channels = self._queues.channels()
        if JOBS_CHANNEL not in channels:
            return self._queues.queue(JOBS_CHANNEL)
        return channels[JOBS_CHANNEL]

    @hidden
    def _new_job(
        self,
        label: str,
        kind: JobKind,
        throttle: Throttle | None,
        *,
        target: str,
        description: str,
        replace: bool,
        streaming: bool = True,
    ) -> _Job:
        """Claim ``label`` (stopping its running job if ``replace``) for a new job.

        ``after`` and a coroutine ``spawn`` pass ``streaming=False`` (they
        cannot be throttled).
        """
        _check_description(description)
        self._claim(label, replace)
        job = _Job(
            label,
            kind,
            throttle,
            streaming,
            _short(target),
            description.strip(),
            path=child_path(self._base_path(), label),
        )
        if label in self._jobs:
            self._finished.append(self._jobs[label])
        self._jobs[label] = job
        return job

    @hidden
    def _claim(self, label: str, replace: bool) -> None:
        """Check ``label`` is free, first stopping its running job if ``replace``."""
        check_name(label, "job", self._queues.channels())
        child = self._child_using(label)
        if child:
            raise ValueError(
                f"name {label!r} is already used by child {child}; pick another label, or "
                f"close the child first (await self.child({label!r}).close())"
            )
        if replace:
            self._stop_now(label)
        current = self._jobs.get(label)
        if current is not None and current.state == "running":
            raise ValueError(
                f"a job labelled {label!r} is already running; pass replace=True to "
                f"restart it, or await self.jobs.cancel({label!r})"
            )

    @hidden
    def _child_using(self, name: str) -> str | None:
        """The path of the agent's open child named ``name``, if it has one (the shared namespace)."""
        using = getattr(self._agent, "_child_using_name", None)
        return using(name) if callable(using) else None

    @hidden
    def _base_path(self) -> str:
        """The agent's session path (``/root`` outside a session tree)."""
        path = getattr(self._agent, "_session_path", None)
        return (path() if callable(path) else None) or ROOT_PATH

    @hidden
    def _start(self, job: _Job, work: Any) -> JobInfo:
        self._channel()
        job.handle = self._queues.spawn(work, channel=JOBS_CHANNEL, label=job.label)
        job.handle.describe = job.status_line
        self._agent._jobs_running[job.label] = job.record()
        job.handle.add_done_callback(lambda handle: self._forget(job, handle))
        return job.info()

    @hidden
    def _forget(self, job: _Job, handle: "JobHandle") -> None:
        """Drop a finished job from ``agent._jobs_running``.

        That dict (label -> ``_Job.record()`` of each running job) is saved
        with the agent's snapshot; ``after_restore`` reads what the previous
        process left there.
        """
        if handle.state == "cancelled" and job.final_state is None:
            return  # stopped by a shutdown, not by the job or the model: it did not survive
        record = self._agent._jobs_running
        if record.get(job.label, {}).get("id") == job.id:
            del record[job.label]

    @hidden
    def _stop_now(self, label: str) -> None:
        """Stop a running job without awaiting it (``replace=True``).

        Kills a ``run`` command's process group at once, requests the task's
        cancellation and withdraws its queued updates, so the old job
        delivers nothing more.
        """
        job = self._jobs.get(label)
        if job is None or job.handle is None or job.state != "running":
            return
        if job.proc is not None:
            _kill_group(job.proc)
        job.handle.cancel_nowait()  # marks the handle cancelled
        # Withdraw its updates still waiting on the channel: none is delivered
        # after the stop. (Once the task is cancelled it puts nothing more.)
        channel = self._queues.channels().get(JOBS_CHANNEL)
        if channel is not None:
            for queued in channel.snapshot():
                if isinstance(queued, JobUpdate) and queued.label == label:
                    channel.remove(queued)
        job.final_state = "cancelled"
        job.finished_at = job.finished_at or _now()

    @hidden
    def after_restore(self) -> None:
        """Report jobs a previous process left running; they did not survive.

        Called after the agent's snapshot was restored in a new process:
        ``agent._jobs_running`` then holds the previous process's running
        jobs. Adds one ``JobsLost`` event (the model sees it in the next turn)
        and clears the record, so it is reported once.
        """
        kept: dict[str, dict[str, str]] = {}
        lost: list[dict[str, str]] = []
        for label, entry in self._agent._jobs_running.items():
            current = self._jobs.get(label)
            if current is not None and current.id == entry.get("id"):
                kept[label] = entry
            else:
                lost.append(entry)
        self._agent._jobs_running = kept
        if not lost:
            return
        events = getattr(self._agent, "event_manager", None)
        if events is not None:
            events.add(
                JobsLost(
                    jobs=[{key: str(entry.get(key, "")) for key in _LOST_KEYS} for entry in lost]
                )
            )

    @hidden
    def _shell_dir(self) -> str:
        """The agent shell's tracked directory (else the agent's, else this process's)."""
        shell = getattr(self._agent, "shell", None)
        return str(getattr(shell, "cwd", None) or getattr(self._agent, "cwd", None) or os.getcwd())

    @hidden
    async def _shell_context(self) -> tuple[str, dict[str, str]]:
        """The agent shell's directory and exported environment, cheaply.

        Reads them from the running ``self.shell`` session when it is idle
        (``BashSession.snapshot_cwd_env``); otherwise falls back to the
        shell's tracked directory and this process's ``os.environ``.
        """
        session = getattr(getattr(self._agent, "shell", None), "session", None)
        snapshot = getattr(session, "snapshot_cwd_env", None)
        if snapshot is not None:
            captured = await snapshot(timeout=5.0)
            if captured is not None:
                return captured
        return self._shell_dir(), dict(os.environ)

    @hidden
    async def _lines(
        self,
        command: str,
        stdin: str | None,
        timeout: float | None,
        cwd: str | Path | None,
        context: tuple[str, dict],
        job: _Job,
    ) -> AsyncIterator[StreamEvent | StreamDone]:
        """Run one command in its own process; yield its lines, then a ``StreamDone``."""
        base, env = context
        directory = Path(base) if cwd is None else (Path(base) / Path(cwd)).resolve()
        proc = job.proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(directory),
            env=env,
            start_new_session=True,
        )
        queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue(_QUEUE_ITEMS)

        async def pump(stream: asyncio.StreamReader, kind: str) -> None:
            # The end marker is put only when the stream ends or fails, never
            # on cancellation: then nobody reads the queue any more, and a put
            # on a full queue would block the cleanup below forever.
            try:
                async for line in bounded_lines(
                    stream,
                    max_line_chars=_LINE_CHARS,
                    chunk_bytes=_CHUNK_BYTES,
                ):
                    await queue.put((kind, line))
            except Exception:
                await queue.put(None)
                raise
            await queue.put(None)

        async def feed() -> None:
            assert proc.stdin is not None and stdin is not None
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                proc.stdin.write(stdin.encode())
                await proc.stdin.drain()
            proc.stdin.close()

        assert proc.stdout is not None and proc.stderr is not None
        tasks = [
            asyncio.create_task(pump(proc.stdout, "stdout")),
            asyncio.create_task(pump(proc.stderr, "stderr")),
        ]
        if stdin is not None:
            tasks.append(asyncio.create_task(feed()))
        deadline = None if timeout is None else time.monotonic() + timeout
        timed_out = False
        open_streams = 2
        try:
            while open_streams:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    timed_out = True
                    break
                try:
                    entry = await asyncio.wait_for(queue.get(), remaining)
                except TimeoutError:
                    timed_out = True
                    break
                if entry is None:
                    open_streams -= 1
                    continue
                kind, text = entry
                yield StreamEvent(kind=kind, text=text)
            if not timed_out:
                # The pipes can close long before the process ends (`cmd > out.txt 2>&1`),
                # so the deadline covers the wait too.
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                try:
                    await asyncio.wait_for(proc.wait(), remaining)
                except TimeoutError:
                    timed_out = True
            if timed_out:
                _kill_group(proc)
                await _stop_pumps(tasks)
                # Lines the pumps had already queued still go out.
                while not queue.empty():
                    entry = queue.get_nowait()
                    if entry is not None:
                        yield StreamEvent(kind=entry[0], text=entry[1])
                await _reap(proc)
            code = 124 if timed_out else (proc.returncode or 0)
            yield StreamDone(kind="done", returncode=code, timed_out=timed_out)
        finally:
            _kill_group(proc)
            await _stop_pumps(tasks)
            with contextlib.suppress(Exception):
                await _reap(proc)

    @hidden
    async def _run_job(
        self,
        job: _Job,
        command: str,
        stdin: str | None,
        timeout: float | None,
        cwd: str | Path | None,
        context: tuple[str, dict],
        log: Any = None,
    ) -> AsyncIterator[JobUpdate]:
        """Run one command; yield progress updates (if throttled), then the done update.

        With ``log`` (an open text file, from ``log_path=``) every line is also
        written there as it arrives; the file is closed when the job ends.
        """
        # Per stream, the kept end of the output (in memory; the full output only with log_path=).
        tails = {
            kind: TruncatingStringIO(limit=_TAIL_CHARS, tail_chars=_TAIL_CHARS)
            for kind in ("stdout", "stderr")
        }
        done = [StreamDone(kind="done", returncode=0, timed_out=False)]

        async def source() -> AsyncIterator[Any]:
            try:
                async for event in self._lines(command, stdin, timeout, cwd, context, job):
                    if log is not None and isinstance(event, StreamEvent):
                        log.write(event.text)
                    yield event
            finally:
                if log is not None:
                    log.close()

        def collect(event: Any) -> Any:
            """Keep the output tail; return the progress item to deliver, or ``_SKIP``."""
            if isinstance(event, StreamDone):
                done[0] = event
                return _SKIP
            tails[event.kind].write(event.text)
            line = _strip_newline(event.text)
            return _StderrLine(line) if event.kind == "stderr" else line

        def finish() -> dict[str, Any]:
            return {
                "returncode": done[0].returncode,
                "timed_out": done[0].timed_out,
                "stdout": tails["stdout"].getvalue(),
                "stderr": tails["stderr"].getvalue(),
                "log_path": job.log_path,
            }

        stream = self._stream_job(job, source(), into="lines", collect=collect, finish=finish)
        async for item in stream:
            yield item

    @hidden
    async def _call_job(self, job: _Job, work: Awaitable[Any]) -> JobUpdate:
        try:
            item = JobUpdate(label=job.label, path=job.path, result=await work, done=True)
        except Exception as exc:
            item = JobUpdate(
                label=job.label, path=job.path, done=True, error=f"{type(exc).__name__}: {exc}"
            )
            job.final_state = "failed"
        finally:
            job.finished_at = _now()
        job.last.append(item)
        return item

    @hidden
    async def _stream_job(
        self,
        job: _Job,
        source: AsyncIterator[Any],
        *,
        into: Literal["lines", "values"],
        collect: Callable[[Any], Any] | None = None,
        finish: Callable[[], dict[str, Any]] | None = None,
    ) -> AsyncIterator[JobUpdate]:
        """Pump ``source`` and yield ``JobUpdate`` batches per the job's live throttle.

        A batch goes to the update's ``into`` field (``lines`` or ``values``).
        Plain streams deliver every value (batched when throttled) and end with
        a done update carrying the last pending values. With ``finish``
        (``run``), values pass ``collect`` first (it returns what to deliver, or
        ``_SKIP``) and are delivered only while throttled; the done update's
        fields are what ``finish`` returns, and values still pending then are
        dropped (``stdout``/``stderr`` cover them). On an error the pending
        values go out first, then a done update with only ``error``.
        """
        inbox: asyncio.Queue[Any] = asyncio.Queue(_QUEUE_ITEMS)
        job.wake = inbox
        failure: list[str] = []

        async def pump() -> None:
            # _END is put when the source ends or fails, not on cancellation
            # (from the finally below): nobody reads the inbox then, and a put
            # on a full inbox would block that cleanup forever.
            try:
                async for value in source:
                    await inbox.put(value)
            except Exception as exc:
                failure.append(f"{type(exc).__name__}: {exc}")
            await inbox.put(_END)

        reader = asyncio.create_task(pump())
        batch: list[Any] = []
        batch_chars = 0
        batch_started: float | None = None

        final: dict[str, Any] = {}

        def update(done: bool = False) -> JobUpdate:
            nonlocal batch, batch_chars, batch_started
            batched: dict[str, list[Any]] = {into: batch}
            if into == "lines":  # a run's stderr lines go to their own field
                batched = {
                    "lines": [v for v in batch if not isinstance(v, _StderrLine)],
                    "stderr_lines": [str(v) for v in batch if isinstance(v, _StderrLine)],
                }
            item = JobUpdate(
                label=job.label,
                path=job.path,
                done=done,
                error=failure[0] if done and failure else None,
                **batched,
                **(final if done else {}),
            )
            batch, batch_chars, batch_started = [], 0, None
            job.pending = 0
            job.last.append(item)
            return item

        try:
            while True:
                if job.rethrottled:
                    # Throttle changed: deliver what was batched under the old setting.
                    job.rethrottled = False
                    if batch:
                        yield update()
                throttle = job.throttle
                timeout = None
                if throttle is not None and throttle.seconds is not None and batch:
                    assert batch_started is not None
                    timeout = max(0.0, batch_started + throttle.seconds - time.monotonic())
                try:
                    value = await asyncio.wait_for(inbox.get(), timeout)
                except TimeoutError:
                    yield update()
                    continue
                if value is _END:
                    break
                if value is _WAKE:
                    continue  # job.rethrottled is handled at the top of the loop
                if collect is not None:
                    value = collect(value)
                    if value is _SKIP:
                        continue
                throttle = job.throttle
                if finish is not None and throttle is None:
                    continue  # a quiet run: its output goes to the done update only
                if not batch:
                    batch_started = time.monotonic()
                batch.append(value)
                batch_chars += _text_chars(value)
                job.pending = len(batch)
                if (
                    throttle is None
                    or (throttle.items is not None and len(batch) >= throttle.items)
                    or len(batch) >= _BATCH_ITEMS
                    or batch_chars >= _BATCH_CHARS
                ):
                    yield update()
                elif (
                    throttle.seconds is not None
                    and batch_started is not None
                    and time.monotonic() - batch_started >= throttle.seconds
                ):
                    yield update()
            if failure:
                job.final_state = "failed"
                if batch:
                    yield update()
            elif finish is not None:
                batch = []  # the final output replaces any pending progress
                final.update(finish())
            job.finished_at = _now()
            yield update(done=True)
        finally:
            job.wake = None
            job.pending = 0  # ended (or cancelled): nothing more is batched
            if job.finished_at is None:
                job.finished_at = _now()
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            aclose = getattr(source, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()

    # ---- model-facing API -----------------------------------------------

    async def run(
        self,
        cmd: str,
        *,
        label: str,
        stdin: str | None = None,
        timeout: float | None = None,
        cwd: str | Path | None = None,
        items: int | None = None,
        seconds: float | None = None,
        description: str = "",
        replace: bool = False,
        log_path: str | Path | None = None,
    ) -> JobInfo:
        """Run a shell command in the background, in its own bash process.

        Like ``self.shell.run`` (same ``stdin``/``cwd``; ``cwd`` is relative to
        the shell's directory) but returns at once and has no timeout unless
        given. ``cd``/``export`` in ``cmd`` do not affect ``self.shell``. When
        it ends, a ``JobUpdate`` with ``done=True`` arrives in
        ``notification["jobs"]`` with ``returncode`` (124 on timeout),
        ``timed_out``, ``stdout`` and ``stderr``. ``stdout`` and ``stderr`` are
        each capped at 20,000 chars: only the last 20,000 of each stream are
        kept, after a ``[... N chars omitted]`` line. For the full output pass
        ``log_path=``: stdout and stderr are written there as lines arrive
        (interleaved; relative to the command's directory; overwritten), and
        the done update and ``JobInfo`` carry the path. With ``items``/``seconds`` it also sends progress while running:
        batches of output lines, ``JobUpdate.lines`` (stdout) and
        ``JobUpdate.stderr_lines`` (see ``throttle``; a line is cut at 8,000
        chars); lines still batched when it ends arrive only in
        ``stdout``/``stderr``.

            await self.jobs.run("uv run pytest -q", label="tests", log_path="logs/tests.log")
            return_result(Waiting(explanation="tests running", on=["tests"]))
        """
        # Every argument is checked before _claim, which stops the old job on replace=True.
        if not isinstance(cmd, str):
            raise TypeError(f"cmd must be a string, got {cmd!r}")
        if stdin is not None and not isinstance(stdin, str):
            raise TypeError(f"stdin must be a string or None, got {type(stdin).__name__}")
        if cwd is not None and not isinstance(cwd, str | Path):
            raise TypeError(f"cwd must be a str or Path, got {cwd!r}")
        if log_path is not None and not isinstance(log_path, str | Path):
            raise TypeError(f"log_path must be a str or Path, got {log_path!r}")
        if timeout is not None:
            _check_seconds(timeout)
        _check_description(description)
        throttle = _throttle(items, seconds)
        self._claim(label, replace)  # fail before the await
        context = await self._shell_context()
        log = None
        if log_path is not None:
            base = Path(context[0]) if cwd is None else (Path(context[0]) / Path(cwd)).resolve()
            resolved = (base / Path(log_path)).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered, so the file can be read while the job runs.
            log = open(resolved, "w", encoding="utf-8", errors="replace", buffering=1)  # noqa: SIM115
        job = self._new_job(
            label, "run", throttle, target=cmd, description=description, replace=replace
        )
        if log is not None:
            job.log_path = log.name
        return self._start(job, self._run_job(job, cmd, stdin, timeout, cwd, context, log))

    def spawn(
        self,
        work: Any,
        *,
        label: str,
        items: int | None = None,
        seconds: float | None = None,
        description: str = "",
        replace: bool = False,
    ) -> JobInfo:
        """Run a coroutine or async iterator in the background.

        A coroutine delivers one done update whose ``result`` is its return
        value (``error`` set if it raised). An async iterator delivers one
        ``JobUpdate`` per yielded value in ``values`` (batched with
        ``items``/``seconds``), then a done update with any values still pending.

            self.jobs.spawn(fetch_report(url), label="report")  # any coroutine
            self.jobs.spawn(watch_queue(), label="watch", seconds=60)  # an async generator
        """
        if inspect.iscoroutinefunction(work) or inspect.isasyncgenfunction(work):
            name = getattr(work, "__name__", "work")
            raise TypeError(f"pass {name}(), not {name}: spawn() takes the call's result")
        try:
            name = getattr(work, "__qualname__", None) or type(work).__name__
            about = {"target": f"{name}()", "description": description, "replace": replace}
            if hasattr(work, "__aiter__"):
                job = self._new_job(label, "spawn", _throttle(items, seconds), **about)
                return self._start(job, self._stream_job(job, work.__aiter__(), into="values"))
            if not isinstance(work, Awaitable):
                raise TypeError("spawn() takes a coroutine or an async iterator")
            if items is not None or seconds is not None:
                raise ValueError("items/seconds apply to async iterators only")
            job = self._new_job(label, "spawn", None, streaming=False, **about)
        except BaseException:
            # Rejected: close the model's coroutine so it is not reported as never awaited.
            if inspect.iscoroutine(work):
                work.close()
            raise
        return self._start(job, self._call_job(job, work))

    def tail(
        self,
        path: str | Path,
        *,
        label: str,
        items: int | None = None,
        seconds: float | None = None,
        description: str = "",
        replace: bool = False,
    ) -> JobInfo:
        """Deliver lines appended to a file from now on, until cancelled.

        Each new line (a str, without the newline) goes to ``JobUpdate.lines``;
        batch with ``items``/``seconds``. A relative ``path`` is relative to the shell's
        directory.

            self.jobs.tail("logs/server.log", label="server_log", seconds=60)
        """
        # Every argument (and the file) is checked before _claim, which stops the
        # old job on replace=True.
        if not isinstance(path, str | Path):
            raise TypeError(f"path must be a str or Path, got {path!r}")
        _check_description(description)
        throttle = _throttle(items, seconds)
        resolved = (Path(self._shell_dir()) / Path(path)).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"no such file: {resolved}")
        if not os.access(resolved, os.R_OK):
            raise PermissionError(f"cannot read {resolved}")
        offset = resolved.stat().st_size  # deliver what is appended from now on
        target = str(resolved) if str(resolved) == str(path) else f"{path} ({resolved})"
        job = self._new_job(
            label, "tail", throttle, target=target, description=description, replace=replace
        )
        return self._start(job, self._stream_job(job, _tail_lines(resolved, offset), into="lines"))

    def after(
        self,
        seconds: float,
        *,
        label: str,
        payload: Any = None,
        description: str = "",
        replace: bool = False,
    ) -> JobInfo:
        """A timer: after ``seconds``, one done update whose ``result`` is ``payload``.

        ``seconds=0`` fires right away; ``cancel(label)`` stops it before it fires.

            self.jobs.after(600, label="check_ci", payload="check CI on PR #462 and report")
        """
        _check_seconds(seconds, zero_ok=True)
        job = self._new_job(
            label,
            "after",
            None,
            target=f"after {seconds:g}s",
            description=description,
            replace=replace,
            streaming=False,
        )
        job.payload = payload
        return self._start(job, self._call_job(job, asyncio.sleep(seconds, result=payload)))

    def every(
        self,
        seconds: float,
        *,
        label: str,
        payload: Any = None,
        description: str = "",
        replace: bool = False,
    ) -> JobInfo:
        """Deliver a tick every ``seconds`` until cancelled.

        Each tick is one entry of ``JobUpdate.values``: ``payload`` if given,
        else the tick count (1, 2, 3, ...); ``throttle(label, ...)`` batches them.

            self.jobs.every(300, label="poll_ci", payload="check CI and report changes")
        """
        _check_seconds(seconds)
        job = self._new_job(
            label,
            "every",
            None,
            target=f"every {seconds:g}s",
            description=description,
            replace=replace,
        )
        job.payload = payload
        return self._start(job, self._stream_job(job, _ticks(seconds, payload), into="values"))

    def throttle(
        self, label: str, *, items: int | None = None, seconds: float | None = None
    ) -> JobInfo:
        """Batch a running job's deliveries: one per ``items`` values or ``seconds``.

        Whichever comes first triggers a delivery; the rest is flushed at the
        end. So that one update never gets too big, a batch is also delivered
        early once it holds 200 lines/values or 20,000 chars of text (a size
        limit per update, unrelated to a ``run``'s 20,000-char output cap).
        Each call replaces the previous setting (omitted arguments become
        None), first delivering what was already batched. ``throttle(label)``
        turns batching off: a ``run`` goes quiet until its done update (whose
        ``stdout``/``stderr`` hold the output), other jobs deliver every value.
        Applies at once, without restarting the job.

            self.jobs.throttle("build", items=200)
        """
        job = self._jobs.get(label)
        if job is None:
            raise KeyError(f"no job labelled {label!r}")
        if not job.streaming:
            raise ValueError(f"job {label!r} is not a streaming job")
        job.throttle = _throttle(items, seconds)
        job.rethrottled = True
        if job.wake is not None:
            with contextlib.suppress(asyncio.QueueFull):  # a full inbox wakes it anyway
                job.wake.put_nowait(_WAKE)
        return job.info()

    def list(self) -> list[JobInfo]:
        """All known jobs (running and recent finished), oldest first."""
        jobs = [*self._finished, *self._jobs.values()]
        return [job.info() for job in jobs]

    def get(self, label: str) -> JobInfo | None:
        """The latest job with ``label`` (``.last`` holds its recent updates), or None."""
        job = self._jobs.get(label)
        return None if job is None else job.info()

    async def cancel(self, label: str) -> bool:
        """Stop a running job (killing its process); True if it was running.

        Its updates still queued are dropped; ones already handed to the
        current turn (in ``notification["jobs"]``) are not taken back.
        """
        job = self._jobs.get(label)
        if job is None or job.handle is None or job.state != "running":
            return False
        self._stop_now(label)
        await job.handle.cancel()  # waits for the task to unwind
        return True

    def __repr__(self) -> str:
        running = sum(1 for job in self._jobs.values() if job.state == "running")
        return f"<JobsApi({running} running)>"


JobInfo.model_rebuild()
