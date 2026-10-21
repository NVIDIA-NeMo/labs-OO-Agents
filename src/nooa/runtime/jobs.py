# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing JobsApi: run long work in the background, by label.

``JobsApi`` is attached to interactive agents as ``self.jobs``. It runs
shell commands in their own bash processes (not the agent's persistent
shell session) and arbitrary coroutines or async iterators, and delivers
their output on the agent's ``jobs`` queue channel. The channel registry
(``self.queue_manager``) stays hidden; this is the model-facing view of it,
as ``self.events`` is for ``self.event_manager``.

Every delivered item carries the job's ``label``, so a turn can wait with
``Waiting(on=["jobs:<label>"])`` (or ``on=["jobs"]`` for any job) and find
the outcome in ``notification["jobs"]``.
"""

import asyncio
import base64
import contextlib
import datetime
import inspect
import os
import signal
import tempfile
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from nooa.agentdoc import (
    FileBackedTruncatingStringIO,
    LineSplitter,
    TruncatingStringIO,
    bounded_lines,
    hidden,
)
from nooa.context_blocks import EventBase
from nooa.skill import Skill
from nooa.tools._results import StreamDone, StreamEvent
from nooa.tools.shell_tools import ShellResult

if TYPE_CHECKING:
    from nooa.runtime.channels import Channel, JobHandle, QueueManager

__all__ = ["JobInfo", "JobUpdate", "JobsApi", "JobsLost", "Throttle"]

JOBS_CHANNEL = "jobs"
# Characters of stdout and of stderr kept in a delivered ShellResult (the end
# of the output); the full output is in the log.
_TAIL_CHARS = 3_000
# Per-job log cap; output beyond it is counted but not written.
_LOG_CAP_BYTES = 50 * 1024 * 1024
# Longest delivered line (run progress, tail), in characters; the rest of the
# line is replaced by a "[... N chars of this line omitted]" marker.
_LINE_CHARS = 8_000
# Bytes read from a pipe or a tailed file at a time.
_CHUNK_BYTES = 64 * 1024
# A batch is delivered early once it holds this many items or chars of text.
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


class JobInfo(BaseModel):
    """A job's status, as returned by ``self.jobs`` methods."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str = Field(description="Name it was started with; items it delivers carry it")
    kind: JobKind = Field(description="The method that started it")
    state: JobStateName = Field(description="running, done, failed or cancelled")
    target: str = Field(default="", description="What runs: command, file, interval or callable")
    description: str = Field(default="", description="Why it runs, as passed with description=")
    payload: Any = Field(
        default=None, description="Value given to after()/every(); delivered as the item"
    )
    started_at: str = Field(description="Start time (ISO, seconds)")
    finished_at: str | None = Field(default=None, description="End time, None while running")
    log_path: str | None = Field(default=None, description="Full stdout+stderr log (run only)")
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


class JobUpdate(BaseModel):
    """What a job delivers to ``notification["jobs"]``: new items while it runs, then its end.

    Every job ends with exactly one ``done=True`` update; a cancelled job
    delivers nothing more. Any job's update starts the next turn; check
    ``u.label`` / ``u.done`` and return ``Waiting`` again if yours has not
    finished.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str = Field(description="The job's label")
    items: list[Any] = Field(
        default_factory=list,
        description=(
            "New values: StreamEvent output lines (throttled run), tail lines (str), yielded"
            " values, ticks; a line item never ends in a newline. On the done update: [ShellResult] (run), [return value]"
            " (coroutine spawn), [payload] (after), else values still pending or []."
            " [] when error is set"
        ),
    )
    done: bool = Field(default=False, description="True on the job's one final update")
    error: str | None = Field(
        default=None,
        description=(
            "Why the job failed ('ExceptionType: message'), on the done update only; items"
            " is then [] (e.g. a spawn whose coroutine raised). None otherwise"
        ),
    )
    log_path: str | None = Field(
        default=None, description="Full stdout+stderr log file (run only, else None)"
    )


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
        log_path: str | None,
        streaming: bool,
        target: str = "",
        description: str = "",
    ):
        self.label = label
        self.kind = kind
        self.target = target
        self.description = description
        self.payload: Any = None
        self.id = uuid.uuid4().hex
        self.proc: asyncio.subprocess.Process | None = None  # set while a run command runs
        self.streaming = streaming
        self.throttle = throttle
        self.log_path = log_path
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
            kind=self.kind,
            state=self.state,
            target=self.target,
            description=self.description,
            payload=self.payload,
            started_at=self.started_at,
            finished_at=self.finished_at,
            log_path=self.log_path,
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
        """One short line for the per-turn status: label · job · state · description or target."""
        about = self.description or self.target
        return _short(
            " · ".join(part for part in (self.label, "job", self.state, about) if part), 100
        )


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


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


def _text_chars(value: Any) -> int:
    """Characters of text in a delivered item (for the batch cap); 0 for non-text values."""
    if isinstance(value, StreamEvent):
        return len(value.text)
    return len(value) if isinstance(value, str) else 0


async def _tail_lines(handle: Any) -> AsyncIterator[str]:
    """Yield complete lines appended to an open binary file, polling for more."""
    splitter = LineSplitter(_LINE_CHARS)
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
    arrives as items in ``notification["jobs"]``, each with ``.label``.

        await self.jobs.run("uv run pytest -q", label="tests", description="full run before pushing")
        return_result(Waiting(explanation="tests running", on=["tests"]))  # or "jobs:tests"
        # next turn: u = notification["jobs"][0]  -> JobUpdate with done=True
        # u.items[0] is the ShellResult (.returncode, .stdout tail); u.log_path the full log

        await self.jobs.run("make build", label="build", items=200, seconds=60)
        # -> JobUpdate batches of output lines while it runs, then the done update

    Any job's update starts the next turn; check ``u.label`` / ``u.done`` and
    return ``Waiting`` again if yours has not finished.

    ``tail(path)`` follows a file, ``after(seconds)`` is a timer and
    ``every(seconds)`` ticks. ``run`` and ``cancel`` are awaited; the other
    methods return at once. Every start method takes an optional
    ``description=``; ``JobInfo.target`` says what runs. Labels are unique
    among running jobs; ``replace=True`` restarts a running job with the same
    label. ``list()``/``get(label)`` show state; ``await cancel(label)`` stops
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
        self._log_dir: Path | None = None
        self._queues.set_cancel_hint(
            JOBS_CHANNEL, "Cancel a job: await self.jobs.cancel('<label>')"
        )

    # ---- internals -------------------------------------------------------

    @hidden
    def _channel(self) -> "Channel[Any]":
        channels = self._queues.channels()
        if JOBS_CHANNEL not in channels:
            return self._queues.queue(JOBS_CHANNEL)
        return channels[JOBS_CHANNEL]

    @hidden
    def _logs(self) -> Path:
        if self._log_dir is None:
            root = Path(tempfile.gettempdir()) / "nooa-jobs"
            root.mkdir(parents=True, exist_ok=True)
            self._log_dir = Path(tempfile.mkdtemp(prefix=f"{os.getpid()}-", dir=root))
        return self._log_dir

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

        Only a ``run`` gets a log; ``after`` and a coroutine ``spawn`` pass
        ``streaming=False`` (they cannot be throttled).
        """
        if not isinstance(description, str):
            raise TypeError("description must be a string")
        self._claim(label, replace)
        log_path = None
        if kind == "run":
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)[:60]
            log_path = str(self._logs() / f"{safe}-{uuid.uuid4().hex[:6]}.log")
        job = _Job(label, kind, throttle, log_path, streaming, _short(target), description.strip())
        if label in self._jobs:
            self._finished.append(self._jobs[label])
        self._jobs[label] = job
        return job

    @hidden
    def _claim(self, label: str, replace: bool) -> None:
        """Check ``label`` is free, first stopping its running job if ``replace``."""
        if replace:
            self._stop_now(label)
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a non-empty string")
        current = self._jobs.get(label)
        if current is not None and current.state == "running":
            raise ValueError(
                f"a job labelled {label!r} is already running; pass replace=True to "
                f"restart it, or await self.jobs.cancel({label!r})"
            )

    @hidden
    def _start(self, job: _Job, work: Any) -> JobInfo:
        self._channel()
        job.handle = self._queues.spawn(work, channel=JOBS_CHANNEL, label=job.label)
        job.handle.describe = job.status_line
        self._agent._jobs_running[job.label] = job.record()
        job.handle._task.add_done_callback(lambda task: self._forget(job, task))
        return job.info()

    @hidden
    def _forget(self, job: _Job, task: asyncio.Task[Any]) -> None:
        """Drop a finished job from ``agent._jobs_running``.

        That dict (label -> ``_Job.record()`` of each running job) is saved
        with the agent's snapshot; ``after_restore`` reads what the previous
        process left there.
        """
        if task.cancelled() and job.final_state is None:
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
        job.handle._task.cancel()
        # Withdraw its updates still waiting on the channel: none is delivered
        # after the stop. (Once the task is cancelled it puts nothing more.)
        channel = self._queues.channels().get(JOBS_CHANNEL)
        if channel is not None:
            for queued in channel.snapshot():
                if isinstance(queued, JobUpdate) and queued.label == label:
                    channel.remove(queued)
        job.handle.state = "cancelled"
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

        Reads them from the running ``self.shell`` session when it is idle;
        otherwise falls back to the shell's tracked directory and this
        process's ``os.environ``.
        """
        shell = getattr(self._agent, "shell", None)
        cwd = self._shell_dir()
        env = dict(os.environ)
        session = getattr(shell, "session", None)
        lock = getattr(session, "_lock", None)
        if (
            session is None
            or not getattr(session, "_ready", False)
            or (lock is not None and lock.locked())
        ):
            return cwd, env
        try:
            out, _err, code, timed_out = await session.run_with_timeout_flag(
                'printf "%s\\n" "$PWD"; env -0 | base64 | tr -d "\\n"', timeout=5.0
            )
        except Exception:
            return cwd, env
        if code != 0 or timed_out:
            return cwd, env
        first, _, encoded = out.partition("\n")
        try:
            raw = base64.b64decode(encoded.strip())
        except ValueError:
            return cwd, env
        captured: dict[str, str] = {}
        for entry in raw.split(b"\0"):
            key, sep, value = entry.partition(b"=")
            if sep and key:
                captured[key.decode(errors="replace")] = value.decode(errors="replace")
        if captured:
            env = captured
        if first.strip():
            cwd = first.strip()
        return cwd, env

    @hidden
    async def _lines(
        self,
        command: str,
        stdin: str | None,
        timeout: float | None,
        cwd: str | Path | None,
        log: FileBackedTruncatingStringIO,
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
            try:
                async for line in bounded_lines(
                    stream,
                    max_line_chars=_LINE_CHARS,
                    chunk_bytes=_CHUNK_BYTES,
                    on_bytes=log.write_bytes,
                ):
                    await queue.put((kind, line))
            finally:
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
                await proc.wait()
            code = 124 if timed_out else (proc.returncode or 0)
            yield StreamDone(kind="done", returncode=code, timed_out=timed_out)
        finally:
            _kill_group(proc)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with contextlib.suppress(Exception):
                await proc.wait()

    @hidden
    async def _run_job(
        self,
        job: _Job,
        command: str,
        stdin: str | None,
        timeout: float | None,
        cwd: str | Path | None,
        context: tuple[str, dict],
    ) -> AsyncIterator[JobUpdate]:
        """Run one command; yield progress updates (if throttled), then the done update."""
        assert job.log_path is not None
        # The combined log (file only) and, per stream, the kept end of the output.
        log = FileBackedTruncatingStringIO(
            limit=0, tail_chars=0, path=job.log_path, max_file_bytes=_LOG_CAP_BYTES
        )
        tails = {
            kind: TruncatingStringIO(
                limit=_TAIL_CHARS, tail_chars=_TAIL_CHARS, full_output_path=job.log_path
            )
            for kind in ("stdout", "stderr")
        }
        done = [StreamDone(kind="done", returncode=0, timed_out=False)]

        async def source() -> AsyncIterator[Any]:
            try:
                async for event in self._lines(command, stdin, timeout, cwd, log, context, job):
                    yield event
            finally:
                log.close()

        def collect(event: Any) -> Any:
            """Keep the output tail; return the progress item to deliver, or ``_SKIP``."""
            if isinstance(event, StreamDone):
                done[0] = event
                return _SKIP
            tails[event.kind].write(event.text)
            return StreamEvent(kind=event.kind, text=_strip_newline(event.text))

        def finish() -> list[Any]:
            return [
                ShellResult(
                    stdout=tails["stdout"].getvalue(),
                    stderr=tails["stderr"].getvalue(),
                    returncode=done[0].returncode,
                    timed_out=done[0].timed_out,
                )
            ]

        async for item in self._stream_job(job, source(), collect=collect, finish=finish):
            yield item

    @hidden
    async def _call_job(self, job: _Job, work: Awaitable[Any]) -> JobUpdate:
        try:
            item = JobUpdate(label=job.label, items=[await work], done=True)
        except Exception as exc:
            item = JobUpdate(label=job.label, done=True, error=f"{type(exc).__name__}: {exc}")
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
        collect: Callable[[Any], Any] | None = None,
        finish: Callable[[], list[Any]] | None = None,
    ) -> AsyncIterator[JobUpdate]:
        """Pump ``source`` and yield ``JobUpdate`` batches per the job's live throttle.

        Plain streams deliver every value (batched when throttled) and end with
        a done update carrying the last pending values. With ``finish``
        (``run``), values pass ``collect`` first (it returns what to deliver, or
        ``_SKIP``) and are delivered only while
        throttled; the done update's items are what ``finish`` returns, and
        values still pending then are dropped (the result covers them). On an
        error the pending values go out first, then a done update with ``[]``.
        """
        inbox: asyncio.Queue[Any] = asyncio.Queue(_QUEUE_ITEMS)
        job.wake = inbox
        failure: list[str] = []

        async def pump() -> None:
            try:
                async for value in source:
                    await inbox.put(value)
            except Exception as exc:
                failure.append(f"{type(exc).__name__}: {exc}")
            finally:
                await inbox.put(_END)

        reader = asyncio.create_task(pump())
        batch: list[Any] = []
        batch_chars = 0
        batch_started: float | None = None

        def update(done: bool = False) -> JobUpdate:
            nonlocal batch, batch_chars, batch_started
            item = JobUpdate(
                label=job.label,
                items=batch,
                done=done,
                log_path=job.log_path,
                error=failure[0] if done and failure else None,
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
                batch = finish()  # the result replaces any pending progress
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
    ) -> JobInfo:
        """Run a shell command in the background, in its own bash process.

        Like ``self.shell.run`` (same ``stdin``/``cwd``; ``cwd`` is relative to
        the shell's directory) but returns at once and has no timeout unless
        given. ``cd``/``export`` in ``cmd`` do not affect ``self.shell``. When
        it ends, a ``JobUpdate`` with ``done=True`` arrives in
        ``notification["jobs"]``: ``items[0]`` is a ``ShellResult`` (output
        tail; on timeout returncode 124), ``log_path`` the full stdout+stderr
        log. With ``items``/``seconds`` it also sends progress while running:
        ``JobUpdate`` batches of ``StreamEvent`` lines (see ``throttle``; a line
        is cut at 8,000 chars); lines still batched when it ends arrive only in
        the ``ShellResult``.

            await self.jobs.run("uv run pytest -q", label="tests")
            return_result(Waiting(explanation="tests running", on=["jobs:tests"]))
        """
        throttle = _throttle(items, seconds)
        self._claim(label, replace)  # fail before the await
        context = await self._shell_context()
        job = self._new_job(
            label, "run", throttle, target=cmd, description=description, replace=replace
        )
        return self._start(job, self._run_job(job, cmd, stdin, timeout, cwd, context))

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

        A coroutine delivers one done update with ``items == [return value]``
        (``error`` set if it raised). An async iterator delivers one
        ``JobUpdate`` per yielded value (batched with ``items``/``seconds``),
        then a done update with any values still pending.

            self.jobs.spawn(fetch_report(url), label="report")  # any coroutine
            self.jobs.spawn(watch_queue(), label="watch", seconds=60)  # an async generator
        """
        if inspect.iscoroutinefunction(work) or inspect.isasyncgenfunction(work):
            name = getattr(work, "__name__", "work")
            raise TypeError(f"pass {name}(), not {name}: spawn() takes the call's result")
        name = getattr(work, "__qualname__", None) or type(work).__name__
        about = {"target": f"{name}()", "description": description, "replace": replace}
        if hasattr(work, "__aiter__"):
            job = self._new_job(label, "spawn", _throttle(items, seconds), **about)
            return self._start(job, self._stream_job(job, work.__aiter__()))
        if not isinstance(work, Awaitable):
            raise TypeError("spawn() takes a coroutine or an async iterator")
        if items is not None or seconds is not None:
            raise ValueError("items/seconds apply to async iterators only")
        job = self._new_job(label, "spawn", None, streaming=False, **about)
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

        Each ``JobUpdate`` item is one line (a str); batch with
        ``items``/``seconds``. A relative ``path`` is relative to the shell's
        directory.

            self.jobs.tail("logs/server.log", label="server-log", seconds=60)
        """
        throttle = _throttle(items, seconds)
        self._claim(label, replace)  # fail before opening the file
        resolved = (Path(self._shell_dir()) / Path(path)).resolve()
        handle = open(resolved, "rb")  # noqa: SIM115 - closed by the generator
        handle.seek(0, os.SEEK_END)
        target = str(resolved) if str(resolved) == str(path) else f"{path} ({resolved})"
        job = self._new_job(
            label, "tail", throttle, target=target, description=description, replace=replace
        )
        return self._start(job, self._stream_job(job, _tail_lines(handle)))

    def after(
        self,
        seconds: float,
        *,
        label: str,
        payload: Any = None,
        description: str = "",
        replace: bool = False,
    ) -> JobInfo:
        """A timer: after ``seconds``, one done update with ``items == [payload]``.

        ``seconds=0`` fires right away; ``cancel(label)`` stops it before it fires.

            self.jobs.after(600, label="check-ci", payload="check CI on PR #462 and report")
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

        Each tick is one ``JobUpdate`` item: ``payload`` if given, else the tick
        count (1, 2, 3, ...); ``throttle(label, ...)`` batches them.

            self.jobs.every(300, label="poll-ci", payload="check CI and report changes")
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
        return self._start(job, self._stream_job(job, _ticks(seconds, payload)))

    def throttle(
        self, label: str, *, items: int | None = None, seconds: float | None = None
    ) -> JobInfo:
        """Batch a running job's deliveries: one per ``items`` values or ``seconds``.

        Whichever comes first triggers a delivery; the rest is flushed at the
        end. A batch is also sent early once it holds 200 items or 20,000 chars. Each call replaces the previous setting (omitted arguments become
        None), first delivering what was already batched. ``throttle(label)``
        turns batching off: a ``run`` goes quiet until its done update (whose
        ``ShellResult`` holds the output), other jobs deliver every item.
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
