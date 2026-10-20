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
import os
import signal
import tempfile
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nooa.agentdoc import hidden
from nooa.context_blocks import EventBase
from nooa.skill import Skill
from nooa.tools._results import StreamDone, StreamEvent
from nooa.tools.shell_tools import ShellResult

if TYPE_CHECKING:
    from nooa.runtime.channels import Channel, JobHandle, QueueManager

__all__ = ["JobInfo", "JobResult", "JobUpdate", "JobsApi", "JobsLost", "Throttle"]

JOBS_CHANNEL = "jobs"
# Output kept in a delivered ShellResult (the full output is in the log).
_TAIL_CHARS = 20_000
# Per-job log cap; output beyond it is counted but not written.
_LOG_CAP_BYTES = 50 * 1024 * 1024
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
    """Batching for a streaming job: one delivery per ``lines`` items or ``seconds``."""

    lines: int | None = None
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
        default=None, description="Value given to after()/every(); after() delivers it as .result"
    )
    started_at: str = Field(description="Start time (ISO, seconds)")
    finished_at: str | None = Field(default=None, description="End time, None while running")
    log_path: str | None = Field(default=None, description="Full stdout+stderr log (run only)")
    throttle: Throttle | None = Field(default=None, description="Current batching, None if off")
    pending: int = Field(default=0, description="Items batched by throttle, not yet delivered")
    last: list[Any] = Field(default_factory=list, description="Most recent delivered items")


class JobResult(BaseModel):
    """Delivered once when a ``run`` or a coroutine ``spawn`` job ends.

    ``result`` is a ``ShellResult`` for ``run`` (stdout/stderr hold the
    output tail; ``log_path`` has all of it), else the coroutine's return
    value. ``error`` is set when the job raised.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    result: Any = None
    log_path: str | None = None
    error: str | None = None


class JobUpdate(BaseModel):
    """Delivered while a job runs: ``tail``, ``every``, async-iterator ``spawn``, or a throttled ``run``.

    ``items`` are the new values (``StreamEvent`` lines for ``run``); one per
    update unless throttled. A ``run`` ends with its ``JobResult``; for other
    jobs the final batch, if any, has ``final=True`` (``JobInfo.state`` shows
    the end). ``error`` is set when the job raised.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    items: list[Any] = Field(default_factory=list)
    final: bool = False
    log_path: str | None = None
    error: str | None = None


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
        """One short line for the per-turn queue status: label · state · description or target."""
        about = self.description or self.target
        return _short(" · ".join(part for part in (self.label, self.state, about) if part), 100)


def _short(text: str, limit: int = _TARGET_CHARS) -> str:
    """``text`` on one line, cut to ``limit`` characters with an ellipsis."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _seconds(value: float) -> str:
    return f"{value:g}s"


def _callable_target(work: Any) -> str:
    name = getattr(work, "__qualname__", None) or type(work).__name__
    return f"{name}()"


def _throttle(lines: int | None, seconds: float | None) -> Throttle | None:
    if lines is not None and lines < 1:
        raise ValueError("lines must be at least 1")
    if seconds is not None and seconds <= 0:
        raise ValueError("seconds must be positive")
    if lines is None and seconds is None:
        return None
    return Throttle(lines=lines, seconds=seconds)


class _Log:
    """Bounded append-only log file for one job (internal)."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "wb")  # noqa: SIM115 - closed in close()
        self._written = 0
        self._dropped = 0

    def write(self, data: bytes) -> None:
        room = _LOG_CAP_BYTES - self._written
        if room > 0:
            chunk = data[:room]
            self._fh.write(chunk)
            self._fh.flush()
            self._written += len(chunk)
        self._dropped += max(0, len(data) - max(room, 0))

    def close(self) -> None:
        if self._dropped:
            self._fh.write(f"\n[log truncated: {self._dropped} more bytes not written]\n".encode())
        self._fh.close()


def _tail(text: str) -> str:
    if len(text) <= _TAIL_CHARS:
        return text
    cut = len(text) - _TAIL_CHARS
    return f"[... {cut} chars omitted; full output in the log]\n" + text[cut:]


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


async def _tail_lines(handle: Any) -> AsyncIterator[str]:
    """Yield complete lines appended to an open binary file, polling for more."""
    partial = b""
    try:
        while True:
            chunk = handle.readline()
            if not chunk:
                await asyncio.sleep(_TAIL_POLL_SECONDS)
                continue
            partial += chunk
            if partial.endswith(b"\n"):
                yield partial[:-1].decode("utf-8", errors="replace").removesuffix("\r")
                partial = b""
    finally:
        handle.close()


async def _ticks(seconds: float) -> AsyncIterator[int]:
    """Yield 1, 2, 3, ... every ``seconds``."""
    tick = 0
    while True:
        await asyncio.sleep(seconds)
        tick += 1
        yield tick


class JobsApi(Skill):
    """Background jobs: long commands and async work that report back by label.

    Use for anything that may take more than a minute or two (test suites,
    builds, evals) instead of blocking a cell. Each ``run`` command gets its
    own fresh bash process, so ``self.shell`` stays free;
    it starts in the shell's current directory with the shell's exported
    environment. Output arrives as items in ``notification["jobs"]``, each
    with ``.label``.

        await self.jobs.run("uv run pytest -q", label="tests", description="full run before pushing")
        return_result(Waiting(explanation="tests running", on=["jobs:tests"]))
        # next turn: r = notification["jobs"][0]  -> JobResult
        # r.result.returncode, r.result.stdout (tail), r.log_path (full log)

        await self.jobs.run("make build", label="build", lines=200, seconds=60)
        # -> JobUpdate batches of output lines while it runs, then the JobResult

    ``tail(path)`` follows a file, ``after(seconds)`` is a timer and
    ``every(seconds)`` ticks. Every start method takes an optional
    ``description=``; ``JobInfo.target`` says what runs. ``replace=True``
    restarts a running job with the same label. Labels are unique among running jobs. ``list()``/``get(label)`` show
    state; ``await cancel(label)`` stops a job (and kills its process).
    """

    def __init__(self, agent: Any):
        self._agent = agent
        self._queues: QueueManager = agent.queue_manager
        self._jobs: dict[str, _Job] = {}
        self._finished: deque[_Job] = deque(maxlen=_KEEP_FINISHED)
        self._log_dir: Path | None = None
        self._queues.set_cancel_hint(JOBS_CHANNEL, "await self.jobs.cancel({label!r})")

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
        log: bool,
        streaming: bool,
        target: str = "",
        description: str = "",
    ) -> _Job:
        if not isinstance(description, str):
            raise TypeError("description must be a string")
        self._check_label(label)
        current = self._jobs.get(label)
        log_path = None
        if log:
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)[:60]
            log_path = str(self._logs() / f"{safe}-{uuid.uuid4().hex[:6]}.log")
        job = _Job(label, kind, throttle, log_path, streaming, _short(target), description.strip())
        if current is not None:
            self._finished.append(current)
        self._jobs[label] = job
        return job

    @hidden
    def _check_label(self, label: str) -> None:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a non-empty string")
        current = self._jobs.get(label)
        if current is not None and current.state == "running":
            raise ValueError(f"a job labelled {label!r} is already running")

    @hidden
    def _start(self, job: _Job, work: Any) -> JobInfo:
        self._channel()
        job.handle = self._queues.spawn(work, channel=JOBS_CHANNEL, label=job.label)
        job.handle.describe = job.status_line
        self._running_record()[job.label] = job.record()
        job.handle._task.add_done_callback(lambda task: self._forget(job, task))
        return job.info()

    @hidden
    def _running_record(self) -> dict[str, dict[str, str]]:
        """``agent._jobs_running``: label -> record of each running job.

        A plain dict on the agent, so it is saved with the agent's snapshot;
        ``after_restore`` reads what the previous process left there.
        """
        record = getattr(self._agent, "_jobs_running", None)
        if not isinstance(record, dict):
            record = {}
            self._agent._jobs_running = record
        return record

    @hidden
    def _forget(self, job: _Job, task: asyncio.Task[Any]) -> None:
        if task.cancelled() and job.final_state is None:
            return  # stopped by a shutdown, not by the job or the model: it did not survive
        record = self._running_record()
        entry = record.get(job.label)
        if isinstance(entry, dict) and entry.get("id") == job.id:
            del record[job.label]

    @hidden
    def _stop_now(self, label: str) -> None:
        """Stop a running job without awaiting it (``replace=True``).

        Kills a ``run`` command's process group at once and requests the
        task's cancellation, so the old job delivers nothing more.
        """
        job = self._jobs.get(label)
        if job is None or job.handle is None or job.state != "running":
            return
        if job.proc is not None:
            _kill_group(job.proc)
        job.handle._task.cancel()
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
        record = self._running_record()
        lost = [
            entry
            for label, entry in record.items()
            if isinstance(entry, dict)
            and not (label in self._jobs and self._jobs[label].id == entry.get("id"))
        ]
        self._agent._jobs_running = {
            label: entry for label, entry in record.items() if entry not in lost
        }
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
                'printf "%s\\n" "$PWD"; env -0 | base64 -w0', timeout=5.0
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
    async def _spawn_process(
        self, command: str, stdin: str | None, cwd: str | Path | None, context: tuple[str, dict]
    ) -> asyncio.subprocess.Process:
        base, env = context
        directory = Path(base) if cwd is None else (Path(base) / Path(cwd)).resolve()
        return await asyncio.create_subprocess_exec(
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

    @hidden
    async def _lines(
        self,
        command: str,
        stdin: str | None,
        timeout: float | None,
        cwd: str | Path | None,
        log: _Log,
        context: tuple[str, dict],
        job: _Job | None = None,
    ) -> AsyncIterator[StreamEvent | StreamDone]:
        """Run one command in its own process; yield its lines, then a ``StreamDone``."""
        proc = await self._spawn_process(command, stdin, cwd, context)
        if job is not None:
            job.proc = proc
        queue: asyncio.Queue[tuple[str, bytes] | None] = asyncio.Queue()

        async def pump(stream: asyncio.StreamReader, kind: str) -> None:
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    log.write(line)
                    queue.put_nowait((kind, line))
            except (ValueError, asyncio.LimitOverrunError):
                rest = await stream.read()
                log.write(rest)
                queue.put_nowait((kind, rest))
            finally:
                queue.put_nowait(None)

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
                kind, data = entry
                yield StreamEvent(kind=kind, text=data.decode("utf-8", errors="replace"))
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
    ) -> AsyncIterator[JobUpdate | JobResult]:
        """Run one command; yield progress ``JobUpdate``s (if throttled), then its ``JobResult``."""
        assert job.log_path is not None
        log = _Log(Path(job.log_path))
        out: list[str] = []
        err: list[str] = []
        done = [StreamDone(kind="done", returncode=0, timed_out=False)]

        async def source() -> AsyncIterator[Any]:
            try:
                async for event in self._lines(command, stdin, timeout, cwd, log, context, job):
                    yield event
            finally:
                log.close()

        def collect(event: Any) -> bool:
            """Keep the output tail; return whether ``event`` is progress to deliver."""
            if isinstance(event, StreamDone):
                done[0] = event
                return False
            target = out if event.kind == "stdout" else err
            target.append(event.text)
            if len(target) > 4096:  # keep memory bounded; the tail is enough
                target[:] = [_tail("".join(target))]
            return True

        def finish(error: str | None) -> JobResult:
            if error is not None:
                return JobResult(label=job.label, log_path=job.log_path, error=error)
            result = ShellResult(
                stdout=_tail("".join(out)),
                stderr=_tail("".join(err)),
                returncode=done[0].returncode,
                timed_out=done[0].timed_out,
            )
            return JobResult(label=job.label, result=result, log_path=job.log_path)

        async for item in self._stream_job(job, source(), collect=collect, finish=finish):
            yield item

    @hidden
    async def _call_job(self, job: _Job, work: Awaitable[Any]) -> JobResult:
        try:
            item = JobResult(label=job.label, result=await work)
        except Exception as exc:
            item = JobResult(label=job.label, error=f"{type(exc).__name__}: {exc}")
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
        collect: Callable[[Any], bool] | None = None,
        finish: Callable[[str | None], JobResult] | None = None,
    ) -> AsyncIterator[JobUpdate | JobResult]:
        """Pump ``source`` and yield ``JobUpdate`` batches per the job's live throttle.

        Plain streams deliver every value (batched when throttled) and end with
        a ``final`` update. With ``finish`` (``run``), values pass ``collect``
        first, are delivered only while throttled, and the job ends with the
        ``JobResult`` that ``finish`` builds, after flushing the batch.
        """
        inbox: asyncio.Queue[Any] = asyncio.Queue()
        job.wake = inbox
        failure: list[str] = []

        async def pump() -> None:
            try:
                async for value in source:
                    inbox.put_nowait(value)
            except Exception as exc:
                failure.append(f"{type(exc).__name__}: {exc}")
            finally:
                inbox.put_nowait(_END)

        reader = asyncio.create_task(pump())
        batch: list[Any] = []
        batch_started: float | None = None

        def update(final: bool = False) -> JobUpdate:
            nonlocal batch, batch_started
            item = JobUpdate(
                label=job.label,
                items=batch,
                final=final,
                log_path=job.log_path,
                error=failure[0] if final and failure else None,
            )
            batch, batch_started = [], None
            job.pending = 0
            job.last.append(item)
            return item

        try:
            while True:
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
                    # Throttle changed: deliver what was batched under the old setting.
                    if batch:
                        yield update()
                    continue
                if collect is not None and not collect(value):
                    continue
                throttle = job.throttle
                if finish is not None and throttle is None:
                    continue  # a quiet run: its output goes to the JobResult only
                if not batch:
                    batch_started = time.monotonic()
                batch.append(value)
                job.pending = len(batch)
                if throttle is None or (
                    throttle.lines is not None and len(batch) >= throttle.lines
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
            job.finished_at = _now()
            if finish is None:
                if batch:  # nothing extra when nothing is left; JobInfo.state shows the end
                    yield update(final=True)
            else:
                if batch:
                    yield update()
                result = finish(failure[0] if failure else None)
                job.last.append(result)
                yield result
        finally:
            job.wake = None
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
        lines: int | None = None,
        seconds: float | None = None,
        description: str = "",
        replace: bool = False,
    ) -> JobInfo:
        """Run a shell command in the background, in its own bash process.

        Like ``self.shell.run`` (same ``stdin``/``cwd``; ``cwd`` is relative to
        the shell's directory) but returns at once and has no timeout unless
        given. ``cd``/``export`` in ``cmd`` do not affect ``self.shell``. When
        it ends, one ``JobResult`` arrives in ``notification["jobs"]``:
        ``.result`` is a ``ShellResult`` (output tail; on timeout returncode
        124), ``.log_path`` the full stdout+stderr log. With ``lines``/``seconds``
        it also sends progress while running: ``JobUpdate`` batches of
        ``StreamEvent`` lines (see ``throttle``).

            await self.jobs.run("uv run pytest -q", label="tests")
            return_result(Waiting(explanation="tests running", on=["jobs:tests"]))
        """
        if replace:
            self._stop_now(label)
        throttle = _throttle(lines, seconds)
        self._check_label(label)
        context = await self._shell_context()
        job = self._new_job(
            label, "run", throttle, log=True, streaming=True, target=cmd, description=description
        )
        return self._start(job, self._run_job(job, cmd, stdin, timeout, cwd, context))

    def spawn(
        self,
        work: Any,
        *,
        label: str,
        lines: int | None = None,
        seconds: float | None = None,
        description: str = "",
        replace: bool = False,
    ) -> JobInfo:
        """Run a coroutine or async iterator in the background.

        A coroutine delivers one ``JobResult`` (``.result`` is its return
        value, ``.error`` set if it raised). An async iterator delivers one
        ``JobUpdate`` per yielded value (batched with ``lines``/``seconds``);
        the final batch, if any, has ``final=True``.

            self.jobs.spawn(fetch_report(url), label="report")  # any coroutine
            self.jobs.spawn(watch_queue(), label="watch", seconds=60)  # an async generator
        """
        if replace:
            self._stop_now(label)
        about = {"target": _callable_target(work), "description": description}
        if hasattr(work, "__aiter__"):
            job = self._new_job(
                label, "spawn", _throttle(lines, seconds), log=False, streaming=True, **about
            )
            return self._start(job, self._stream_job(job, work.__aiter__()))
        if not asyncio.iscoroutine(work) and not isinstance(work, Awaitable):
            raise TypeError("spawn() takes a coroutine or an async iterator")
        if lines is not None or seconds is not None:
            raise ValueError("lines/seconds apply to async iterators only")
        job = self._new_job(label, "spawn", None, log=False, streaming=False, **about)
        return self._start(job, self._call_job(job, work))

    def tail(
        self,
        path: str | Path,
        *,
        label: str,
        lines: int | None = None,
        seconds: float | None = None,
        description: str = "",
        replace: bool = False,
    ) -> JobInfo:
        """Deliver lines appended to a file from now on, until cancelled.

        Each ``JobUpdate`` item is one line (without its newline); batch with
        ``lines``/``seconds``. A relative ``path`` is relative to the shell's
        directory.

            self.jobs.tail("logs/server.log", label="server-log", seconds=60)
        """
        if replace:
            self._stop_now(label)
        throttle = _throttle(lines, seconds)
        self._check_label(label)
        resolved = (Path(self._shell_dir()) / Path(path)).resolve()
        handle = open(resolved, "rb")  # noqa: SIM115 - closed by the generator
        handle.seek(0, os.SEEK_END)
        target = str(resolved) if str(resolved) == str(path) else f"{path} ({resolved})"
        job = self._new_job(
            label,
            "tail",
            throttle,
            log=False,
            streaming=True,
            target=target,
            description=description,
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
        """Deliver one ``JobResult`` after ``seconds``: a timer; ``.result`` is ``payload``.

        ``cancel(label)`` stops it before it fires.

            self.jobs.after(600, label="check-ci", payload="check CI on PR #462 and report")
        """
        if replace:
            self._stop_now(label)
        job = self._new_job(
            label,
            "after",
            None,
            log=False,
            streaming=False,
            target=f"after {_seconds(seconds)}",
            description=description,
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
        """Deliver a tick (1, 2, 3, ...) every ``seconds`` until cancelled.

        Ticks are ``JobUpdate`` items; ``throttle(label, ...)`` batches them.
        ``payload`` is kept on the job (``get(label).payload``).

            self.jobs.every(300, label="poll-ci", payload="check CI and report changes")
        """
        if replace:
            self._stop_now(label)
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        job = self._new_job(
            label,
            "every",
            None,
            log=False,
            streaming=True,
            target=f"every {_seconds(seconds)}",
            description=description,
        )
        job.payload = payload
        return self._start(job, self._stream_job(job, _ticks(seconds)))

    def throttle(
        self, label: str, *, lines: int | None = None, seconds: float | None = None
    ) -> JobInfo:
        """Batch a running job's deliveries: one per ``lines`` items or ``seconds``.

        Whichever comes first triggers a delivery; the rest is flushed at the
        end. Each call replaces the previous setting (omitted arguments become
        None), first delivering what was already batched. ``throttle(label)``
        turns batching off: a ``run`` goes quiet until its ``JobResult``, other
        jobs deliver every item. Applies at once, without restarting the job.

            self.jobs.throttle("build", lines=200)
        """
        job = self._jobs.get(label)
        if job is None:
            raise KeyError(f"no job labelled {label!r}")
        if not job.streaming:
            raise ValueError(f"job {label!r} is not a streaming job")
        job.throttle = _throttle(lines, seconds)
        if job.wake is not None:
            job.wake.put_nowait(_WAKE)
        return job.info()

    def list(self) -> list[JobInfo]:
        """All known jobs (running and recent finished), oldest first."""
        jobs = [*self._finished, *self._jobs.values()]
        return [job.info() for job in jobs]

    def get(self, label: str) -> JobInfo | None:
        """The latest job with ``label`` (``.last`` holds its recent items), or None."""
        job = self._jobs.get(label)
        return None if job is None else job.info()

    async def cancel(self, label: str) -> bool:
        """Stop a running job (killing its process); True if it was running."""
        job = self._jobs.get(label)
        if job is None or job.handle is None or job.state != "running":
            return False
        job.final_state = "cancelled"  # first: the done callback then forgets the job
        if job.proc is not None:
            _kill_group(job.proc)
        await job.handle.cancel()
        if job.finished_at is None:
            job.finished_at = _now()
        return True

    def __repr__(self) -> str:
        running = sum(1 for job in self._jobs.values() if job.state == "running")
        return f"<JobsApi({running} running)>"
