# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shell work for the work manager: one persistent bash per ``Shell`` object.

``Shell(cwd)`` is inert configuration; its bash starts on first use. Each
``run(command, timeout=...)`` has its own deadline.

- ``sh = Shell()``; ``await sh.run("cd src")``; ``await sh.run("make")``: the
  same object, the same bash (cwd, exported and unexported variables carry
  over). Another ``Shell()`` is another bash.
- ``job = manager.launch(sh, name="sh")``: the same shell as a job;
  ``await job.send("make")`` runs one command in it, like ``sh.run`` with
  its default timeout; for another, launch the call:
  ``manager.launch(sh.run("make", timeout=60))``.
  Commands run one at a time, in order (FIFO), whichever way they arrive.

Lifecycle: ``await shell.aclose()`` or ``async with Shell() as sh:`` ends
bash and its background processes; a dropped Shell ends with its object; an
owner (``ShellOwner`` in ``shell_owner``, which an agent session sets) closes
the Shells started under it; a launched Shell closes with its job. Inside a
harness's ``preserve_background_services()`` scope, closing ends bash only:
that external harness owns (and must clean up) the background processes.

A command's stdout and stderr are read through pipes into files in the job's
scratch directory, each up to an output limit (past it, bytes are counted
but not kept, and the summary says so); outside a job they are not kept. A
``CommandResult`` carries a bounded summary of each stream and the file path
when the summary was cut.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import itertools
import math
import os
import shlex
import signal
import sys
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Annotated, Any

from nooa import hidden
from nooa.agentdoc import spec
from nooa.tools import BashSession
from nooa_atom.agent.work_manager import describe, job_dir

__all__ = ["CommandResult", "Shell", "ShellOwner", "default_cwd", "shell_owner"]

default_cwd: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nooa_work_shell_cwd", default=None
)
"""Where a ``Shell()`` without ``cwd`` starts (an agent sets its workspace); else the process's."""

# A stream's summary: whole when at most this many bytes, else head and tail.
_SUMMARY_BYTES = 10_000
_HEAD_BYTES = 2_000
_TAIL_BYTES = _SUMMARY_BYTES - _HEAD_BYTES
# Bytes of each stream kept on disk per command, by default.
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024
# Seconds a cancelled persistent command gets to end after SIGTERM, then SIGKILL.
_INTERRUPT_GRACE = 2.0
# Seconds a command may run by default (``Shell.run``'s ``timeout``).
_DEFAULT_TIMEOUT = 1800.0
# Seconds past a command's deadline before the bash session's own timeout fires.
_BACKSTOP_SECONDS = 60.0
# Skips each command once an interrupt set __nooa_stop, except the epilogue's first.
# `!` keeps a failing test from tripping errexit; ${...-} keeps nounset quiet.
_GUARD = "! [[ -n ${__nooa_stop-} && $BASH_COMMAND != __nooa_rc=* ]]"
# The SIGUSR1 trap: note the user's options, DEBUG trap and extdebug; then skip the rest.
_ON_INTERRUPT = (
    "__nooa_e=$-; set +e; __nooa_stop=1; __nooa_ud=$(trap -p DEBUG); "
    "__nooa_xd=$(shopt -p extdebug); shopt -s extdebug; trap " + shlex.quote(_GUARD) + " DEBUG"
)
# Runs after the command under the user's options: the first line turns them off
# (its trace to /dev/null); then it restores the user's traps and reports
# "status options user-usr1-trap" and returns the command's status.
_EPILOGUE = """\
{ __nooa_rc=$? __nooa_f=${__nooa_e-$-} __nooa_was=${__nooa_stop-} __nooa_stop=; set +euxv; } 2>/dev/null
__nooa_p=; if [[ -o pipefail ]]; then __nooa_p=p; set +o pipefail; fi
if [[ -n $__nooa_was ]]; then eval "${__nooa_ud:-trap - DEBUG}"; eval "$__nooa_xd"; fi
case "$(trap -p USR1)" in *__nooa_stop=1*) trap - USR1; __nooa_t=0 ;; '') __nooa_t=0 ;; *) __nooa_t=1 ;; esac
printf '%s %s %s\\n' "$__nooa_rc" "-$__nooa_f$__nooa_p" "$__nooa_t" >STATUSFILE
eval "unset __nooa_rc __nooa_f __nooa_was __nooa_stop __nooa_p __nooa_t __nooa_e __nooa_ud __nooa_xd; (exit $__nooa_rc)"
"""
# Seconds to wait for output still in the pipes.
_DRAIN_SECONDS = 0.5
# How a command ends when its Shell is closed (``aclose``) before or while it runs.
_CLOSED_BEFORE = "the shell was closed before this command started; it did not run"
_CLOSED_BEFORE_ENTRY = "the shell was closed before it was entered; no bash started"
_CLOSED_STARTING = "the shell was closed while bash was starting; the command did not run"
_CLOSED_DURING = "the shell was closed while this command ran; the command was stopped"


@dataclass
class CommandResult:
    """One command's outcome: exit code, output, and the shell's directory after it.

    ``stdout``/``stderr`` are whole up to 10 kB, else head and tail around a
    marker. ``stdout_path``/``stderr_path`` name longer managed captures, kept
    while the job is listed, capped per stream by ``max_output`` (default 64 MiB).
    The marker reports cap overflow. Unmanaged calls keep no files. ``cwd`` is
    bash's directory after the command (a ``cd`` lasts).
    """

    returncode: int
    timed_out: bool
    # Shown cut to 2,000 characters in notifications and pprint (display only):
    # the full summary stays on the object; *_path files retain up to max_output.
    stdout: Annotated[str, spec(max_string=2000)]
    stderr: Annotated[str, spec(max_string=2000)]
    cwd: str
    stdout_path: str | None = None
    stderr_path: str | None = None


class _Capture:
    """One stream into a file, up to ``limit`` bytes; the last bytes also in memory."""

    def __init__(self, path: str | None, limit: int) -> None:
        self.path = path
        self.limit = limit
        self.file: IO[bytes] = open(path, "w+b") if path else tempfile.TemporaryFile()  # noqa: SIM115
        self.written = 0
        self.total = 0
        self.tail = bytearray()

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self.limit - self.written
        if room > 0:
            part = chunk[:room]
            self.file.write(part)
            self.written += len(part)
        self.tail += chunk
        del self.tail[:-_TAIL_BYTES]

    async def pump(self, stream: asyncio.StreamReader) -> None:
        with contextlib.suppress(Exception):
            while chunk := await stream.read(_CHUNK_BYTES):
                self.feed(chunk)

    def summary(self) -> tuple[str, str | None]:
        """The stream's text, or its head and tail around a marker; and the path if cut."""
        self.file.flush()
        self.file.seek(0)
        if self.total <= _SUMMARY_BYTES:
            return self.file.read(self.total).decode("utf-8", errors="replace"), None
        head = self.file.read(_HEAD_BYTES).decode("utf-8", errors="replace")
        tail = bytes(self.tail[-_TAIL_BYTES:]).decode("utf-8", errors="replace")
        omitted = self.total - _HEAD_BYTES - len(self.tail[-_TAIL_BYTES:])
        if self.total > self.written:
            where = (
                f"over the {self.limit:,}-byte output limit: {self.path} holds only the "
                f"first {self.written:,} of {self.total:,} bytes"
                if self.path
                else f"over the output limit; output not kept ({self.total:,} bytes)"
            )
        else:
            where = f"full output: {self.path}" if self.path else "full output not kept"
        return f"{head}\n[... {omitted:,} bytes omitted; {where}]\n{tail}", self.path

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.file.close()


def _captures(directory: Path | None, stem: str, limit: int) -> tuple[_Capture, _Capture]:
    if directory is None:
        return _Capture(None, limit), _Capture(None, limit)
    return (
        _Capture(str(directory / f"{stem}.stdout"), limit),
        _Capture(str(directory / f"{stem}.stderr"), limit),
    )


def _result(code: int, timed_out: bool, out: _Capture, err: _Capture, cwd: str) -> CommandResult:
    stdout, stdout_path = out.summary()
    stderr, stderr_path = err.summary()
    return CommandResult(code, timed_out, stdout, stderr, cwd, stdout_path, stderr_path)


def _start_time(pid: int) -> str | None:
    """A process's start time (Linux), so a reused pid is never mistaken for it."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _descendants(pid: int) -> dict[int, str]:
    """``pid``'s descendant processes (pid -> start time), from /proc; {} when unavailable."""
    if sys.platform != "linux":
        return {}
    children: dict[int, list[tuple[int, str]]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append((int(entry.name), fields[19]))
        except (OSError, IndexError, ValueError):
            continue
    found: dict[int, str] = {}
    pending = list(children.get(pid, []))
    while pending:
        child, started = pending.pop()
        if child not in found:
            found[child] = started
            pending.extend(children.get(child, []))
    return found


def _signal(targets: dict[int, str], sig: int) -> None:
    for pid, started in targets.items():
        if _start_time(pid) != started:
            continue  # gone, or the pid was reused
        with contextlib.suppress(OSError):
            os.kill(pid, sig)


def _close_fds(fds: set[int]) -> None:
    while fds:
        with contextlib.suppress(OSError):
            os.close(fds.pop())


async def _open_fifo(path: Path) -> tuple[asyncio.StreamReader, asyncio.BaseTransport, int]:
    """A FIFO read through asyncio; our own write end keeps it from ending before bash opens it."""
    os.mkfifo(path, 0o600)
    read_fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    write_fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    reader = asyncio.StreamReader(limit=2**20)
    loop = asyncio.get_running_loop()
    transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(read_fd, "rb", 0)
    )
    return reader, transport, write_fd


def _cancelling() -> bool:
    """Whether the current task itself is being cancelled."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _closed_error(message: str) -> BaseException:
    """The error for work its Shell's close ended: the caller's own cancel wins."""
    return asyncio.CancelledError() if _cancelling() else RuntimeError(message)


def _check_timeout(value: Any) -> float:
    """``value`` as a command deadline: a non-bool int or float, finite and positive."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"timeout must be seconds (an int or float), got {value!r}")
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"timeout must be a finite, positive number of seconds, got {value!r}")
    return float(value)


class ShellOwner:
    """Shells started under one owner, closed together: an agent session closes its own.

    A ``Shell`` that starts bash while ``shell_owner`` names an owner joins
    it (weakly: a dropped Shell ends with its object). ``await owner.aclose()``
    ends every one still open; a closed Shell starts a new bash if used again.
    """

    __nosnapshot__ = True

    def __init__(self) -> None:
        self._shells: weakref.WeakSet[Shell] = weakref.WeakSet()
        self._closed = False

    def adopt(self, shell: Shell) -> None:
        if self._closed:  # closing is terminal: no bash starts under a closed owner
            raise RuntimeError("the session that owns this shell has closed; no new bash starts")
        self._shells.add(shell)

    def __len__(self) -> int:
        return sum(1 for shell in list(self._shells) if shell._session is not None)

    async def aclose(self) -> None:
        """Close every open Shell this owner adopted (also one still starting); final."""
        self._closed = True
        shells = list(self._shells)
        self._shells.clear()
        results = await asyncio.gather(*(s.aclose() for s in shells), return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]


shell_owner: contextvars.ContextVar[ShellOwner | None] = contextvars.ContextVar(
    "nooa_work_shell_owner", default=None
)
"""Who closes the Shells started here (an agent session sets its own); None: only their users."""


class Shell:
    """Persistent bash. Common managed use::

        sh = Shell()
        r = await self.jobs.launch(sh.run("pwd"), name="pwd")
        print(r.stdout)

    Reuse ``sh``: FIFO commands preserve cwd, variables, options and traps.
    A new Shell starts fresh in the workspace; bash starts on first use.

    Timeout returns 124 with ``timed_out=True``. Timeout/cancel skips remaining
    commands, usually retaining state. Unstoppable builtins or a USR1 trap may
    force a restart. ``set -e`` or ``exit N`` can end bash; the next command starts
    fresh. Other options persist; results report lost state.

    Shell output is completion-only: no output items or early notifications.
    Await or iterate a command job, or use Waiting; its final ``CommandResult``
    holds bounded summaries and managed capture paths, subject to output limits
    and job retention.

    Close with ``await sh.aclose()``, ``async with Shell()``, dropping it or
    cancelling its resource Job; session shutdown also closes it. Closing ends
    bash/background processes, except under ``preserve_background_services()``:
    only bash ends and the harness owns cleanup. Keep ``sh`` while a background
    server must run. A closed Shell restarts on its next use.
    """

    __nosnapshot__ = True  # a live bash is not data

    cwd: str | Path | None  # Where it starts; current_dir is where the shell is now

    def __init__(
        self, cwd: str | Path | None = None, *, max_output: int = _MAX_OUTPUT_BYTES
    ) -> None:
        self.cwd = cwd
        self._max_output = max_output
        self._session: BashSession | None = None
        self._starting = False
        self._entered = False  # launched (or ``async with``): one owner at a time
        self._lock = asyncio.Lock()  # FIFO: commands run one at a time, in order
        # Bumped by aclose: work from before it (queued, running, starting) is void.
        self._generation = 0
        self._start_job: asyncio.Task[int] | None = None  # bash starting; aclose cancels it
        self._closing: asyncio.Task[None] | None = None  # the cleanup every closer awaits
        self._bash: tuple[int, str | None] | None = None  # pid and start time of bash
        self._own_dir: tempfile.TemporaryDirectory[str] | None = None
        self._lingering: set[asyncio.Task[None]] = set()
        self._count = itertools.count(1)
        self._note = ""
        self._pidfile: Path | None = None
        # Shell state as last confirmed by bash itself (never guessed from config).
        self._confirmed_cwd: str | None = None
        self._busy: str | None = None  # the command running now
        self._lost = False  # restarted since the last command: state was lost
        # The user's shell options (of "euxvp", p: pipefail) and whether a user
        # USR1 trap is set, as bash last reported them (see ``_wrap``).
        self._options = ""
        self._user_usr1 = False

    def __repr__(self) -> str:
        return f"Shell(cwd={self.cwd!r}; {self.summary()})"

    @property
    def current_dir(self) -> str | None:
        """The shell's directory as bash last reported it (None before it started)."""
        return self._confirmed_cwd

    @hidden
    def summary(self) -> str:
        """The shell's state in one line, without asking bash (it may be busy)."""
        if self._session is None:
            if self._starting:
                return "starting"
            return "not started" if self._confirmed_cwd is None else "closed"
        lost = " · restarted, variables lost" if self._lost else ""
        if self._busy is not None:
            return f"running $ {self._busy} · was in {self._confirmed_cwd}{lost}"
        return f"idle · in {self._confirmed_cwd}{lost}"

    def _directory(self) -> str:
        if self.cwd is not None:
            return str(Path(self.cwd).resolve())
        return default_cwd.get() or os.getcwd()

    async def run(
        self, command: str, *, stdin: str | None = None, timeout: float = _DEFAULT_TIMEOUT
    ) -> CommandResult:
        """Run one command in this object's bash (started on first use); state carries over.

        ``stdin`` is the command's input (else it reads nothing); ``timeout``
        is its deadline in seconds (finite, positive) from when it starts.
        """
        timeout = _check_timeout(timeout)
        describe(f"$ {command}")
        await self._closed()
        generation = self._generation
        async with self._lock:
            if self._generation != generation:
                raise RuntimeError(_CLOSED_BEFORE)
            if self._session is None:
                await self._start(generation)
            self._busy = command
            try:
                return await self._send_locked(command, stdin, timeout, generation)
            finally:
                self._busy = None
                session = self._session
                if session is not None:
                    self._confirmed_cwd = str(session.cwd)  # bash reports it after each command

    async def send(self, command: str) -> CommandResult:
        """A launched Shell's request (``job.send(command)``): ``run`` with its default timeout."""
        return await self.run(command)

    async def _start(self, generation: int) -> None:
        """Start bash; the caller holds the lock."""
        owner = shell_owner.get()
        if owner is not None:  # before bash exists: the owner's close also ends this start
            owner.adopt(self)
        self._starting = True
        try:
            session = BashSession(cwd=self._directory())
            pid = await self._start_session(session, generation, _CLOSED_STARTING)
        finally:
            self._starting = False
        self._session, self._note = session, ""
        self._bash = (pid, _start_time(pid))
        self._confirmed_cwd, self._busy, self._lost = str(session.cwd), None, False
        self._reset_tracked()

    async def _start_session(self, session: BashSession, generation: int, closed: str) -> int:
        """Start ``session`` and return bash's pid, in a task ``aclose`` cancels.

        If the Shell was closed meanwhile, nothing started stays: the session is
        closed and this raises ``RuntimeError(closed)`` (``CancelledError`` if the
        caller itself was cancelled).
        """
        if self._generation != generation:  # closed already: start nothing
            raise _closed_error(closed)
        job = self._start_job = asyncio.ensure_future(self._boot(session))
        try:
            try:
                pid = await job
            except asyncio.CancelledError:
                if self._generation == generation or _cancelling():
                    raise
                raise RuntimeError(closed) from None
            if self._generation != generation:  # closed just after bash came up
                raise _closed_error(closed)
            return pid
        finally:
            if self._start_job is job:
                self._start_job = None
            if self._generation != generation:
                await asyncio.shield(session.close())

    async def _boot(self, session: BashSession) -> int:
        try:
            await session.start()
            return await self._bash_pid(session)
        except BaseException:
            await asyncio.shield(session.close())
            raise

    # ---- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> Shell:
        """Start bash now (``async with Shell() as sh``, or a launch); one entry at a time."""
        if self._entered:
            raise RuntimeError("this Shell is already running; launch another Shell()")
        self._entered = True  # claimed before the first await: a second entry is refused
        try:
            await self._closed()
            generation = self._generation
            async with self._lock:
                if self._generation != generation:
                    raise RuntimeError(_CLOSED_BEFORE_ENTRY)
                if self._session is None:
                    await self._start(generation)
        except BaseException:
            self._entered = False
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._entered = False
        await self.aclose()

    async def aclose(self) -> None:
        """End bash, its background processes (see the class doc) and scratch files.

        Safe to repeat; concurrent calls wait for one cleanup. A later ``run``
        starts a new bash.
        """
        # Never waits for the lock: the running command is stopped by closing its
        # bash, queued commands are refused, and a start (first use or an
        # interrupt's restart) is cancelled, so no bash comes back for them. One
        # cleanup task per close; every closer (user, owner, job) awaits it,
        # shielded, and new commands wait for it before they start.
        closing = self._closing
        if closing is None or closing.done():
            self._generation += 1
            start, self._start_job = self._start_job, None
            if start is not None:
                start.cancel()
            session, self._session = self._session, None
            pumps, self._lingering = self._lingering, set()
            own_dir, self._own_dir = self._own_dir, None
            closing = self._closing = asyncio.ensure_future(
                self._cleanup(start, session, pumps, own_dir)
            )
        await asyncio.shield(closing)

    @staticmethod
    async def _cleanup(
        start: asyncio.Task[int] | None,
        session: BashSession | None,
        pumps: set[asyncio.Task[None]],
        own_dir: tempfile.TemporaryDirectory[str] | None,
    ) -> None:
        for pump in pumps:
            pump.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)
        if start is not None:
            await asyncio.gather(start, return_exceptions=True)  # it closes what it started
        if session is not None:
            await session.close()
        if own_dir is not None:
            own_dir.cleanup()

    async def _closed(self) -> None:
        """Wait for a close under way to finish (its outcome is the closers')."""
        while (closing := self._closing) is not None and not closing.done():
            await asyncio.wait({closing})

    @staticmethod
    async def _bash_pid(session: BashSession) -> int:
        out, _err, code = await session.run("echo $$", timeout=10.0)
        if code != 0 or not out.strip().isdigit():
            raise RuntimeError(f"could not start bash (exit {code})")
        return int(out.strip())

    def _bash_alive(self) -> bool:
        return self._bash is not None and _start_time(self._bash[0]) == self._bash[1]

    async def _send_locked(
        self, command: str, stdin: str | None, timeout: float, generation: int
    ) -> CommandResult:
        session = self._session
        assert session is not None
        # A preceding queued command may have changed cwd. Report it only once
        # this command owns the shell, not when its coroutine was submitted.
        describe(f"$ {command} · cwd {session.cwd}")
        directory = kept = job_dir()  # output files are kept only in a job's directory
        if directory is None:
            if self._own_dir is None:
                self._own_dir = tempfile.TemporaryDirectory(prefix="nooa-shell-")
            directory = Path(self._own_dir.name)
        stem = f"cmd-{next(self._count)}"
        out, err = _captures(kept, stem, self._max_output)
        fifos = [directory / f"{stem}.out.fifo", directory / f"{stem}.err.fifo"]
        source = "/dev/null"
        if stdin is not None:  # a file, written before the command starts
            stdin_path = directory / f"{stem}.stdin"
            fd = os.open(stdin_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(stdin.encode())
            source = str(stdin_path)
            fifos.append(stdin_path)  # removed with the FIFOs
        pidfile = self._pidfile = directory / f"{stem}.pid"
        opened: list[tuple[asyncio.StreamReader, asyncio.BaseTransport, int]] = []
        write_fds: set[int] = set()
        pumps: list[tuple[asyncio.Task[None], asyncio.BaseTransport, _Capture]] = []
        try:
            for fifo in fifos[:2]:
                opened.append(await _open_fifo(fifo))
                write_fds.add(opened[-1][2])
            for capture, (reader, transport, _) in zip((out, err), opened, strict=True):
                pumps.append((asyncio.create_task(capture.pump(reader)), transport, capture))
            statusfile = directory / f"{stem}.status"
            fifos.append(statusfile)  # removed with the FIFOs
            wrapped = self._wrap(command, pidfile, statusfile, source, fifos[0], fifos[1])
            # Earlier background jobs of this bash: an interrupt never signals them.
            before = _descendants(self._bash[0]) if self._bash and self._bash_alive() else {}
            # Our own deadline (the session's is a backstop): a timeout stops the
            # whole command the same way a cancel does.
            inner = asyncio.ensure_future(
                session.run_with_timeout_flag(wrapped, timeout + _BACKSTOP_SECONDS)
            )
            restart_notes: list[str] | None = None
            try:
                try:
                    stray_out, stray_err, code, timed_out = await asyncio.wait_for(
                        asyncio.shield(inner), timeout
                    )
                except TimeoutError:
                    restarted = await self._interrupt(inner, before, session, generation)
                    self._check_open(generation)
                    if restarted:
                        stray_out = stray_err = ""
                        restart_notes, self._note = [self._note], ""
                    else:
                        stray_out, stray_err, _, _ = inner.result()
                    code, timed_out = 124, True
            except asyncio.CancelledError:
                if await self._interrupt(inner, before, session, generation):
                    raise asyncio.CancelledError(self._note) from None
                raise
            except Exception:
                self._check_open(generation)  # closed: say so, not how the close broke it
                raise
            finally:
                _close_fds(write_fds)  # bash's ends close with the group: then EOF
            self._check_open(generation)
            await self._settle([pump for pump, _, _ in pumps])
            self._read_status(statusfile)
            ended = False
            if code == -1:  # no status from bash: did it exit (`exit 7`, errexit)?
                status = await self._exit_status(session)
                if status is not None:
                    ended, code = True, status
            if restart_notes is None:
                notes = self._check_bash(pidfile, code, timed_out, ended)
            else:  # restarted at the deadline: this result says so
                notes = restart_notes
            result = _result(code, timed_out, out, err, str(session.cwd))
        finally:
            _close_fds(write_fds)
            for pump, transport, capture in pumps:
                if pump.done():
                    transport.close()
                    capture.close()
                else:  # a background job still writes: bounded by the output limit
                    self._lingering.add(pump)
                    pump.add_done_callback(
                        lambda _t, tr=transport, cap=capture: (tr.close(), cap.close())
                    )
                    pump.add_done_callback(self._lingering.discard)
            for _, transport, _ in opened[len(pumps) :]:
                transport.close()
            for capture in (out, err)[len(pumps) :]:
                capture.close()
            for fifo in fifos:
                with contextlib.suppress(OSError):
                    fifo.unlink()  # an open reader keeps working; the name is not needed
            self._pidfile = None
        extra = "\n".join(text for text in (stray_out, stray_err, *notes) if text)
        if extra:
            result.stderr = f"{result.stderr}\n{extra}" if result.stderr else extra
        return result

    # ---- the command wrapper ---------------------------------------------------
    #
    # Each command runs in bash as: prologue, the command in a { } group (not a
    # subshell, so its state stays), epilogue. Between commands the user's
    # errexit/nounset/xtrace/verbose/pipefail are off, so neither the wrapper
    # nor BashSession's protocol can trip them; the epilogue reports them (and
    # the exit status) in a status file and the next prologue sets them again,
    # only in the same bash. No variable, trap or option of ours is left set.
    #
    # Interrupting: while a command runs, bash has a SIGUSR1 trap (unless the
    # user has one). An interrupt sends SIGUSR1, then stops the running process;
    # the trap sets a DEBUG trap with extdebug (a failing DEBUG trap skips the
    # command) so the rest of the command is skipped. The epilogue puts the
    # user's DEBUG trap and extdebug setting back. With a user USR1 trap there
    # is no safe way to skip the rest, so an interrupt restarts bash (reported).
    # Under the user's `set -e` bash exits when the stopped process fails; that,
    # too, is reported as a lost shell.

    def _wrap(
        self, command: str, pidfile: Path, statusfile: Path, source: str, out: Path, err: Path
    ) -> str:
        # The prologue and the command are one parse unit: a command that does
        # not parse (an open quote or here-document) runs nothing of either.
        # The command is spliced in as text, not eval'd: under `set -e` a final
        # `[ -f x ] && . x` must not end bash as a failed `eval` would.
        # (BashSession sends all of this base64-encoded, so stdin cannot cut it.)
        lines = [f"{{ echo $$ >{shlex.quote(str(pidfile))}"]
        if not self._user_usr1:
            lines.append(f"trap {shlex.quote(_ON_INTERRUPT)} USR1")
        restore = self._options_restore()
        if restore and self._bash is not None:
            lines.append(f'if [ "$$" = {self._bash[0]} ]; then {restore}; fi')
        lines += [
            f"{{ {command}\n}}"
            f" <{shlex.quote(source)} >{shlex.quote(str(out))} 2>{shlex.quote(str(err))}\n}}",
            _EPILOGUE.replace("STATUSFILE", shlex.quote(str(statusfile))),
        ]
        return "\n".join(lines)

    def _options_restore(self) -> str:
        flags = "".join(f for f in "euxv" if f in self._options)
        parts = [f"set -{flags}"] if flags else []
        if "p" in self._options:
            parts.append("set -o pipefail")
        return "; ".join(parts)

    def _read_status(self, statusfile: Path) -> None:
        """Take the options and USR1-trap state the epilogue reported (none: bash ended first)."""
        try:
            _rc, flags, usr1 = statusfile.read_text().split()
        except (OSError, ValueError):
            return
        self._options = "".join(f for f in "euxvp" if f in flags)
        self._user_usr1 = usr1 == "1"

    def _reset_tracked(self) -> None:
        self._options, self._user_usr1 = "", False

    @staticmethod
    async def _exit_status(session: BashSession) -> int | None:
        """The status bash exited with (``exit 7``, or errexit); None if it has not exited."""
        proc = getattr(session, "_process", None)
        if proc is None:
            return None
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(proc.wait()), 1.0)
        returncode = proc.returncode
        if returncode is None:
            return None
        return 128 - returncode if returncode < 0 else returncode

    async def _settle(self, pumps: list[asyncio.Task[None]]) -> None:
        """Wait briefly for the output still in the pipes."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*pumps, return_exceptions=True)), _DRAIN_SECONDS
            )

    def _check_bash(self, pidfile: Path, code: int, timed_out: bool, ended: bool) -> list[str]:
        """Notice a bash that was replaced before, or died during, this command."""
        notes = []
        if self._note:
            notes.append(self._note)
            self._note = ""
        self._lost = False
        try:
            ran_in = int(pidfile.read_text().strip())
        except (OSError, ValueError):
            ran_in = None
        if ran_in is not None and self._bash is None:
            self._bash = (ran_in, _start_time(ran_in))  # the new bash after a reported loss
        elif ran_in is not None and ran_in != self._bash[0]:
            self._lost = True
            notes.append(
                "[the shell had restarted before this command: cwd kept, "
                "environment and shell variables lost]"
            )
            self._bash = (ran_in, _start_time(ran_in))
        if ended or (timed_out and not self._bash_alive()):
            self._lost = True
            self._bash = None
            self._reset_tracked()
            status = f" (exit {code})" if ended else ""
            notes.append(
                f"[bash ended during this command{status}; the next command starts a new "
                "shell: cwd kept, environment, shell variables, options and traps lost]"
            )
        return notes

    def _check_open(self, generation: int) -> None:
        if self._generation != generation:
            raise _closed_error(_CLOSED_DURING)

    async def _interrupt(
        self,
        inner: asyncio.Future[Any],
        before: dict[int, str],
        session: BashSession,
        generation: int,
    ) -> bool | None:
        """Stop the running command; restart bash (state lost, True) only if it will not stop.

        None: the Shell was closed meanwhile, so there is nothing to restart.
        """
        bash = self._running_bash()
        if self._user_usr1:
            bash = None  # no trap of ours to skip the rest with: restart, said so
        elif bash is not None:
            _signal({bash: _start_time(bash) or ""}, signal.SIGUSR1)  # skip the rest of it
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if bash is None:
                break
            targets = {pid: s for pid, s in _descendants(bash).items() if pid not in before}
            if not targets:
                break  # a shell builtin is running (or /proc is unavailable)
            _signal(targets, sig)
            with contextlib.suppress(TimeoutError, Exception):
                await asyncio.wait_for(asyncio.shield(inner), _INTERRUPT_GRACE)
            if inner.done():
                return False
        inner.cancel()
        await asyncio.gather(inner, return_exceptions=True)
        if self._generation != generation:
            return None  # closed: aclose ended bash; no restart
        await session.close()
        if self._generation != generation:  # closed while that close ran
            return None
        pid = await self._start_session(session, generation, _CLOSED_DURING)  # keeps the cwd
        self._bash, self._lost = (pid, _start_time(pid)), True
        self._reset_tracked()
        self._note = (
            "[the shell restarted: an interrupted command did not stop; "
            "cwd kept, environment and shell variables lost]"
        )
        return True

    def _running_bash(self) -> int | None:
        """The pid of the bash running the current command, checked against pid reuse."""
        pidfile = self._pidfile
        try:
            pid = int(pidfile.read_text().strip()) if pidfile is not None else None
        except (OSError, ValueError):
            pid = None
        if pid is not None and self._bash is not None and pid == self._bash[0]:
            return pid if self._bash_alive() else None
        if pid is not None and _start_time(pid) is not None:
            return pid  # bash was replaced just before this command; it wrote its own pid
        return None
