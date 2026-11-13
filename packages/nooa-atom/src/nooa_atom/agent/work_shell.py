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

A command's stdout and stderr are each read through a pipe into a bounded
``nooa.snapshot.TextStream`` (bytes): the first 256 KiB and the last 768 KiB
stay in memory, and the whole stream also goes to an overflow file
(``nooa.overflow``), up to the output limit. A ``CommandResult`` carries each stream in ``render_output``'s notation
(whole up to 1 MiB, else ``HEAD``, one ``[... N bytes omitted; ...]`` line
that names the file, ``TAIL``) and the file path when the text was cut (short
output keeps no file). With the agent's ``TruncationConfig.overflow_files``
off (``ShellOwner(overflow_files=False)``) no file is written: a cut stream
keeps its head and tail, its line says ``full output not kept``, and the
path is None.

Retention of those files: an ``OverflowDir`` keeps at most 20 files (the
newest) and deletes the oldest closed files when it needs room, so a path in
an earlier result may no longer exist (opening it raises
``FileNotFoundError``); the in-memory head and tail of every result are
unaffected. Files are deleted only when their folder closes, by that cap, or
at interpreter exit, never when an object is garbage collected.

- In an agent session (a Shell started under an owner): one ``OverflowDir``
  per agent, ``ShellOwner.overflow_dir()``, shared by every output source of
  that agent: its Shells' direct runs, the commands inside its launched jobs
  and its cell output. The 20-file cap is per agent, whatever produced the
  files; each file holds up to ``min(max_output, 64 MiB)`` bytes. Its files
  outlive a closed or dropped Shell and its jobs, and go when the session
  (the owner) closes.
- Without an agent (plain library use): one private ``OverflowDir`` per Shell
  (files of up to ``max_output`` bytes) for its direct runs and the commands
  it runs inside launched jobs, removed when that Shell closes (``aclose``,
  ``async with`` exit, its job's end for a launched Shell). A Shell dropped
  unclosed keeps them until interpreter exit.
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
from typing import Annotated, Any, Literal

from nooa import hidden
from nooa.agentdoc import spec
from nooa.overflow import OverflowDir, OverflowFile
from nooa.runtime.cell_deadline import deadline_cancel
from nooa.snapshot import Text, TextStream, cut, render_output, render_python
from nooa.tools import BashSession
from nooa_atom.agent.metrics import atom_metrics
from nooa_atom.agent.work_manager import (
    _progress_sink,
    current_name,
    describe,
    job_dir,
)
from nooa_atom.agent.work_progress import _ShellProgress

__all__ = ["CommandResult", "Shell", "ShellOwner", "default_cwd", "shell_owner"]

# How bash ended during a command: its own exit (``exit N``, errexit) or a signal.
_Ended = Literal["exit", "signal"] | None

default_cwd: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nooa_work_shell_cwd", default=None
)
"""Where a ``Shell()`` without ``cwd`` starts (an agent sets its workspace); else the process's."""

# A stream's summary: whole when at most this many bytes, else head and tail.
_SUMMARY_BYTES = 1_048_576  # a stream's text is whole in the result up to 1 MiB
_HEAD_BYTES = _SUMMARY_BYTES // 4
_TAIL_BYTES = _SUMMARY_BYTES - _HEAD_BYTES
# Bytes of each stream kept on disk per command, by default.
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024
# Seconds a cancelled persistent command gets to end after SIGTERM, then SIGKILL.
_INTERRUPT_GRACE = 2.0
# Seconds a command may run by default (``Shell.run``'s ``timeout``).
_DEFAULT_TIMEOUT = 600.0
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


def _deadline_note(command: str) -> str:
    """The cell deadline's note for the Shell command the cell was awaiting."""
    shown = render_python(cut(command, head=80, tail=0))  # whole: its repr
    return (
        f"It stopped the Shell command {shown}; the Shell is still usable. A command's own "
        "timeout= returns exit 124 instead. For long commands, use "
        "self.jobs.launch(sh.run(cmd), name=...) and return Waiting."
    )


_CLOSED_BEFORE_ENTRY = "the shell was closed before it was entered; no bash started"
_CLOSED_STARTING = "the shell was closed while bash was starting; the command did not run"
_CLOSED_DURING = "the shell was closed while this command ran; the command was stopped"


@dataclass
class CommandResult:
    """One command's outcome: exit code, output, and bash's directory after it.

    ``stdout``/``stderr``: whole up to 1 MiB, else head+tail around a
    ``[... N bytes omitted; ...]`` line; ``*_path``: that file (to 64 MiB),
    kept until the session closes (outside one: its Shell) or evicted.
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
    """One stream as a bounded ``TextStream`` (bytes): head and tail in memory, all of it in a file.

    ``file``: the stream's overflow file (from an ``OverflowDir``), or
    ``None`` to keep memory only.
    """

    def __init__(self, file: OverflowFile | None) -> None:
        self.stream = TextStream(unit="bytes", head=_HEAD_BYTES, tail=_TAIL_BYTES, overflow=file)
        self.progress: Any = None  # a managed job's preview stream (work_progress)

    def feed(self, chunk: bytes) -> None:
        with contextlib.suppress(ValueError):  # closed: late output has no reader
            self.stream.write(chunk)

    async def pump(self, stream: asyncio.StreamReader) -> None:
        with contextlib.suppress(Exception):
            while chunk := await stream.read(_CHUNK_BYTES):
                self.feed(chunk)
                if self.progress is not None:
                    self.progress.feed(chunk)
            if self.progress is not None:
                self.progress.feed(b"", final=True)

    def text(self, *, final: bool) -> Text:
        """The stream so far (``final``: closed, its file kept only if cut)."""
        return self.stream.close() if final else self.stream.snapshot()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.stream.close()


def _summary(text: Text) -> tuple[str, str | None]:
    """A stream's text in ``render_output``'s notation, and its file path if it was cut."""
    full = text.full
    return render_output(text), (full.path if full is not None and not text.complete else None)


_OVERFLOW_FILES = 20
"""How many overflow files one folder keeps (the newest)."""


def _new_overflow_dir(per_file: int) -> OverflowDir:
    """A private ``OverflowDir`` of ``_OVERFLOW_FILES`` files, each up to ``per_file`` bytes (the per-capture cap)."""
    return OverflowDir(max_files=_OVERFLOW_FILES, file_bytes=max(per_file, 0))


def _captures(folder: OverflowDir | None, limit: int) -> tuple[_Capture, _Capture]:
    """stdout's and stderr's captures, each with a file in ``folder`` (at most ``limit`` bytes)."""

    def file() -> OverflowFile | None:
        if folder is None or limit <= 0:
            return None
        # A folder shared by Shells with different limits (an agent's): this Shell's own.
        return folder.file(max_bytes=limit)

    return _Capture(file()), _Capture(file())


def _result(
    code: int, timed_out: bool, out: _Capture, err: _Capture, cwd: str, *, final: tuple[bool, bool]
) -> CommandResult:
    stdout, stdout_path = _summary(out.text(final=final[0]))
    stderr, stderr_path = _summary(err.text(final=final[1]))
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


def _timeout_note(timeout: float) -> str:
    """How to go on after a command ran into its deadline."""
    return (
        f"[timed out after {timeout:g} s: pass a larger timeout= for a longer command, or "
        "run it in the background: self.jobs.launch(sh.run(cmd, timeout=...), name=...)]"
    )


# Seconds a command may wait for its turn before its result says what it waited for.
_QUEUE_NOTE_SECONDS = 5.0
_BLOCKER_CHARS = 80


def _blocker(command: str | None, job: str | None) -> str:
    """The command running in a Shell, as a note names it: `cmd` (job name)."""
    flat = " ".join((command or "a command").split())
    text = cut(flat, head=_BLOCKER_CHARS, tail=0)
    shown = f"`{flat}`" if text.complete else render_python(text)
    return shown + (f" (job {job})" if job else "")


def _queued_note(blocker: str, waited: float) -> str:
    """What a command queued behind a running one waited for, and how to avoid it."""
    head = f"waited {waited:.0f} s for its turn behind {blocker} in this Shell"
    return f"[{head}; a Shell runs one command at a time: use another Shell() for other commands]"


def _busy_message(blocker: str) -> str:
    """Why a command awaited in a cell was refused behind a launched job's command."""
    return (
        f"this Shell is busy running {blocker}, which was launched in the background: "
        "a Shell runs one command at a time. Use another Shell() for this command "
        "(e.g. sh2 = Shell()), or wait for the job."
    )


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

    def __init__(self, *, overflow_files: bool = True) -> None:
        self._shells: weakref.WeakSet[Shell] = weakref.WeakSet()
        self._closed = False
        # The agent's ``TruncationConfig.overflow_files``: False writes no file at all.
        self.overflow_files = overflow_files
        self._overflow: OverflowDir | None = None  # the agent's output files (``overflow_dir``)

    def overflow_dir(self) -> OverflowDir | None:
        """The agent's one ``OverflowDir`` (20 files, up to 64 MiB each); None once closed.

        Also None when ``overflow_files`` is False (the agent's
        ``TruncationConfig.overflow_files``): its Shells, jobs and cells
        then keep memory only, and write nothing to disk.

        Shared by every output source of the agent: cell output, its Shells'
        direct runs and the commands of its launched jobs, so the 20-file cap
        is per agent. Its files go when the owner closes (or at interpreter
        exit); the folder keeps the newest, so an older path may no longer
        exist (``FileNotFoundError``).
        """
        if self._closed or not self.overflow_files:
            return None
        if self._overflow is None:
            self._overflow = _new_overflow_dir(_MAX_OUTPUT_BYTES)
        return self._overflow

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
        try:
            results = await asyncio.gather(*(s.aclose() for s in shells), return_exceptions=True)
        finally:  # also when cancelled: the folder closes (a still-running writer may go on)
            overflow, self._overflow = self._overflow, None
            if overflow is not None:  # also the output of Shells dropped unclosed
                overflow.close()
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]


shell_owner: contextvars.ContextVar[ShellOwner | None] = contextvars.ContextVar(
    "nooa_work_shell_owner", default=None
)
"""Who closes the Shells started here (an agent session sets its own); None: only their users."""


class Shell:
    """Persistent bash::

        sh = Shell()
        r = await sh.run("pytest -q", timeout=600)
        print(r.returncode, r.stdout)

    Reuse ``sh``: FIFO commands preserve cwd, variables, options and traps.
    A new Shell starts a fresh ``bash`` in the workspace directory on first use.
    A Shell runs one command at a time: use another Shell for a second concurrent command.

    ``timeout`` defaults to 600 s. Timeout returns 124 with ``timed_out=True``
    and a note in stderr on how to go on. Timeout/cancel skips remaining
    commands, usually retaining state. Unstoppable builtins or a USR1 trap may
    force a restart. ``set -e`` or ``exit N`` can end bash; the next command starts
    fresh. Other options persist; results report lost state.

    Completion-only unless launched with ``progress=Throttle(...)``; the final
    CommandResult supersedes unread ShellProgress items.

    Close: ``await sh.aclose()``, ``async with Shell()``, drop it or cancel its
    job; the session closes it too. Closing ends bash and its background
    processes (under ``preserve_background_services()`` only bash). Keep ``sh``
    while a background server must run. A closed Shell restarts on next use.
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
        self._owner: ShellOwner | None = None  # who adopted it at its last start
        # Without an owner, the capture files of its direct runs and launched
        # commands: one private folder, closed by aclose (or at exit). Under an
        # owner they go in the agent's folder (``overflow_dir``).
        self._overflow: OverflowDir | None = None
        self._lingering: set[asyncio.Task[None]] = set()
        self._count = itertools.count(1)
        self._note = ""
        self._pidfile: Path | None = None
        # Shell state as last confirmed by bash itself (never guessed from config).
        self._confirmed_cwd: str | None = None
        self._busy: str | None = None  # the command running now
        self._busy_job: str | None = None  # the job it runs in (None: a direct run)
        self._queued = 0  # commands waiting for their turn behind it
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
            queued = f" · {self._queued} queued behind it" if self._queued else ""
            return f"running $ {self._busy} · was in {self._confirmed_cwd}{lost}{queued}"
        return f"idle · in {self._confirmed_cwd}{lost}"

    def _directory(self) -> str:
        if self.cwd is not None:
            return str(Path(self.cwd).resolve())
        return default_cwd.get() or os.getcwd()

    async def run(
        self, command: str, *, stdin: str | None = None, timeout: float = _DEFAULT_TIMEOUT
    ) -> CommandResult:
        """Run one command in this object's bash (started on first use); state carries over.

        ``stdin``: its input (else none); ``timeout``: deadline, seconds from its start.
        Awaited in a cell while this Shell runs a background job: fails at once; use another Shell().
        """
        timeout = _check_timeout(timeout)
        describe(f"$ {command}")
        await self._closed()
        generation = self._generation
        note = ""
        if self._lock.locked() and self._busy is not None:
            blocker = _blocker(self._busy, self._busy_job)
            if self._busy_job is not None and current_name() is None:
                # Awaited in a cell behind a background job (often a never-ending tail):
                # waiting would block the turn, so say so at once instead.
                atom_metrics().shell_busy()
                raise RuntimeError(_busy_message(blocker))
            atom_metrics().shell_queued()  # FIFO: wait behind it
            describe(f"$ {command} · queued behind {blocker}")
            loop = asyncio.get_running_loop()
            queued_at = loop.time()
            self._queued += 1
            try:
                await self._lock.acquire()
            finally:
                self._queued -= 1
            waited = loop.time() - queued_at
            if waited >= _QUEUE_NOTE_SECONDS:
                note = _queued_note(blocker, waited)
        else:
            await self._lock.acquire()
        try:
            if self._generation != generation:
                raise RuntimeError(_CLOSED_BEFORE)
            # Busy from first use on: a cell's run behind a job starting bash fails too.
            self._busy, self._busy_job = command, current_name()
            try:
                if self._session is None:
                    await self._start(generation)
                result = await self._send_locked(command, stdin, timeout, generation)
            finally:
                self._busy = self._busy_job = None
                session = self._session
                if session is not None:
                    self._confirmed_cwd = str(session.cwd)  # bash reports it after each command
        finally:
            self._lock.release()
        if note:
            result.stderr = f"{result.stderr}\n{note}" if result.stderr else note
        return result

    async def send(self, command: str) -> CommandResult:
        """A launched Shell's request (``job.send(command)``): ``run`` with its default timeout."""
        return await self.run(command)

    async def _start(self, generation: int) -> None:
        """Start bash; the caller holds the lock."""
        owner = shell_owner.get()
        if owner is not None:  # before bash exists: the owner's close also ends this start
            owner.adopt(self)
        self._owner = owner
        self._starting = True
        try:
            session = BashSession(cwd=self._directory())
            pid = await self._start_session(session, generation, _CLOSED_STARTING)
        finally:
            self._starting = False
        self._session, self._note = session, ""
        self._bash = (pid, _start_time(pid))
        self._confirmed_cwd, self._lost = str(session.cwd), False
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
        """End bash, its background processes (see the class doc) and scratch files; repeatable."""
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
            overflow, self._overflow = self._overflow, None
            closing = self._closing = asyncio.ensure_future(
                self._cleanup(start, session, pumps, own_dir, overflow)
            )
        await asyncio.shield(closing)

    @staticmethod
    async def _cleanup(
        start: asyncio.Task[int] | None,
        session: BashSession | None,
        pumps: set[asyncio.Task[None]],
        own_dir: tempfile.TemporaryDirectory[str] | None,
        overflow: OverflowDir | None = None,
    ) -> None:
        for pump in pumps:
            pump.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)
        if overflow is not None:
            overflow.close()
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

    def _overflow_dir(self) -> OverflowDir | None:
        """The ``OverflowDir`` of a command's captures (20 files, each up to ``max_output`` bytes).

        ``None`` under an owner whose ``overflow_files`` is off: memory only.

        Started under an agent session (an owner): the session's one folder,
        shared with every Shell, job and cell of that agent (files up to
        ``min(max_output, 64 MiB)``), until the session closes. Otherwise this
        Shell's one private folder, for its direct runs and its commands in
        launched jobs alike, until the Shell closes.
        """
        owner = self._owner
        if owner is not None and not owner.overflow_files:
            return None
        if owner is not None and (shared := owner.overflow_dir()) is not None:
            return shared
        if self._overflow is None:
            self._overflow = _new_overflow_dir(self._max_output)
        return self._overflow

    async def _send_locked(
        self, command: str, stdin: str | None, timeout: float, generation: int
    ) -> CommandResult:
        session = self._session
        assert session is not None
        # A preceding queued command may have changed cwd. Report it only once
        # this command owns the shell, not when its coroutine was submitted.
        describe(f"$ {command} · cwd {session.cwd}")
        directory = job_dir()  # the FIFOs and pid file: a job's directory, else the Shell's own
        if directory is None:
            if self._own_dir is None:
                self._own_dir = tempfile.TemporaryDirectory(prefix="nooa-shell-")
            directory = Path(self._own_dir.name)
        stem = f"cmd-{next(self._count)}"
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
        # The captures hand out files: made last before the ``try`` whose
        # ``finally`` closes them, so nothing above can fail with one unclosed.
        out, err = _captures(self._overflow_dir(), self._max_output)
        opened: list[tuple[asyncio.StreamReader, asyncio.BaseTransport, int]] = []
        write_fds: set[int] = set()
        pumps: list[tuple[asyncio.Task[None], asyncio.BaseTransport, _Capture]] = []
        progress = None
        try:
            sink = _progress_sink()
            if sink is not None:  # a managed job: previews follow its Throttle (None: quiet)
                progress = _ShellProgress(sink, sink.policy)
                out.progress, err.progress = progress.stdout, progress.stderr
            for fifo in fifos[:2]:
                opened.append(await _open_fifo(fifo))
                write_fds.add(opened[-1][2])
            for capture, (reader, transport, _) in zip((out, err), opened, strict=True):
                pumps.append((asyncio.create_task(capture.pump(reader)), transport, capture))
            statusfile = directory / f"{stem}.status"
            fifos.append(statusfile)  # removed with the FIFOs
            wrapped = self._wrap(command, pidfile, statusfile, source, fifos[0], fifos[1])
            # A dead bash restarts in its kept cwd (BashSession resets on the next
            # command): if that directory is gone, start where this Shell started.
            cwd_note = "" if self._bash and self._bash_alive() else self._fallback_cwd(session)
            # Earlier background jobs of this bash: an interrupt never signals them.
            before = _descendants(self._bash[0]) if self._bash and self._bash_alive() else {}
            # Our own deadline (the session's is a backstop): a timeout stops the
            # whole command the same way a cancel does.
            inner = asyncio.ensure_future(
                session.run_with_timeout_flag(wrapped, timeout + _BACKSTOP_SECONDS)
            )
            if progress is not None:
                progress.live = lambda: not inner.done() and self._generation == generation
            restart_notes: list[str] | None = None
            try:
                try:
                    stray_out, stray_err, code, timed_out = await asyncio.wait_for(
                        asyncio.shield(inner), timeout
                    )
                except TimeoutError:
                    if progress is not None:
                        progress.close()
                    restarted = await self._interrupt(inner, before, session, generation)
                    self._check_open(generation)
                    if restarted:
                        stray_out = stray_err = ""
                        restart_notes, self._note = [self._note], ""
                    else:
                        stray_out, stray_err, _, _ = inner.result()
                    code, timed_out = 124, True
            except asyncio.CancelledError:
                if progress is not None:
                    progress.close()
                if (deadline := deadline_cancel()) is not None:
                    deadline.note(_deadline_note(command))
                if await self._interrupt(inner, before, session, generation):
                    raise asyncio.CancelledError(self._note) from None
                raise
            except Exception:
                self._check_open(generation)  # closed: say so, not how the close broke it
                raise
            finally:
                if progress is not None:
                    progress.close()  # terminal, not EOF: background writers may keep pipes open
                _close_fds(write_fds)  # bash's ends close with the group: then EOF
            self._check_open(generation)
            await self._settle([pump for pump, _, _ in pumps])
            self._read_status(statusfile)
            ended: _Ended = None
            if code == -1:  # no status from bash: did it exit (`exit 7`, errexit)?
                exit_status = await self._exit_status(session)
                if exit_status is not None:
                    code, ended = exit_status
            if restart_notes is None:
                notes = self._check_bash(pidfile, code, timed_out, ended)
            else:  # restarted at the deadline: this result says so
                notes = restart_notes
            if cwd_note:
                notes.insert(0, cwd_note)
            if timed_out:
                notes.append(_timeout_note(timeout))
            # A stream whose writer is gone is final (closed); one a background
            # job still writes to is read as it is now, and closed when it ends.
            final = (pumps[0][0].done(), pumps[1][0].done())
            result = _result(code, timed_out, out, err, str(session.cwd), final=final)
        finally:
            if progress is not None:
                progress.close()
            _close_fds(write_fds)
            for pump, transport, capture in pumps:
                if pump.done():
                    transport.close()
                    capture.close()
                else:  # a background job still writes: bounded by head, tail and the file cap
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
    async def _exit_status(session: BashSession) -> tuple[int, _Ended] | None:
        """How bash ended: its status and ``exit`` (its own ``exit N`` or errexit) or
        ``signal`` (killed from outside); None if it has not exited."""
        proc = getattr(session, "_process", None)
        if proc is None:
            return None
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(proc.wait()), 1.0)
        returncode = proc.returncode
        if returncode is None:
            return None
        return (128 - returncode, "signal") if returncode < 0 else (returncode, "exit")

    async def _settle(self, pumps: list[asyncio.Task[None]]) -> None:
        """Wait briefly for the output still in the pipes."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*pumps, return_exceptions=True)), _DRAIN_SECONDS
            )

    def _check_bash(self, pidfile: Path, code: int, timed_out: bool, ended: _Ended) -> list[str]:
        """Notice a bash that was replaced before, or died during, this command.

        ``ended``: how bash ended during it (``exit``: the command's own
        ``exit N`` or errexit), else None.
        """
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
            if ended == "exit":
                atom_metrics().shell_exit_hint(code)
                notes.append(
                    f"[this command ended the Shell's bash (exit {code}); the next command "
                    "starts a fresh bash (cwd kept; environment, variables, options and traps "
                    "reset). Use a subshell `( ... )` or a return code instead of exit.]"
                )
                return notes
            status = f" (exit {code})" if ended else ""
            notes.append(
                f"[bash ended during this command{status}; the next command starts a new "
                "shell: cwd kept, environment, shell variables, options and traps lost]"
            )
        return notes

    def _fallback_cwd(self, session: BashSession) -> str:
        """Before bash (re)starts: move a kept cwd that no longer exists; the note, or "".

        The fallback is this Shell's starting directory, else the workspace,
        else the process's cwd (the first that exists); counted in harness metrics.
        """
        kept = str(session.cwd)
        if os.path.isdir(kept):
            return ""
        candidates = (self._directory(), default_cwd.get(), os.getcwd())
        fallback = next((c for c in candidates if c and os.path.isdir(c)), None)
        if fallback is None:
            return ""
        session.cwd = fallback  # where BashSession starts bash next
        self._confirmed_cwd = str(session.cwd)
        atom_metrics().shell_cwd_fallback(f"{kept} -> {session.cwd}")
        return (
            f"[the shell's kept cwd {kept} no longer exists: the new bash started in {session.cwd}]"
        )

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
        cwd_note = self._fallback_cwd(session)
        pid = await self._start_session(session, generation, _CLOSED_DURING)  # keeps the cwd
        self._bash, self._lost = (pid, _start_time(pid)), True
        self._reset_tracked()
        self._note = (
            "[the shell restarted: an interrupted command did not stop; "
            "cwd kept, environment and shell variables lost]"
        )
        if cwd_note:
            self._note += "\n" + cwd_note
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
