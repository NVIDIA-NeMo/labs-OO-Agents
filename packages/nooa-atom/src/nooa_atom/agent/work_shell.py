# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shell work for the work manager: one-shot commands and a persistent session.

``Shell(cwd, timeout)`` is inert configuration; nothing starts until it is used.

- ``await Shell().run("make")``: one command in its own bash process.
- ``sh = manager.launch(Shell(), name="sh")``: a persistent session (cwd,
  exported and unexported variables survive); ``await sh.send("cd src")``
  runs one command in it. Commands run one at a time, in order.

A command's stdout and stderr are read through pipes into files in the job's
scratch directory, each up to an output limit (past it, bytes are counted
but not kept, and the summary says so). A ``CommandResult`` carries a
bounded summary of each stream and the file path when the summary was cut.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import itertools
import os
import shlex
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from nooa.tools._bash_session import BashSession
from nooa_atom.agent.work_manager import describe, job_dir

__all__ = ["CommandResult", "Shell"]

# A stream's summary: whole when at most this many bytes, else head and tail.
_SUMMARY_BYTES = 10_000
_HEAD_BYTES = 2_000
_TAIL_BYTES = _SUMMARY_BYTES - _HEAD_BYTES
# Bytes of each stream kept on disk per command, by default.
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024
# Seconds a cancelled persistent command gets to end after SIGTERM, then SIGKILL.
_INTERRUPT_GRACE = 2.0
# Seconds to wait for a killed process, and for output still in the pipes.
_REAP_SECONDS = 5.0
_DRAIN_SECONDS = 0.5


@dataclass
class CommandResult:
    """One command's outcome: exit code, output summaries, directory after it.

    ``cwd`` is the shell's directory after the command (a one-shot command's
    own ``cd`` does not last). ``*_path`` is set when that summary was cut.
    """

    returncode: int
    timed_out: bool
    stdout: str
    stderr: str
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


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)


async def _reap(proc: asyncio.subprocess.Process) -> None:
    """Wait for a (killed) process, but never forever."""
    with contextlib.suppress(TimeoutError, Exception):
        await asyncio.wait_for(asyncio.shield(proc.wait()), _REAP_SECONDS)


async def _settle_pumps(pumps: list[asyncio.Task[None]]) -> None:
    """Let the pipes drain briefly; a background process still holding them is cut off."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(asyncio.gather(*pumps, return_exceptions=True), _DRAIN_SECONDS)
    for pump in pumps:
        pump.cancel()
    await asyncio.gather(*pumps, return_exceptions=True)


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


class Shell:
    """Shell commands: ``await Shell().run(cmd)`` once, or launch it for a session.

    ``cwd`` defaults to the process's directory when used; ``timeout``
    (seconds) bounds each command. A timed-out command ends with
    ``returncode`` 124 and ``timed_out=True``.
    """

    cwd: str | Path | None  # Where it starts; current_dir is where the session is now
    timeout: float  # Seconds per command

    def __init__(
        self,
        cwd: str | Path | None = None,
        *,
        timeout: float = 1800.0,
        max_output: int = _MAX_OUTPUT_BYTES,
    ) -> None:
        self.cwd = cwd
        self.timeout = timeout
        self._max_output = max_output
        self._session: BashSession | None = None
        self._entering = False
        self._lock: asyncio.Lock | None = None
        self._bash: tuple[int, str | None] | None = None  # pid and start time of bash
        self._own_dir: tempfile.TemporaryDirectory[str] | None = None
        self._lingering: set[asyncio.Task[None]] = set()
        self._count = itertools.count(1)
        self._note = ""
        self._pidfile: Path | None = None
        # Session state as last confirmed by bash itself (never guessed from config).
        self._confirmed_cwd: str | None = None
        self._busy: str | None = None  # the command running now
        self._lost = False  # restarted since the last command: state was lost

    def __repr__(self) -> str:
        return f"Shell(cwd={self.cwd!r}, timeout={self.timeout:g}; {self.summary()})"

    @property
    def current_dir(self) -> str | None:
        """The session's directory as bash last reported it (None before it started)."""
        return self._confirmed_cwd

    def summary(self) -> str:
        """The session's state in one line, without asking bash (it may be busy)."""
        if self._session is None:
            if self._entering:
                return "starting"
            return "not started" if self._confirmed_cwd is None else "closed"
        lost = " · restarted, variables lost" if self._lost else ""
        if self._busy is not None:
            return f"running $ {self._busy} · was in {self._confirmed_cwd}{lost}"
        return f"idle · in {self._confirmed_cwd}{lost}"

    def _directory(self) -> str:
        return str(Path(self.cwd).resolve()) if self.cwd is not None else os.getcwd()

    async def run(self, command: str, *, stdin: str | None = None) -> CommandResult:
        """Run one command in a fresh bash process and return its result."""
        directory = self._directory()
        describe(f"$ {command} · in {directory}")
        out, err = _captures(job_dir(), f"cmd-{next(self._count)}", self._max_output)
        # Our own pipes, not asyncio's: Process.wait() then waits for bash
        # alone, not for a background process that inherited stdout/stderr.
        fds: set[int] = set()
        transports: list[asyncio.BaseTransport] = []
        pumps: list[asyncio.Task[None]] = []
        try:
            readers = []
            write_ends = []
            for _ in range(2):
                read_fd, write_fd = os.pipe()
                fds.update((read_fd, write_fd))
                readers.append(read_fd)
                write_ends.append(write_fd)
            try:
                proc = await asyncio.create_subprocess_exec(
                    "bash",
                    "-c",
                    command,
                    stdin=(
                        asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL
                    ),
                    stdout=write_ends[0],
                    stderr=write_ends[1],
                    cwd=directory,
                    start_new_session=True,
                )
            finally:
                for fd in write_ends:  # bash holds its own copies now
                    fds.discard(fd)
                    os.close(fd)
            loop = asyncio.get_running_loop()
            for capture, read_fd in zip((out, err), readers, strict=True):
                reader = asyncio.StreamReader(limit=2**20)
                transport, _ = await loop.connect_read_pipe(
                    lambda r=reader: asyncio.StreamReaderProtocol(r), os.fdopen(read_fd, "rb", 0)
                )
                fds.discard(read_fd)  # the transport owns it
                transports.append(transport)
                pumps.append(asyncio.create_task(capture.pump(reader)))

            async def feed() -> None:
                assert proc.stdin is not None and stdin is not None
                with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                    proc.stdin.write(stdin.encode())
                    await proc.stdin.drain()
                    proc.stdin.close()

            timed_out = False
            try:
                # One deadline for feeding stdin and for the command itself.
                work = [proc.wait(), *([feed()] if stdin is not None else [])]
                await asyncio.wait_for(asyncio.gather(*work), self.timeout)
            except TimeoutError:
                timed_out = True
                _kill_group(proc)
                await _reap(proc)
            except BaseException:
                _kill_group(proc)
                await _reap(proc)
                await _settle_pumps(pumps)
                raise
            await _settle_pumps(pumps)
            code = 124 if timed_out else (proc.returncode or 0)
            return _result(code, timed_out, out, err, directory)
        finally:
            for pump in pumps:
                pump.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            for transport in transports:  # a background writer now gets EPIPE
                transport.close()
            _close_fds(fds)
            out.close()
            err.close()

    # ---- persistent session ------------------------------------------------

    async def __aenter__(self) -> Shell:
        if self._session is not None or self._entering:
            raise RuntimeError("this Shell is already running; launch another Shell()")
        self._entering = True  # claimed before the first await: a second entry is refused
        session = BashSession(cwd=self._directory())
        try:
            await session.start()
            pid = await self._bash_pid(session)
        except BaseException:
            await asyncio.shield(session.close())
            raise
        finally:
            self._entering = False
        self._session, self._lock, self._note = session, asyncio.Lock(), ""
        self._bash = (pid, _start_time(pid))
        self._confirmed_cwd, self._busy, self._lost = str(session.cwd), None, False
        return self

    async def __aexit__(self, *exc: Any) -> None:
        session, self._session = self._session, None
        for pump in self._lingering:
            pump.cancel()
        await asyncio.gather(*self._lingering, return_exceptions=True)
        self._lingering.clear()
        if session is not None:
            await session.close()
        if self._own_dir is not None:
            self._own_dir.cleanup()
            self._own_dir = None

    @staticmethod
    async def _bash_pid(session: BashSession) -> int:
        out, _err, code = await session.run("echo $$", timeout=10.0)
        if code != 0 or not out.strip().isdigit():
            raise RuntimeError(f"could not start bash (exit {code})")
        return int(out.strip())

    async def send(self, command: str) -> CommandResult:
        """Run one command in the session; state carries over to the next."""
        if self._session is None or self._lock is None:
            raise RuntimeError("the shell session is not running")
        describe(f"$ {command}")
        async with self._lock:
            self._busy = command
            try:
                return await self._send_locked(command)
            finally:
                self._busy = None
                session = self._session
                if session is not None:
                    self._confirmed_cwd = str(session.cwd)  # bash reports it after each command

    def _bash_alive(self) -> bool:
        return self._bash is not None and _start_time(self._bash[0]) == self._bash[1]

    async def _send_locked(self, command: str) -> CommandResult:
        session = self._session
        assert session is not None
        directory = job_dir()
        if directory is None:
            if self._own_dir is None:
                self._own_dir = tempfile.TemporaryDirectory(prefix="nooa-shell-")
            directory = Path(self._own_dir.name)
        stem = f"cmd-{next(self._count)}"
        out, err = _captures(directory, stem, self._max_output)
        fifos = [directory / f"{stem}.out.fifo", directory / f"{stem}.err.fifo"]
        pidfile = self._pidfile = directory / f"{stem}.pid"
        opened: list[tuple[asyncio.StreamReader, asyncio.BaseTransport, int]] = []
        write_fds: set[int] = set()
        pumps: list[tuple[asyncio.Task[None], asyncio.BaseTransport, _Capture]] = []
        try:
            for fifo in fifos:
                opened.append(await _open_fifo(fifo))
                write_fds.add(opened[-1][2])
            for capture, (reader, transport, _) in zip((out, err), opened, strict=True):
                pumps.append((asyncio.create_task(capture.pump(reader)), transport, capture))
            blob = base64.b64encode(command.encode()).decode()
            # The pid first (it identifies the bash that runs it), then a group,
            # not a subshell: cd/export/variables stay in the session.
            wrapped = (
                f"echo $$ >{shlex.quote(str(pidfile))}\n"
                f'{{ eval "$(printf %s {blob} | base64 -d)"\n}}'
                f" >{shlex.quote(str(fifos[0]))} 2>{shlex.quote(str(fifos[1]))}"
            )
            # Earlier background jobs of this bash: a cancel never signals them.
            before = _descendants(self._bash[0]) if self._bash and self._bash_alive() else {}
            inner = asyncio.ensure_future(session.run_with_timeout_flag(wrapped, self.timeout))
            try:
                stray_out, stray_err, code, timed_out = await asyncio.shield(inner)
            except asyncio.CancelledError:
                await self._interrupt(inner, before)
                raise
            finally:
                _close_fds(write_fds)  # bash's ends close with the group: then EOF
            await self._settle([pump for pump, _, _ in pumps])
            notes = self._check_bash(pidfile, code, timed_out)
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

    async def _settle(self, pumps: list[asyncio.Task[None]]) -> None:
        """Wait briefly for the output still in the pipes."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*pumps, return_exceptions=True)), _DRAIN_SECONDS
            )

    def _check_bash(self, pidfile: Path, code: int, timed_out: bool) -> list[str]:
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
        if (code == -1 or timed_out) and not self._bash_alive():
            self._lost = True
            self._bash = None
            notes.append(
                "[bash ended during this command; the next command starts a new shell: "
                "cwd kept, environment and shell variables lost]"
            )
        return notes

    async def _interrupt(self, inner: asyncio.Future[Any], before: dict[int, str]) -> None:
        """Stop the running command; restart bash (state lost) only if it will not stop."""
        bash = self._running_bash()
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
                return
        inner.cancel()
        await asyncio.gather(inner, return_exceptions=True)
        session = self._session
        assert session is not None
        await session.close()
        await session.start()  # keeps the last cwd
        pid = await self._bash_pid(session)
        self._bash, self._lost = (pid, _start_time(pid)), True
        self._note = (
            "[the shell restarted: an interrupted command did not stop; "
            "cwd kept, environment and shell variables lost]"
        )
        raise asyncio.CancelledError(self._note)

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
