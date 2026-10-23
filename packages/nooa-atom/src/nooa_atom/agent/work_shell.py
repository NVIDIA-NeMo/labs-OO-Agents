# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shell work for the work manager: one-shot commands and a persistent session.

``Shell(cwd, timeout)`` is inert configuration; nothing starts until it is used.

- ``await Shell().run("make")``: one command in its own bash process.
- ``sh = manager.launch(Shell(), name="sh")``: a persistent session (cwd,
  exported and unexported variables survive); ``await sh.send("cd src")``
  runs one command in it. Commands run one at a time, in order.

Full stdout and stderr go to files in the job's scratch directory; a
``CommandResult`` carries a bounded summary of each and the file path when
the summary was cut.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
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
# Seconds a cancelled persistent command gets to end after SIGTERM, then SIGKILL.
_INTERRUPT_GRACE = 2.0


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


def _summary(handle: IO[bytes], path: str | None) -> tuple[str, str | None]:
    """The stream's text, or its head and tail around a marker; and the path if cut."""
    size = handle.seek(0, os.SEEK_END)
    handle.seek(0)
    if size <= _SUMMARY_BYTES:
        return handle.read().decode("utf-8", errors="replace"), None
    head = handle.read(_HEAD_BYTES)
    tail_bytes = _SUMMARY_BYTES - _HEAD_BYTES
    handle.seek(size - tail_bytes)
    tail = handle.read(tail_bytes)
    where = f"full output: {path}" if path else "full output not kept"
    marker = f"\n[... {size - _SUMMARY_BYTES} bytes omitted; {where}]\n"
    text = head.decode("utf-8", errors="replace") + marker
    return text + tail.decode("utf-8", errors="replace"), path


def _output_files(directory: Path | None) -> tuple[IO[bytes], IO[bytes], str | None, str | None]:
    """Files for one command's stdout and stderr: in ``directory``, else anonymous."""
    if directory is None:
        return tempfile.TemporaryFile(), tempfile.TemporaryFile(), None, None
    fd_out, out_path = tempfile.mkstemp(prefix="cmd-", suffix=".stdout", dir=directory)
    fd_err, err_path = tempfile.mkstemp(prefix="cmd-", suffix=".stderr", dir=directory)
    return os.fdopen(fd_out, "w+b"), os.fdopen(fd_err, "w+b"), out_path, err_path


def _result(
    code: int, timed_out: bool, out: IO[bytes], err: IO[bytes], paths: tuple, cwd: str
) -> CommandResult:
    stdout, stdout_path = _summary(out, paths[0])
    stderr, stderr_path = _summary(err, paths[1])
    return CommandResult(code, timed_out, stdout, stderr, cwd, stdout_path, stderr_path)


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)


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
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[19] != started:
                continue  # the pid was reused
            os.kill(pid, sig)
        except (OSError, IndexError):
            continue


class Shell:
    """Shell commands: ``await Shell().run(cmd)`` once, or launch it for a session.

    ``cwd`` defaults to the process's directory when used; ``timeout``
    (seconds) bounds each command. A timed-out command ends with
    ``returncode`` 124 and ``timed_out=True``.
    """

    cwd: str | Path | None  # Where it starts; see current_dir for where it is
    timeout: float  # Seconds per command

    def __init__(self, cwd: str | Path | None = None, *, timeout: float = 1800.0) -> None:
        self.cwd = cwd
        self.timeout = timeout
        self._session: BashSession | None = None
        self._lock: asyncio.Lock | None = None
        self._pid: int | None = None
        self._own_dir: tempfile.TemporaryDirectory[str] | None = None
        self._note = ""
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
        out, err, *paths = _output_files(job_dir())
        with out, err:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                cwd=directory,
                start_new_session=True,
            )
            timed_out = False
            try:
                if stdin is not None and proc.stdin is not None:
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        proc.stdin.write(stdin.encode())
                        await proc.stdin.drain()
                    proc.stdin.close()
                try:
                    await asyncio.wait_for(proc.wait(), self.timeout)
                except TimeoutError:
                    timed_out = True
                    _kill_group(proc)
                    await proc.wait()
            except BaseException:
                _kill_group(proc)
                await asyncio.shield(proc.wait())
                raise
            code = 124 if timed_out else (proc.returncode or 0)
            return _result(code, timed_out, out, err, tuple(paths), directory)

    # ---- persistent session ------------------------------------------------

    async def __aenter__(self) -> Shell:
        if self._session is not None:
            raise RuntimeError("this Shell is already running; launch another Shell()")
        session = BashSession(cwd=self._directory())
        try:
            await session.start()
            self._pid = await self._bash_pid(session)
        except BaseException:
            await asyncio.shield(session.close())
            raise
        self._session, self._lock, self._note = session, asyncio.Lock(), ""
        self._confirmed_cwd, self._busy, self._lost = str(session.cwd), None, False
        return self

    async def __aexit__(self, *exc: Any) -> None:
        session, self._session = self._session, None
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

    async def _send_locked(self, command: str) -> CommandResult:
        session = self._session
        assert session is not None and self._pid is not None
        directory = job_dir()
        if directory is None:
            if self._own_dir is None:
                self._own_dir = tempfile.TemporaryDirectory(prefix="nooa-shell-")
            directory = Path(self._own_dir.name)
        out, err, out_path, err_path = _output_files(directory)
        with out, err:
            blob = base64.b64encode(command.encode()).decode()
            # A group, not a subshell: cd/export/variables stay in the session.
            wrapped = (
                f'{{ eval "$(printf %s {blob} | base64 -d)"\n}}'
                f" >{shlex.quote(str(out_path))} 2>{shlex.quote(str(err_path))}"
            )
            before = _descendants(self._pid)  # earlier background jobs: never interrupted
            inner = asyncio.ensure_future(session.run_with_timeout_flag(wrapped, self.timeout))
            try:
                stray_out, stray_err, code, timed_out = await asyncio.shield(inner)
            except asyncio.CancelledError:
                await self._interrupt(inner, before)
                raise
            notes = []
            if self._note:
                notes.append(self._note)
                self._note = ""
            self._lost = False
            if timed_out and (pid := await self._bash_pid(session)) != self._pid:
                self._pid, self._lost = pid, True
                notes.append(
                    "[the shell restarted after the timeout: cwd kept, "
                    "environment and shell variables lost]"
                )
            result = _result(code, timed_out, out, err, (out_path, err_path), str(session.cwd))
        extra = "\n".join(text for text in (stray_out, stray_err, *notes) if text)
        if extra:
            result.stderr = f"{result.stderr}\n{extra}" if result.stderr else extra
        return result

    async def _interrupt(self, inner: asyncio.Future[Any], before: dict[int, str]) -> None:
        """Stop the running command; restart bash (state lost) only if it will not stop."""
        assert self._pid is not None
        for sig in (signal.SIGTERM, signal.SIGKILL):
            targets = {
                pid: started
                for pid, started in _descendants(self._pid).items()
                if pid not in before
            }
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
        self._pid = await self._bash_pid(session)
        self._lost = True
        self._note = (
            "[the shell restarted: an interrupted command did not stop; "
            "cwd kept, environment and shell variables lost]"
        )
        raise asyncio.CancelledError(self._note)
