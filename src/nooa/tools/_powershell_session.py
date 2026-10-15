# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Windows PowerShell sessions with private control and owned process trees.

PowerShell 7 (pwsh) is preferred; Windows PowerShell 5.1 is the fallback. User
commands run in one dot-sourced scope. A timeout, cancellation, or dead host
restarts that scope on the next command, preserving the last confirmed cwd and
rerunning init_command. Completed background trees survive this recovery.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from nooa.agentdoc import TruncatingStringIO

if TYPE_CHECKING:
    from nooa.tools._windows_job import WindowsJob

logger = logging.getLogger(__name__)
_MAX_OUTPUT_CHARS = 30_000


@dataclass(frozen=True)
class _Completion:
    code: int
    timed_out: bool = False


# Duplicated asyncio pipe handles retain their original IOCP association. Read
# them with OVERLAPPED events, not a new proactor: the low event-handle bit keeps
# completion notifications out of the old process's completion port.
_DRAIN_SCRIPT = """
import ctypes, sys
from ctypes import wintypes
from concurrent.futures import ThreadPoolExecutor

class Overlapped(ctypes.Structure):
    _fields_ = [
        ('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
        ('Offset', wintypes.DWORD), ('OffsetHigh', wintypes.DWORD),
        ('hEvent', wintypes.HANDLE),
    ]

kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
kernel.CreateEventW.restype = wintypes.HANDLE
kernel.ResetEvent.argtypes = [wintypes.HANDLE]
kernel.ResetEvent.restype = wintypes.BOOL
kernel.ReadFile.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(Overlapped),
]
kernel.ReadFile.restype = wintypes.BOOL
kernel.GetOverlappedResult.argtypes = [
    wintypes.HANDLE, ctypes.POINTER(Overlapped), ctypes.POINTER(wintypes.DWORD), wintypes.BOOL,
]
kernel.GetOverlappedResult.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL

def drain(handle):
    event = kernel.CreateEventW(None, True, False, None)
    if not event:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        buffer = ctypes.create_string_buffer(65536)
        while True:
            kernel.ResetEvent(event)
            operation = Overlapped(hEvent=event | 1)
            count = wintypes.DWORD()
            complete = kernel.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), ctypes.byref(operation))
            if not complete:
                error = ctypes.get_last_error()
                if error in (109, 232):
                    return
                if error != 997:
                    raise ctypes.WinError(error)
                if not kernel.GetOverlappedResult(handle, ctypes.byref(operation), ctypes.byref(count), True):
                    error = ctypes.get_last_error()
                    if error in (109, 232):
                        return
                    raise ctypes.WinError(error)
            if not count.value:
                return
    finally:
        kernel.CloseHandle(event)
        kernel.CloseHandle(handle)

with ThreadPoolExecutor(max_workers=len(sys.argv) - 1) as workers:
    list(workers.map(drain, map(int, sys.argv[1:])))
"""


def _handoff_output(proc: asyncio.subprocess.Process) -> None:
    """Keep services' stdout/stderr writable after the owner and its loop exit."""
    import _winapi

    duplicates: list[int] = []
    try:
        # asyncio exposes these pipe handles only on the subprocess transport.
        transport: asyncio.SubprocessTransport = cast(Any, proc)._transport
        current = _winapi.GetCurrentProcess()
        for fd in (1, 2):
            pipe = transport.get_pipe_transport(fd)
            assert isinstance(pipe, asyncio.ReadTransport)
            pipe.pause_reading()
            handle = pipe.get_extra_info("pipe").fileno()
            duplicates.append(
                _winapi.DuplicateHandle(
                    current, handle, current, 0, True, _winapi.DUPLICATE_SAME_ACCESS
                )
            )
        startup = subprocess.STARTUPINFO()
        startup.lpAttributeList = {"handle_list": duplicates}
        drainer = subprocess.Popen(
            [sys.executable, "-c", _DRAIN_SCRIPT, *map(str, duplicates)],
            startupinfo=startup,
            creationflags=subprocess.CREATE_NO_WINDOW,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Keep the Popen owner alive and reap its handle when the last writer
        # exits. A daemon waiter does not tie preserved services to this loop
        # or prevent the caller's process from exiting.
        threading.Thread(target=drainer.wait, daemon=True).start()
    finally:
        for handle in duplicates:
            _winapi.CloseHandle(handle)


def _close_expired_transport(transport: asyncio.BaseTransport) -> None:
    """Release a proactor transport after its owning event loop was closed.

    At that point IOCP has cancelled pending I/O, but transport.close() cannot
    schedule connection_lost. Release the Windows handle and mark the pipe
    closed before subprocess.close() visits it on Python 3.12. Python 3.13
    skips those pipes when the loop is closed. CPython's private fields are
    necessary here because the public close path needs a live event loop.
    """
    handle = getattr(transport, "_sock", None)
    if handle is not None:
        handle.close()
        transport._sock = None  # type: ignore[attr-defined]
    transport._closing = True  # type: ignore[attr-defined]
    server = getattr(transport, "_server", None)
    if server is not None:
        if sys.version_info >= (3, 13):
            server._detach(transport)
        else:
            server._detach()
        transport._server = None  # type: ignore[attr-defined]


class PowerShellSession:
    """A persistent native shell implementing the same interface as BashSession.

    Calls from one event loop are serialized. Native stdin receives UTF-8 bytes
    verbatim and EOF (or just EOF when omitted). Both native and PowerShell
    failures have nonzero status. ``run_stream`` yields output while executing.
    ``close`` waits for an active call; cancel that call to interrupt it early.

    Background work is owned until close, unless keep_background_on_close or
    preserve_background_services explicitly transfers ownership to the caller.
    Use Start-Process -WindowStyle Hidden for a detached Windows service;
    PowerShell Start-Job jobs depend on their PowerShell host's lifetime.
    """

    shell_name: Literal["powershell"] = "powershell"

    def __init__(
        self,
        cwd: str | Path = ".",
        init_command: str | None = None,
        *,
        keep_background_on_close: bool = False,
        executable: str | Path | None = None,
    ) -> None:
        self._cwd = Path(cwd).resolve()
        self._init_command = init_command
        self._executable = str(executable) if executable is not None else None
        self._keep_background_on_close = keep_background_on_close
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._job: WindowsJob | None = None
        self._jobs: list[WindowsJob] = []
        self._background: list[WindowsJob] = []
        self._started = False
        self._started_on_loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()
        from nooa.tools.shell_lifecycle import _current_scope

        if scope := _current_scope.get():
            scope.adopt(self)

    @property
    def cwd(self) -> Path:
        return self._cwd

    def _ensure_lock(self) -> None:
        if (
            self._started_on_loop is not None
            and self._started_on_loop is not asyncio.get_running_loop()
        ):
            self._lock = asyncio.Lock()

    async def __aenter__(self) -> PowerShellSession:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def start(self) -> None:
        self._ensure_lock()
        async with self._lock:
            await self._ensure_started()

    async def _ensure_started(self) -> None:
        if (
            self._started
            and self._process is not None
            and self._process.returncode is None
            and self._started_on_loop is asyncio.get_running_loop()
        ):
            return
        if sys.platform != "win32":
            raise RuntimeError("PowerShellSession requires native Windows; use BashSession on Unix")
        from nooa.tools._windows_job import WindowsJob

        if self._process is not None:
            await self._retire_current()
        executable = self._executable or shutil.which("pwsh") or shutil.which("powershell")
        if not executable:
            raise FileNotFoundError(
                "PowerShell was not found; install PowerShell 7 (pwsh) or Windows PowerShell 5.1"
            )
        loop = asyncio.get_running_loop()
        accepted: asyncio.Future[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
            loop.create_future()
        )
        token = secrets.token_hex(32)
        connections: list[asyncio.StreamWriter] = []

        async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            connections.append(writer)
            try:
                hello = await asyncio.wait_for(reader.readline(), 5.0)
                if hello.rstrip(b"\r\n") == token.encode() and not accepted.done():
                    accepted.set_result((reader, writer))
                    return
            except (TimeoutError, OSError):
                pass
            writer.close()

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            self._job = WindowsJob()
            self._jobs.append(self._job)
            env = os.environ.copy()
            env.update(TERM="dumb", NO_COLOR="1", PYTHONIOENCODING="utf-8")
            self._process = await asyncio.create_subprocess_exec(
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(Path(__file__).with_name("_powershell_host.ps1")),
                str(port),
                token,
                cwd=str(self._cwd),
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self._started_on_loop = loop
            self._job.assign(self._process.pid)
            self._reader, self._writer = await asyncio.wait_for(accepted, 20.0)
            self._writer.write(b"start\n")
            await self._writer.drain()
            ready = json.loads(await asyncio.wait_for(self._reader.readline(), 20.0))
            if ready.get("ready") is not True:
                raise RuntimeError(f"PowerShell startup failed: {ready.get('error', ready)}")
            self._started = True
        except BaseException:
            await self._retire_current()
            raise
        finally:
            server.close()
            # Python 3.13's wait_closed() also waits for accepted connections.
            # This connection is the live session, so only stop accepting here.
            for writer in connections:
                if writer is not self._writer:
                    writer.close()
        if self._init_command:
            try:
                stdout, stderr, code, timed_out = await self._collect(self._init_command, 60.0)
                if code != 0 or timed_out:
                    raise RuntimeError(
                        f"PowerShell init_command failed ({code}): {stderr or stdout}"
                    )
            except BaseException:
                await self._retire_current()
                raise

    async def run(
        self,
        command: str,
        timeout: float = 30.0,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> tuple[str, str, int]:
        stdout, stderr, code, _ = await self.run_with_timeout_flag(
            command, timeout, stdin=stdin, cwd=cwd
        )
        return stdout, stderr, code

    async def run_with_timeout_flag(
        self,
        command: str,
        timeout: float = 30.0,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> tuple[str, str, int, bool]:
        self._ensure_lock()
        async with self._lock:
            await self._ensure_started()
            return await self._collect(command, timeout, stdin=stdin, cwd=cwd)

    async def _collect(
        self,
        command: str,
        timeout: float,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> tuple[str, str, int, bool]:
        buffers = {
            name: TruncatingStringIO(limit=_MAX_OUTPUT_CHARS) for name in ("stdout", "stderr")
        }
        code, timed_out = -1, False
        async with aclosing(self._execute(command, timeout, stdin=stdin, cwd=cwd)) as stream:
            async for name, chunk in stream:
                if name == "__done__":
                    code_text, timeout_text = chunk.split(",")
                    code, timed_out = int(code_text), bool(int(timeout_text))
                else:
                    buffers[name].write(chunk)
        return (
            buffers["stdout"].getvalue().strip(),
            buffers["stderr"].getvalue().strip(),
            code,
            timed_out,
        )

    async def run_stream(
        self,
        command: str,
        timeout: float = 30.0,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> AsyncGenerator[tuple[str, str]]:
        self._ensure_lock()
        async with self._lock:
            await self._ensure_started()
            async with aclosing(self._execute(command, timeout, stdin=stdin, cwd=cwd)) as stream:
                async for event in stream:
                    yield event

    async def _execute(
        self,
        command: str,
        timeout: float,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> AsyncGenerator[tuple[str, str]]:
        proc, reader, writer = self._process, self._reader, self._writer
        assert proc is not None and proc.stdout is not None and proc.stderr is not None
        assert reader is not None and writer is not None
        request_id = secrets.token_hex(16)
        input_path: Path | None = None
        if stdin is not None:
            fd, name = tempfile.mkstemp(prefix="nooa-stdin-")
            input_path = Path(name)
            with os.fdopen(fd, "wb") as output:
                output.write(stdin.encode("utf-8"))
        request = {
            "id": request_id,
            "command": command,
            "stdin_path": str(input_path) if input_path is not None else None,
            "cwd": str(cwd) if cwd is not None else None,
        }
        queue: asyncio.Queue[tuple[str, str | _Completion | Exception]] = asyncio.Queue(maxsize=32)
        marker = f"\x1eNOOA_{request_id}\x1f".encode("ascii")
        host_exited = asyncio.Event()
        completed = False

        async def pump(stream: asyncio.StreamReader, name: str) -> None:
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            pending = b""
            exited = asyncio.create_task(host_exited.wait())
            reading: asyncio.Task[bytes] | None = None
            # Once a dead host has no marker (e.g. TerminateProcess), a live
            # background writer must not hold this command open indefinitely.
            # This exceeds the bounded reader/OS buffers left by the host.
            remaining_after_exit = 1_048_576
            try:
                while True:
                    reading = asyncio.create_task(stream.read(8192))
                    await asyncio.wait((reading, exited), return_when=asyncio.FIRST_COMPLETED)
                    try:
                        data = (
                            await asyncio.wait_for(reading, 0.1)
                            if host_exited.is_set()
                            else await reading
                        )
                    except (TimeoutError, OSError):
                        break
                    if not data:
                        break
                    pending += data
                    end = pending.find(marker)
                    if end >= 0:
                        if text := decoder.decode(pending[:end], final=True):
                            await queue.put((name, text))
                        pending = b""
                        break
                    # Keep a possible split marker; the decoder keeps split UTF-8.
                    start = pending.rfind(marker[:1])
                    safe = (
                        start if start >= 0 and marker.startswith(pending[start:]) else len(pending)
                    )
                    if text := decoder.decode(pending[:safe]):
                        await queue.put((name, text))
                    pending = pending[safe:]
                    if host_exited.is_set():
                        remaining_after_exit -= len(data)
                        if remaining_after_exit <= 0:
                            break
                if text := decoder.decode(pending, final=True):
                    await queue.put((name, text))
                await queue.put(("__pipe_done__", name))
            except Exception as error:
                await queue.put(("__error__", error))
            finally:
                exited.cancel()
                if reading is not None:
                    reading.cancel()
                    await asyncio.gather(reading, return_exceptions=True)
                await asyncio.gather(exited, return_exceptions=True)

        async def control() -> None:
            try:
                # This timeout belongs to a producer task, never to the task
                # consuming run_stream(). Keeping it across a yield would
                # cancel the caller while it handles an output chunk.
                async with asyncio.timeout(timeout):
                    writer.write((json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8"))
                    await writer.drain()
                    try:
                        line = await reader.readline()
                    except ConnectionResetError:
                        line = b""
                    reply = json.loads(line) if line else None
                    if reply is not None:
                        if reply.get("id") != request_id:
                            raise RuntimeError(
                                "PowerShell control response did not match the command"
                            )
                        result = _Completion(int(reply["code"]))
                        if reply.get("cwd"):
                            self._cwd = Path(reply["cwd"])
                        self._remember_background()
                    else:
                        # A socket reset can precede the process-exit callback.
                        # proc.wait() also waits for pipes held by background work.
                        while proc.returncode is None:
                            await asyncio.sleep(0.01)
                        host_exited.set()
                        result = _Completion(proc.returncode)
                await queue.put(("__control__", result))
            except TimeoutError:
                try:
                    await self._retire_current()
                    host_exited.set()
                    await queue.put(("__control__", _Completion(124, timed_out=True)))
                except Exception as error:
                    await queue.put(("__error__", error))
            except Exception as error:
                await queue.put(("__error__", error))

        tasks = [
            asyncio.create_task(pump(proc.stdout, "stdout")),
            asyncio.create_task(pump(proc.stderr, "stderr")),
            asyncio.create_task(control()),
        ]
        finished_pipes: set[str] = set()
        result: _Completion | None = None
        try:
            while result is None or len(finished_pipes) != 2:
                name, value = await queue.get()
                if name == "__error__":
                    assert isinstance(value, Exception)
                    raise value
                if name == "__pipe_done__":
                    assert isinstance(value, str)
                    finished_pipes.add(value)
                elif name == "__control__":
                    assert isinstance(value, _Completion)
                    result = value
                else:
                    assert isinstance(value, str)
                    yield name, value
            completed = not host_exited.is_set()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if not completed and self._process is proc:
                await self._retire_current()
            if input_path is not None:
                input_path.unlink(missing_ok=True)
        assert result is not None
        yield "__done__", f"{result.code},{int(result.timed_out)}"

    def _remember_background(self) -> None:
        assert self._job is not None and self._process is not None
        self._prune_jobs()
        self._background.extend(
            self._job.collect_background(self._process.pid, tuple(self._background))
        )

    def _prune_jobs(self) -> None:
        for groups in (self._background, self._jobs):
            for job in list(groups):
                if job is not self._job and not job.pids():
                    job.close()
                    groups.remove(job)

    async def _retire_current(self) -> None:
        """End unfinished work; retain completed background trees in their old job."""
        proc = self._process
        if self._job is not None:
            self._job.terminate_foreground(tuple(self._background))
        if proc is not None:
            if any(job.pids() for job in self._background):
                _handoff_output(proc)
            loop_closed = self._started_on_loop is not None and self._started_on_loop.is_closed()
            if self._writer is not None:
                if loop_closed:
                    _close_expired_transport(self._writer.transport)
                else:
                    self._writer.close()
            # A background process can hold the pipe open forever. Close our
            # copies after handing off the readers, before waiting on the host.
            transport: asyncio.SubprocessTransport = proc._transport  # type: ignore[attr-defined]
            if loop_closed:
                for fd in (1, 2):
                    if pipe := transport.get_pipe_transport(fd):
                        _close_expired_transport(pipe)
            transport.close()
            try:
                if loop_closed:
                    # The closed proactor cannot observe process exit or update
                    # Popen.returncode. Reap the terminated host directly rather
                    # than awaiting a Process.wait() future owned by that loop.
                    child = transport.get_extra_info("subprocess")
                    if child is not None:
                        await asyncio.to_thread(child.wait, timeout=5.0)
                elif self._started_on_loop is asyncio.get_running_loop():
                    await asyncio.wait_for(proc.wait(), 5.0)
            except (TimeoutError, subprocess.TimeoutExpired):
                raise RuntimeError("PowerShell did not exit after its job was terminated") from None
        self._process = None
        self._reader = None
        self._writer = None
        self._job = None
        self._started = False
        self._prune_jobs()

    async def reset(self) -> None:
        self._ensure_lock()
        async with self._lock:
            await self._retire_current()
            await self._ensure_started()

    async def close(self) -> None:
        self._ensure_lock()
        async with self._lock:
            await self._retire_current()
            for job in self._jobs:
                job.close(keep_processes=self._keep_background_on_close)
            for job in self._background:
                job.close()
            self._jobs.clear()
            self._background.clear()
            self._started_on_loop = None

    def __del__(self) -> None:
        try:
            if self._keep_background_on_close:
                if self._job is not None:
                    self._job.terminate_foreground(tuple(self._background))
                if self._process is not None and any(j.pids() for j in self._background):
                    _handoff_output(self._process)
            for job in self._jobs:
                job.close(keep_processes=self._keep_background_on_close)
        except Exception:
            logger.debug("PowerShell destructor cleanup failed", exc_info=True)
