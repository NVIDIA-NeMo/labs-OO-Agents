# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent bash session for SWE tools.

Maintains a long-running bash subprocess with sentinel-based output capture.
Uses a dedicated control file descriptor (fd 3) for sentinels so that
stdout/stderr are 100% user-owned.

Architecture:
  stdin  -> bash (commands only)
  FIFOs  <- isolated per-command stdout/stderr for run() and run_stream()
  fd 3   <- exit code + cwd + sentinel (control channel)
  stdout/stderr of the interactive shell itself are discarded
"""

import asyncio
import codecs
import json
import logging
import os
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

from nooa.agentdoc import FileBackedTruncatingStringIO, TruncatingStringIO

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 30_000
MAX_OUTPUT_ARTIFACT_BYTES = 10 * 1024 * 1024
MAX_OUTPUT_ARTIFACT_TOTAL_BYTES = 100 * 1024 * 1024
OUTPUT_ARTIFACT_TTL_SECONDS = 24 * 60 * 60
_STREAM_CANCEL_INT_GRACE = 0.5
_STREAM_CANCEL_TERM_GRACE = 0.5
_STREAM_CANCEL_KILL_GRACE = 1.0

_artifact_stores: dict[tuple[str, int], "_OutputArtifactStore"] = {}
_artifact_stores_lock = threading.Lock()


def _process_start_token(pid: int) -> str:
    """Return a PID-reuse-resistant process start token when available."""
    try:
        # The parenthesized comm field may contain spaces, so count fields
        # only after its closing parenthesis. Linux field 22 (starttime) is
        # index 19 in that suffix because it begins with field 3 (state).
        suffix = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if len(suffix) > 19:
            return suffix[19]
    except (FileNotFoundError, PermissionError, OSError):
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _owner_is_alive(pid: int, start_token: str) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    current_token = _process_start_token(pid)
    return not (start_token and current_token and current_token != start_token)


def _write_owner_metadata(path: Path, metadata: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, sort_keys=True))
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _reap_expired_artifact_stores(root: Path) -> None:
    """Reap only stores whose owning process is proven dead and TTL expired."""
    now = time.time()
    for directory in root.glob("nooa-shell-output-*"):
        try:
            directory_stat = directory.lstat()
        except OSError:
            continue
        # The temp root may be shared. Never follow symlinks or reap a store
        # that is not private and owned by this OS user.
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or directory_stat.st_uid != os.getuid()
            or stat.S_IMODE(directory_stat.st_mode) & 0o077
        ):
            continue
        owner_path = directory / "owner.json"
        try:
            metadata = json.loads(owner_path.read_text())
            pid = int(metadata["pid"])
            start_token = str(metadata.get("start_token", ""))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue

        if _owner_is_alive(pid, start_token):
            if metadata.get("orphaned_at") is not None:
                metadata["orphaned_at"] = None
                try:
                    _write_owner_metadata(owner_path, metadata)
                except OSError:
                    pass
            continue

        orphaned_at = metadata.get("orphaned_at")
        if not isinstance(orphaned_at, (int, float)):
            metadata["orphaned_at"] = now
            try:
                _write_owner_metadata(owner_path, metadata)
            except OSError:
                pass
            continue
        if now - float(orphaned_at) >= OUTPUT_ARTIFACT_TTL_SECONDS:
            shutil.rmtree(directory, ignore_errors=True)


class _OutputArtifactStore:
    """Private, process-owned storage with a non-evicting aggregate quota."""

    def __init__(self, root: Path) -> None:
        _reap_expired_artifact_stores(root)
        self.directory = Path(
            tempfile.mkdtemp(prefix=f"nooa-shell-output-{os.getpid()}-", dir=root)
        )
        os.chmod(self.directory, 0o700)
        self._lock = threading.Lock()
        self._bytes_retained = 0
        _write_owner_metadata(
            self.directory / "owner.json",
            {
                "pid": os.getpid(),
                "start_token": _process_start_token(os.getpid()),
                "orphaned_at": None,
                "created_at": time.time(),
            },
        )

    @property
    def remaining_bytes(self) -> int:
        with self._lock:
            return max(0, MAX_OUTPUT_ARTIFACT_TOTAL_BYTES - self._bytes_retained)

    def reserve(self, requested: int) -> int:
        with self._lock:
            available = max(0, MAX_OUTPUT_ARTIFACT_TOTAL_BYTES - self._bytes_retained)
            granted = min(max(0, requested), available)
            self._bytes_retained += granted
            return granted

    def release(self, count: int) -> None:
        with self._lock:
            self._bytes_retained = max(0, self._bytes_retained - max(0, count))


def _get_output_artifact_store() -> _OutputArtifactStore | None:
    root = Path(tempfile.gettempdir())
    key = (str(root.resolve()), os.getpid())
    with _artifact_stores_lock:
        store = _artifact_stores.get(key)
        if store is not None:
            return store
        try:
            store = _OutputArtifactStore(root)
        except OSError:
            logger.warning("Failed to initialize shell output artifact store", exc_info=True)
            return None
        _artifact_stores[key] = store
        return store


def _new_output_buffer(prefix: str) -> TruncatingStringIO:
    store = _get_output_artifact_store()
    if store is None:
        return TruncatingStringIO(limit=MAX_OUTPUT_CHARS)
    file_limit = min(MAX_OUTPUT_ARTIFACT_BYTES, store.remaining_bytes)
    return FileBackedTruncatingStringIO(
        limit=MAX_OUTPUT_CHARS,
        dir=str(store.directory),
        prefix=prefix,
        file_limit_bytes=file_limit,
        byte_reserver=store.reserve,
        byte_releaser=store.release,
    )


class _CommandOutputPipes:
    """Isolated stdout/stderr pipes for one persistent-shell command.

    The persistent bash redirects a brace group into these FIFOs. Background
    descendants may retain their write ends, but closing this object severs
    those writers instead of allowing their output to bleed into a later
    command on the shell's process-wide stdout/stderr pipes.
    """

    def __init__(
        self,
        *,
        directory: Path,
        paths: dict[str, Path],
        readers: dict[str, asyncio.StreamReader],
        transports: list[asyncio.BaseTransport],
        keepalive_fds: list[int],
    ) -> None:
        self.directory = directory
        self.paths = paths
        self.readers = readers
        self.transports = transports
        self.keepalive_fds = keepalive_fds

    @classmethod
    async def open(cls) -> "_CommandOutputPipes":
        directory = Path(tempfile.mkdtemp(prefix="nooa-bash-output-"))
        paths = {name: directory / name for name in ("stdout", "stderr")}
        readers: dict[str, asyncio.StreamReader] = {}
        transports: list[asyncio.BaseTransport] = []
        keepalive_fds: list[int] = []
        flags = os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
        try:
            loop = asyncio.get_running_loop()
            for name, path in paths.items():
                os.mkfifo(path, 0o600)
                read_fd = os.open(path, os.O_RDONLY | flags)
                read_file = None
                try:
                    # Keep a writer open until cleanup so the reader cannot see
                    # premature EOF before bash opens its redirected stream.
                    keepalive_fds.append(os.open(path, os.O_WRONLY | flags))
                    read_file = os.fdopen(read_fd, "rb", 0)
                    reader = asyncio.StreamReader()
                    transport, _ = await loop.connect_read_pipe(
                        lambda reader=reader: asyncio.StreamReaderProtocol(reader),
                        read_file,
                    )
                except BaseException:
                    if read_file is None:
                        os.close(read_fd)
                    else:
                        read_file.close()
                    raise
                readers[name] = reader
                transports.append(transport)
            return cls(
                directory=directory,
                paths=paths,
                readers=readers,
                transports=transports,
                keepalive_fds=keepalive_fds,
            )
        except BaseException:
            for fd in keepalive_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            for transport in transports:
                transport.close()
            for path in paths.values():
                path.unlink(missing_ok=True)
            directory.rmdir()
            raise

    def redirect(self, script: str) -> str:
        stdout_path = shlex.quote(str(self.paths["stdout"]))
        stderr_path = shlex.quote(str(self.paths["stderr"]))
        return f"{{\n{script}\n}} > {stdout_path} 2> {stderr_path}\n"

    def close_keepalives(self) -> None:
        for fd in self.keepalive_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.keepalive_fds.clear()

    async def close(self) -> None:
        self.close_keepalives()
        for transport in self.transports:
            transport.close()
        self.transports.clear()
        # Let connect_read_pipe process transport closure before unlinking.
        await asyncio.sleep(0)
        for path in self.paths.values():
            path.unlink(missing_ok=True)
        try:
            self.directory.rmdir()
        except OSError:
            pass


class BashSession:
    """A persistent bash shell session with dedicated control channel.

    Commands are serialized via an internal asyncio.Lock — concurrent
    ``run()`` / ``run_stream()`` calls from the same event loop will queue
    and execute one at a time.  This is safe but sequential; for true
    parallelism, create multiple BashSession instances.

    Usage::

        session = BashSession(cwd="/my/project")
        await session.start()
        stdout, stderr, code = await session.run("ls -la")
        stdout, stderr, code = await session.run("cd src && pwd")  # cd persists!
        await session.close()
    """

    def __init__(self, cwd: str | Path = ".", init_command: str | None = None) -> None:
        self._cwd = Path(cwd).resolve()
        # Optional shell snippet run once every time the session (re)starts —
        # before any user command — to set up the environment (e.g. activating a
        # conda env). Re-run on reset() because a fresh bash loses prior env.
        self._init_command = init_command
        self._running_init = False
        self._process: asyncio.subprocess.Process | None = None
        self._control_reader: asyncio.StreamReader | None = None
        self._control_transport: asyncio.BaseTransport | None = None
        self._started = False
        self._started_on_loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()
        self._last_successful_command: float | None = None
        self._last_command: str = ""
        self._start_count: int = 0

    @property
    def cwd(self) -> Path:
        """Current working directory of the session."""
        return self._cwd

    def __del__(self) -> None:
        """Best-effort cleanup: kill the bash subprocess if still running."""
        proc = self._process
        if proc is not None and proc.returncode is None:
            try:
                # During interpreter shutdown, module globals (os, signal) may
                # be None, causing TypeError. Broad except handles all cases.
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    async def __aenter__(self) -> "BashSession":
        """Support ``async with BashSession() as session:`` usage."""
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def _diagnose_death(self, context: str) -> str:
        """Capture diagnostic info about why bash died. Logs at ERROR level."""
        proc = self._process
        parts = [f"[BASH_DEATH] context={context}"]
        if self._last_successful_command is not None:
            parts.append(
                f"  last_successful_cmd_ago={time.time() - self._last_successful_command:.1f}s"
            )
        else:
            parts.append("  last_successful_cmd_ago=never")
        parts.append(f"  last_command={self._last_command[:200]!r}")
        parts.append(f"  start_count={self._start_count}")
        parts.append(f"  cwd={self._cwd}")
        if proc is None:
            parts.append("  proc=None")
        else:
            parts.append(f"  proc.pid={proc.pid}")
            parts.append(f"  proc.returncode={proc.returncode}")
            if proc.returncode is not None and proc.returncode < 0:
                sig_num = -proc.returncode
                try:
                    sig_name = signal.Signals(sig_num).name
                except (ValueError, AttributeError):
                    sig_name = f"signal {sig_num}"
                parts.append(f"  killed_by={sig_name}")
            # Try to read /proc/<pid>/status before it disappears
            try:
                with open(f"/proc/{proc.pid}/status") as f:
                    for line in f:
                        if any(k in line for k in ("State:", "SigPnd:", "SigCgt:")):
                            parts.append(f"  /proc/status: {line.strip()}")
            except (FileNotFoundError, PermissionError, OSError):
                parts.append("  /proc/status: unavailable (process reaped)")
        # Check cwd accessibility (detects virtiofs / mount failures)
        try:
            os.stat(str(self._cwd))
            parts.append("  cwd_stat=OK")
        except OSError as e:
            parts.append(f"  cwd_stat=FAILED: {e}")
        # FD count of parent — detects FD leaks that can trigger OOM-killer
        try:
            fd_count = len(os.listdir("/proc/self/fd"))
            parts.append(f"  parent_fd_count={fd_count}")
        except OSError:
            pass
        diag = "\n".join(parts)
        logger.error(diag)
        try:
            from nooa.runtime.harness_metrics import get_harness_metrics

            get_harness_metrics().shell_death(context, diag)
        except Exception:
            pass  # telemetry must not break recovery
        return diag

    async def start(self) -> None:
        """Start the bash subprocess with a dedicated control fd."""
        if self._started:
            return

        # Reap expired stores even when this shell never produces large output.
        _get_output_artifact_store()

        self._start_count += 1
        env = os.environ.copy()
        env["PS1"] = ""
        env["TERM"] = "dumb"

        # Create pipe for control channel (fd 3 inside bash).
        ctrl_r, ctrl_w = os.pipe()
        try:
            self._process = await asyncio.create_subprocess_exec(
                "/bin/bash",
                "--norc",
                "--noprofile",
                "--noediting",
                "-i",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(self._cwd),
                env=env,
                start_new_session=True,
                pass_fds=(ctrl_w,),
            )
        except Exception:
            os.close(ctrl_r)
            os.close(ctrl_w)
            raise

        # Dup the write end to fd 3 inside bash, then close the original.
        assert self._process.stdin is not None
        self._process.stdin.write(f"exec 3>&{ctrl_w} {ctrl_w}>&-\n".encode())
        await self._process.stdin.drain()
        os.close(ctrl_w)

        # Wrap the read end in an asyncio StreamReader.
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=2**20)
        transport, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader),
            os.fdopen(ctrl_r, "rb", 0),
        )
        self._control_reader = reader
        self._control_transport = transport
        self._started = True
        self._started_on_loop = asyncio.get_running_loop()

        # Drain startup — send a no-op through the control channel.
        sentinel = f"__CTRL_{secrets.token_hex(8)}__"
        self._process.stdin.write(f"echo {sentinel} >&3\n".encode())
        await self._process.stdin.drain()
        await self._read_control_until(sentinel, timeout=5.0)

        # Run the one-time init command (env setup) before any user command.
        # Its output is intentionally discarded, including any artifact file.
        if self._init_command and not self._running_init:
            self._running_init = True
            try:
                await self._run_unlocked(
                    self._init_command,
                    timeout=60.0,
                    retain_artifacts=False,
                )
            finally:
                self._running_init = False

    def _ensure_lock_on_current_loop(self) -> None:
        """Recreate the lock if the event loop changed since it was created."""
        if (
            self._started_on_loop is not None
            and self._started_on_loop is not asyncio.get_running_loop()
        ):
            self._lock = asyncio.Lock()

    async def run(self, command: str, timeout: float = 30.0) -> tuple[str, str, int]:
        """Run a command and return (stdout, stderr, exit_code).

        The session persists state: cd, export, etc. carry over.
        Concurrent calls are serialized via an internal lock.

        On timeout, exit_code is 124 — same as the ``timeout(1)`` command.
        Use ``run_with_timeout_flag()`` if you need to distinguish a real
        timeout from a command that exits 124 naturally.
        """
        self._ensure_lock_on_current_loop()
        async with self._lock:
            stdout, stderr, code, _ = await self._run_unlocked(command, timeout)
            return stdout, stderr, code

    async def run_with_timeout_flag(
        self, command: str, timeout: float = 30.0
    ) -> tuple[str, str, int, bool]:
        """Like run(), but returns a 4th element: whether the command timed out."""
        self._ensure_lock_on_current_loop()
        async with self._lock:
            return await self._run_unlocked(command, timeout)

    async def _run_unlocked(
        self,
        command: str,
        timeout: float,
        *,
        retain_artifacts: bool = True,
    ) -> tuple[str, str, int, bool]:
        """Actual run implementation (caller must hold self._lock).

        Returns (stdout, stderr, exit_code, timed_out).
        """
        stdout_buf = _new_output_buffer("nooa_shell_stdout_")
        stderr_buf = _new_output_buffer("nooa_shell_stderr_")
        stream = self._run_stream_unlocked(command, timeout)
        exit_code = -1
        timed_out = False
        try:
            async for stream_name, value in stream:
                if stream_name == "stdout":
                    stdout_buf.write(value)
                elif stream_name == "stderr":
                    stderr_buf.write(value)
                elif stream_name == "__done__":
                    code_text, timed_out_text = value.split(",", 1)
                    exit_code = int(code_text)
                    timed_out = bool(int(timed_out_text))

            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()
        except BaseException:
            for buf in (stdout_buf, stderr_buf):
                if isinstance(buf, FileBackedTruncatingStringIO):
                    buf.cleanup()
            raise
        finally:
            await stream.aclose()

        for buf in (stdout_buf, stderr_buf):
            if not isinstance(buf, FileBackedTruncatingStringIO):
                continue
            if retain_artifacts and buf.was_truncated:
                buf.close()
            else:
                buf.cleanup()

        return stdout.strip(), stderr.strip(), exit_code, timed_out

    async def run_stream(
        self, command: str, timeout: float = 30.0
    ) -> AsyncIterator[tuple[str, str]]:
        """Run a command and yield (stream_name, chunk) pairs as output arrives.

        stream_name is 'stdout' or 'stderr'. After the command finishes,
        yields ('__done__', 'exit_code,timed_out_flag') where timed_out_flag
        is '1' if the command timed out, '0' otherwise.

        Concurrent calls are serialized via an internal lock.

        Each call uses isolated stdout/stderr pipes terminated by random
        per-command markers. Output from a background process after its
        foreground command completes is excluded and cannot bleed into the
        next call. Closing an active stream interrupts only processes created
        by that command and preserves the persistent shell and earlier
        background jobs. Persistent background jobs must redirect stdout and
        stderr; command-private output pipes close at foreground completion.
        Deliberately detached/reparented daemons are outside portable process
        ownership. The shell is reset only if scoped recovery fails.
        """
        self._ensure_lock_on_current_loop()
        async with self._lock:
            stream = self._run_stream_unlocked(command, timeout)
            try:
                async for item in stream:
                    yield item
            finally:
                await stream.aclose()

    async def _run_stream_unlocked(
        self, command: str, timeout: float
    ) -> AsyncIterator[tuple[str, str]]:
        """Actual run_stream implementation (caller must hold self._lock)."""
        if not self._started:
            await self.start()
        elif self._started_on_loop is not asyncio.get_running_loop():
            await self._reset_for_loop_change()

        self._last_command = command
        sentinel = f"__CTRL_{secrets.token_hex(8)}__"
        output_marker_text = f"__NOOA_OUTPUT_{secrets.token_hex(16)}__"
        output_marker = output_marker_text.encode()

        proc = self._process
        ctrl = self._control_reader
        if proc is None or proc.stdin is None or ctrl is None or proc.returncode is not None:
            self._diagnose_death("run_stream_pre_check")
            await self.reset()
            proc = self._process
            ctrl = self._control_reader
            if proc is None or proc.stdin is None or ctrl is None:
                raise RuntimeError("Bash session failed to restart")

        # Snapshot processes already owned by this shell so cancellation can
        # spare background servers from earlier commands while terminating the
        # descendants that remain owned by this command. Deliberately detached
        # processes are outside portable BashSession ownership.
        protected_descendants = await self._descendant_pids(proc.pid)
        output_pipes = await _CommandOutputPipes.open()
        quoted_marker = shlex.quote(output_marker_text)
        script = output_pipes.redirect(
            # eval preserves top-level shell semantics (unlike source or a
            # function) while keeping parse errors inside the private pipes.
            f"builtin eval -- {shlex.quote(command)}\n"
            "_nemo_ec=$?\n"
            f"printf '%s' {quoted_marker}\n"
            f"printf '%s' {quoted_marker} >&2\n"
            "echo $_nemo_ec >&3\n"
            "pwd >&3\n"
            f"echo {sentinel} >&3"
        )

        control_task: asyncio.Task[tuple[list[str], bool]] | None = None
        output_tasks: dict[asyncio.Task[bytes], tuple[str, asyncio.StreamReader]] = {}
        decoders = {
            "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
            "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
        }
        pending_bytes = {"stdout": b"", "stderr": b""}
        output_complete: set[str] = set()
        command_complete = False
        stream_abandoned = False
        ctrl_lines: list[str] = []
        timed_out = False
        interrupted = False
        shell_died = False
        reset_required = False

        def consume_output(name: str, chunk: bytes) -> tuple[str, bool]:
            """Decode bytes up to this command's marker, preserving split UTF-8."""
            data = pending_bytes[name] + chunk
            marker_index = data.find(output_marker)
            if marker_index >= 0:
                payload = data[:marker_index]
                pending_bytes[name] = b""
                return decoders[name].decode(payload, final=True), True

            # Retain only the longest suffix that could actually be the start
            # of a marker split across reads. Ordinary short output should be
            # yielded immediately rather than waiting for marker_length bytes.
            possible_prefix = min(len(data), len(output_marker) - 1)
            while possible_prefix and not data.endswith(output_marker[:possible_prefix]):
                possible_prefix -= 1
            safe_length = len(data) - possible_prefix
            payload = data[:safe_length]
            pending_bytes[name] = data[safe_length:]
            return decoders[name].decode(payload), False

        try:
            try:
                proc.stdin.write(script.encode())
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                self._diagnose_death(f"run_stream_write: {exc}")
                await self.reset()
                proc = self._process
                ctrl = self._control_reader
                if proc is None or proc.stdin is None or ctrl is None:
                    raise RuntimeError("Bash session failed to restart") from exc
                protected_descendants = await self._descendant_pids(proc.pid)
                proc.stdin.write(script.encode())
                await proc.stdin.drain()

            # One pending read per stream gives natural backpressure without an
            # unbounded Python queue.
            control_task = asyncio.create_task(
                self._read_control_until(sentinel, timeout),
                name="bash-command-control",
            )
            output_tasks = {
                asyncio.create_task(reader.read(4096)): (name, reader)
                for name, reader in output_pipes.readers.items()
            }

            while not command_complete or len(output_complete) < 2:
                waiters = [*output_tasks]
                if not command_complete:
                    waiters.append(control_task)
                done, _ = await asyncio.wait(
                    waiters,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if control_task in done:
                    ctrl_lines, timed_out = control_task.result()
                    if timed_out:
                        recovered = await self._interrupt_active_command(
                            proc,
                            control_task,
                            protected_descendants,
                            exit_code=124,
                        )
                        if recovered is None:
                            reset_required = True
                            command_complete = True
                        else:
                            ctrl_lines = recovered
                            command_complete = True
                            interrupted = True
                    else:
                        command_complete = True
                        if not ctrl_lines:
                            shell_died = True
                    output_pipes.close_keepalives()

                for task in tuple(done):
                    if task not in output_tasks:
                        continue
                    name, stream = output_tasks.pop(task)
                    chunk = task.result()
                    if not chunk:
                        if interrupted or shell_died or reset_required:
                            pending = pending_bytes[name]
                            pending_bytes[name] = b""
                            text = decoders[name].decode(pending, final=True)
                            output_complete.add(name)
                            if text:
                                yield (name, text)
                            continue
                        raise RuntimeError(f"{name} closed before the command output marker")
                    text, reached_marker = consume_output(name, chunk)
                    if reached_marker:
                        output_complete.add(name)
                    else:
                        output_tasks[asyncio.create_task(stream.read(4096))] = (name, stream)
                    if text:
                        yield (name, text)

                if interrupted or shell_died or reset_required:
                    # The normal marker lines were aborted. Drain bytes already
                    # available from the private pipes, then finish at EOF.
                    for task, (name, stream) in tuple(output_tasks.items()):
                        output_tasks.pop(task)
                        try:
                            chunk = await asyncio.wait_for(task, timeout=0.2)
                            if pending_bytes[name]:
                                chunk = pending_bytes[name] + chunk
                                pending_bytes[name] = b""
                            while chunk:
                                text = decoders[name].decode(chunk)
                                if text:
                                    yield (name, text)
                                chunk = await asyncio.wait_for(stream.read(4096), timeout=0.2)
                        except TimeoutError:
                            pass
                        tail = decoders[name].decode(pending_bytes[name], final=True)
                        pending_bytes[name] = b""
                        if tail:
                            yield (name, tail)
                        output_complete.add(name)
                    break
        except BaseException:
            stream_abandoned = True
            raise
        finally:
            try:
                if (
                    stream_abandoned
                    and not command_complete
                    and self._process is proc
                    and control_task is not None
                ):
                    recovered = await self._interrupt_active_command(
                        proc,
                        control_task,
                        protected_descendants,
                        exit_code=130,
                    )
                    if recovered is not None:
                        ctrl_lines = recovered
                        command_complete = True
                    else:
                        reset_required = True
                if command_complete:
                    self._update_cwd_from_control(ctrl_lines)
            finally:
                cleanup_tasks = [
                    *(task for task in (control_task,) if task is not None),
                    *output_tasks,
                ]
                for task in cleanup_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                await output_pipes.close()

            if shell_died or reset_required:
                if reset_required:
                    self._diagnose_death("run_stream_cancel_recovery_failed")
                await self.reset()

        # Parse exit code
        if shell_died:
            await proc.wait()
            exit_code = proc.returncode if proc.returncode is not None else -1
        else:
            exit_code = -1 if not ctrl_lines else 0
        if ctrl_lines and not shell_died:
            try:
                exit_code = int(ctrl_lines[0].strip())
            except (ValueError, IndexError):
                pass
            self._update_cwd_from_control(ctrl_lines)

        if timed_out:
            exit_code = 124
        elif ctrl_lines:
            self._last_successful_command = time.time()

        yield ("__done__", f"{exit_code},{1 if timed_out else 0}")

    def _update_cwd_from_control(self, ctrl_lines: list[str]) -> None:
        if len(ctrl_lines) >= 2:
            candidate = ctrl_lines[1].strip()
            if candidate.startswith("/"):
                self._cwd = Path(candidate)

    @staticmethod
    async def _process_parent_map() -> dict[int, int] | None:
        """Return ``pid -> ppid`` from portable ``ps``, or None on failure."""
        try:
            ps = await asyncio.create_subprocess_exec(
                "ps",
                "-axo",
                "pid=,ppid=",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            return None
        try:
            stdout, _ = await asyncio.wait_for(ps.communicate(), timeout=1.0)
        except TimeoutError:
            ps.kill()
            await ps.wait()
            return None
        except BaseException:
            if ps.returncode is None:
                ps.kill()
                await ps.wait()
            raise
        if ps.returncode != 0:
            return None

        parents: dict[int, int] = {}
        for raw_line in stdout.splitlines():
            fields = raw_line.split()
            if len(fields) != 2:
                continue
            try:
                pid, ppid = (int(field) for field in fields)
            except ValueError:
                continue
            parents[pid] = ppid
        return parents

    @staticmethod
    def _descendants_from_parents(root_pid: int, parents: dict[int, int]) -> set[int]:
        children: dict[int, list[int]] = {}
        for pid, ppid in parents.items():
            children.setdefault(ppid, []).append(pid)

        descendants: set[int] = set()
        stack = list(children.get(root_pid, ()))
        while stack:
            pid = stack.pop()
            if pid in descendants:
                continue
            descendants.add(pid)
            stack.extend(children.get(pid, ()))
        return descendants

    async def _descendant_pids(self, root_pid: int) -> set[int] | None:
        """Snapshot all descendants, retaining failure as an explicit None."""
        parents = await self._process_parent_map()
        if parents is None:
            return None
        return self._descendants_from_parents(root_pid, parents)

    async def _signal_command_descendants(
        self,
        shell_pid: int,
        protected_descendants: set[int] | None,
        sig: signal.Signals,
    ) -> set[int] | None:
        """Signal only descendants created after the streamed command began.

        Descendants of processes present in the pre-command snapshot are also
        protected, so workers spawned later by an existing background server
        are not mistaken for children of the active command.
        """
        if protected_descendants is None:
            return None
        parents = await self._process_parent_map()
        if parents is None:
            return None

        all_descendants = self._descendants_from_parents(shell_pid, parents)
        protected_now = set(protected_descendants) & all_descendants
        for protected_pid in tuple(protected_now):
            protected_now.update(self._descendants_from_parents(protected_pid, parents))
        targets = all_descendants - protected_now

        def depth(pid: int) -> int:
            value = 0
            seen: set[int] = set()
            while pid in parents and pid != shell_pid and pid not in seen:
                seen.add(pid)
                pid = parents[pid]
                value += 1
            return value

        signalled: set[int] = set()
        for pid in sorted(targets, key=depth, reverse=True):
            try:
                os.kill(pid, sig)
                signalled.add(pid)
            except (ProcessLookupError, PermissionError):
                pass
        return signalled

    async def _interrupt_active_command(
        self,
        proc: asyncio.subprocess.Process,
        control_task: asyncio.Task[tuple[list[str], bool]],
        protected_descendants: set[int] | None,
        *,
        exit_code: int,
    ) -> list[str] | None:
        """Abort the active command as Ctrl-C, preserving the interactive shell."""
        if control_task.done() and not control_task.cancelled():
            try:
                lines, timed_out = control_task.result()
            except Exception:
                lines, timed_out = [], False
            if not timed_out and len(lines) >= 2:
                return lines

        if not control_task.done():
            done, _ = await asyncio.wait({control_task}, timeout=0.02)
            if done:
                try:
                    lines, timed_out = control_task.result()
                except Exception:
                    lines, timed_out = [], False
                if not timed_out and len(lines) >= 2:
                    return lines

        if not control_task.done():
            control_task.cancel()
        await asyncio.gather(control_task, return_exceptions=True)

        if protected_descendants is None or proc.stdin is None:
            return None

        owned_targets: set[int] = set()

        recovery_sentinel = f"__CTRL_RECOVER_{secrets.token_hex(8)}__"
        recovery_script = (
            f"printf '%s\\n' {exit_code} >&3\n"
            "pwd >&3\n"
            f"printf '%s\\n' {shlex.quote(recovery_sentinel)} >&3\n"
        )

        async def signal_phase(sig: signal.Signals) -> bool:
            signalled = await self._signal_command_descendants(
                proc.pid,
                protected_descendants,
                sig,
            )
            if signalled is None:
                return False
            owned_targets.update(signalled)
            try:
                # Interactive Bash handles SIGINT like Ctrl-C: it aborts the
                # current eval/list, remains alive, and reads the next command.
                os.kill(proc.pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                return False
            return True

        if not await signal_phase(signal.SIGINT):
            return None
        try:
            proc.stdin.write(recovery_script.encode())
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return None

        recovery_task = asyncio.create_task(
            self._read_control_until(
                recovery_sentinel,
                _STREAM_CANCEL_INT_GRACE
                + _STREAM_CANCEL_TERM_GRACE
                + _STREAM_CANCEL_KILL_GRACE
                + 1.0,
            ),
            name="bash-command-recovery",
        )

        async def finish_recovery(lines: list[str], timed_out: bool) -> list[str] | None:
            if timed_out or len(lines) < 2:
                return None
            # SIGINT may be ignored by asynchronous descendants. Once Bash is
            # synchronized and no next command can start under the held lock,
            # finish terminating any remaining command-owned processes.
            late_targets = await self._signal_command_descendants(
                proc.pid,
                protected_descendants,
                signal.SIGKILL,
            )
            if late_targets is not None:
                owned_targets.update(late_targets)
            # Some descendants may have reparented when interactive Bash
            # aborted its foreground list. They are still command-owned PIDs
            # captured immediately before interruption.
            for pid in owned_targets:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            return lines

        try:
            for sig, grace in (
                (signal.SIGINT, _STREAM_CANCEL_INT_GRACE),
                (signal.SIGTERM, _STREAM_CANCEL_TERM_GRACE),
                (signal.SIGKILL, _STREAM_CANCEL_KILL_GRACE),
            ):
                if sig != signal.SIGINT and not await signal_phase(sig):
                    return None
                done, _ = await asyncio.wait({recovery_task}, timeout=grace)
                if done:
                    lines, timed_out = recovery_task.result()
                    return await finish_recovery(lines, timed_out)
            return None
        finally:
            if not recovery_task.done():
                recovery_task.cancel()
            await asyncio.gather(recovery_task, return_exceptions=True)

    async def _read_control_until(self, sentinel: str, timeout: float) -> tuple[list[str], bool]:
        """Read lines from control fd until sentinel. Returns (lines, timed_out)."""
        ctrl = self._control_reader
        assert ctrl is not None
        lines: list[str] = []
        timed_out = False
        while True:
            try:
                raw = await asyncio.wait_for(ctrl.readline(), timeout=timeout)
            except TimeoutError:
                timed_out = True
                break
            if not raw:
                self._diagnose_death("control_fd_eof")
                break  # EOF — bash died
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if sentinel in line:
                break
            lines.append(line)

        return lines, timed_out

    async def _reset_for_loop_change(self) -> None:
        """Reset after detecting that the event loop changed (gl-212)."""
        logger.warning("BashSession: event loop changed — resetting (env/aliases lost)")
        try:
            from nooa.runtime.harness_metrics import get_harness_metrics

            get_harness_metrics().shell_death(
                "loop_change_reset",
                f"BashSession reset due to event loop change (gl-212). "
                f"cwd={self._cwd}, start_count={self._start_count}",
            )
        except Exception:
            pass
        await self.reset()

    async def reset(self) -> None:
        """Kill the current session and start a fresh one, preserving cwd."""
        cwd = self._cwd
        await self.close()
        self._cwd = cwd
        await self.start()

    async def close(self) -> None:
        """Terminate the bash session cleanly."""
        same_loop = self._started_on_loop is asyncio.get_running_loop()
        if self._control_transport is not None:
            try:
                self._control_transport.close()
            except Exception:
                pass  # Transport may be bound to a dead loop (gl-212)
            self._control_transport = None
        self._control_reader = None

        if self._process is not None and self._process.returncode is None:
            if same_loop:
                # Interactive bash ignores SIGTERM while idle. Ask it to exit
                # through stdin first, then fall back to TERM/KILL.
                if self._process.stdin is not None:
                    try:
                        self._process.stdin.write(b"exit\n")
                        await self._process.stdin.drain()
                        await asyncio.wait_for(self._process.wait(), timeout=0.5)
                    except (BrokenPipeError, ConnectionResetError, OSError, TimeoutError):
                        pass
                if self._process.returncode is not None:
                    self._process = None
                    self._started = False
                    self._started_on_loop = None
                    self._lock = asyncio.Lock()
                    return
                try:
                    pgid = os.getpgid(self._process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=3.0)
                except TimeoutError:
                    try:
                        pgid = os.getpgid(self._process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
            else:
                # Cross-loop (gl-212): transport is dead, just kill immediately.
                try:
                    pgid = os.getpgid(self._process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    try:
                        self._process.kill()
                    except Exception:
                        pass
        self._process = None
        self._started = False
        self._started_on_loop = None
        self._lock = asyncio.Lock()
