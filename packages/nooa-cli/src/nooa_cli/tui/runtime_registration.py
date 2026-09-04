# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime registration and graceful self-restart support for the TUI.

A small JSON record lets an external, developer-owned supervisor discover TUI
processes without guessing from ``ps`` output.  SIGUSR1 asks the TUI to leave
through its normal shutdown path; the caller re-execs only after snapshots,
SQLite ownership, agent resources, and terminal state have been released.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_RUNTIME_DIR_ENV = "NOOA_TUI_RUNTIME_DIR"


def runtime_directory() -> Path:
    """Return the shared directory containing live-TUI records."""
    configured = os.environ.get(_RUNTIME_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".nooa" / "run" / "tui"


def _source_root() -> Path:
    """Find the source/install root that supplied ``nooa_cli``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() or (
            (parent / "pyproject.toml").exists() and (parent / "packages" / "nooa-cli").exists()
        ):
            return parent
    return here.parent


def _source_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def process_identity(pid: int) -> str | None:
    """Return a stable identity for this incarnation of *pid*, when available."""
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        # Field 22 is the process start time in clock ticks since boot.  Split
        # after the final ')' because process names may contain spaces.
        fields = stat_path.read_text().rsplit(")", 1)[1].split()
        return f"proc:{fields[19]}"
    except (OSError, IndexError):
        pass
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    started = " ".join(result.stdout.split())
    return f"ps:{started}" if started else None


def explicit_resume_argv(argv: list[str], session_id: str) -> list[str]:
    """Return *argv* with exactly one explicit ``--continue SESSION`` option."""
    result: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument.startswith("--continue="):
            index += 1
            continue
        if argument in {"-c", "--continue"}:
            index += 1
            if index < len(argv) and not argv[index].startswith("-"):
                index += 1
            continue
        result.append(argument)
        index += 1
    return [*result, "--continue", session_id]


class TUIRuntimeRegistration:
    """Own one atomic live-process record for an active TUI session."""

    def __init__(self, *, session_id: str, working_dir: str) -> None:
        self.session_id = session_id
        self.working_dir = str(Path(working_dir).expanduser().resolve())
        self.pid = os.getpid()
        self.token = secrets.token_hex(16)
        self.source_root = _source_root()
        self.started_at = time.time()
        original_argv = list(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
        self.restart_argv = explicit_resume_argv(original_argv, session_id)
        self.path = runtime_directory() / f"{self.pid}.json"
        self.lock_path = self.path.with_suffix(".lock")
        self._lock_fd: int | None = None
        self._installed_loop: Any | None = None
        self._previous_handler: Any = None

    def _payload(self) -> dict[str, Any]:
        identity = process_identity(self.pid)
        if identity is None:
            raise OSError("cannot determine a stable process identity")
        return {
            "schema_version": 1,
            "pid": self.pid,
            "token": self.token,
            "process_identity": identity,
            "restart_signal": "SIGUSR1",
            "session_id": self.session_id,
            "working_dir": self.working_dir,
            "source_root": str(self.source_root),
            "source_revision": _source_revision(self.source_root),
            "argv": self.restart_argv,
            "started_at": self.started_at,
        }

    def publish(self) -> None:
        """Atomically publish this process as restart-capable."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        if self._lock_fd is None:
            self._lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.ftruncate(self._lock_fd, 0)
                os.write(self._lock_fd, self.token.encode())
            except BaseException:
                os.close(self._lock_fd)
                self._lock_fd = None
                raise
        temporary = self.path.with_suffix(f".json.{self.token}.tmp")
        payload = (json.dumps(self._payload(), sort_keys=True) + "\n").encode()
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
            temporary.replace(self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def update_session(self, session_id: str) -> None:
        """Republish this process with a new active session and resume target."""
        self.session_id = session_id
        self.restart_argv = explicit_resume_argv(self.restart_argv, session_id)
        self.publish()

    def install_restart_signal(self, loop: Any, request_restart: Any) -> bool:
        """Route SIGUSR1 onto *loop*; return false on unsupported platforms."""
        if not hasattr(signal, "SIGUSR1") or not hasattr(loop, "add_signal_handler"):
            return False
        try:
            self._previous_handler = signal.getsignal(signal.SIGUSR1)
            loop.add_signal_handler(signal.SIGUSR1, request_restart)
        except (NotImplementedError, RuntimeError, ValueError):
            return False
        self._installed_loop = loop
        return True

    def close(self) -> None:
        """Remove signal routing and records without deleting a successor's."""
        if self._installed_loop is not None:
            try:
                self._installed_loop.remove_signal_handler(signal.SIGUSR1)
                if self._previous_handler is not None:
                    signal.signal(signal.SIGUSR1, self._previous_handler)
            except (OSError, RuntimeError, ValueError):
                pass
            self._installed_loop = None
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, ValueError, TypeError):
            payload = None
        if (
            isinstance(payload, dict)
            and payload.get("pid") == self.pid
            and payload.get("token") == self.token
        ):
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass


def reexec_tui(argv: list[str]) -> None:
    """Replace the current process with its recorded Python invocation."""
    if not argv:
        raise ValueError("restart argv must not be empty")
    os.execvp(argv[0], argv)
