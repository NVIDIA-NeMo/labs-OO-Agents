# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guardrail leak-vs-closed pairs for the OS-enforced sandbox primitives.

Each guardrail gets two tests forked into child processes: one shows the exploit
succeeding *without* the guard (the leak), the paired one shows the *same*
exploit refused *with* the guard installed. The guards themselves live in
``nooa.runtime.sandbox.guards`` and are applied by a child on itself, so these
tests exercise the exact code path the worker uses.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import tempfile

import pytest

from nooa.runtime.sandbox import guards
from nooa.runtime.sandbox.config import LandlockRule
from nooa.runtime.sandbox.guards import (
    apply_env_allowlist,
    apply_landlock,
    apply_rlimits,
    apply_seccomp_no_inet,
    probe_capabilities,
)

pytestmark = pytest.mark.sandbox

CAPS = probe_capabilities()

# A minimal read-only system set so the forked child can still import/run Python.
_SYSTEM_READ = [
    LandlockRule(p, write=False)
    for p in ("/usr", "/lib", "/lib64", "/etc", "/proc", "/dev", os.path.dirname(os.__file__))
]


def run_child(fn) -> tuple[str, int]:
    """Fork, run ``fn`` in the child, return (message, term_signal).

    ``message`` is ``"OK"`` if ``fn`` returns, else ``"<ExcType>: <msg>"``.
    ``term_signal`` is the signal that killed the child (0 if it exited cleanly).
    """
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child process
        os.close(r)
        try:
            fn()
            os.write(w, b"OK")
        except BaseException as exc:  # noqa: BLE001 - report every failure mode
            os.write(w, f"{type(exc).__name__}: {exc}".encode()[:300])
        finally:
            os._exit(0)
    os.close(w)
    msg = os.read(r, 4096).decode()
    os.close(r)
    _, status = os.waitpid(pid, 0)
    sig = os.WTERMSIG(status) if os.WIFSIGNALED(status) else 0
    return msg, sig


# --- guardrail 2a: memory ---------------------------------------------------
@pytest.mark.skipif(
    sys.platform != "linux" or not CAPS.rlimit, reason="Linux RLIMIT_AS unavailable"
)
def test_memory_leak_without_cap():
    msg, _ = run_child(lambda: bytearray(400 * 1024 * 1024))
    assert msg == "OK"  # 400 MiB allocates fine when unbounded


@pytest.mark.skipif(
    sys.platform != "linux" or not CAPS.rlimit, reason="Linux RLIMIT_AS unavailable"
)
def test_memory_closed_with_cap():
    def child():
        apply_rlimits(max_memory_mb=128)
        bytearray(1024 * 1024 * 1024)  # 1 GiB > 128 MiB cap

    msg, _ = run_child(child)
    assert "MemoryError" in msg


# --- guardrail 2b: cpu ------------------------------------------------------
@pytest.mark.skipif(not CAPS.rlimit, reason="RLIMIT unavailable")
def test_cpu_closed_with_cap():
    def child():
        apply_rlimits(max_cpu_seconds=1)
        while True:
            pass

    msg, sig = run_child(child)
    # RLIMIT_CPU raises SIGXCPU at the soft limit and SIGKILL at the hard limit;
    # with soft==hard the spin loop is terminated near the cap either way.
    assert sig in (signal.SIGXCPU, signal.SIGKILL), (msg, sig)


# --- guardrail 3a: file read ------------------------------------------------
@pytest.mark.skipif(CAPS.landlock_abi < 1, reason="Landlock unavailable")
def test_file_read_leak_without_sandbox():
    with tempfile.TemporaryDirectory() as secret:
        path = os.path.join(secret, "s.txt")
        with open(path, "w") as fh:
            fh.write("TOPSECRET")
        msg, _ = run_child(lambda: open(path).read())
        assert msg == "OK"


@pytest.mark.skipif(CAPS.landlock_abi < 1, reason="Landlock unavailable")
def test_file_read_closed_with_sandbox():
    with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as secret:
        secret_path = os.path.join(secret, "s.txt")
        with open(secret_path, "w") as fh:
            fh.write("TOPSECRET")

        def child():
            apply_landlock(_SYSTEM_READ + [LandlockRule(ws, write=True)])
            open(secret_path).read()

        msg, _ = run_child(child)
        assert "PermissionError" in msg


# --- guardrail 3b: file write -----------------------------------------------
@pytest.mark.skipif(CAPS.landlock_abi < 1, reason="Landlock unavailable")
def test_file_write_closed_but_workspace_allowed():
    with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as outside:
        outside_path = os.path.join(outside, "evil.txt")

        def write_outside():
            apply_landlock(_SYSTEM_READ + [LandlockRule(ws, write=True)])
            open(outside_path, "w").write("x")

        def write_inside():
            apply_landlock(_SYSTEM_READ + [LandlockRule(ws, write=True)])
            open(os.path.join(ws, "ok.txt"), "w").write("x")

        outside_msg, _ = run_child(write_outside)
        inside_msg, _ = run_child(write_inside)
        assert "PermissionError" in outside_msg
        assert inside_msg == "OK"


# --- environment ------------------------------------------------------------
def _exec_time_env_keys() -> set[str]:
    with open("/proc/self/environ", "rb") as fh:
        entries = fh.read().split(b"\0")
    return {e.split(b"=", 1)[0].decode() for e in entries if b"=" in e}


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/environ is Linux-only")
def test_env_leak_without_allowlist():
    # A fork sees the parent's exec-time environment both ways a cell would look.
    assert {"PATH"} < _exec_time_env_keys(), "test needs more than PATH in its environment"

    def child():
        leaked = _exec_time_env_keys() - {"PATH"}
        assert leaked, "expected inherited variables in /proc/self/environ"
        assert set(os.environ) - {"PATH"}, "expected inherited variables in os.environ"

    msg, _ = run_child(child)
    assert msg == "OK", msg


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/environ is Linux-only")
def test_env_closed_with_allowlist():
    path = os.environ["PATH"]

    def child():
        apply_env_allowlist(("PATH",))
        assert set(os.environ) == {"PATH"}, sorted(os.environ)
        assert os.environ["PATH"] == path
        assert _exec_time_env_keys() == set(), sorted(_exec_time_env_keys())
        # Subprocesses get the allowlisted environment, not the original block.
        import subprocess

        out = subprocess.run(["env"], capture_output=True, text=True, check=True).stdout
        assert {line.split("=", 1)[0] for line in out.splitlines()} == {"PATH"}, out

    msg, _ = run_child(child)
    assert msg == "OK", msg


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/environ is Linux-only")
def test_env_allowlist_fails_closed_when_block_not_zeroed(monkeypatch):
    # No usable env_start/env_end (e.g. a /proc that reports zeros): refuse to run.
    monkeypatch.setattr(guards, "_env_block_bounds", lambda: (0, 0))
    msg, _ = run_child(lambda: apply_env_allowlist(("PATH",)))
    assert msg.startswith("RuntimeError"), msg


# --- guardrail 4: network ---------------------------------------------------
@pytest.mark.skipif(not CAPS.seccomp, reason="seccomp unavailable")
def test_network_leak_without_sandbox():
    def child():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.close()

    msg, _ = run_child(child)
    assert msg == "OK"


@pytest.mark.skipif(not CAPS.seccomp, reason="seccomp unavailable")
def test_network_closed_with_sandbox():
    def child():
        apply_seccomp_no_inet()
        # AF_UNIX must keep working (the parent broker relies on it).
        u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        u.close()
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # -> PermissionError

    msg, _ = run_child(child)
    assert "PermissionError" in msg


def test_probe_capabilities_reports_mechanisms():
    caps = guards.probe_capabilities()
    assert isinstance(caps.landlock_abi, int)
    assert isinstance(caps.seccomp, bool)
    assert isinstance(caps.rlimit, bool)
