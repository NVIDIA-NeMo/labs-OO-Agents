# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guardrail leak-vs-closed pairs for the OS-enforced sandbox primitives.

Each guardrail gets two tests forked into child processes: one shows the exploit
succeeding *without* the guard (the leak), the paired one shows the *same*
exploit refused *with* the guard installed. The guards themselves live in
``nooa.runtime.sandbox.guards`` and are applied by a child on itself, so these
tests exercise the exact code path the worker uses.

The environment tests are the exception: they run in a fresh interpreter
instead of a fork. They check the environment block fixed at ``exec``, which a
fork would inherit from whatever environment pytest was launched with.
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
# These tests depend on the exec-time environment block, which pytest's own
# process inherits from whoever launched it. Each one therefore runs in a fresh
# interpreter exec'd with a known block: PATH plus a sentinel variable.
_SENTINEL = "NOOA_TEST_ENV_SENTINEL"

_ENV_PRELUDE = """
import os, subprocess, sys
from nooa.runtime.sandbox import guards

def exec_time_env_keys(raw=None):
    if raw is None:
        with open("/proc/self/environ", "rb") as fh:
            raw = fh.read()
    return {e.split(b"=", 1)[0].decode() for e in raw.split(b"\\0") if b"=" in e}
"""


def run_in_controlled_env(body: str) -> None:
    """Run ``body`` in a new interpreter whose exec-time environment is known.

    The child sees only ``PATH`` and the sentinel in ``/proc/self/environ``,
    and imports from the same ``sys.path`` as this process.
    ``os.environ`` may additionally hold variables Python sets at startup
    (e.g. ``LC_CTYPE`` from locale coercion), so the checks key off the block.
    """
    import subprocess
    import textwrap

    env = {"PATH": os.environ.get("PATH", os.defpath), _SENTINEL: "1"}
    # pytest's ``pythonpath`` setting does not reach the child. Hand over this
    # process's import path in the script rather than PYTHONPATH, which would
    # add a variable to the block under test.
    import_path = f"import sys\nsys.path[:0] = {sys.path!r}\n"
    script = import_path + _ENV_PRELUDE + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/environ is Linux-only")
def test_env_leak_without_allowlist():
    # A fork sees the parent's exec-time environment both ways a cell would look.
    run_in_controlled_env(f"""
        assert exec_time_env_keys() == {{"PATH", "{_SENTINEL}"}}, exec_time_env_keys()
        pid = os.fork()
        if pid == 0:
            ok = "{_SENTINEL}" in exec_time_env_keys() and "{_SENTINEL}" in os.environ
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0, "fork did not inherit the variable"
    """)


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/environ is Linux-only")
def test_env_closed_with_allowlist():
    run_in_controlled_env("""
        path = os.environ["PATH"]
        guards.apply_env_allowlist(("PATH",))
        assert set(os.environ) == {"PATH"}, sorted(os.environ)
        assert os.environ["PATH"] == path
        assert exec_time_env_keys() == set(), sorted(exec_time_env_keys())
        # Subprocesses get the allowlisted environment, not the original block.
        # Read the child's exec-time block, which is exactly what it was given.
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.stdout.buffer.write(open('/proc/self/environ', 'rb').read())"],
            capture_output=True, check=True,
        ).stdout
        assert exec_time_env_keys(out) == {"PATH"}, out
    """)


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/environ is Linux-only")
def test_env_allowlist_keeps_allowed_variable_set_from_c():
    # A C library's setenv() reaches libc's environment but not os.environ.
    run_in_controlled_env("""
        import ctypes
        libc = ctypes.CDLL(None)
        libc.getenv.restype = ctypes.c_char_p
        libc.setenv(b"FROM_C", b"kept", 1)
        libc.setenv(b"FROM_C_DENIED", b"dropped", 1)
        assert "FROM_C" not in os.environ
        guards.apply_env_allowlist(("PATH", "FROM_C"))
        assert os.environ.get("FROM_C") == "kept", sorted(os.environ)
        assert libc.getenv(b"FROM_C") == b"kept"
        assert libc.getenv(b"FROM_C_DENIED") is None
    """)


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/environ is Linux-only")
def test_env_allowlist_fails_closed_when_block_not_zeroed():
    # No usable env_start/env_end (e.g. a /proc that reports zeros): refuse to run.
    # The controlled block is non-empty, so there is something left to detect.
    run_in_controlled_env("""
        guards._env_block_bounds = lambda: (0, 0)
        try:
            guards.apply_env_allowlist(("PATH",))
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")
    """)


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
