# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container-side entrypoint, launched by ``win_launcher`` as ``python -I -m``.

It exists as a separate module for one reason: the container process is created
by ``CreateProcessW``, not by ``multiprocessing``, so there is no pickle-based
``target`` to call. Everything after the handle is reconstructed is exactly the
same code the fork path uses — ``worker_main`` builds the namespace, installs
whatever guards remain, and serves the op loop.

Ordering inside the container matters:

1. reconstruct the pipe connection from the inherited handle;
2. verify we really are in a container (a launch outside one means the
   launcher's confinement silently did not apply — refuse to run);
3. tell the parent the baseline memory, so it can translate the configured
   headroom into an absolute Job Object ceiling;
4. build the namespace and serve.

Step 3 happens before the namespace build on purpose: the baseline must describe
the *bare interpreter*, so that importing the agent module and its dependencies
counts against the cell's budget rather than against the baseline.
"""

from __future__ import annotations

import os
import sys


def _current_rss_bytes() -> int:
    """Private working set of this process, via psapi (no psutil dependency)."""
    import ctypes
    from ctypes import wintypes

    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
    except OSError:
        return 0

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    # Without explicit argtypes ctypes marshals the handle as a C int, and a
    # 64-bit process handle overflows it (ArgumentError: int too long).
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        k32.GetCurrentProcess(), ctypes.byref(counters), ctypes.sizeof(counters)
    ):
        return 0
    return int(counters.WorkingSetSize)


def main() -> int:
    if len(sys.argv) < 2:
        _crash_log(
            "usage: python -m nooa.runtime.sandbox.win_worker "
            "<pipe-handle> [name] [log] [tokenizer-cache]"
        )
        return 2
    handle = int(sys.argv[1])
    expected_container = sys.argv[2] if len(sys.argv) > 2 else None
    # The launcher creates this process with no inherited std handles, so a
    # traceback printed to stderr goes nowhere. A crash log in the (granted)
    # workspace is the only channel that survives a startup failure.
    crash_path = sys.argv[3] if len(sys.argv) > 3 else None
    # Set from argv, before any import: argv exists from interpreter startup,
    # whereas a value sent over the pipe only arrives after ``_run`` connects.
    # Any import chain that touches tiktoken at module scope would otherwise
    # read the environment before the path was known.
    #
    # ``CUSTOM_TIKTOKEN_CACHE_DIR`` is what matters, not ``TIKTOKEN_CACHE_DIR``:
    # litellm overwrites the latter at import time to point at its bundled
    # tokenizers, and only honors the former. Setting both keeps plain tiktoken
    # users on the same directory.
    if len(sys.argv) > 4 and sys.argv[4]:
        os.environ["TIKTOKEN_CACHE_DIR"] = sys.argv[4]
        os.environ["CUSTOM_TIKTOKEN_CACHE_DIR"] = sys.argv[4]
    try:
        return _run(handle, expected_container, crash_path)
    except BaseException:  # noqa: BLE001 - the report is the whole point
        import traceback

        _crash_log(traceback.format_exc(), crash_path)
        return 9


def _cache_entries(path: str | None) -> str:
    """List a cache directory's entries, or say why it cannot be read."""
    if not path:
        return "unset"
    try:
        return repr(sorted(os.listdir(path)))
    except OSError as exc:
        return f"unreadable ({type(exc).__name__}: {exc})"


def _cache_hash_facts(path: str | None) -> str:
    """Compare the cached file's digest with this interpreter's expectation.

    ``read_file_cached`` deletes a mismatching entry and re-downloads, so a
    stale or foreign cache entry produces exactly the same symptom as a missing
    one. The filename is only ``sha1(blob-url)``, which both sides agree on —
    the contents are what can disagree.
    """
    if not path:
        return "unset"
    try:
        import hashlib

        entries = sorted(os.listdir(path))
    except OSError as exc:
        return f"unlistable ({type(exc).__name__}: {exc})"
    if not entries:
        return "empty"

    actual = ""
    for name in entries:
        try:
            with open(os.path.join(path, name), "rb") as handle:
                actual = hashlib.sha256(handle.read()).hexdigest()
        except OSError as exc:
            return f"unreadable ({type(exc).__name__}: {exc})"
        break

    expected = ""
    try:
        import inspect
        import re

        from tiktoken_ext import openai_public

        source = inspect.getsource(openai_public.cl100k_base)
        matched = re.search(r"expected_hash=\"([0-9a-f]{64})\"", source)
        expected = matched.group(1) if matched else ""
    except Exception as exc:  # noqa: BLE001 - diagnostics must never fail hard
        return f"actual={actual} expected=unavailable({type(exc).__name__})"
    verdict = "match" if actual and actual == expected else "MISMATCH"
    return f"actual={actual} expected={expected} {verdict}"


def _cache_readable(path: str | None) -> str:
    """Open each cache file and report which ones this process can read.

    ``os.path.isdir`` and ``os.listdir`` can both succeed on a directory whose
    entries are still unreadable to this token; ``open`` is the only check that
    matches what tiktoken will actually do.
    """
    if not path:
        return "unset"
    try:
        entries = sorted(os.listdir(path))
    except OSError as exc:
        return f"unlistable ({type(exc).__name__}: {exc})"
    results = []
    for name in entries:
        try:
            with open(os.path.join(path, name), "rb") as handle:
                handle.read(1)
            results.append(f"{name}=readable")
        except OSError as exc:
            results.append(f"{name}=DENIED({type(exc).__name__})")
    return ", ".join(results) if results else "empty"


def _crash_log(message: str, path: str | None = None) -> None:
    if path:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
            return
        except OSError:
            pass
    try:
        print(message, file=sys.stderr)
    except Exception:
        pass


def _run(handle: int, expected_container: str | None, crash_path: str | None = None) -> int:
    from multiprocessing.connection import PipeConnection

    from nooa.runtime.sandbox import wire
    from nooa.runtime.sandbox.win_api import app_container_sid, is_app_container

    try:
        conn = PipeConnection(handle, readable=True, writable=True)
    except OSError as exc:
        _crash_log(f"win_worker: cannot adopt pipe handle {handle}: {exc}")
        return 3

    # Refuse to run unconfined. If the launcher's SECURITY_CAPABILITIES was
    # dropped (a test harness, a copy-pasted argv), every cell would execute
    # with the parent's full permissions while the caller believes otherwise —
    # fail loudly instead of being quietly wrong.
    try:
        if not is_app_container():
            _send_fatal(conn, wire, "process is not in an AppContainer")
            return 4
        actual = app_container_sid()
        if expected_container and actual is None:
            _send_fatal(conn, wire, "container SID is unreadable")
            return 4
    except OSError as exc:
        _send_fatal(conn, wire, f"token query failed: {exc}")
        return 4

    from nooa.runtime.sandbox.worker import worker_main

    try:
        init = conn.recv()  # parent -> worker: the parent is trusted
    except (EOFError, OSError) as exc:
        _crash_log(f"win_worker: no init message: {exc}")
        return 5

    # ``-I`` strips cwd and PYTHONPATH, and the container is confined from
    # creation, so the launcher must grant and name the agent's import roots
    # explicitly. Inserting them here — before the blueprint imports the agent
    # class — is the only point where both are true.
    for root in reversed(list(init.get("import_roots") or [])):
        if root not in sys.path:
            sys.path.insert(0, root)

    # Tokenizer data cannot be fetched here (no network), so the parent
    # pre-populates a granted directory and passes it on argv (set in ``main``
    # before any import). Report what this process actually sees over the pipe.
    # The workspace crash log is not a reliable channel: tests create it under
    # tmp_path and delete it when the test ends, so CI never sees the diagnosis.
    _cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    wire.send(
        conn,
        {
            "type": "startup",
            "tokenizer_cache": _cache_dir,
            "tokenizer_cache_exists": bool(_cache_dir) and os.path.isdir(_cache_dir),
            "tokenizer_cache_entries": _cache_entries(_cache_dir),
            # "exists" and "readable" are different questions, and the gap
            # between them is exactly what made this fail silently: tiktoken
            # calls os.path.exists, which reports False on a denied stat.
            "tokenizer_cache_readable": _cache_readable(_cache_dir),
            "tokenizer_cache_hash": _cache_hash_facts(_cache_dir),
            "argv": list(sys.argv),
        },
    )

    # ``ready`` is sent from the callback, not here: the parent's first-cell
    # deadline must not include this process's cold import cost. The memory
    # baseline is likewise taken *after* the namespace build, because the
    # configured headroom is relative to a worker that is ready to run cells —
    # the same baseline the Linux backend inherits through ``fork``.
    def _on_ready() -> None:
        wire.send(
            conn,
            {
                "type": "ready",
                "baseline_bytes": _current_rss_bytes(),
                "container_sid": actual,
                "pid": os.getpid(),
            },
        )

    worker_main(conn, init, on_ready=_on_ready)
    return 0


def _send_fatal(conn, wire, message: str) -> None:
    try:
        wire.send(conn, {"type": "fatal", "error": message})
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
