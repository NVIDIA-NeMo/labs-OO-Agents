# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Python on Linux/macOS with the POSIX-only stdlib hidden, as on native Windows.

Usage::

    uv run --frozen python scripts/winsim.py -c "import nooa"
    uv run --frozen python scripts/winsim.py path/to/script.py [args...]

What it does before running the code:

* blocks POSIX-only modules (``sys.modules[name] = None``: fcntl, resource,
  termios, pty, pwd, grp, tty, readline, curses, ...), so importing them raises
  ``ImportError`` exactly like on Windows;
* deletes the ``signal`` constants and functions Windows lacks (every ``SIG*``
  except SIGABRT/SIGFPE/SIGILL/SIGINT/SIGSEGV/SIGTERM/SIGBREAK, plus alarm,
  setitimer, pthread_kill, ...);
* deletes the ``os`` functions Windows lacks (fork, killpg, getpgid, setsid,
  register_at_fork, getuid, ...);
* sets ``sys.platform = "win32"``.

It does NOT fake ``os.name``, ``os.sep``, path semantics, the event loop
(Proactor), ``subprocess``/``multiprocessing`` internals or file locking, so it
only proves "no POSIX-only *name* is touched on this code path". Stdlib modules
that branch on ``sys.platform`` at import time (asyncio, subprocess, ...) are
imported *before* the switch so they keep their POSIX internals.
"""

from __future__ import annotations

import runpy
import sys

BLOCKED_MODULES = (
    "fcntl",
    "resource",
    "termios",
    "tty",
    "pty",
    "pwd",
    "grp",
    "readline",
    "curses",
    "_curses",
    "syslog",
    "posix_ipc",
    "multiprocessing.popen_fork",
    "multiprocessing.popen_forkserver",
)

# signal names that exist on Windows (CPython signalmodule.c)
WINDOWS_SIGNALS = {"SIGABRT", "SIGFPE", "SIGILL", "SIGINT", "SIGSEGV", "SIGTERM", "SIGBREAK"}
POSIX_SIGNAL_FUNCS = (
    "alarm",
    "pause",
    "setitimer",
    "getitimer",
    "siginterrupt",
    "pthread_kill",
    "pthread_sigmask",
    "sigpending",
    "sigwait",
    "sigwaitinfo",
    "sigtimedwait",
    "pidfd_send_signal",
    "ITIMER_REAL",
    "ITIMER_VIRTUAL",
    "ITIMER_PROF",
    "SIG_BLOCK",
    "SIG_UNBLOCK",
    "SIG_SETMASK",
)
POSIX_OS_FUNCS = (
    "fork",
    "forkpty",
    "killpg",
    "getpgid",
    "setpgid",
    "getpgrp",
    "setpgrp",
    "setsid",
    "getsid",
    "register_at_fork",
    "getuid",
    "geteuid",
    "getgid",
    "getegid",
    "setuid",
    "setgid",
    "getgroups",
    "getlogin_r",
    "wait",
    "wait3",
    "wait4",
    "WNOHANG",
    "WUNTRACED",
    # sched_getaffinity is kept: Linux os.process_cpu_count() calls it internally.
    "sched_setaffinity",
    "sysconf",
    "confstr",
    "pathconf",
    "fpathconf",
    "uname",
    "getloadavg",
    "mkfifo",
    "mknod",
    "chown",
    "fchown",
    "lchown",
    "statvfs",
    "fstatvfs",
    "nice",
    "openpty",
    "ttyname",
    "ctermid",
    "posix_spawn",
    "posix_spawnp",
    "pidfd_open",
    "memfd_create",
    "eventfd",
    "splice",
    "sendfile",
)
# Imported before sys.platform flips so their platform branches stay POSIX.
PREIMPORT = (
    "os",
    "signal",
    "subprocess",
    "asyncio",
    "asyncio.subprocess",
    "multiprocessing",
    "multiprocessing.connection",
    "multiprocessing.process",
    "multiprocessing.context",
    "socket",
    "selectors",
    "threading",
    "tempfile",
    "shutil",
    "platform",
    "sqlite3",
    "concurrent.futures",
    "ssl",
    "uuid",
    "ctypes",
    "locale",
    "mimetypes",
    "getpass",
    "webbrowser",
    "pathlib",
    "logging.handlers",
    "http.client",
    "urllib.request",
    "faulthandler",
    "zoneinfo",
    "importlib.metadata",
)
# Third-party packages with their own ``sys.platform == "win32"`` branches that
# import Windows-only modules (msvcrt, winreg, ...). They support Windows
# themselves; importing them first keeps the simulation about nooa's code.
PREIMPORT_THIRD_PARTY = ("click", "mcp", "mcp.client.stdio")


def install() -> None:
    import importlib
    import os
    import signal

    for name in PREIMPORT + PREIMPORT_THIRD_PARTY:
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    for name in BLOCKED_MODULES:
        sys.modules[name] = None  # type: ignore[assignment]
    for name in list(vars(signal)):
        if name.startswith("SIG") and not name.startswith("SIG_") and name not in WINDOWS_SIGNALS:
            delattr(signal, name)
    for name in POSIX_SIGNAL_FUNCS:
        if hasattr(signal, name):
            delattr(signal, name)
    for name in POSIX_OS_FUNCS:
        if hasattr(os, name):
            delattr(os, name)
    # Stdlib code that checks sys.platform at *call* time and would reach
    # Windows-only C functions that do not exist on Linux: neutralise it.
    import ssl

    ssl.SSLContext._load_windows_store_certs = lambda self, storename, purpose: None  # type: ignore[attr-defined]
    sys.platform = "win32"


def main(argv: list[str]) -> None:
    install()
    if len(argv) >= 2 and argv[0] == "-c":
        sys.argv = ["-c", *argv[2:]]
        code = argv[1]
        namespace = {"__name__": "__main__"}
        exec(compile(code, "<winsim -c>", "exec"), namespace)  # noqa: S102
    elif argv:
        sys.argv = list(argv)
        runpy.run_path(argv[0], run_name="__main__")
    else:
        raise SystemExit("usage: winsim.py -c CODE | winsim.py SCRIPT [ARGS...]")


if __name__ == "__main__":
    main(sys.argv[1:])
