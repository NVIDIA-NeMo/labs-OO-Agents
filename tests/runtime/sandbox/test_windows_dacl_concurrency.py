# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A DACL edit is read-modify-write, so two of them must not interleave.

``grant`` reads the DACL, adds an ACE, and writes the whole DACL back. Two
grants that interleave therefore lose one of them: both read the same original,
each writes its own version, and the last write wins. The same applies across
processes — two editors each run their own ACP process against one runtime
root — which is why the lock is a named mutex, not only a thread lock.

These tests need no AppContainer and no runtime root: they edit the DACL of a
plain temp directory. They are Windows-only because the functions are.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

pytestmark = [
    pytest.mark.skipif(os.name != "nt", reason="DACL editing is Windows-only"),
]

# Distinct, harmless SIDs. The restricted-code SIDs are never used by a real
# process here; only the DACL text matters.
_SIDS = [
    "S-1-5-21-1111111111-2222222222-3333333333-1001",
    "S-1-5-21-1111111111-2222222222-3333333333-1002",
    "S-1-5-21-1111111111-2222222222-3333333333-1003",
]


def _sid(text: str):
    from nooa.runtime.sandbox.win_api import sid_from_string

    sid = sid_from_string(text)
    assert sid is not None, f"{text} is not a valid SID"
    return sid


def _contains(path: str, text: str) -> bool:
    from nooa.runtime.sandbox.win_api import read_dacl

    return text in read_dacl(path)


def test_concurrent_grants_on_one_path_all_survive(tmp_path):
    """Interleaved grants for different SIDs must not lose an ACE.

    Measured before the lock: 15 rounds of "grant A and revoke A on one thread
    while granting B on another" lost B's ACE 10 times.
    """
    from nooa.runtime.sandbox import win_api

    target = str(tmp_path)
    sids = [_sid(text) for text in _SIDS]

    for round_index in range(20):
        errors: list[BaseException] = []

        def worker(sid, collected=errors):
            try:
                win_api.grant(target, sid, win_api.READ_EXECUTE)
            except BaseException as exc:  # noqa: BLE001 - reported below
                collected.append(exc)

        threads = [threading.Thread(target=worker, args=(sid,)) for sid in sids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, errors
        missing = [text for text in _SIDS if not _contains(target, text)]
        assert not missing, f"round {round_index} lost the ACE for {missing}"

        for sid in sids:
            win_api.revoke(target, sid)
        leftover = [text for text in _SIDS if _contains(target, text)]
        assert not leftover, f"round {round_index} failed to revoke {leftover}"


def test_grant_and_revoke_on_the_same_sid_leave_a_consistent_dacl(tmp_path):
    """A grant racing a revoke must end in one of the two states, never both.

    The failure this guards is a half-written DACL: an ACE that the revoke read
    but did not write back, or a grant that lands after its own revoke.
    """
    from nooa.runtime.sandbox import win_api

    target = str(tmp_path)
    sid = _sid(_SIDS[0])
    errors: list[BaseException] = []

    def grant_loop():
        try:
            for _ in range(30):
                win_api.grant(target, sid, win_api.READ_EXECUTE)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    def revoke_loop():
        try:
            for _ in range(30):
                win_api.revoke(target, sid)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=grant_loop), threading.Thread(target=revoke_loop)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    # Whatever the final state, a revoke afterwards must leave none behind.
    win_api.revoke(target, sid)
    assert not _contains(target, _SIDS[0])


def test_the_mutex_is_created_once_and_shared(tmp_path):
    """The cross-process lock must be one named object, not one per call.

    Creating a fresh mutex per edit would serialise nothing: each process would
    hold a different object.
    """
    from nooa.runtime.sandbox import win_api

    first = win_api._dacl_mutex()
    second = win_api._dacl_mutex()
    assert first is not None, "no DACL mutex could be created on this host"
    assert first == second


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
def test_a_second_process_cannot_edit_between_the_read_and_the_write(tmp_path):
    """The lock must be visible to another process, not just another thread.

    Runs a child that takes the same named mutex and holds it, then asserts this
    process blocks until the child releases it. A thread-only lock would return
    immediately.
    """
    import subprocess
    import textwrap
    import time

    from nooa.runtime.sandbox import win_api

    script = textwrap.dedent(
        """
        import ctypes, sys, time
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        k32.CreateMutexW.restype = wintypes.HANDLE
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.ReleaseMutex.argtypes = [wintypes.HANDLE]

        handle = k32.CreateMutexW(None, False, sys.argv[1])
        if not handle:
            sys.exit(2)
        k32.WaitForSingleObject(handle, 0xFFFFFFFF)
        print("held", flush=True)
        time.sleep(float(sys.argv[2]))
        k32.ReleaseMutex(handle)
        """
    )
    # Ask the module which namespace it actually opened: Global\ needs a
    # privilege an ordinary user lacks, so on most machines this is Local\.
    mutex = win_api._dacl_mutex()
    assert mutex is not None, "no DACL mutex could be created on this host"
    name = win_api._DACL_MUTEX_USED
    assert name, "the module opened a mutex but did not record its name"
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", script, name, "1.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "held", "the child never took the mutex"
        started = time.monotonic()
        win_api._k32.WaitForSingleObject(mutex, win_api.INFINITE)
        win_api._k32.ReleaseMutex(mutex)
        elapsed = time.monotonic() - started
    finally:
        child.kill()
        child.wait(timeout=10)
    assert elapsed > 0.4, (
        f"acquired the mutex in {elapsed:.2f}s while another process held it; "
        "the lock is not shared across processes"
    )


# --- argv quoting ------------------------------------------------------------
def _split_like_windows(command_line: str) -> list[str]:
    """Split ``command_line`` with the same parser the child process uses."""
    import ctypes
    from ctypes import wintypes

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    shell32.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    count = ctypes.c_int()
    argv = shell32.CommandLineToArgvW(command_line, ctypes.byref(count))
    assert argv, f"CommandLineToArgvW rejected {command_line!r}"
    try:
        return [argv[index] for index in range(count.value)]
    finally:
        ctypes.WinDLL("kernel32").LocalFree(argv)


@pytest.mark.parametrize(
    "argument",
    [
        "",
        "plain",
        "with space",
        "with\ttab",
        'embedded "quote"',
        'ends with a quote"',
        "trailing-backslash\\",
        "two-trailing-backslashes\\\\",
        'backslash-before-quote\\"',
        r"C:\Program Files\nooa\runtime",
        "中文 路径",
        r"spaces and \ backslashes \ mixed",
        '"',
        '""',
        "\\",
        "\\\\",
    ],
)
def test_argv_round_trips_through_commandlinetoargvw(argument):
    """What the launcher builds must split back into the arguments it was given.

    A lost empty argument shifts every later one into the wrong slot — measured
    with the launcher's own argv, where a missing workspace made the tokenizer
    cache path land in the crash-log position. Quotes and trailing backslashes
    split the argument or end its quoting early.
    """
    from nooa.runtime.sandbox.win_api import _command_line

    argv = ["prog.exe", argument]
    assert _split_like_windows(_command_line(argv)) == argv


def test_an_empty_argument_keeps_its_slot():
    """The specific failure the launcher hit: an empty workspace argument."""
    from nooa.runtime.sandbox.win_api import _command_line

    argv = ["python.exe", "-I", "worker.py", "", r"C:\cache"]
    split = _split_like_windows(_command_line(argv))
    assert split == argv, f"the empty slot moved: {split}"
    assert split[3] == ""
    assert split[4] == r"C:\cache"


def test_a_denied_global_mutex_falls_back_to_the_session_namespace(monkeypatch):
    r"""``Global\`` needs a privilege an ordinary user lacks; ``Local\`` does not.

    The first version returned ``None`` on the Global refusal, so on any machine
    without SeCreateGlobalPrivilege the edit ran unserialised while the
    docstring said it was locked. A refusal of one namespace says nothing about
    the next.
    """
    import ctypes

    from nooa.runtime.sandbox import win_api

    real = win_api._k32.CreateMutexW
    attempts: list[str] = []

    def refuse_global(attributes, initial, name):
        attempts.append(name)
        if name.startswith("Global"):
            ctypes.set_last_error(win_api.ERROR_ACCESS_DENIED)
            return None
        return real(attributes, initial, name)

    monkeypatch.setattr(win_api._k32, "CreateMutexW", refuse_global)
    monkeypatch.setattr(win_api, "_DACL_MUTEX", None)
    monkeypatch.setattr(win_api, "_DACL_MUTEX_USED", None)

    handle = win_api._dacl_mutex()

    assert attempts == [win_api._DACL_MUTEX_NAME, win_api._DACL_MUTEX_LOCAL_NAME], (
        f"the Global refusal did not fall through to Local: {attempts}"
    )
    assert handle is not None, "no mutex was opened even though Local was available"
    assert win_api._DACL_MUTEX_USED == win_api._DACL_MUTEX_LOCAL_NAME
