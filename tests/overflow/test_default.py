# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OverflowDir.default(): one folder per process, made on first use, fork-safe, None when it cannot be made."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
from pathlib import Path

import pytest

from nooa.overflow import OverflowDir
from nooa.overflow import _overflow as impl

needs_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(impl, "_default", None)
    monkeypatch.setattr(impl, "_default_failed", False)
    yield tmp_path
    if impl._default is not None:
        impl._default.close()


def test_made_on_first_use_and_shared(fresh) -> None:
    assert list(fresh.iterdir()) == []
    folder = OverflowDir.default()
    assert folder is not None and OverflowDir.default() is folder
    assert Path(folder.path).parent == fresh
    assert (folder.max_files, folder.file_bytes) == (
        impl.DEFAULT_MAX_FILES,
        impl.DEFAULT_FILE_BYTES,
    )


def test_closed_default_is_replaced(fresh) -> None:
    first = OverflowDir.default()
    assert first is not None
    first.close()
    second = OverflowDir.default()
    assert second is not None and second is not first and not second.closed


def test_none_when_it_cannot_be_made_logged_once_then_retried(fresh, monkeypatch, caplog) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(fresh / "missing"))
    with caplog.at_level(logging.WARNING, logger="nooa.overflow._overflow"):
        assert OverflowDir.default() is None
        assert OverflowDir.default() is None
    assert len([r for r in caplog.records if "default overflow folder" in r.message]) == 1
    (fresh / "missing").mkdir()
    folder = OverflowDir.default()  # a later call tries again
    assert folder is not None and Path(folder.path).parent == fresh / "missing"


@needs_fork
@pytest.mark.parametrize("held", [False, True])
def test_a_forked_child_makes_its_own(tmp_path: Path, held: bool) -> None:
    code = f"""
        import json, os, threading, time
        from nooa.overflow import OverflowDir
        from nooa.overflow import _overflow as impl

        parent = OverflowDir.default()
        ready, release = threading.Event(), threading.Event()

        def locker():
            with impl._default_lock:
                ready.set()
                release.wait()

        thread = None
        if {held!r}:
            thread = threading.Thread(target=locker)
            thread.start()
            ready.wait()
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:
            child = OverflowDir.default()
            writer = child.open() if child is not None else None
            os.write(write, json.dumps(
                [child is not None and child.path != parent.path, writer is not None]
            ).encode())
            os._exit(0)
        os.close(write)
        release.set()
        if thread is not None:
            thread.join()
        deadline = time.monotonic() + 10
        done = False
        while not done and time.monotonic() < deadline:
            done = os.waitpid(pid, os.WNOHANG)[0] != 0
            time.sleep(0.05)
        if not done:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            print(json.dumps("blocked"))
        else:
            print(os.read(read, 1000).decode())
        parent.close()
    """
    result = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [True, True]


def test_a_removal_warning_under_the_default_lock_does_not_deadlock(fresh, monkeypatch) -> None:
    # #580 review (CodeRabbit r4627396374): ``default()`` holds ``_default_lock``
    # while it makes the folder, and the exit cleanup may begin in between (its
    # ``_exiting`` check read False, then the constructor finds it True). The
    # constructor then removes the folder it just made and warns when that fails
    # -- after the folder's lock, but still under ``_default_lock``. A logging
    # handler that calls ``OverflowDir.default()`` waited on a lock its own
    # thread held. The constructor takes the caller's sink now, so the warning
    # is emitted once ``_default_lock`` is released.
    real_rmtree = shutil.rmtree
    real_folder = impl._Folder

    def broken_rmtree(path, **kwargs):
        if not str(path).startswith(str(fresh)):
            return real_rmtree(path, **kwargs)

        recall = kwargs.get("onexc")  # the folder's own "could not remove" report
        if recall is not None:
            recall(os.unlink, path, OSError("busy"))
        return None

    class ExitStartsHere(real_folder):
        def __init__(self, max_files: int, file_bytes: int) -> None:
            super().__init__(max_files, file_bytes)
            monkeypatch.setattr(impl, "_exiting", True)  # the exit cleanup begins now

    monkeypatch.setattr(impl, "_exiting", False)
    monkeypatch.setattr(impl, "_Folder", ExitStartsHere)
    monkeypatch.setattr(shutil, "rmtree", broken_rmtree)
    records: list[str] = []
    reentered: list[bool] = []

    class BackIntoDefault(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            records.append(message)
            if "could not remove" in message and not reentered:
                reentered.append(True)
                OverflowDir.default()  # waits on _default_lock
                records.append("came back")

    logger = logging.getLogger("nooa.overflow._overflow")
    handler = BackIntoDefault()
    logger.addHandler(handler)
    made: list[OverflowDir | None] = []
    try:
        caller = threading.Thread(target=lambda: made.append(OverflowDir.default()), daemon=True)
        caller.start()
        caller.join(timeout=30)
        assert not caller.is_alive()  # was a deadlock
    finally:
        logger.removeHandler(handler)
        monkeypatch.setattr(shutil, "rmtree", real_rmtree)  # not the folder's own cleanup
        if impl._default is not None:
            real_rmtree(impl._default.path, ignore_errors=True)
            monkeypatch.setattr(impl, "_default", None)
    assert "came back" in records  # the handler's own call finished
    assert any("could not remove" in message for message in records)
    assert made and made[0] is not None and made[0].closed  # made, then removed at once
