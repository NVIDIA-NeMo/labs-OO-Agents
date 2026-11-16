# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OverflowDir.default(): one folder per process, made on first use, fork-safe, None when it cannot be made."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
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
