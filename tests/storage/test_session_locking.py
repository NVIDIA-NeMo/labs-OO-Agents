# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session ownership through real spawned processes on every supported platform."""

import multiprocessing
import os
import socket

import pytest

from nooa.storage import (
    SessionAlreadyActiveError,
    SQLiteStorageManager,
    delete_sqlite_database,
    read_lock_owner,
)


def _hold_session(path, ready, release):
    with SQLiteStorageManager(path) as storage:
        storage.save_snapshot_json('{"saved": true}')
        ready.set()
        if not release.wait(30):
            raise TimeoutError("Parent did not release the test session")


@pytest.mark.parametrize("crash", [False, True])
def test_session_excludes_other_processes_and_recovers_after_exit(tmp_path, crash):
    from tests.storage.test_session_locking import _hold_session as holder

    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    path = tmp_path / "会话 with spaces.db"
    lock_path = str(path.with_suffix(".lock"))
    child = ctx.Process(target=holder, args=(path, ready, release))
    try:
        child.start()
        assert ready.wait(15), "Child did not acquire the session"
        assert read_lock_owner(lock_path) == (child.pid, socket.gethostname())

        with pytest.raises(SessionAlreadyActiveError) as error:
            SQLiteStorageManager(path)
        assert error.value.owner_pid == child.pid
        assert read_lock_owner(lock_path)[0] == child.pid

        with pytest.raises(SessionAlreadyActiveError):
            delete_sqlite_database(path)
        assert path.exists()

        if crash:
            child.terminate()
        else:
            release.set()
        child.join(15)
        assert not child.is_alive()
        if not crash:
            assert child.exitcode == 0
            assert read_lock_owner(lock_path) == (None, None)

        with SQLiteStorageManager(path) as reopened:
            assert read_lock_owner(lock_path) == (os.getpid(), socket.gethostname())
            assert reopened.get_latest_snapshot_id() is not None
        assert read_lock_owner(lock_path) == (None, None)
        assert delete_sqlite_database(path)
        assert path.with_suffix(".lock").exists()
        assert read_lock_owner(lock_path) == (None, None)
    finally:
        if child.pid is not None:
            # Terminating a waiter can leave multiprocessing.Event's condition
            # counters incomplete. Do not set that event after a forced exit.
            if child.is_alive():
                child.terminate()
                child.join(5)


def test_two_handles_in_one_process_cannot_own_the_same_session(tmp_path):
    path = tmp_path / "session.db"
    with SQLiteStorageManager(path):
        with pytest.raises(SessionAlreadyActiveError) as error:
            SQLiteStorageManager(path)
        assert error.value.owner_pid == os.getpid()
    with SQLiteStorageManager(path):
        pass
