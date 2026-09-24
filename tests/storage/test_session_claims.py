# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SQLite ownership claim and liveness probe.

These exercise ``SQLiteStorageManager`` and ``is_sqlite_database_active``
directly, independent of any session layer built on top of them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import pytest

import nooa.storage.sqlite as sqlite_storage
from nooa.storage import SessionAlreadyActiveError


def test_concurrent_liveness_probes_never_self_collide(tmp_path):
    """is_sqlite_database_active() must probe with a shared flock, not an
    exclusive one: an exclusive-lock probe races for mutual exclusion
    against *other concurrent probes*, not just against a real owner, and
    reports a healthy, never-opened database as falsely "active". A shared
    lock never conflicts with another shared lock.
    """
    path = tmp_path / "never-opened.db"
    path.touch()
    (path.with_suffix(".lock")).touch()

    results: list[bool] = []
    lock = threading.Lock()

    def probe() -> None:
        active = sqlite_storage.is_sqlite_database_active(path)
        with lock:
            results.append(active)

    for _ in range(20):
        threads = [threading.Thread(target=probe) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not any(results), f"{sum(results)} of {len(results)} probes falsely reported active"


def test_liveness_probe_does_not_break_a_real_opener(tmp_path):
    """A liveness probe running concurrently with a real owner's own
    open/close cycle must not itself cause SessionAlreadyActiveError for
    that legitimate opener.
    """
    db_path = tmp_path / "probe-target.db"
    failures: list[BaseException] = []
    stop = threading.Event()

    def probe_loop() -> None:
        while not stop.is_set():
            sqlite_storage.is_sqlite_database_active(db_path)
            time.sleep(0.0005)

    probers = [threading.Thread(target=probe_loop) for _ in range(4)]
    for t in probers:
        t.start()
    try:
        for _ in range(50):
            try:
                sqlite_storage.SQLiteStorageManager(db_path).close()
                sqlite_storage.delete_sqlite_database(db_path)
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)
                break
    finally:
        stop.set()
        for t in probers:
            t.join()

    assert not failures, failures


def test_claim_owner_is_confirmed_dead_requires_matching_identity(tmp_path):
    """A recorded PID alone is not enough to declare an owner dead: the same
    number can belong to an unrelated live process in a different PID
    namespace (sandbox vs. host) or after a reboot recycles it. The claim
    must also record -- and this check must also match -- the owner's
    PID-namespace and boot identity before trusting os.kill(pid, 0) at all.
    """
    identity = sqlite_storage._owner_identity()
    if identity is None:
        pytest.skip("procfs identity is unavailable")

    claim_path = tmp_path / "claim.active"
    claim_path.mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()

    # Matching identity + a provably dead PID: confirmed dead.
    (claim_path / "owner-match.json").write_text(
        json.dumps({"token": "match", "pid": dead.pid, "identity": identity})
    )
    assert sqlite_storage.claim_owner_is_confirmed_dead(claim_path) is True
    (claim_path / "owner-match.json").unlink()

    # Same dead PID, but a different identity (a different namespace/boot):
    # must refuse to answer rather than risk probing an unrelated live PID.
    mismatched = dict(identity)
    mismatched["boot_id"] = "not-" + str(mismatched["boot_id"])
    (claim_path / "owner-mismatch.json").write_text(
        json.dumps({"token": "mismatch", "pid": dead.pid, "identity": mismatched})
    )
    assert sqlite_storage.claim_owner_is_confirmed_dead(claim_path) is False
    (claim_path / "owner-mismatch.json").unlink()

    # A legacy claim with no identity field at all: also refuse to answer.
    (claim_path / "owner-legacy.json").write_text(json.dumps({"token": "legacy", "pid": dead.pid}))
    assert sqlite_storage.claim_owner_is_confirmed_dead(claim_path) is False


def test_forked_child_close_does_not_remove_parent_claim(tmp_path, monkeypatch):
    """A forked copy cannot release the original process's ownership marker."""
    path = tmp_path / "fork-safe.db"
    manager = sqlite_storage.SQLiteStorageManager(path)
    claim = manager._session_claim
    assert claim is not None
    claim_path = sqlite_storage._claim_path(path)
    parent_pid = os.getpid()

    monkeypatch.setattr(sqlite_storage.os, "getpid", lambda: parent_pid + 1)
    claim.close()
    assert claim_path.is_dir()

    monkeypatch.setattr(sqlite_storage.os, "getpid", lambda: parent_pid)
    manager.close()
    assert not claim_path.exists()


def test_claim_cleanup_failure_still_releases_local_lock(tmp_path, monkeypatch):
    """A marker unlink error cannot leak the manager's flock descriptor."""
    path = tmp_path / "cleanup-failure.db"
    manager = sqlite_storage.SQLiteStorageManager(path)
    claim = manager._session_claim
    assert claim is not None
    owner_path = claim._owner_path
    claim_path = sqlite_storage._claim_path(path)
    original_unlink = sqlite_storage.Path.unlink

    def fail_owner_unlink(target, *args, **kwargs):
        if target == owner_path:
            raise PermissionError("owner cleanup denied")
        return original_unlink(target, *args, **kwargs)

    monkeypatch.setattr(sqlite_storage.Path, "unlink", fail_owner_unlink)
    manager.close()

    assert manager._lock_fd is None
    assert manager._session_claim is None
    # The claim remains fail-closed, but after explicit recovery the released
    # flock must permit a replacement manager in this same process.
    monkeypatch.undo()
    owner_path.unlink()
    claim_path.rmdir()
    replacement = sqlite_storage.SQLiteStorageManager(path)
    replacement.close()


def test_old_owner_does_not_remove_replacement_claim(tmp_path):
    """Token checking keeps stale cleanup from deleting a successor's claim."""
    path = tmp_path / "replaced.db"
    manager = sqlite_storage.SQLiteStorageManager(path)
    claim = manager._session_claim
    assert claim is not None
    claim_path = sqlite_storage._claim_path(path)
    displaced = claim_path.with_name("replaced.displaced")
    claim_path.rename(displaced)
    claim_path.mkdir()
    replacement_owner = claim_path / "owner-replacement.json"
    replacement_owner.write_text('{"token": "replacement", "pid": 456}')

    manager.close()

    assert claim_path.is_dir()
    assert json.loads(replacement_owner.read_text())["token"] == "replacement"
    replacement_owner.unlink()
    claim_path.rmdir()
    next(displaced.iterdir()).unlink()
    displaced.rmdir()


@pytest.mark.parametrize("owner", [{"token": "unknown", "pid": 123}, [], None, "invalid", 7])
def test_orphaned_shared_claim_requires_explicit_recovery(tmp_path, monkeypatch, owner):
    """An orphaned claim never silently admits a potentially live old writer."""
    path = tmp_path / "orphaned.db"
    sqlite_storage.SQLiteStorageManager(path).close()

    claim_path = sqlite_storage._claim_path(path)
    claim_path.mkdir()
    owner_path = claim_path / "owner-unknown.json"
    owner_path.write_text(json.dumps(owner))
    monkeypatch.setattr(sqlite_storage.fcntl, "flock", lambda *_args: None)

    assert sqlite_storage.is_sqlite_database_active(path)
    with pytest.raises(SessionAlreadyActiveError, match="remove .*orphaned.active"):
        sqlite_storage.SQLiteStorageManager(path)

    owner_path.unlink()
    claim_path.rmdir()
    sqlite_storage.SQLiteStorageManager(path).close()
