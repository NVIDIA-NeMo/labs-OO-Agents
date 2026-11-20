# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real owned-connection coordinated transactions; ordinary rollback, not crash proof."""

import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import pytest

from nooa.context_blocks import Metadata
from nooa.runtime.event_manager import EventManager
from nooa.storage.sqlite import (
    SQLiteStorageManager,
    _append_coordinated_batches,
    _UnsupportedCoordinatedBatch,
)


class InstrumentedConnection(sqlite3.Connection):
    failure = None
    stage = None
    detach_failure = False
    attach_hook = None

    def execute(self, sql, *args):
        if self.detach_failure and sql.startswith("DETACH"):
            raise RuntimeError("cleanup sentinel")
        if self.stage == "begin" and sql.startswith("BEGIN IMMEDIATE"):
            raise self.failure
        result = super().execute(sql, *args)
        if sql.startswith("ATTACH") and self.attach_hook is not None:
            hook, self.attach_hook = self.attach_hook, None
            hook()
        return result

    def executemany(self, sql, *args):
        if self.stage == "write" and 'nooa_batch_1".active_tags' in sql:
            raise self.failure
        return super().executemany(sql, *args)

    def commit(self):
        if self.stage == "commit":
            raise self.failure
        return super().commit()


@pytest.fixture
def stores(tmp_path, monkeypatch):
    original = sqlite3.connect

    def connect(*args, **kwargs):
        return original(*args, **kwargs, factory=InstrumentedConnection)

    monkeypatch.setattr(sqlite3, "connect", connect)
    with ExitStack() as stack:
        yield [
            stack.enter_context(
                SQLiteStorageManager(
                    tmp_path / f"s{i}.db",
                    journal_mode="delete",
                    check_same_thread=False,
                )
            )
            for i in range(3)
        ]


def batches(stores):
    return [(s, [Metadata(description="one"), Metadata(description="two")]) for s in stores]


def state(stores):
    return [
        (
            s._conn.execute("SELECT * FROM events ORDER BY insertion_order").fetchall(),
            s._conn.execute("SELECT * FROM active_tags ORDER BY position").fetchall(),
            s._backend._next_tag_num,
            s._backend._insertion_counter,
        )
        for s in stores
    ]


def assert_untouched(stores, batch, before):
    assert state(stores) == before
    assert all(event.tag is None for _, events in batch for event in events)


def test_actual_commit_events_tags_counters_no_observers(stores):
    seen = []
    for s in stores:
        EventManager(backend=s.event_backend).on("*", seen.append)
    batch = batches(stores)
    assert _append_coordinated_batches(batch) == (["1", "2"],) * 3
    assert seen == []
    for s, events in batch:
        assert list(s.event_backend.all_events()) == events
        assert s.event_backend.active_tags() == ["1", "2"]
        assert (s._backend._next_tag_num, s._backend._insertion_counter) == (3, 2)
        assert len(s._conn.execute("PRAGMA database_list").fetchall()) == 1
    print(
        "SQLITE",
        sqlite3.sqlite_version,
        "LIMIT",
        stores[0]._conn.getlimit(sqlite3.SQLITE_LIMIT_ATTACHED),
    )


@pytest.mark.parametrize("table", ["events", "active_tags"])
def test_second_file_trigger_rolls_back_every_file_same_inputs_retry(stores, table):
    stores[1]._conn.execute(
        f"CREATE TRIGGER fail BEFORE INSERT ON {table} BEGIN SELECT RAISE(ABORT, 'sentinel'); END"
    )
    batch, before = batches(stores), state(stores)
    with pytest.raises(sqlite3.IntegrityError):
        _append_coordinated_batches(batch)
    assert_untouched(stores, batch, before)
    stores[1]._conn.execute("DROP TRIGGER fail")
    assert _append_coordinated_batches(batch) == (["1", "2"],) * 3


@pytest.mark.parametrize("stage", ["serialize", "setter", "begin", "write", "commit"])
@pytest.mark.parametrize("control", [False, True])
def test_original_error_or_control_identity_even_when_detach_fails(
    stores, monkeypatch, stage, control
):
    class Control(BaseException):
        pass

    failure = Control("control") if control else RuntimeError("error")
    batch, before = batches(stores), state(stores)
    coordinator = stores[0]._conn
    coordinator.detach_failure = True
    if stage == "serialize":

        def fail(event):
            raise failure

        monkeypatch.setattr(stores[1]._backend, "_serialize", fail)
    elif stage == "setter":

        class Reject(Metadata):
            def __setattr__(self, name, value):
                if name == "tag" and value is not None:
                    raise failure
                super().__setattr__(name, value)

        batch[1][1][1] = Reject(description="setter")
    else:
        coordinator.stage, coordinator.failure = stage, failure
    with pytest.raises(type(failure)) as caught:
        _append_coordinated_batches(batch)
    assert caught.value is failure
    assert_untouched(stores, batch, before)
    assert stores[0]._batch_cleanup_errors == ("detach-cleanup",)
    coordinator.stage = None
    coordinator.detach_failure = False
    for _, name, _ in coordinator.execute("PRAGMA database_list").fetchall():
        if name not in {"main", "temp"}:
            coordinator.execute(f'DETACH DATABASE "{name}"')


def test_success_detach_failure_is_success_with_retained_diagnostic(stores):
    stores[0]._conn.detach_failure = True
    batch = batches(stores)
    assert _append_coordinated_batches(batch) == (["1", "2"],) * 3
    assert stores[0]._batch_cleanup_errors == ("detach-cleanup",)
    before, again = state(stores), batches(stores)
    with pytest.raises(_UnsupportedCoordinatedBatch, match="existing-attachments"):
        _append_coordinated_batches(again)
    assert_untouched(stores, again, before)
    stores[0]._conn.detach_failure = False


def test_real_commit_busy_rolls_back_all_files(stores):
    reader = sqlite3.connect(stores[1].path)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM events").fetchall()
        batch, before = batches(stores), state(stores)
        with pytest.raises(sqlite3.OperationalError):
            _append_coordinated_batches(batch)
        assert_untouched(stores, batch, before)
        assert not stores[0]._conn.in_transaction
    finally:
        reader.rollback()
        reader.close()


@pytest.mark.parametrize(
    "invalid",
    [
        "wal",
        "sync",
        "active",
        "closed",
        "raw_closed",
        "thread",
        "attachments",
        "identity",
        "duplicate",
        "memory",
    ],
)
def test_invalid_layout_no_tags_or_other_file_writes(stores, tmp_path, invalid):
    s = stores[1]
    batch = batches(stores)
    first_before = state([stores[0]])
    counters = [(v._backend._next_tag_num, v._backend._insertion_counter) for v in stores]
    extra = None
    if invalid == "wal":
        s._conn.execute("PRAGMA journal_mode=WAL")
    elif invalid == "sync":
        s._conn.execute("PRAGMA synchronous=NORMAL")
    elif invalid == "active":
        s._conn.execute("BEGIN")
    elif invalid == "closed":
        s.close()
    elif invalid == "raw_closed":
        s._conn.close()  # deliberately NO flag before the operation
    elif invalid == "thread":
        s._check_same_thread = True
    elif invalid == "attachments":
        s._conn.execute("ATTACH DATABASE ':memory:' AS unrelated")
    elif invalid == "identity":
        s._backend._lock = threading.RLock()
    elif invalid == "duplicate":
        batch[1] = (stores[0], batch[1][1])
    else:
        extra = SQLiteStorageManager(":memory:", check_same_thread=False)
        batch[1] = (extra, batch[1][1])
    try:
        with pytest.raises((sqlite3.ProgrammingError, _UnsupportedCoordinatedBatch)):
            _append_coordinated_batches(batch)
        assert state([stores[0]]) == first_before
        assert [
            (v._backend._next_tag_num, v._backend._insertion_counter) for v in stores
        ] == counters
        assert all(e.tag is None for _, events in batch for e in events)
    finally:
        if invalid == "raw_closed":
            s._closed = True  # teardown only AFTER actual raw-closed proof
        elif invalid == "active":
            s._conn.rollback()
        if extra:
            extra.close()


def test_different_device_safe_mock_not_real_fs_proof(stores, monkeypatch):
    original = Path.stat

    def stat(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == stores[1]._canonical_path:
            values = list(value)
            values[2] += 1
            return os.stat_result(values)
        return value

    monkeypatch.setattr(Path, "stat", stat)
    value = stores[1]._canonical_path.stat()
    stores[1]._file_identity = (value.st_dev, value.st_ino)
    batch, before = batches(stores), state(stores)
    with pytest.raises(_UnsupportedCoordinatedBatch, match="same-device"):
        _append_coordinated_batches(batch)
    assert_untouched(stores, batch, before)


def test_actual_capacity_boundary_and_rejection(tmp_path):
    with ExitStack() as stack:
        first = stack.enter_context(
            SQLiteStorageManager(tmp_path / "00.db", journal_mode="delete", check_same_thread=False)
        )
        limit = first._conn.getlimit(sqlite3.SQLITE_LIMIT_ATTACHED)
        stores = [first] + [
            stack.enter_context(
                SQLiteStorageManager(
                    tmp_path / f"{i:02}.db", journal_mode="delete", check_same_thread=False
                )
            )
            for i in range(1, limit + 2)
        ]
        assert _append_coordinated_batches(batches(stores[:-1])) == (["1", "2"],) * (limit + 1)
        batch, before = batches(stores), state(stores)
        with pytest.raises(_UnsupportedCoordinatedBatch, match="attachment-capacity"):
            _append_coordinated_batches(batch)
        assert_untouched(stores, batch, before)
        print("SUPPORTED_FILES", limit + 1, "REJECTED_FILES", limit + 2)


@pytest.mark.parametrize("restore", [False, True])
def test_original_create_restore_uri_special_path_reconnect_cwd(tmp_path, monkeypatch, restore):
    folder = tmp_path / "original"
    folder.mkdir()
    monkeypatch.chdir(folder)
    name = "literal ?#% ü.db"
    if restore:
        with SQLiteStorageManager(name, journal_mode="delete"):
            pass
    with SQLiteStorageManager(
        name, must_exist=restore, journal_mode="delete", check_same_thread=False
    ) as s:
        assert s.path == Path(name)  # public spelling unchanged
        frozen = s._canonical_path
        monkeypatch.chdir(tmp_path)
        s._reconnect()
        assert s._canonical_path == frozen
        assert s._conn.execute("PRAGMA database_list").fetchone()[2] == str(frozen)
        missing = tmp_path / "absent?#.db"
        with pytest.raises(sqlite3.OperationalError):
            s._conn.execute("ATTACH DATABASE ? AS absent", (missing.as_uri() + "?mode=rw",))
        assert not missing.exists()
        with SQLiteStorageManager(
            tmp_path / "other?#.db", journal_mode="delete", check_same_thread=False
        ) as other:
            assert _append_coordinated_batches(batches([s, other])) == (["1", "2"],) * 2
        assert not (tmp_path / name).exists()


def test_concurrency_reversed_coordinators_and_existing_single_writer(stores):
    barrier = threading.Barrier(3)

    def writer(reverse):
        barrier.wait()
        for _ in range(8):
            _append_coordinated_batches(batches(list(reversed(stores)) if reverse else stores))

    def single():
        barrier.wait()
        for _ in range(8):
            stores[1].event_backend.append_batch([Metadata(description="single")])
            stores[1].save_snapshot_json("{}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(writer, False), pool.submit(writer, True), pool.submit(single)]
        for f in futures:
            f.result(timeout=30)
    for i, s in enumerate(stores):
        count = 40 if i == 1 else 32
        assert len(s.event_backend) == count
        assert s.event_backend.active_tags() == [str(i) for i in range(1, count + 1)]
        assert s._conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("at_attach", [False, True])
def test_replaced_original_file_prewrite_refusal(stores, index, at_attach):
    from nooa.storage.sqlite import _ensure_schema

    storage = stores[index]
    batch, before = batches(stores), state(stores)
    path = storage._canonical_path
    original = storage._file_identity

    def replace():
        path.unlink()
        with sqlite3.connect(path) as conn:
            _ensure_schema(conn)
        assert (path.stat().st_dev, path.stat().st_ino) != original

    if at_attach:
        stores[0]._conn.attach_hook = replace
    else:
        replace()
    try:
        with pytest.raises(_UnsupportedCoordinatedBatch, match="replaced-file"):
            _append_coordinated_batches(batch)
        assert_untouched(stores, batch, before)
        with sqlite3.connect(path) as replacement:
            assert replacement.execute("SELECT COUNT(*) FROM events").fetchone() == (0,)
            assert replacement.execute("SELECT COUNT(*) FROM active_tags").fetchone() == (0,)
        with pytest.raises(_UnsupportedCoordinatedBatch, match="replaced-file"):
            storage._reconnect()
        assert storage._file_identity == original
        assert storage._backend._next_tag_num == before[index][2]
    finally:
        # Teardown stale/unlinked actual connection AFTER proof, not a guard.
        storage._conn.close()
        storage._closed = True
