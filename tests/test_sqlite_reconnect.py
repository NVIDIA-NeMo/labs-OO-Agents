# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SQLiteStorageManager reconnect-on-IOERR behavior."""

import sqlite3

import pytest

from nooa.context_blocks import Metadata
from nooa.storage.sqlite import SQLiteStorageManager


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path for a temporary SQLite database."""
    return tmp_path / "test.db"


@pytest.fixture
def storage(tmp_db):
    """Create a SQLiteStorageManager with a real file-backed DB."""
    sm = SQLiteStorageManager(tmp_db)
    yield sm
    sm.close()


def _make_event(tag: str = "1") -> Metadata:
    return Metadata(tag=tag, event_type="Metadata")


class TestReconnect:
    """Tests for _reconnect() and retry-on-IOERR in store()."""

    def test_reconnect_replaces_connection(self, storage):
        """_reconnect() should produce a new working connection."""
        old_conn = storage._conn
        storage._reconnect()
        assert storage._conn is not old_conn
        assert storage._backend._conn is storage._conn
        # New connection should work
        row = storage._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert row[0] >= 0

    def test_reconnect_preserves_data(self, storage):
        """Data stored before reconnect should be readable after."""
        event = _make_event("1")
        storage._backend.store("1", event)
        storage._reconnect()
        retrieved = storage._backend.get("1")
        assert retrieved is not None
        assert retrieved.tag == "1"

    def test_store_retries_on_disk_io_error(self, storage):
        """store() should reconnect and retry once on disk I/O error."""
        call_count = 0
        original_do_store = storage._backend._do_store

        def failing_store(tag, event, data, order):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise sqlite3.OperationalError("disk I/O error")
            return original_do_store(tag, event, data, order)

        storage._backend._do_store = failing_store
        event = _make_event("2")
        storage._backend.store("2", event)

        # Should have retried and succeeded
        assert call_count == 2
        retrieved = storage._backend.get("2")
        assert retrieved is not None

    def test_store_raises_on_non_io_error(self, storage):
        """store() should NOT retry on non-I/O OperationalErrors."""

        def always_fail(tag, event, data, order):
            raise sqlite3.OperationalError("database is locked")

        storage._backend._do_store = always_fail
        event = _make_event("3")
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            storage._backend.store("3", event)

    def test_store_raises_if_retry_also_fails(self, storage):
        """If reconnect doesn't help, the second failure should propagate."""

        def always_fail(tag, event, data, order):
            raise sqlite3.OperationalError("disk I/O error")

        storage._backend._do_store = always_fail
        event = _make_event("4")
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            storage._backend.store("4", event)

    def test_store_handles_integrity_error_on_retry(self, storage):
        """If data was committed before I/O error, retry gets IntegrityError — treat as success."""
        call_count = [0]
        original_do_store = storage._backend._do_store

        def commit_then_ioerr(tag, event, data, order):
            call_count[0] += 1
            if call_count[0] == 1:
                # Data commits, then I/O error on return path
                original_do_store(tag, event, data, order)
                raise sqlite3.OperationalError("disk I/O error")
            return original_do_store(tag, event, data, order)

        storage._backend._do_store = commit_then_ioerr
        event = _make_event("integrity")
        # Should not raise — IntegrityError on retry means first write succeeded
        storage._backend.store("integrity", event)
        retrieved = storage._backend.get("integrity")
        assert retrieved is not None

    def test_store_does_not_retry_without_callback(self, tmp_db):
        """Without _on_io_error set, store() should raise immediately."""
        sm = SQLiteStorageManager(tmp_db)
        sm._backend._on_io_error = None  # Explicitly clear

        def always_fail(tag, event, data, order):
            raise sqlite3.OperationalError("disk I/O error")

        sm._backend._do_store = always_fail
        event = _make_event("5")
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            sm._backend.store("5", event)
        sm.close()


class TestBusyTimeout:
    """Test that busy_timeout PRAGMA is set."""

    def test_busy_timeout_is_set(self, storage):
        """busy_timeout PRAGMA should be configured to 5000 ms on open."""
        row = storage._conn.execute("PRAGMA busy_timeout").fetchone()
        assert row[0] == 5000

    def test_busy_timeout_persists_after_reconnect(self, storage):
        """busy_timeout should be reconfigured on a fresh connection after reconnect."""
        storage._reconnect()
        row = storage._conn.execute("PRAGMA busy_timeout").fetchone()
        assert row[0] == 5000


class TestOpenConnection:
    """Test _open_connection helper."""

    def test_wal_mode(self, storage):
        """WAL journal mode should be set on connection open."""
        row = storage._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"
