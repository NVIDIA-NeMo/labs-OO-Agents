# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SQLite corruption resilience (P0) and virtiofs detection (P1)."""

import sqlite3
import subprocess
from unittest.mock import patch

import pytest

from nooa.context_blocks import Metadata
from nooa.storage.sqlite import SQLiteStorageManager, _is_virtiofs


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def storage(tmp_db):
    sm = SQLiteStorageManager(tmp_db)
    yield sm
    sm.close()


def _make_event(tag: str = "1") -> Metadata:
    return Metadata(tag=tag, event_type="Metadata")


# ─── P0: Graceful handling of corrupt events ──────────────────────────────


class TestCorruptEventResilience:
    """get(), get_by_id(), active_tags(), and all_events() survive corruption."""

    def test_get_returns_none_on_operational_error(self, storage):
        """get() returns None when a row triggers OperationalError."""
        event = _make_event("1")
        storage._backend.store("1", event)

        # Corrupt the data column with invalid UTF-8 bytes
        storage._conn.execute("UPDATE events SET data = CAST(X'80808080' AS TEXT) WHERE tag = '1'")
        storage._conn.commit()

        # Should not raise — returns None
        result = storage._backend.get("1")
        assert result is None

    def test_get_by_id_returns_none_on_corrupt_data(self, storage):
        """get_by_id() returns None when deserialization fails."""
        event = _make_event("1")
        storage._backend.store("1", event)
        event_id = event.id

        # Write invalid JSON
        storage._conn.execute("UPDATE events SET data = '{not valid json' WHERE tag = '1'")
        storage._conn.commit()

        result = storage._backend.get_by_id(event_id)
        assert result is None

    def test_get_returns_none_on_zeroed_blob_data(self, storage):
        """get() returns None when event data is a zeroed blob."""
        event = _make_event("1")
        storage._backend.store("1", event)

        # Store a zeroed blob where JSON event text is expected.
        storage._conn.execute("UPDATE events SET data = zeroblob(4096) WHERE tag = '1'")
        storage._conn.commit()

        result = storage._backend.get("1")
        assert result is None

    def test_get_returns_none_on_non_object_json(self, storage):
        """get() treats valid non-object JSON as corrupt event data."""
        for tag, payload in (("1", "null"), ("2", "[]")):
            storage._backend.store(tag, _make_event(tag))
            storage._conn.execute("UPDATE events SET data = ? WHERE tag = ?", (payload, tag))
        storage._conn.commit()

        assert storage._backend.get("1") is None
        assert storage._backend.get("2") is None

    def test_unrelated_database_errors_propagate(self, storage):
        """Read APIs do not convert non-corruption DB failures into missing events."""
        conn = storage._conn
        original_execute = conn.execute

        class BusyConnection:
            def execute(self, sql, *args, **kwargs):
                if "SELECT data FROM events" in sql:
                    raise sqlite3.OperationalError("database is locked")
                return original_execute(sql, *args, **kwargs)

        storage._backend._conn = BusyConnection()
        try:
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                storage._backend.get("1")
        finally:
            storage._backend._conn = conn

    def test_active_tags_returns_empty_on_corruption(self, storage):
        """active_tags() returns [] for known corruption errors."""
        event = _make_event("1")
        storage._backend.store("1", event)

        # Verify it works normally first
        tags = storage._backend.active_tags()
        assert "1" in tags

        conn = storage._conn
        original_execute = conn.execute

        class CorruptActiveTagsConnection:
            def execute(self, sql, *args, **kwargs):
                if "FROM active_tags" in sql:
                    raise sqlite3.DatabaseError("database disk image is malformed")
                return original_execute(sql, *args, **kwargs)

        storage._backend._conn = CorruptActiveTagsConnection()
        try:
            assert storage._backend.active_tags() == []
        finally:
            storage._backend._conn = conn

    def test_all_events_skips_corrupt_rows(self, storage):
        """all_events() yields readable events, skips corrupt ones."""
        # Store 3 events
        for i in range(1, 4):
            storage._backend.store(str(i), _make_event(str(i)))

        # Corrupt the middle event
        storage._conn.execute("UPDATE events SET data = '{broken}' WHERE tag = '2'")
        storage._conn.commit()

        events = list(storage._backend.all_events())
        # Should get 2 events (skipping the corrupt one)
        assert len(events) == 2

    def test_all_events_returns_empty_when_all_rows_are_corrupt(self, storage):
        """all_events() skips every row when all event data is corrupt."""
        storage._backend.store("1", _make_event("1"))
        storage._backend.store("2", _make_event("2"))
        storage._backend.store("3", _make_event("3"))

        # Corrupt ALL event data so the entire query yields nothing readable
        storage._conn.execute("UPDATE events SET data = zeroblob(100)")
        storage._conn.commit()

        events = list(storage._backend.all_events())
        # All events are corrupt — should skip all, returning empty
        assert events == []

    def test_session_survives_single_corrupt_event(self, storage):
        """A session with one corrupt event can still read the others."""
        # Store events
        for i in range(1, 6):
            storage._backend.store(str(i), _make_event(str(i)))

        # Corrupt one in the middle
        storage._conn.execute("UPDATE events SET data = CAST(X'FF' AS TEXT) WHERE tag = '3'")
        storage._conn.commit()

        # get() on the corrupt tag returns None
        assert storage._backend.get("3") is None

        # Other events are fine
        assert storage._backend.get("1") is not None
        assert storage._backend.get("2") is not None
        assert storage._backend.get("4") is not None
        assert storage._backend.get("5") is not None

    def test_backend_initialization_tolerates_unreadable_event_aggregates(self, storage):
        """Backend construction does not fail if startup aggregate queries hit corruption."""
        conn = storage._conn
        original_execute = conn.execute

        class CorruptingConnection:
            def execute(self, sql, *args, **kwargs):
                if "MAX(insertion_order)" in sql or "FROM events" in sql:
                    raise sqlite3.DatabaseError("database disk image is malformed")
                return original_execute(sql, *args, **kwargs)

        from nooa.storage.sqlite import SQLiteEventBackend

        backend = SQLiteEventBackend(CorruptingConnection())
        assert backend._insertion_counter == 0
        assert backend._next_tag_num == 1


# ─── P1: virtiofs detection ───────────────────────────────────────────────


class TestVirtiofsDetection:
    """_is_virtiofs() correctly identifies virtiofs mounts."""

    def setup_method(self):
        _is_virtiofs.cache_clear()

    def test_memory_db_is_not_virtiofs(self):
        assert _is_virtiofs(":memory:") is False

    @patch("subprocess.run")
    def test_detects_virtiofs_in_df_output(self, mock_run):
        mock_run.return_value = type(
            "Result",
            (),
            {
                "stdout": "Filesystem     Type  1K-blocks  Used Available Use% Mounted on\n"
                "bind-abc123    virtiofs 971350180 886768724  84581456  92% /Users/dev\n",
                "returncode": 0,
            },
        )()
        assert _is_virtiofs("/Users/dev/project/test.db") is True

    @patch("subprocess.run")
    def test_non_virtiofs_returns_false(self, mock_run):
        mock_run.return_value = type(
            "Result",
            (),
            {
                "stdout": "Filesystem     Type  1K-blocks  Used Available Use% Mounted on\n"
                "/dev/sda1      ext4  971350180 886768724  84581456  92% /\n",
                "returncode": 0,
            },
        )()
        assert _is_virtiofs("/home/user/test.db") is False

    @patch("subprocess.run", side_effect=FileNotFoundError("df not found"))
    def test_returns_false_on_error(self, mock_run):
        assert _is_virtiofs("/some/path/test.db") is False

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="df", timeout=2))
    def test_returns_false_on_timeout(self, mock_run):
        assert _is_virtiofs("/some/path/test.db") is False

    def test_synchronous_full_on_virtiofs(self, tmp_db):
        """Storage manager sets DELETE + synchronous=FULL when virtiofs detected."""
        with patch("nooa.storage.sqlite._is_virtiofs", return_value=True):
            sm = SQLiteStorageManager(tmp_db)
            try:
                row = sm._conn.execute("PRAGMA synchronous").fetchone()
                # FULL = 2
                assert row[0] == 2
                row2 = sm._conn.execute("PRAGMA journal_mode").fetchone()
                assert row2[0] == "delete"
            finally:
                sm.close()

    def test_virtiofs_does_not_enable_wal_before_delete(self, tmp_db):
        """virtiofs setup chooses DELETE directly instead of WAL then DELETE."""
        seen_pragmas = []
        real_connect = sqlite3.connect

        class RecordingConnection:
            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, *args, **kwargs):
                if sql.startswith("PRAGMA journal_mode") or sql.startswith("PRAGMA synchronous"):
                    seen_pragmas.append(sql)
                return self._conn.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        def recording_connect(*args, **kwargs):
            return RecordingConnection(real_connect(*args, **kwargs))

        with (
            patch("nooa.storage.sqlite._is_virtiofs", return_value=True),
            patch("sqlite3.connect", side_effect=recording_connect),
        ):
            sm = SQLiteStorageManager(tmp_db)
            sm.close()

        assert "PRAGMA journal_mode=WAL" not in seen_pragmas
        assert "PRAGMA journal_mode=DELETE" in seen_pragmas
        assert "PRAGMA synchronous=FULL" in seen_pragmas

    def test_synchronous_full_on_ext4(self, tmp_db):
        """Storage manager uses synchronous=FULL on non-virtiofs disks too.

        WAL+NORMAL leaves a disk-full window that can corrupt the file; FULL
        closes it. Regression for the disk-full corruption crash.
        """
        with patch("nooa.storage.sqlite._is_virtiofs", return_value=False):
            sm = SQLiteStorageManager(tmp_db)
            try:
                row = sm._conn.execute("PRAGMA synchronous").fetchone()
                assert row[0] == 2  # FULL
                row2 = sm._conn.execute("PRAGMA journal_mode").fetchone()
                assert row2[0] == "wal"
            finally:
                sm.close()


# ─── P0: Write path survives corruption (regression: uncaught DatabaseError) ──


class TestStoreCorruptionResilience:
    """store() must not let a corrupt DB escape as an uncaught task exception.

    Regression for the disk-full → "database disk image is malformed" crash:
    the read paths degrade gracefully via _is_corruption_error(), but store()
    only caught OperationalError("disk I/O error"). A bare
    sqlite3.DatabaseError("... malformed") escaped uncaught from a spawned
    background task.
    """

    def test_store_raises_typed_corruption_error(self, storage):
        """store() converts a corruption DatabaseError into CorruptDatabaseError."""
        from nooa.storage.sqlite import CorruptDatabaseError

        def boom(*args, **kwargs):
            raise sqlite3.DatabaseError("database disk image is malformed")

        storage._backend._do_store = boom

        event = _make_event("9")
        with pytest.raises(CorruptDatabaseError):
            storage._backend.store("9", event)

    def test_store_propagates_non_corruption_errors(self, storage):
        """store() does not mask unrelated DB failures as corruption."""

        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        storage._backend._do_store = boom

        event = _make_event("9")
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            storage._backend.store("9", event)

    def test_store_retry_path_corruption_is_typed(self, storage):
        """A corruption DatabaseError on the reconnected retry is also typed, not uncaught.

        Reproduces the double-failure: disk-I/O error on the first _do_store,
        then corruption on the retry after reconnect.
        """
        from nooa.storage.sqlite import CorruptDatabaseError

        calls = {"n": 0}

        def first_io_then_corrupt(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("disk I/O error")
            raise sqlite3.DatabaseError("database disk image is malformed")

        storage._backend._do_store = first_io_then_corrupt
        storage._backend._on_io_error = lambda: None

        with pytest.raises(CorruptDatabaseError):
            storage._backend.store("9", _make_event("9"))
