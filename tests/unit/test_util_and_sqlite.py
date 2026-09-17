# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for util/ package and storage/sqlite.py."""

from __future__ import annotations

import logging
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# util/_context.py
# ---------------------------------------------------------------------------


class TestContext:
    """Tests for nooa.util._context."""

    def setup_method(self):
        """Reset context var before each test."""
        from nooa.util._context import _current_agent_var

        # Reset to None between tests
        _current_agent_var.set(None)

    def test_current_agent_raises_when_not_set(self):
        from nooa.util._context import _current_agent

        with pytest.raises(RuntimeError, match="No agent in context"):
            _current_agent()

    def test_current_agent_returns_agent_when_set(self):
        from nooa.util._context import _current_agent, _current_agent_var

        fake_agent = MagicMock()
        _current_agent_var.set(fake_agent)
        assert _current_agent() is fake_agent

    def test_runtime_var_exists(self):
        """The _current_runtime_var ContextVar is also exported."""
        from nooa.util._context import _current_runtime_var

        assert _current_runtime_var is not None


# ---------------------------------------------------------------------------
# util/quickstart.py — example classes (module-level side effects are mocked)
# ---------------------------------------------------------------------------


class TestArtwork:
    """Tests for quickstart.Artwork."""

    def _make(self):
        # Import must be done under patch to avoid module-level side effects
        with (
            patch("dotenv.load_dotenv"),
            patch("nooa.unifiedllm.registry.get_llm_client"),
        ):
            from nooa.util import quickstart

            return quickstart.Artwork("Starry Night", "van Gogh", 1_000_000.0)

    def test_init_attributes(self):
        artwork = self._make()
        assert artwork.title == "Starry Night"
        assert artwork.artist == "van Gogh"

    def test_get_appraisal(self):
        artwork = self._make()
        appraisal = artwork.get_appraisal()
        assert appraisal["title"] == "Starry Night"
        assert appraisal["artist"] == "van Gogh"
        assert appraisal["value"] == 1_000_000.0
        assert appraisal["currency"] == "USD"

    def test_repr_and_eq_not_broken_by_dataclass(self):
        # Regression: a no-op @dataclass (no field annotations) generated a
        # repr that dropped all state and an __eq__ under which every instance
        # compared equal. With the decorator removed these are plain classes:
        # distinct instances must not be equal and repr must not falsely
        # collapse them.
        with (
            patch("dotenv.load_dotenv"),
            patch("nooa.unifiedllm.registry.get_llm_client"),
        ):
            from nooa.util import quickstart

            a = quickstart.Artwork("Starry Night", "van Gogh", 1_000_000.0)
            b = quickstart.Artwork("Mona Lisa", "da Vinci", 2_000_000.0)
        assert a != b
        assert a == a
        assert "Artwork" in repr(a)


class TestStockHolding:
    """Tests for quickstart.StockHolding."""

    def _make(self):
        with (
            patch("dotenv.load_dotenv"),
            patch("nooa.unifiedllm.registry.get_llm_client"),
        ):
            from nooa.util import quickstart

            return quickstart.StockHolding("AAPL", 100, 150.0)

    def test_init_attributes(self):
        stock = self._make()
        assert stock.symbol == "AAPL"

    def test_get_total_value(self):
        stock = self._make()
        assert stock.get_total_value() == 15000.0


class TestJewelry:
    """Tests for quickstart.Jewelry."""

    def _make(self):
        with (
            patch("dotenv.load_dotenv"),
            patch("nooa.unifiedllm.registry.get_llm_client"),
        ):
            from nooa.util import quickstart

            return quickstart.Jewelry("Diamond Ring", 2.5, 10_000.0)

    def test_init_attributes(self):
        jewelry = self._make()
        assert jewelry.description == "Diamond Ring"

    def test_compute_value(self):
        jewelry = self._make()
        assert jewelry.compute_value() == 25_000.0


class TestCollectible:
    """Tests for quickstart.Collectible."""

    def _make(self, condition="mint"):
        with (
            patch("dotenv.load_dotenv"),
            patch("nooa.unifiedllm.registry.get_llm_client"),
        ):
            from nooa.util import quickstart

            return quickstart.Collectible("Baseball Card", 1000.0, condition)

    def test_init_attributes(self):
        c = self._make()
        assert c.name == "Baseball Card"
        assert c.condition == "mint"

    def test_estimate_value_mint(self):
        assert self._make("mint").estimate_value() == 1000.0

    def test_estimate_value_excellent(self):
        assert self._make("excellent").estimate_value() == pytest.approx(850.0)

    def test_estimate_value_good(self):
        assert self._make("good").estimate_value() == pytest.approx(700.0)

    def test_estimate_value_fair(self):
        assert self._make("fair").estimate_value() == pytest.approx(500.0)

    def test_estimate_value_unknown_condition(self):
        """Unknown condition defaults to 0.5 multiplier."""
        assert self._make("poor").estimate_value() == pytest.approx(500.0)


class TestAutorun:
    """Tests for quickstart.autorun decorator."""

    def test_autorun_calls_asyncio_run(self):
        with (
            patch("dotenv.load_dotenv"),
            patch("nooa.unifiedllm.registry.get_llm_client"),
        ):
            from nooa.util import quickstart

            async def my_func():
                return 42

            with patch("asyncio.run") as mock_run, patch("builtins.print"):
                quickstart.autorun(my_func)
                assert mock_run.call_count == 1
                # The argument is a coroutine from calling my_func()
                call_arg = mock_run.call_args[0][0]
                import inspect

                assert inspect.iscoroutine(call_arg)
                call_arg.close()  # clean up the coroutine

    def test_autorun_returns_original_function(self):
        with (
            patch("dotenv.load_dotenv"),
            patch("nooa.unifiedllm.registry.get_llm_client"),
        ):
            from nooa.util import quickstart

            async def my_func():
                return 42

            with patch("asyncio.run"), patch("builtins.print"):
                result = quickstart.autorun(my_func)
                assert result is my_func

    def test_autorun_prints_example_output(self):
        with (
            patch("dotenv.load_dotenv"),
            patch("nooa.unifiedllm.registry.get_llm_client"),
        ):
            from nooa.util import quickstart

            async def my_func():
                pass

            with patch("asyncio.run"), patch("builtins.print") as mock_print:
                quickstart.autorun(my_func)
                mock_print.assert_called_once_with("\n\nEXAMPLE OUTPUT:")


# ---------------------------------------------------------------------------
# storage/sqlite.py — SQLiteEventBackend
# ---------------------------------------------------------------------------


def _make_backend():
    """Create an in-memory SQLiteEventBackend for testing."""
    from nooa.storage.sqlite import SQLiteEventBackend, _ensure_schema

    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)
    return SQLiteEventBackend(conn), conn


class TestEnsureSchema:
    """Tests for _ensure_schema."""

    def test_creates_tables(self):
        from nooa.storage.sqlite import _ensure_schema

        conn = sqlite3.connect(":memory:")
        _ensure_schema(conn)
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "events" in tables
        assert "active_tags" in tables
        assert "snapshots" in tables
        assert "schema_version" in tables

    def test_inserts_schema_version(self):
        from nooa.storage.sqlite import _SCHEMA_VERSION, _ensure_schema

        conn = sqlite3.connect(":memory:")
        _ensure_schema(conn)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row[0] == _SCHEMA_VERSION

    def test_schema_version_mismatch_raises(self):
        from nooa.storage.sqlite import _SCHEMA_VERSION, _ensure_schema

        conn = sqlite3.connect(":memory:")
        _ensure_schema(conn)
        # Tamper with the version
        conn.execute("UPDATE schema_version SET version = ?", (_SCHEMA_VERSION + 1,))
        conn.commit()
        with pytest.raises(RuntimeError, match="schema version mismatch"):
            _ensure_schema(conn)

    def test_idempotent_on_second_call(self):
        """Calling _ensure_schema twice on the same DB should not raise."""
        from nooa.storage.sqlite import _ensure_schema

        conn = sqlite3.connect(":memory:")
        _ensure_schema(conn)
        _ensure_schema(conn)  # should not raise


class TestSQLiteEventBackendStore:
    """Tests for SQLiteEventBackend.store and get."""

    def test_store_and_get_message(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        msg = Message(content="hello world")
        backend.store("tag1", msg)
        result = backend.get("tag1")
        assert result is not None
        assert result.content == "hello world"

    def test_store_adds_to_active_tags(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        msg = Message(content="test")
        backend.store("mytag", msg)
        assert "mytag" in backend.active_tags()

    def test_store_multiple_preserves_order(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        for i in range(5):
            backend.store(f"tag{i}", Message(content=f"msg{i}"))
        tags = backend.active_tags()
        assert tags == ["tag0", "tag1", "tag2", "tag3", "tag4"]

    def test_len_counts_events(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        assert len(backend) == 0
        backend.store("t1", Message(content="a"))
        assert len(backend) == 1
        backend.store("t2", Message(content="b"))
        assert len(backend) == 2

    def test_get_returns_none_for_missing(self):
        backend, _ = _make_backend()
        assert backend.get("nonexistent") is None

    def test_get_by_id(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        msg = Message(content="by_id_test")
        backend.store("tag_id", msg)
        result = backend.get_by_id(msg.id)
        assert result is not None
        assert result.content == "by_id_test"

    def test_get_by_id_returns_none_for_missing(self):
        backend, _ = _make_backend()
        assert backend.get_by_id("nonexistent-id") is None


class TestSQLiteEventBackendUpdate:
    """Tests for SQLiteEventBackend.update."""

    def test_update_field(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        msg = Message(content="original")
        backend.store("tag1", msg)
        ok = backend.update("tag1", content="updated")
        assert ok is True
        result = backend.get("tag1")
        assert result.content == "updated"

    def test_update_metadata(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        msg = Message(content="test")
        backend.store("tag1", msg)
        ok = backend.update("tag1", metadata={"key": "value"})
        assert ok is True
        result = backend.get("tag1")
        assert result.metadata.get("key") == "value"

    def test_update_returns_false_for_missing(self):
        backend, _ = _make_backend()
        assert backend.update("nonexistent", content="x") is False

    def test_update_nonexistent_field_is_ignored(self):
        """Updating a field that doesn't exist on the event is silently ignored."""
        from nooa.events import Message

        backend, _ = _make_backend()
        msg = Message(content="test")
        backend.store("tag1", msg)
        ok = backend.update("tag1", nonexistent_field="value")
        assert ok is True  # update succeeds even for unknown fields


class TestSQLiteEventBackendRemove:
    """Tests for SQLiteEventBackend.remove."""

    def test_remove_existing(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        backend.store("tag1", Message(content="test"))
        ok = backend.remove("tag1")
        assert ok is True
        assert backend.get("tag1") is None
        assert "tag1" not in backend.active_tags()

    def test_remove_nonexistent_returns_false(self):
        backend, _ = _make_backend()
        assert backend.remove("nonexistent") is False

    def test_remove_decrements_len(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        backend.store("t1", Message(content="a"))
        backend.store("t2", Message(content="b"))
        assert len(backend) == 2
        backend.remove("t1")
        assert len(backend) == 1


class TestSQLiteEventBackendSetStatus:
    """Tests for SQLiteEventBackend.set_status."""

    def test_set_status_to_archived(self):
        from nooa.context_blocks import EventStatus
        from nooa.events import Message

        backend, _ = _make_backend()
        msg = Message(content="test")
        backend.store("tag1", msg)
        ok = backend.set_status("tag1", EventStatus.ARCHIVED)
        assert ok is True
        result = backend.get("tag1")
        assert result.status == EventStatus.ARCHIVED

    def test_set_status_returns_false_for_missing(self):
        from nooa.context_blocks import EventStatus

        backend, _ = _make_backend()
        assert backend.set_status("nonexistent", EventStatus.ACTIVE) is False


class TestSQLiteEventBackendActiveTags:
    """Tests for active_tags, insert_active_tag, remove_active_tag."""

    def test_active_tags_empty(self):
        backend, _ = _make_backend()
        assert backend.active_tags() == []

    def test_insert_active_tag_at_position(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        backend.store("tag0", Message(content="a"))
        backend.store("tag2", Message(content="b"))
        # Insert a tag at position 1 (between existing positions)
        backend.insert_active_tag("tag1", 1)
        tags = backend.active_tags()
        assert tags.index("tag1") < tags.index("tag2")

    def test_remove_active_tag_existing(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        backend.store("tag1", Message(content="x"))
        ok = backend.remove_active_tag("tag1")
        assert ok is True
        assert "tag1" not in backend.active_tags()

    def test_remove_active_tag_nonexistent(self):
        backend, _ = _make_backend()
        assert backend.remove_active_tag("nonexistent") is False


class TestSQLiteEventBackendAllEvents:
    """Tests for all_events() iterator."""

    def test_all_events_empty(self):
        backend, _ = _make_backend()
        assert list(backend.all_events()) == []

    def test_all_events_in_insertion_order(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        msgs = [Message(content=f"msg{i}") for i in range(3)]
        for i, m in enumerate(msgs):
            backend.store(f"tag{i}", m)
        events = list(backend.all_events())
        assert len(events) == 3
        assert events[0].content == "msg0"
        assert events[1].content == "msg1"
        assert events[2].content == "msg2"


class TestSQLiteEventBackendFindTag:
    """Tests for find_tag()."""

    def test_find_tag_existing(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        msg = Message(content="test")
        backend.store("mytag", msg)
        assert backend.find_tag(msg) == "mytag"

    def test_find_tag_nonexistent_returns_none(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        assert backend.find_tag(Message(content="x")) is None


class TestSQLiteEventBackendClear:
    """Tests for clear()."""

    def test_clear_removes_all_events_and_tags(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        for i in range(3):
            backend.store(f"tag{i}", Message(content=f"msg{i}"))
        backend.clear()
        assert len(backend) == 0
        assert backend.active_tags() == []

    def test_clear_resets_insertion_counter(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        backend.store("t1", Message(content="a"))
        backend.clear()
        # After clear, insertion counter resets to 0
        assert backend._insertion_counter == 0


class TestSQLiteEventBackendDeserialization:
    """Tests for _deserialize with unknown event types."""

    def test_unknown_event_type_falls_back_to_metadata(self, caplog):
        from nooa.context_blocks import Metadata

        backend, _ = _make_backend()
        import json

        raw = json.dumps(
            {
                "event_type": "totally_unknown_type_xyz",
                "id": "test-id",
                "status": "active",
            }
        )
        with caplog.at_level(logging.WARNING, logger="nooa.storage.sqlite"):
            event = backend._deserialize(raw)
        assert isinstance(event, Metadata)
        assert "Unknown event_type" in caplog.text

    def test_known_event_type_uses_correct_class(self):
        from nooa.events import Message

        backend, _ = _make_backend()
        import json

        raw = json.dumps(
            {"event_type": "Message", "id": "test-id", "status": "active", "content": "hi"}
        )
        event = backend._deserialize(raw)
        assert isinstance(event, Message)
        assert event.content == "hi"


class TestSQLiteEventBackendRegisterEventType:
    """Tests for register_event_type()."""

    def test_register_new_type(self):
        from nooa.context_blocks import EventBase

        backend, _ = _make_backend()

        class MyEvent(EventBase):
            event_type: str = "my_custom_event"
            data: str = ""

        backend.register_event_type(MyEvent)
        assert backend._registry["my_custom_event"] is MyEvent

    def test_register_same_type_no_warning(self, caplog):
        """Re-registering the same class for same key should not warn."""
        from nooa.events import Message

        backend, _ = _make_backend()
        with caplog.at_level(logging.WARNING, logger="nooa.storage.sqlite"):
            backend.register_event_type(Message)
        assert "overwrites" not in caplog.text

    def test_register_different_class_same_key_warns(self, caplog):
        """Registering a different class for an existing key should warn."""
        from nooa.context_blocks import EventBase

        backend, _ = _make_backend()

        class FakeMessage(EventBase):
            event_type: str = "Message"  # Same as Message!
            content: str = ""

        with caplog.at_level(logging.WARNING, logger="nooa.storage.sqlite"):
            backend.register_event_type(FakeMessage)
        assert "overwrites" in caplog.text


class TestSQLiteStorageManager:
    """Tests for SQLiteStorageManager."""

    def test_in_memory_creation(self):
        from nooa.storage.sqlite import SQLiteStorageManager

        sm = SQLiteStorageManager(":memory:")
        assert sm.event_backend is not None
        sm.close()

    def test_file_based_creation(self, tmp_path):
        from nooa.storage.sqlite import SQLiteStorageManager

        db_path = tmp_path / "test.db"
        sm = SQLiteStorageManager(db_path)
        assert sm.event_backend is not None
        sm.close()
        assert db_path.exists()

    def test_context_manager(self):
        from nooa.storage.sqlite import SQLiteStorageManager

        with SQLiteStorageManager(":memory:") as sm:
            assert sm.event_backend is not None
        # After __exit__, connection is closed (further operations would fail)

    def test_get_latest_snapshot_id_empty(self):
        from nooa.storage.sqlite import SQLiteStorageManager

        sm = SQLiteStorageManager(":memory:")
        assert sm.get_latest_snapshot_id() is None
        sm.close()

    def test_event_backend_property(self):
        from nooa.runtime.event_backend import EventBackend
        from nooa.storage.sqlite import SQLiteStorageManager

        sm = SQLiteStorageManager(":memory:")
        assert isinstance(sm.event_backend, EventBackend)
        sm.close()

    def test_backend_uses_insertion_counter(self):
        """Insertion counter increments with each stored event."""
        from nooa.events import Message
        from nooa.storage.sqlite import SQLiteStorageManager

        sm = SQLiteStorageManager(":memory:")
        backend = sm._backend
        initial = backend._insertion_counter
        backend.store("t1", Message(content="a"))
        assert backend._insertion_counter == initial + 1
        backend.store("t2", Message(content="b"))
        assert backend._insertion_counter == initial + 2
        sm.close()

    def test_close_closes_the_connection_when_commit_fails(self):
        """A raising commit() must not leak the connection.

        close() sets _closed before committing, so a connection missed here can
        never be closed by a later close() call -- it early-returns.
        """
        import sqlite3

        from nooa.storage.sqlite import SQLiteStorageManager

        sm = SQLiteStorageManager(":memory:")
        real_conn = sm._conn
        try:
            fake_conn = MagicMock(spec=sqlite3.Connection)
            fake_conn.commit.side_effect = sqlite3.OperationalError("disk I/O error")
            sm._conn = fake_conn
            with pytest.raises(sqlite3.OperationalError):
                sm.close()
            assert fake_conn.close.called
        finally:
            real_conn.close()

    def test_restore_snapshot_not_found_raises(self):
        from nooa.errors.storage import SnapshotNotFoundError
        from nooa.storage.sqlite import SQLiteStorageManager

        sm = SQLiteStorageManager(":memory:")
        mock_agent = MagicMock()
        with pytest.raises(SnapshotNotFoundError, match="not found"):
            sm.restore_snapshot("nonexistent-snapshot-id", mock_agent)
        sm.close()
