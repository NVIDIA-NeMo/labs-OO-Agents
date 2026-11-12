# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic mandatory recorder is separate from best-effort observers."""

import sqlite3

import pytest

from nooa.context_blocks import Metadata
from nooa.runtime.event_manager import EventManager
from nooa.storage import SQLiteStorageManager


def test_sqlite_batch_failure_rolls_back_events_and_active_tags(tmp_path):
    path = tmp_path / "batch.db"
    with SQLiteStorageManager(path) as storage:
        manager = EventManager(backend=storage.event_backend)
        seen = []
        manager.on("*", seen.append)
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TRIGGER fail_second BEFORE INSERT ON events "
                "WHEN NEW.tag = '2' BEGIN SELECT RAISE(ABORT, 'second failed'); END"
            )
        events = [Metadata(description="first"), Metadata(description="second")]
        with pytest.raises(sqlite3.IntegrityError, match="second failed"):
            manager.record_batch(events)
        assert [event.tag for event in events] == [None, None]
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM events").fetchone() == (0,)
            assert connection.execute("SELECT COUNT(*) FROM active_tags").fetchone() == (0,)
            connection.execute("DROP TRIGGER fail_second")
        assert seen == []
        # Retry the same caller objects after rollback, not newly created ones.
        assert manager.record_batch(events) == ["1", "2"]
        assert manager.all_events() == events
        assert seen == []  # no observers inside mandatory recording
        assert manager.add(Metadata(description="third")) == "3"


def test_memory_batch_supported_and_unsupported_backend_fails_explicitly():
    manager = EventManager()
    events = [Metadata(description="one"), Metadata(description="two")]
    assert manager.record_batch(events) == ["1", "2"]
    assert manager.all_events() == events

    class Unsupported:
        pass

    manager.set_backend(Unsupported())
    with pytest.raises(TypeError, match="atomic batch"):
        manager.record_batch([Metadata(description="no fallback")])


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("invalid", ["tagged", "reused", "alias", "duplicate_id", "stored_id"])
def test_batch_rejects_nonfresh_events_without_any_mutation(tmp_path, backend, invalid):
    from contextlib import ExitStack

    with ExitStack() as stack:
        manager = EventManager()
        if backend == "sqlite":
            storage = stack.enter_context(SQLiteStorageManager(tmp_path / "fresh.db"))
            manager.set_backend(storage.event_backend)
        stored = Metadata(description="existing")
        manager.record_batch([stored])
        fresh = Metadata(description="fresh")
        if invalid == "tagged":
            bad = Metadata(description="tagged", tag="99")
            batch = [fresh, bad]
        elif invalid == "reused":
            batch = [fresh, stored]
        elif invalid == "alias":
            batch = [fresh, fresh]
        elif invalid == "duplicate_id":
            batch = [fresh, fresh.model_copy()]
        else:
            batch = [fresh, Metadata(description="existing ID", id=stored.id)]
        before = [(event.id, event.tag) for event in batch]
        with pytest.raises(ValueError, match="fresh"):
            manager.record_batch(batch)
        assert [(event.id, event.tag) for event in batch] == before
        assert manager.all_events() == [stored]
        assert manager.keys() == ["1"]
        assert manager.record_batch([fresh]) == ["2"]
        assert manager.get("1") == stored and manager.get("2") == fresh


@pytest.mark.parametrize("failure", ["serialize", "active_tags"])
def test_sqlite_batch_other_failures_preserve_same_objects_for_retry(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "retry.db"
    with SQLiteStorageManager(path) as storage:
        backend = storage.event_backend
        manager = EventManager(backend=backend)
        events = [Metadata(description="one"), Metadata(description="two")]
        serialize = backend._serialize
        if failure == "serialize":

            def fail(event):
                if event.description == "two":
                    raise ValueError("serialization failed")
                return serialize(event)

            monkeypatch.setattr(backend, "_serialize", fail)
            expected = ValueError
        else:
            with sqlite3.connect(path) as connection:
                connection.execute(
                    "CREATE TRIGGER fail_active BEFORE INSERT ON active_tags "
                    "WHEN NEW.tag = '2' BEGIN SELECT RAISE(ABORT, 'active failed'); END"
                )
            expected = sqlite3.IntegrityError
        with pytest.raises(expected):
            manager.record_batch(events)
        assert [event.tag for event in events] == [None, None]
        assert manager.all_events() == [] and manager.keys() == []
        monkeypatch.setattr(backend, "_serialize", serialize)
        if failure == "active_tags":
            with sqlite3.connect(path) as connection:
                connection.execute("DROP TRIGGER fail_active")
        assert manager.record_batch(events) == ["1", "2"]
        assert manager.all_events() == events


@pytest.mark.parametrize("backend_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("invalid", ["runtime", "summary", "archived", "not_event"])
def test_direct_backend_batch_also_rejects_invalid_inputs(tmp_path, backend_kind, invalid):
    from contextlib import ExitStack

    from nooa.context_blocks import EventStatus
    from nooa.events import Summary
    from nooa.runtime.event_backend import InMemoryBackend
    from nooa.runtime.turn_loop import TurnBegan

    with ExitStack() as stack:
        backend = InMemoryBackend()
        if backend_kind == "sqlite":
            backend = stack.enter_context(
                SQLiteStorageManager(tmp_path / "direct.db")
            ).event_backend
        candidates = {
            "runtime": TurnBegan(),
            "summary": Summary(summary_tag="1..2", replaced_range=(1, 2), children_tags=[]),
            "archived": Metadata(description="archived", status=EventStatus.ARCHIVED),
            "not_event": object(),
        }
        fresh = Metadata(description="fresh")
        with pytest.raises(ValueError, match="fresh"):
            backend.append_batch([fresh, candidates[invalid]])
        assert fresh.tag is None and list(backend.all_events()) == []
        assert backend.append_batch([fresh]) == ["1"]


def _batch_backend(stack, tmp_path, backend_kind):
    from nooa.runtime.event_backend import InMemoryBackend

    if backend_kind == "sqlite":
        return stack.enter_context(SQLiteStorageManager(tmp_path / "batch.db")).event_backend
    return InMemoryBackend()


def _counters(backend):
    return backend._next_tag_num, getattr(backend, "_insertion_counter", None)


@pytest.mark.parametrize("backend_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("entrypoint", ["manager", "backend"])
@pytest.mark.parametrize("failing", ["validator", "setter", "frozen_model"])
def test_batch_failing_tag_setter_stores_nothing_and_resets_tags(
    tmp_path, backend_kind, entrypoint, failing
):
    """Tags are set with the model's own setter first; a failure stores nothing."""
    from contextlib import ExitStack

    from pydantic import field_validator

    class ValidatedMetadata(Metadata):
        model_config = {"validate_assignment": True}

        @field_validator("tag")
        @classmethod
        def reject_third(cls, value):
            if value == "3":
                raise ValueError("no tag 3")
            return value

    class SetterMetadata(Metadata):
        def __setattr__(self, name, value):
            if name == "tag" and value == "3":
                raise ValueError("no tag 3")
            return super().__setattr__(name, value)

    class FrozenMetadata(Metadata):
        model_config = {"frozen": True}

    cls = {
        "validator": ValidatedMetadata,
        "setter": SetterMetadata,
        "frozen_model": FrozenMetadata,
    }[failing]
    with ExitStack() as stack:
        backend = _batch_backend(stack, tmp_path, backend_kind)
        manager = EventManager(backend=backend)
        backend.store("1", Metadata(description="existing", tag="1"))
        seen = []
        manager.on("*", seen.append)
        first = cls(description="first", event_type=cls.__name__)
        if failing == "frozen_model":
            first = Metadata(description="first")
        events = [first, cls(description="second", event_type=cls.__name__)]
        counters = _counters(backend)
        append = manager.record_batch if entrypoint == "manager" else backend.append_batch
        with pytest.raises(ValueError):
            append(events)
        assert [event.tag for event in events] == [None, None]
        assert seen == []
        assert _counters(backend) == counters
        assert backend.active_tags() == ["1"] and len(backend) == 1
        if backend_kind == "sqlite":
            with sqlite3.connect(tmp_path / "batch.db") as connection:
                assert connection.execute("SELECT tag FROM events").fetchall() == [("1",)]
                assert connection.execute("SELECT tag FROM active_tags").fetchall() == [("1",)]
        # The same fresh first event can still be recorded.
        assert append([events[0]]) == ["2"]
        assert events[0].tag == "2"


@pytest.mark.parametrize("failure", ["serialize", "commit"])
def test_sqlite_batch_serialize_or_commit_failure_resets_all_tags(tmp_path, monkeypatch, failure):
    with SQLiteStorageManager(tmp_path / "fail.db") as storage:
        backend = storage.event_backend
        events = [Metadata(description="one"), Metadata(description="two")]
        counters = _counters(backend)
        if failure == "serialize":
            real = backend._serialize

            def serialize(event):
                if event is events[1]:
                    raise RuntimeError("serialize failed")
                return real(event)

            monkeypatch.setattr(backend, "_serialize", serialize)
            expected = RuntimeError
        else:
            # Fails the second executemany inside the transaction.
            with sqlite3.connect(tmp_path / "fail.db") as connection:
                connection.execute(
                    "CREATE TRIGGER fail_active BEFORE INSERT ON active_tags "
                    "BEGIN SELECT RAISE(ABORT, 'commit failed'); END"
                )
            expected = sqlite3.IntegrityError
        with pytest.raises(expected):
            backend.append_batch(events)
        assert [event.tag for event in events] == [None, None]
        assert _counters(backend) == counters
        assert list(backend.all_events()) == [] and backend.active_tags() == []
        monkeypatch.undo()
        if failure == "commit":
            with sqlite3.connect(tmp_path / "fail.db") as connection:
                connection.execute("DROP TRIGGER fail_active")
        assert backend.append_batch(events) == ["1", "2"]


@pytest.mark.parametrize("backend_kind", ["memory", "sqlite"])
def test_batch_accepts_validate_assignment_model_that_accepts_tag(tmp_path, backend_kind):
    """Previously rejected up front; now the model's own setter just runs."""
    from contextlib import ExitStack

    from pydantic import field_validator

    calls = []

    class ValidatedMetadata(Metadata):
        model_config = {"validate_assignment": True}

        @field_validator("tag")
        @classmethod
        def record(cls, value):
            calls.append(value)
            return value

    with ExitStack() as stack:
        backend = _batch_backend(stack, tmp_path, backend_kind)
        events = [
            ValidatedMetadata(description=name, event_type="ValidatedMetadata")
            for name in ("one", "two")
        ]
        calls.clear()
        assert EventManager(backend=backend).record_batch(events) == ["1", "2"]
        assert calls == ["1", "2"]
        assert [event.tag for event in events] == ["1", "2"]
        assert backend.active_tags() == ["1", "2"]
        assert backend.get("2").tag == "2"


@pytest.mark.parametrize("backend_kind", ["memory", "sqlite"])
def test_batch_accepts_ordinary_subclass_with_unrelated_frozen_field(tmp_path, backend_kind):
    from contextlib import ExitStack

    from pydantic import Field

    from nooa.runtime.event_backend import InMemoryBackend

    class OrdinaryMetadata(Metadata):
        description: str = Field(frozen=True)

    # Warm the normal tag-handler cache, then verify normal publication semantics.
    warm = OrdinaryMetadata(description="warm")
    warm.tag = "99"
    with ExitStack() as stack:
        backend = InMemoryBackend()
        if backend_kind == "sqlite":
            backend = stack.enter_context(
                SQLiteStorageManager(tmp_path / "ordinary.db")
            ).event_backend
        event = OrdinaryMetadata(description="ordinary")
        assert backend.append_batch([event]) == ["1"]
        assert event.tag == "1" and "tag" in event.model_fields_set
        assert backend.get("1") == event
        if backend_kind == "memory":
            assert backend.get("1") is event


@pytest.mark.parametrize("backend_kind", ["memory", "sqlite"])
def test_record_batch_validates_each_batch_once(tmp_path, monkeypatch, backend_kind):
    """The manager leaves full validation to the backend: one pass per batch.

    The manager and the backend both validated, so a sqlite batch ran the
    per-event ID lookup twice.
    """
    from contextlib import ExitStack, contextmanager

    from nooa.runtime import event_backend

    real = event_backend.tagged_batch
    calls = []

    @contextmanager
    def counting(events, backend, **kwargs):
        calls.append(len(events))
        with real(events, backend, **kwargs) as tags:
            yield tags

    monkeypatch.setattr(event_backend, "tagged_batch", counting)
    with ExitStack() as stack:
        manager = EventManager()
        if backend_kind == "sqlite":
            storage = stack.enter_context(SQLiteStorageManager(tmp_path / "once.db"))
            manager = EventManager(backend=storage.event_backend)
        events = [Metadata(description="one"), Metadata(description="two")]
        assert manager.record_batch(events) == ["1", "2"]
        assert calls == [2]
        # Validation still runs: a reused event is rejected by the backend.
        with pytest.raises(ValueError, match="fresh events"):
            manager.record_batch([events[0]])
        assert calls == [2, 1]
