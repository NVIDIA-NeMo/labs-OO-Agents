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


@pytest.mark.parametrize("backend_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("entrypoint", ["manager", "backend"])
@pytest.mark.parametrize(
    "unsupported",
    [
        "frozen_model",
        "frozen_tag",
        "assignment_validation",
        "setter",
        "handler",
        "instance_handler",
        "cached_handler",
        "descriptor",
        "classvar",
    ],
)
def test_batch_rejects_unsafe_tag_publication_before_any_mutation(
    tmp_path, monkeypatch, backend_kind, entrypoint, unsupported
):
    from contextlib import ExitStack
    from typing import ClassVar

    from pydantic import Field, field_validator

    from nooa.runtime.event_backend import InMemoryBackend

    calls = []

    class FrozenMetadata(Metadata):
        model_config = {"frozen": True}

    class FrozenTagMetadata(Metadata):
        tag: str | None = Field(default=None, frozen=True)

    class ValidatedMetadata(Metadata):
        model_config = {"validate_assignment": True}

        @field_validator("tag")
        @classmethod
        def reject_tag(cls, value):
            calls.append("validator")
            raise ValueError("no tag assignments")

    class SetterMetadata(Metadata):
        def __setattr__(self, name, value):
            if name == "tag":
                calls.append("setter")
                raise ValueError("no tag assignments")
            return super().__setattr__(name, value)

    class InheritedSetterMetadata(SetterMetadata):
        pass

    class HandlerMetadata(Metadata):
        def _setattr_handler(self, name, value):
            if name == "tag":
                calls.append("handler")
                raise ValueError("no tag assignments")
            return super()._setattr_handler(name, value)

    class DescriptorMetadata(Metadata):
        pass

    class ClassVarMetadata(Metadata):
        tag: ClassVar[str | None] = None

    def reject(*args):
        calls.append("custom handler")
        raise ValueError("no tag assignments")

    classes = {
        "frozen_model": FrozenMetadata,
        "frozen_tag": FrozenTagMetadata,
        "assignment_validation": ValidatedMetadata,
        "setter": InheritedSetterMetadata,
        "handler": HandlerMetadata,
        "descriptor": DescriptorMetadata,
        "classvar": ClassVarMetadata,
    }
    # Explicit discriminator avoids the frozen subclass's initialization setter.
    cls = classes.get(unsupported, Metadata)
    bad = cls(description="unsupported", event_type=cls.__name__)
    if unsupported == "descriptor":
        monkeypatch.setattr(
            DescriptorMetadata, "tag", property(lambda self: None, reject), raising=False
        )
    elif unsupported == "instance_handler":
        bad.__dict__["_setattr_handler"] = reject
    elif unsupported == "cached_handler":
        if not hasattr(cls, "__pydantic_setattr_handlers__"):
            pytest.skip("Pydantic version has no cached assignment handlers")
        monkeypatch.setitem(cls.__pydantic_setattr_handlers__, "tag", reject)

    with ExitStack() as stack:
        backend = InMemoryBackend()
        path = tmp_path / "unsafe.db"
        if backend_kind == "sqlite":
            backend = stack.enter_context(SQLiteStorageManager(path)).event_backend
        manager = EventManager(backend=backend)
        # Use normal add for setup: custom handler cache is intentionally unsafe
        # for batch publication, but setup itself never assigns a tag.
        existing = Metadata(description="existing", tag="1")
        backend.store("1", existing)
        fresh = Metadata(description="fresh")
        events = [fresh, bad]
        snapshots = [(dict(event.__dict__), set(event.model_fields_set)) for event in events]
        tags_before = backend.active_tags()
        counters_before = (backend._next_tag_num, getattr(backend, "_insertion_counter", None))
        seen = []
        manager.on("*", seen.append)
        if backend_kind == "sqlite":
            monkeypatch.setattr(backend, "_serialize", reject)
        append = manager.record_batch if entrypoint == "manager" else backend.append_batch
        with pytest.raises(ValueError, match="safely writable tags"):
            append(events)
        assert calls == [] and seen == []
        assert [event.tag for event in events] == [None, None]
        assert [
            (dict(event.__dict__), set(event.model_fields_set)) for event in events
        ] == snapshots
        assert backend.active_tags() == tags_before
        assert (
            backend._next_tag_num,
            getattr(backend, "_insertion_counter", None),
        ) == counters_before
        assert len(backend) == 1
        if backend_kind == "sqlite":
            with sqlite3.connect(path) as connection:
                assert connection.execute("SELECT tag FROM events").fetchall() == [("1",)]
                assert connection.execute("SELECT tag FROM active_tags").fetchall() == [("1",)]
        monkeypatch.undo()
        assert append([fresh, Metadata(description="next")]) == ["2", "3"]
        assert fresh.tag == "2" and "tag" in fresh.model_fields_set
        assert backend.active_tags() == ["1", "2", "3"]


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
