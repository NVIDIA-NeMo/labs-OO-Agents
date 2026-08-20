# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for event persistence ordering and retry identity."""

import pytest

from nooa.context_blocks import EventBase
from nooa.events import Task
from nooa.runtime.event_backend import InMemoryBackend
from nooa.runtime.event_manager import EventManager


class FailBeforeStoreBackend(InMemoryBackend):
    """Fail once before writing the event."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def store(self, tag: str, event: EventBase) -> None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("store failed before commit")
        super().store(tag, event)


class CommitThenRaiseBackend(InMemoryBackend):
    """Commit once, then lose the acknowledgement returned by store()."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def store(self, tag: str, event: EventBase) -> None:
        super().store(tag, event)
        if not self.failed:
            self.failed = True
            raise RuntimeError("acknowledgement lost after commit")


def test_recorded_event_is_persisted_before_observer_notification() -> None:
    manager = EventManager()
    visible_during_callback: list[bool] = []
    manager.on(
        "Task", lambda event: visible_during_callback.append(manager.get(event.id) is not None)
    )

    tag = manager.add(Task(prompt="ordered"))

    assert visible_during_callback == [True]
    assert manager.get(tag) is not None


def test_store_failure_does_not_notify_an_unpersisted_event() -> None:
    backend = FailBeforeStoreBackend()
    manager = EventManager(backend)
    notifications: list[str] = []
    manager.on("Task", lambda event: notifications.append(event.id))
    event = Task(prompt="retry")

    with pytest.raises(RuntimeError, match="before commit"):
        manager.add(event)

    assert notifications == []
    assert len(backend) == 0

    tag = manager.add(event)
    assert notifications == [event.id]
    assert manager.get(tag) is not None


def test_committed_event_retry_reuses_tag_and_notifies_once() -> None:
    backend = CommitThenRaiseBackend()
    manager = EventManager(backend)
    notifications: list[str] = []
    manager.on("Task", lambda event: notifications.append(event.id))
    event = Task(prompt="ambiguous acknowledgement")

    with pytest.raises(RuntimeError, match="after commit"):
        manager.add(event)

    retry_tag = manager.add(event)

    assert retry_tag == event.tag
    assert len(backend) == 1
    assert notifications == [event.id]


def test_observer_failure_does_not_undo_persisted_event() -> None:
    manager = EventManager()

    def failing_observer(_event: Task) -> None:
        raise RuntimeError("observer failed")

    manager.on("Task", failing_observer)
    tag = manager.add(Task(prompt="observer failure"))

    assert manager.get(tag) is not None
