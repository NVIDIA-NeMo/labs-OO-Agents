# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility tests for session lifecycle event names."""

from nooa.events import TuiSessionCleared, TuiSessionResumed
from nooa.runtime.event_manager import EventManager
from nooa.sessions import SessionCleared, SessionResumed


def test_old_tui_resume_subscriber_receives_neutral_event():
    manager = EventManager()
    received = []
    manager.on("TuiSessionResumed", received.append)

    event = SessionResumed(session_id="one", restored=True)
    manager.add(event)

    assert received == [event]
    assert "handler_aliases" not in type(event).model_fields


def test_old_tui_clear_subscriber_receives_neutral_event():
    manager = EventManager()
    received = []
    manager.on("TuiSessionCleared", received.append)

    event = SessionCleared(session_id="two")
    manager.add(event)

    assert received == [event]


def test_neutral_resume_subscriber_receives_old_tui_event():
    manager = EventManager()
    received = []
    manager.on("SessionResumed", received.append)

    event = TuiSessionResumed(session_id="one", restored=True)
    manager.add(event)

    assert received == [event]


def test_neutral_clear_subscriber_receives_old_tui_event():
    manager = EventManager()
    received = []
    manager.on("SessionCleared", received.append)

    event = TuiSessionCleared(session_id="two")
    manager.add(event)

    assert received == [event]
