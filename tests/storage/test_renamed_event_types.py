# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rows stored under a renamed event class still load, as the current class."""

import json

import pytest

from nooa.events import SessionCleared, SessionResumed
from nooa.storage.sqlite import SQLiteEventBackend


@pytest.mark.parametrize(
    ("event", "old_type"),
    [
        (SessionResumed(session_id="s1", restored=True), "TuiSessionResumed"),
        (SessionCleared(session_id="s2"), "TuiSessionCleared"),
    ],
)
def test_an_old_session_event_row_loads_as_the_renamed_class(sqlite_conn, event, old_type):
    backend = SQLiteEventBackend(sqlite_conn)
    tag = "1"
    backend.store(tag, event)
    data = json.loads(sqlite_conn.execute("SELECT data FROM events").fetchone()[0])
    data["event_type"] = old_type  # as a session stored before the rename
    sqlite_conn.execute(
        "UPDATE events SET event_type = ?, data = ? WHERE tag = ?",
        (old_type, json.dumps(data), tag),
    )
    loaded = SQLiteEventBackend(sqlite_conn).get(tag)
    assert type(loaded) is type(event)
    assert loaded.event_type == type(event).__name__
    assert loaded.session_id == event.session_id
