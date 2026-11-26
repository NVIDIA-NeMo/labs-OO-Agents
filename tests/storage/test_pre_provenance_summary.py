# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A summary stored before summaries recorded provenance still loads."""

import json

from nooa.events import Summary
from nooa.storage.sqlite import SQLiteEventBackend


def test_a_summary_without_source_call_ids_loads_with_unknown_provenance(sqlite_conn):
    backend = SQLiteEventBackend(sqlite_conn)
    tag = "1..2"
    summary = Summary(
        summary_tag=tag, replaced_range=(1, 2), summary_text="recap", source_call_ids=["c1"]
    )
    backend.store(tag, summary)
    data = json.loads(sqlite_conn.execute("SELECT data FROM events").fetchone()[0])
    del data["source_call_ids"]  # as a session stored before provenance
    sqlite_conn.execute("UPDATE events SET data = ? WHERE tag = ?", (json.dumps(data), tag))

    backend = SQLiteEventBackend(sqlite_conn)
    loaded = backend.get(tag)
    assert isinstance(loaded, Summary)
    assert loaded.summary_text == "recap"
    assert loaded.related_call_ids == frozenset()
    assert [type(event) for event in backend.all_events()] == [Summary]
