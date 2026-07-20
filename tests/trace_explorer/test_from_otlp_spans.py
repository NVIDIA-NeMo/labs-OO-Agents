# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public TraceExplorer.from_otlp_spans builder (#340).

Verifies that the builder produces a working explorer and that its per-call
``quiet`` handling does not mutate the process-wide quiet mode — the race the
explorer routes used to be exposed to.
"""

import pytest

from nooa.trace_explorer.explorer import (
    TraceExplorer,
    get_quiet_mode,
    set_quiet_mode,
)


def _make_otlp_spans():
    return [
        {
            "traceId": "t1",
            "spanId": "root_agent_01",
            "name": "Router.handle",
            "kind": 1,
            "startTimeUnixNano": "1000000000",
            "endTimeUnixNano": "9000000000",
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                {"key": "agent.name", "value": {"stringValue": "Router"}},
                {"key": "agent.method", "value": {"stringValue": "handle"}},
                {"key": "agent.call_id", "value": {"stringValue": "call_root"}},
            ],
            "status": {"code": 1},
            "events": [],
            "_resource": {},
        },
    ]


def test_from_otlp_spans_builds_explorer():
    explorer = TraceExplorer.from_otlp_spans(
        _make_otlp_spans(),
        trace_file="viewer://sess-1",
    )
    assert isinstance(explorer, TraceExplorer)
    assert explorer.trace_file == "viewer://sess-1"
    assert len(explorer.sessions) >= 1


def test_from_otlp_spans_does_not_touch_global_quiet_mode():
    # Whatever the ambient quiet mode is, from_otlp_spans must restore it.
    for ambient in (False, True):
        set_quiet_mode(ambient)
        try:
            TraceExplorer.from_otlp_spans(
                _make_otlp_spans(),
                trace_file="viewer://sess-1",
                quiet=not ambient,
            )
            assert get_quiet_mode() is ambient
        finally:
            set_quiet_mode(False)


def test_from_otlp_spans_extract_eval_toggle():
    explorer = TraceExplorer.from_otlp_spans(
        _make_otlp_spans(),
        trace_file="viewer://sess-1",
        extract_eval=False,
    )
    assert explorer.eval_result is None


def test_from_otlp_spans_empty():
    explorer = TraceExplorer.from_otlp_spans([], trace_file="viewer://empty")
    assert explorer.sessions == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
