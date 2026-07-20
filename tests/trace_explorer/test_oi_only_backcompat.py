# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The trace-explorer must parse OpenInference-only traces.

The explorer was migrated to read OpenInference-standard I/O attrs
(``input.value`` / ``output.value``) first, falling back to the legacy
nooa-native attrs (``code``, ``result``, ``agent.args``/``kwargs``,
``agent.result``, ``generation.result``, ``tool.arguments``). These tests build
synthetic spans that carry **only** the OI attrs (no native I/O attrs at all) —
simulating a future OI-only export — and assert the explorer still recovers
code, stdout, returned value, agent args/kwargs and results.

The existing native-attr tests in ``test_explorer.py`` remain the
backwards-compat guard; this file is the forwards-compat guard.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nooa.trace_explorer import TraceExplorer
from nooa.trace_explorer.explorer import ExecutionTurn


def _otlp_attrs(flat: dict) -> list[dict]:
    out = []
    for k, v in flat.items():
        if isinstance(v, bool):
            out.append({"key": k, "value": {"boolValue": v}})
        elif isinstance(v, int):
            out.append({"key": k, "value": {"intValue": str(v)}})
        else:
            out.append({"key": k, "value": {"stringValue": str(v)}})
    return out


def _span(name: str, span_id: str, attrs: dict, *, parent: str | None = None, status_ok=True):
    span = {
        "name": name,
        "spanId": span_id,
        "startTimeUnixNano": "1000000000",
        "endTimeUnixNano": "2000000000",
        "attributes": _otlp_attrs(attrs),
        "status": {"code": 1 if status_ok else 2},
        "events": [],
    }
    if parent:
        span["parentSpanId"] = parent
    return span


def _write(spans: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for s in spans:
        f.write(json.dumps(s) + "\n")
    f.close()
    return Path(f.name)


@pytest.mark.asyncio
async def test_oi_only_execution_turn_recovers_code_and_output():
    """An OI-only code_execution span (input.value/output.value, NO native
    ``code``/``result``) must still yield code + stdout + returned_value."""
    exec_output = json.dumps({"stdout": "hello\n", "returned_value": "3"})
    spans = [
        # generation span carries its result only as output.value (no generation.result)
        _span(
            "generation",
            "gen1",
            {
                "generation.id": "aaaaaa000000",
                "agent.name": "TestAgent",
                "agent.method": "run",
                "openinference.span.kind": "CHAIN",
                "output.value": "{'stock': 7}",
                "output.mime_type": "text/plain",
            },
        ),
        # code_execution span carries code/result ONLY in OI attrs
        _span(
            "code_execution",
            "exec1",
            {
                "agent.name": "TestAgent",
                "generation.id": "aaaaaa000000",
                "openinference.span.kind": "TOOL",
                "tool.name": "python_executor",
                "input.value": json.dumps({"code": "result = 1 + 2"}),
                "input.mime_type": "application/json",
                "output.value": exec_output,
                "output.mime_type": "application/json",
            },
        ),
    ]
    trace_file = _write(spans)
    try:
        trace = await TraceExplorer.from_file(trace_file)
        session = trace.sessions[0]
        exec_turns = [t for t in session.turns if isinstance(t, ExecutionTurn)]
        assert exec_turns, f"no execution turn parsed; turns={session.turns}"
        turn = exec_turns[0]
        # code recovered from input.value {"code": ...}
        assert turn.code == "result = 1 + 2"
        # stdout + returned_value recovered from output.value (NOT native `result`)
        assert turn.stdout == "hello\n"
        assert turn.returned_value == "3"
        # generation result recovered from output.value
        assert session.result == "{'stock': 7}"
    finally:
        trace_file.unlink()


@pytest.mark.asyncio
async def test_oi_only_agent_span_recovers_args_kwargs_result():
    """An OI-only AGENT span (input.value={"args","kwargs"}, output.value, NO
    native ``agent.args``/``agent.kwargs``/``agent.result``) must still yield the
    session's args, kwargs and result."""
    spans = [
        _span(
            "method.run",
            "agent1",
            {
                "openinference.span.kind": "AGENT",
                "agent.name": "InventoryAgent",
                "agent.method": "run",
                "agent.call_id": "call-1",
                "input.value": json.dumps({"args": ["how many widgets?"], "kwargs": {"k": 1}}),
                "input.mime_type": "application/json",
                "output.value": "{'stock': 7}",
                "output.mime_type": "text/plain",
            },
        ),
    ]
    trace_file = _write(spans)
    try:
        trace = await TraceExplorer.from_file(trace_file)
        session = trace.sessions[0]
        assert session.agent_name == "InventoryAgent"
        assert session.args == ["how many widgets?"]
        assert session.kwargs == {"k": 1}
        # output.value parsed back (it is valid-ish python/JSON repr string)
        assert session.result in ("{'stock': 7}", {"stock": 7})
    finally:
        trace_file.unlink()


@pytest.mark.asyncio
async def test_oi_only_tool_execution_preview_recovers_arguments():
    """The session output preview reads a return_result tool span's arguments.
    With an OI-only trace those live in ``input.value`` (not ``tool.arguments``);
    the preview must still recover the returned result (explorer.py ~3033)."""
    spans = [
        _span(
            "generation",
            "gen1",
            {
                "generation.id": "cccccc000000",
                "agent.name": "TestAgent",
                "agent.method": "run",
                "openinference.span.kind": "CHAIN",
                "output.value": "done",
            },
        ),
        _span(
            "tool_execution.return_result",
            "tool1",
            {
                "agent.name": "TestAgent",
                "generation.id": "cccccc000000",
                "openinference.span.kind": "TOOL",
                "tool.name": "return_result",
                "input.value": json.dumps({"result": "the-final-answer"}),
                "input.mime_type": "application/json",
                "output.value": "the-final-answer",
            },
        ),
    ]
    trace_file = _write(spans)
    try:
        trace = await TraceExplorer.from_file(trace_file)
        session = trace.sessions[0]
        preview = trace._get_session_output_preview(session)
        assert preview == "the-final-answer", f"preview not recovered from input.value: {preview!r}"
    finally:
        trace_file.unlink()


@pytest.mark.asyncio
async def test_native_only_trace_still_parses_backwards_compat():
    """Backwards-compat guard: a legacy native-only code_execution span (no OI
    attrs) must still parse via the fallback path."""
    exec_output = json.dumps({"stdout": "hi\n", "returned_value": "42"})
    spans = [
        _span(
            "generation",
            "gen1",
            {
                "generation.id": "bbbbbb000000",
                "agent.name": "TestAgent",
                "agent.method": "run",
                "openinference.span.kind": "LLM",  # old traces had LLM here
                "generation.result": "legacy-result",
            },
        ),
        _span(
            "code_execution",
            "exec1",
            {
                "agent.name": "TestAgent",
                "generation.id": "bbbbbb000000",
                "openinference.span.kind": "TOOL",
                "code": "x = 6 * 7",
                "result": exec_output,
            },
        ),
    ]
    trace_file = _write(spans)
    try:
        trace = await TraceExplorer.from_file(trace_file)
        session = trace.sessions[0]
        exec_turns = [t for t in session.turns if isinstance(t, ExecutionTurn)]
        assert exec_turns, "native-only execution turn should still parse"
        assert exec_turns[0].code == "x = 6 * 7"
        assert exec_turns[0].returned_value == "42"
        assert session.result == "legacy-result"
    finally:
        trace_file.unlink()
