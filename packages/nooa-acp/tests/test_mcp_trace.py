# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MCP diagnostic records the handoff without capturing private payloads."""

import asyncio
import json

import pytest
from acp.connection import StreamDirection, StreamEvent
from nooa_acp._mcp_trace import MCPHandoffTrace


def test_trace_selects_only_mcp_names_and_transports(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = MCPHandoffTrace(path)
    secret = "DO-NOT-RECORD-THIS"
    trace(
        StreamEvent(
            StreamDirection.INCOMING,
            {
                "method": "session/new",
                "id": secret,
                "params": {
                    "cwd": secret,
                    "_meta": {"secret": secret},
                    "mcpServers": [
                        {
                            "name": "stdio_probe",
                            "command": secret,
                            "args": [secret],
                            "env": [{"name": "API_KEY", "value": secret}],
                        },
                        {
                            "name": "http_probe",
                            "type": "http",
                            "url": secret,
                            "headers": [{"name": "Authorization", "value": secret}],
                        },
                    ],
                },
            },
        )
    )
    for direction, method in (
        (StreamDirection.INCOMING, "session/prompt"),
        (StreamDirection.OUTGOING, "session/update"),
        (StreamDirection.OUTGOING, "session/new"),
    ):
        trace(StreamEvent(direction, {"method": method, "params": {"secret": secret}}))

    content = path.read_text()
    assert secret not in content
    started, request = map(json.loads, content.splitlines())
    assert started == {"pid": request["pid"], "event": "trace_started"}
    assert request == {
        "pid": started["pid"],
        "event": "session/new",
        "mcpServersField": "list",
        "servers": [
            {"name": "stdio_probe", "transport": "stdio"},
            {"name": "http_probe", "transport": "http"},
        ],
    }


@pytest.mark.parametrize(
    ("params", "state"),
    [
        ({}, "missing"),
        ({"mcpServers": []}, "list"),
        ({"mcpServers": None}, "null"),
        ({"mcpServers": "invalid-secret-payload"}, "invalid"),
    ],
)
def test_trace_distinguishes_missing_empty_and_malformed_fields(tmp_path, params, state):
    path = tmp_path / "trace.jsonl"
    trace = MCPHandoffTrace(path)
    trace(StreamEvent(StreamDirection.INCOMING, {"method": "session/load", "params": params}))
    record = json.loads(path.read_text().splitlines()[-1])
    assert record["mcpServersField"] == state
    assert record["servers"] == []


def test_trace_requires_opt_in(monkeypatch):
    monkeypatch.delenv("NOOA_ACP_MCP_TRACE", raising=False)
    assert MCPHandoffTrace.from_env() is None


def test_trace_handles_a_write_failure_on_a_later_event(tmp_path, caplog):
    """A write failure on a later event -- not the constructor's own initial
    write -- must disable tracing and log a warning. Breaking the path only
    after successful construction keeps this from passing merely because
    __init__'s own eager write already failed and disabled tracing before
    the event under test ever ran (which happened when both writes used the
    same already-broken path: __call__'s "if not self._enabled: return"
    guard would short-circuit and never reach _write again).
    """
    trace = MCPHandoffTrace(tmp_path / "trace.jsonl")
    assert trace._enabled is True
    assert len(caplog.records) == 0

    trace._path = tmp_path  # Cannot append to a directory.
    trace(StreamEvent(StreamDirection.INCOMING, {"method": "session/new"}))
    assert trace._enabled is False
    assert len(caplog.records) == 1
    assert "Cannot write ACP MCP handoff trace" in caplog.text


async def test_trace_write_does_not_block_the_running_event_loop(tmp_path):
    """__call__ runs inline on acp.connection's sync receive loop; the actual
    file write must be offloaded to a worker thread from within a running
    event loop rather than blocking it, while still landing durably.
    """
    path = tmp_path / "trace.jsonl"
    trace = MCPHandoffTrace(path)
    trace(StreamEvent(StreamDirection.INCOMING, {"method": "session/new"}))
    # The offloaded write is fire-and-forget; give the executor thread a
    # scheduling tick to finish before checking the file landed.
    for _ in range(50):
        if path.exists() and path.read_text().count("\n") >= 2:
            break
        await asyncio.sleep(0.01)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["event"] for line in lines] == ["trace_started", "session/new"]
