# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MCP diagnostic records the handoff without capturing private payloads."""

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


def test_trace_requires_opt_in_and_handles_file_failure(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("NOOA_ACP_MCP_TRACE", raising=False)
    assert MCPHandoffTrace.from_env() is None

    monkeypatch.setenv("NOOA_ACP_MCP_TRACE", str(tmp_path))  # Cannot append to a directory.
    trace = MCPHandoffTrace.from_env()
    assert trace is not None
    trace(StreamEvent(StreamDirection.INCOMING, {"method": "session/new"}))
    assert len(caplog.records) == 1
    assert "Cannot write ACP MCP handoff trace" in caplog.text
