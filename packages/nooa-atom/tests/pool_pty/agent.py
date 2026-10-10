# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic stdio ACP fixture; never calls a model or reads user configuration."""

import json
import sys
import time
from pathlib import Path

root = Path(__file__).parent
case = json.loads((root / "case.json").read_text())
pending = None


def record(direction, rpc):
    with (root / "wire.jsonl").open("a") as log:
        log.write(json.dumps({"t": time.monotonic(), "direction": direction, "rpc": rpc}) + "\n")


def send(rpc):
    record("out", rpc)
    print(json.dumps(rpc), flush=True)


def reply(req, result):
    send({"jsonrpc": "2.0", "id": req["id"], "result": result})


for line in sys.stdin:
    req = json.loads(line)
    record("in", req)
    method = req.get("method")
    if method == "initialize":
        reply(
            req,
            {
                "protocolVersion": 1,
                "agentInfo": {"name": "Synthetic Pool Probe", "version": "1"},
                "agentCapabilities": {"loadSession": False},
                "authMethods": [],
            },
        )
    elif method == "session/new":
        reply(req, {"sessionId": "synthetic-session"})
    elif method == "session/prompt":
        pending = req
        send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "synthetic-session",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": case["visible_request"]},
                    },
                },
            }
        )
        send(
            {
                "jsonrpc": "2.0",
                "id": "elicit-1",
                "method": "_poolside/elicitation",
                "params": case["params"],
            }
        )
    elif req.get("id") == "elicit-1":
        if pending:
            reply(pending, {"stopReason": "end_turn"})
            pending = None
    elif "id" in req:
        reply(req, {})
