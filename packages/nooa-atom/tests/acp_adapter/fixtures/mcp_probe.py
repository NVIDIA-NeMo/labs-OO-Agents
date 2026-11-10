# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free stdio MCP fixture for forwarding and persistence tests.

Launch with --journal /absolute/path/to/log.jsonl.
Only protocol replies go to stdout. The journal records starts and tool calls.
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    token = uuid.uuid4().hex

    def record(event: str, **fields) -> dict:
        data = {"event": event, "pid": os.getpid(), "server_token": token, **fields}
        args.journal.parent.mkdir(parents=True, exist_ok=True)
        with args.journal.open("a") as stream:
            stream.write(json.dumps(data) + "\n")
        return data

    record("start")
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        method = request.get("method")
        params = request.get("params", {})
        if method == "initialize":
            result = {
                "protocolVersion": params["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "nooa-mcp-probe", "version": "1.0.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "probe",
                        "description": "Echo a nonce and prove an MCP tool call reached this server.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"nonce": {"type": "string"}},
                            "required": ["nonce"],
                            "additionalProperties": False,
                        },
                    }
                ]
            }
        elif method == "tools/call" and params.get("name") == "probe":
            nonce = params.get("arguments", {}).get("nonce")
            if not isinstance(nonce, str):
                result = {
                    "isError": True,
                    "content": [{"type": "text", "text": "nonce must be a string"}],
                }
            else:
                data = record("probe", nonce=nonce)
                result = {"content": [{"type": "text", "text": json.dumps(data)}]}
        else:
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "error": {
                            "code": -32601,
                            "message": f"Unsupported probe method: {method}",
                        },
                    }
                ),
                flush=True,
            )
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)


if __name__ == "__main__":
    main()
