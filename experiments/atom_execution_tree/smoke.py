# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure actual Atom stdio/worker tool streams with execution observation off/on."""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import ToolCallProgress, ToolCallStart

KEY = "nooa.dev/execution"


class Client:
    def __init__(self):
        self.updates = []
        self.root_done = asyncio.Event()

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)
        node = (getattr(update, "field_meta", None) or {}).get(KEY)
        if (
            node
            and node["parentSpanId"] is None
            and isinstance(update, ToolCallProgress)
            and update.status == "completed"
        ):
            self.root_done.set()


async def run(output):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for enabled in (False, True):
        with tempfile.TemporaryDirectory(prefix="atom-tree-") as scratch:
            client = Client()
            args = ["--execution-tree"] if enabled else []
            async with spawn_agent_process(
                client,
                sys.executable,
                str(Path(__file__).with_name("demo.py")),
                *args,
                cwd=scratch,
                use_unstable_protocol=True,
            ) as (connection, _):
                await connection.initialize(PROTOCOL_VERSION)
                session = await connection.new_session(scratch)
                for index in range(2):
                    before = len(client.updates)
                    client.root_done.clear()
                    response = await asyncio.wait_for(
                        connection.prompt(session.session_id, [text_block("run demo")]), 60
                    )
                    if enabled:
                        await asyncio.wait_for(client.root_done.wait(), 10)
                    if index == 1:
                        await connection.close_session(session.session_id)
                    assert response.stop_reason == "end_turn"
                    updates = client.updates[before:]
                    cards = [u for u in updates if isinstance(u, ToolCallStart)]
                    nodes = {
                        u.tool_call_id: u.field_meta[KEY]
                        for u in cards
                        if u.field_meta and KEY in u.field_meta
                    }
                    if enabled:
                        assert {n["nodeType"] for n in nodes.values()} == {
                            "method",
                            "python",
                            "file",
                            "terminal",
                        }
                        assert all(
                            n["parentSpanId"] is None or n["parentSpanId"] in nodes
                            for n in nodes.values()
                        )
                        failed = [
                            u
                            for u in updates
                            if isinstance(u, ToolCallProgress)
                            and u.status == "failed"
                            and u.field_meta
                            and KEY in u.field_meta
                        ]
                        assert (
                            len(failed) == 1
                            and failed[0].field_meta[KEY]["name"] == "DemoVerifier.check"
                        )
                    else:
                        assert not nodes
                    assert (
                        (Path(scratch) / "atom_tree_demo.py")
                        .read_text()
                        .startswith("# Atom execution-tree demo artifact")
                    )
                    serialized = [
                        u.model_dump(mode="json", by_alias=True, exclude_none=True) for u in updates
                    ]
                    filename = f"{'tree' if enabled else 'flat'}-{index}.json"
                    (output / filename).write_text(json.dumps(serialized, indent=2) + "\n")
                    result = {
                        "enabled": enabled,
                        "turn": index + 1,
                        "cards": len(cards),
                        "nodes": len(nodes),
                        "wire_bytes": sum(len(json.dumps(u).encode()) for u in serialized),
                        "capture": filename,
                    }
                    results.append(result)
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    asyncio.run(run(parser.parse_args().output))
