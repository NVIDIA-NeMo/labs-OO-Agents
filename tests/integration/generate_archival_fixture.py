#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a test fixture: sqlite DB with events filling ~95% of a 262K context window.

Fills events into an agent with SQLiteEventBackend, measures tokens, saves DB when ≥95%.
Run ONCE. The e2e test loads the DB at runtime (instant — no deserialization needed).

Usage:
    python tests/integration/generate_archival_fixture.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nooa import Agent
from nooa.context_blocks.events import ResultStatus, ToolCallEvent, ToolResult
from nooa.events import PythonOutput
from nooa.runtime.actor import _current_llm_var, _current_method_var
from nooa.storage.sqlite import SQLiteStorageManager
from nooa.unifiedllm import CompletionClient

_MODEL_NAME = "openai/nvidia/nvidia/Nemotron-3-Nano-30B-A3B"
_API_BASE = "https://inference-api.nvidia.com/v1"
_CONTEXT_WINDOW = 262_144
_TARGET_FRACTION = 0.95

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DB = FIXTURE_DIR / "archival_95pct.db"


def _fill_events(agent, n_events: int, payload_words: int = 200):
    """Add n_events tool-call + output pairs."""
    base = len(list(agent.event_manager.keys()))
    for i in range(n_events):
        idx = base + i
        tc_id = f"call_{idx}"
        payload = f"data_{idx} " * payload_words
        agent.event_manager.add(
            ToolCallEvent(
                tool_call_id=tc_id,
                name="execute_python",
                arguments={"code": payload},
                result=ToolResult(
                    tool_call_id=tc_id,
                    content=payload,
                    result_status=ResultStatus.COMPLETE,
                ),
            )
        )
        agent.event_manager.add(
            PythonOutput(
                tool_call_id=tc_id,
                execution_count=idx,
                stdout=payload,
                stderr="",
                execution_status=ResultStatus.COMPLETE,
            )
        )


async def _measure_tokens(agent, llm):
    method = type(agent).respond
    llm_token = _current_llm_var.set(llm)
    method_token = _current_method_var.set(method)
    try:
        await agent.runtime._build_messages(method, call_args=(agent, "hi"), call_kwargs={})
    finally:
        _current_llm_var.reset(llm_token)
        _current_method_var.reset(method_token)
    return agent.runtime._last_context_stats


async def main():
    api_key = os.environ.get("NVIDIA_INTERNAL_API_KEY", "")
    if not api_key:
        print("ERROR: NVIDIA_INTERNAL_API_KEY not set")
        sys.exit(1)

    # Create the fixture DB
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    if FIXTURE_DB.exists():
        FIXTURE_DB.unlink()

    llm = CompletionClient(
        model=_MODEL_NAME,
        api_base=_API_BASE,
        api_key=api_key,
        temperature=0,
    )
    ctx_window = llm.context_window or _CONTEXT_WINDOW
    target_tokens = int(ctx_window * _TARGET_FRACTION)
    print(f"Target: {target_tokens:,} tokens ({_TARGET_FRACTION:.0%} of {ctx_window:,})")

    # Create agent with SQLite-backed event manager
    storage = SQLiteStorageManager(str(FIXTURE_DB))

    class A(Agent, llm=llm):
        async def respond(self, prompt: str) -> str:
            """Respond to {prompt}."""
            ...

    agent = A()
    agent.event_manager.set_backend(storage.event_backend)

    batch = 50
    while True:
        _fill_events(agent, batch, payload_words=200)
        n_events = len(list(agent.event_manager.keys()))
        print(f"  Events: {n_events}...", end=" ", flush=True)

        stats = await _measure_tokens(agent, llm)
        print(f"tokens: {stats.total_tokens:,} / {target_tokens:,}")

        if stats.total_tokens >= target_tokens:
            break
        if n_events > 10000:
            print("ERROR: Could not reach target in 10000 events")
            sys.exit(1)

    # Save metadata alongside the DB
    import json

    meta_path = FIXTURE_DIR / "archival_95pct_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "model": _MODEL_NAME,
                "context_window": ctx_window,
                "target_fraction": _TARGET_FRACTION,
                "total_tokens": stats.total_tokens,
                "n_events": n_events,
            },
            f,
            indent=2,
        )

    storage.close()
    print(f"\nFixture saved: {FIXTURE_DB}")
    print(f"  Events: {n_events}")
    print(f"  Tokens: {stats.total_tokens:,}")
    print(f"  DB size: {FIXTURE_DB.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    asyncio.run(main())
