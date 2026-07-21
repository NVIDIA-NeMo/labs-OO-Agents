# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Presentation utilities for NVIDIA OO Agents Jupyter slides.

Usage in notebook:
    from presentation_utils import setup, trace, agent_pprint, list_traces
    setup()

    from nooa.unifiedllm import get_llm_client
    llm = get_llm_client("claude-sonnet-4-5-20250514")
"""

from pathlib import Path
from uuid import uuid4

from IPython.display import Markdown, display

TRACE_DIR = Path("traces/presentation")

# --- Tracing ---


def setup():
    """Load env and enable tracing."""
    from dotenv import load_dotenv

    load_dotenv(override=True)

    from nooa.tracing import enable_tracing, exporters

    enable_tracing(exporters=[exporters.jsonl(TRACE_DIR)])
    print(f"Tracing enabled: {TRACE_DIR}")


def trace(name: str):
    """Point tracing to a new session for this demo."""
    from nooa.tracing import set_session

    uid = uuid4().hex[:8]
    set_session(f"{name}_{uid}")


def list_traces():
    """List all trace files with span counts."""

    for trace_file in sorted(
        f
        for f in TRACE_DIR.glob("*.jsonl")
        if not f.name.endswith(".annotations.jsonl") and not f.name.endswith(".noo-eval.jsonl")
    ):
        with open(trace_file) as f:
            span_count = sum(1 for _ in f)
        print(f"  {trace_file.name:50s} ({span_count} spans)")


# --- Pretty-print agent events ---

_active_unsubs: list = []


def agent_pprint(agent):
    """Install callbacks to pretty-print code and execution output."""
    for unsub in _active_unsubs:
        unsub()
    _active_unsubs.clear()

    def on_tool_call(event):
        if event.name == "execute_python":
            code = event.arguments.get("code", "")
            display(Markdown(f"**Generated Code:**\n```python\n{code}\n```"))
        elif event.name == "return_result":
            display(Markdown(f"**Return Result:** `{event.arguments}`"))

    def on_python_output(event):
        parts = []
        if event.stdout:
            parts.append(f"```\n{event.stdout}\n```")
        if event.error:
            parts.append(f"**Error:** `{event.error}`")
        if event.value is not None:
            parts.append(f"**Result:** `{event.value}`")
        if parts:
            display(Markdown("**Execution Output:**\n\n" + "\n\n".join(parts)))

    _active_unsubs.append(agent.event_manager.on("ToolCallEvent", on_tool_call))
    _active_unsubs.append(agent.event_manager.on("PythonOutput", on_python_output))
    print(f"Verbose output enabled for {type(agent).__name__}")
