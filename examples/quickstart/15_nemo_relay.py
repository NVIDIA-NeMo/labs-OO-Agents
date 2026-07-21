# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 15: NeMo Flow integration — guardrails, intercepts, and trajectory export.

NeMo Flow (the ``nemo_relay`` package — renamed from ``nemo_flow``) is a
multi-language agent runtime that adds execution scope management, lifecycle
events, and a configurable middleware pipeline (guardrails/intercepts) to every
LLM call and tool execution in NVIDIA OO Agents.

This example shows how to:
  1. Register an LLM request intercept (inject a system header into every request)
  2. Register a tool conditional-execution guardrail (block dangerous code)
  3. Subscribe to lifecycle events for console logging
  4. Export an ATIF v1.6 trajectory after the run
  5. Demonstrate nested generation methods (inner LLM+tool calls through NeMo Flow)
  6. Demonstrate how different return types appear in the NeMo Flow pipeline and ATIF

This version uses the event-middleware integration — NeMo Flow is wired through
``event_manager.intercept()`` rather than global hooks.

Prerequisites:
  Install NVIDIA OO Agents with the nemo-flow extra (public PyPI wheels):

    uv sync --extra nemo-relay

Usage:
  uv run python examples/quickstart/15_nemo_relay.py
"""

import json

try:
    import nemo_relay as nemo_flow  # renamed upstream from nemo_flow
except ModuleNotFoundError:
    import sys

    print("SKIP: nemo_relay is not installed.\nInstall with: uv sync --extra nemo-relay")
    sys.exit(0)
from pydantic import BaseModel

from nooa.nemo_flow_middleware import nemo_flow_scope
from nooa.util.quickstart import *

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


class ResearchAgent(Agent, llm=llm):
    """An agent that researches topics and summarises findings.

    Demonstrates nested generation: summarize() calls fact_check() which is
    itself a generation method.  Both the outer and inner LLM + tool calls
    flow through the NeMo Flow middleware pipeline, producing a nested ATIF trace.
    """

    def get_current_year(self) -> int:
        """Return the current year (deterministic helper)."""
        import datetime

        return datetime.datetime.now().year

    async def fact_check(self, claim: str) -> str:
        """Evaluate whether {claim} is plausible given your knowledge.

        Return exactly one word: "plausible" or "dubious".
        """
        ...

    async def summarize(self, topic: str) -> str:
        """Research {topic} using the available tools.

        Steps:
        1. Use self.get_current_year() to anchor your research to the present.
        2. Write a concise 2–3 sentence summary.
        3. Call await self.fact_check(summary) to verify the summary is plausible.
        4. Return the summary.
        """
        ...


# ---------------------------------------------------------------------------
# Agent that intentionally triggers the guardrail
# ---------------------------------------------------------------------------


class GuardrailDemoAgent(Agent, llm=llm):
    """Agent whose method is designed to trigger the code-safety guardrail.

    The prompt explicitly instructs the LLM to use os.system(), which the
    guardrail will block before execution.
    """

    async def run_shell_command(self) -> str:
        """Execute os.system('echo hello') and return the output.

        IMPORTANT: You MUST write exactly this code:
            import os
            os.system('echo hello')
            return_result('done')
        """
        ...


# ---------------------------------------------------------------------------
# Return-type demo: shows how different return types appear in NeMo Flow / ATIF
# ---------------------------------------------------------------------------


class Summary(BaseModel):
    """A structured summary with a title and key bullet points."""

    title: str
    points: list[str]


class ReturnTypeDemoAgent(Agent, llm=llm):
    """Agent that demonstrates how different return types are serialized by NeMo Flow.

    BestEffortAnyCodec is applied to the tool's returned_value before handing
    it to the NeMo Flow pipeline.  Simple / structured types appear as readable JSON
    in ATIF and guardrails; complex objects fall back to a pickle blob.
    """

    async def return_int(self) -> int:
        """Compute 6 * 7. Call return_result() with the integer result."""
        ...

    async def return_dict(self) -> dict:
        """Compute basic stats for [1, 3, 5, 7, 9] and call return_result() with a
        dict containing keys 'mean', 'min', 'max'."""
        ...

    async def return_pydantic(self) -> Summary:
        """Return a Summary about NVIDIA OO Agents with title 'NVIDIA OO Agents' and 2 short key points."""
        ...

    async def return_string(self) -> str:
        """Return the string 'Hello from NeMo Flow!' via return_result()."""
        ...


# ---------------------------------------------------------------------------
# 1. LLM request intercept — inject a tracing header into every LLM request
# ---------------------------------------------------------------------------


def _add_tracing_header(
    name: str,
    request: nemo_flow.LLMRequest,
    annotated: nemo_flow.AnnotatedLLMRequest | None,
) -> tuple[nemo_flow.LLMRequest, nemo_flow.AnnotatedLLMRequest | None]:
    """Inject a custom header into every outgoing LLM request.

    Signature: (name, request, annotated) -> (request, annotated) per nemo_flow intercept API.
    """
    request.headers["X-NemoOO-Trace"] = "nemo-flow-poc"
    return request, annotated


nemo_flow.intercepts.register_llm_request(  # type: ignore[attr-defined]
    "add-tracing-header",
    1,
    False,
    _add_tracing_header,
)

# ---------------------------------------------------------------------------
# 2. Tool conditional-execution guardrail — block dangerous code patterns
# ---------------------------------------------------------------------------

_BLOCKED_PATTERNS = ("os.system", "subprocess.run", "subprocess.call", "shell=True")


def _code_safety_check(tool_name: str, args: dict) -> str | None:
    """Return a rejection reason if the code is dangerous, else None."""
    code = args.get("code", "")
    for pattern in _BLOCKED_PATTERNS:
        if pattern in code:
            return f"Execution of '{pattern}' is not permitted by policy"
    return None  # Allow


nemo_flow.guardrails.register_tool_conditional_execution(  # type: ignore[attr-defined]
    "code-safety",
    10,
    _code_safety_check,
)

# ---------------------------------------------------------------------------
# 3. Event subscriber — console logging of all lifecycle events
# ---------------------------------------------------------------------------


def _log_event(event: nemo_flow.Event) -> None:
    """Print a one-line summary of every NeMo Flow lifecycle event."""
    parts = [f"[{event.kind}]", f"name={event.name!r}"]
    category = getattr(event, "category", None)
    scope_category = getattr(event, "scope_category", None)
    if category:
        parts.append(f"category={category}")
    if scope_category:
        parts.append(f"scope_category={scope_category}")
    data = getattr(event, "data", None)
    if data is not None:
        preview = json.dumps(data, default=str)
        if len(preview) > 120:
            preview = preview[:117] + "..."
        parts.append(f"data={preview}")
    print("  NEMO_FLOW " + " | ".join(parts))


nemo_flow.subscribers.register("event-logger", _log_event)  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Event capture: collect tool outputs at runtime from NeMo Flow events
# ---------------------------------------------------------------------------

_captured_tool_outputs: list[dict] = []


def _capture_tool_output(event: nemo_flow.Event) -> None:
    """Capture tool-call End events and their serialized output for display."""
    if (
        isinstance(event, nemo_flow.ScopeEvent)
        and event.category == "tool"
        and event.scope_category == "end"
        and event.data is not None
    ):
        _captured_tool_outputs.append({"tool": event.name, "output": event.data})


nemo_flow.subscribers.register("tool-output-capture", _capture_tool_output)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@autorun
async def main():
    # 4. ATIF exporter — set up here so its lifetime is explicit and scoped to main()
    exporter = nemo_flow.AtifExporter(
        session_id="qs-13-session",
        agent_name="research-agent",
        agent_version="0.1.0",
    )
    exporter.register("atif-exporter")

    print("Running ResearchAgent under NeMo Flow runtime...\n")

    # nemo_flow_scope() installs NeMo Flow middleware (agent_call, llm_call,
    # execute_python) on the agent's event_manager and wraps everything in a
    # root Agent scope.  All LLM calls and code executions inside this block
    # go through the NeMo Flow pipeline.
    agent = ResearchAgent()
    async with nemo_flow_scope(agent, "research-agent"):
        result = await agent.summarize("recent advances in quantum error correction")

    print("\n--- Result ---")
    print(result)

    # Note: ATIF exports a flat steps[] array — the scope hierarchy
    # (Agent → Function:summarize → Function:fact_check) tracked by NeMo Flow
    # internally is not represented in the exported JSON.
    # Event dispatch is async in nemo_relay; flush before reading the export.
    nemo_flow.subscribers.flush()  # type: ignore[attr-defined]
    trajectory_json = exporter.export_json()
    trajectory = json.loads(trajectory_json)
    trajectory_path = "/tmp/nemo_flow_trajectory_research.json"
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2, default=str)
    print(f"\n--- Research trajectory → {trajectory_path} ---")

    # -----------------------------------------------------------------------
    # Guardrail demo: prove that the code-safety guardrail blocks execution
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Guardrail demo: code-safety guardrail blocks os.system()")
    print("=" * 70)

    guardrail_agent = GuardrailDemoAgent()
    try:
        async with nemo_flow_scope(guardrail_agent, "guardrail-demo"):
            await guardrail_agent.run_shell_command()
        print("\n  ERROR: guardrail did not trigger (unexpected)")
    except Exception as exc:
        print(f"\n  BLOCKED by guardrail: {exc}")
        print("  The code-safety guardrail rejected the os.system() call before execution.")

    # -----------------------------------------------------------------------
    # Return-type demo: run each method and inspect what NeMo Flow captures
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Return-type demo: how different return types appear in NeMo Flow / ATIF")
    print("=" * 70)

    demo_agent = ReturnTypeDemoAgent()
    demo_cases = [
        ("return_int", demo_agent.return_int),
        ("return_dict", demo_agent.return_dict),
        ("return_pydantic", demo_agent.return_pydantic),
        ("return_string", demo_agent.return_string),
    ]

    traj: dict = {}
    for method_name, method in demo_cases:
        _captured_tool_outputs.clear()

        async with nemo_flow_scope(demo_agent, f"demo-{method_name}"):
            value = await method()

        nemo_flow.subscribers.flush()  # type: ignore[attr-defined]
        traj = json.loads(exporter.export_json())

        # Tool outputs are captured by _capture_tool_output subscriber
        tool_outputs = [o for o in _captured_tool_outputs if o["tool"] == "execute_python"]
        nemo_flow_output = tool_outputs[-1]["output"] if tool_outputs else None

        print(f"\n  {method_name}() → Python value: {value!r}")
        if nemo_flow_output is not None:
            nemo_flow_view = json.dumps(nemo_flow_output, default=str)
            if len(nemo_flow_view) > 120:
                nemo_flow_view = nemo_flow_view[:117] + "..."
            if isinstance(nemo_flow_output, dict) and "__nv_pickle__" in nemo_flow_output:
                tag = "pickle blob (complex type — opaque to guardrails)"
            else:
                tag = "readable JSON"
            print(f"  NeMo Flow sees ({tag}): {nemo_flow_view}")
        else:
            print("  NeMo Flow sees: (no tool output captured)")

    demo_traj_path = "/tmp/nemo_flow_trajectory_demo.json"
    with open(demo_traj_path, "w") as f:
        json.dump(traj, f, indent=2, default=str)
    print(f"\n--- Last demo trajectory → {demo_traj_path} ---")

    # Cleanup
    exporter.deregister("atif-exporter")
    nemo_flow.subscribers.deregister("event-logger")  # type: ignore[attr-defined]
    nemo_flow.subscribers.deregister("tool-output-capture")  # type: ignore[attr-defined]
    nemo_flow.intercepts.deregister_llm_request("add-tracing-header")  # type: ignore[attr-defined]
    nemo_flow.guardrails.deregister_tool_conditional_execution("code-safety")  # type: ignore[attr-defined]

    print("\n" + "=" * 70)
    print("What happened under the hood:")
    print("  - nemo_flow_scope() installed middleware on the agent's event_manager")
    print("  - agent_call middleware pushed a ScopeType.Function scope per method")
    print("  - Nested generation: summarize() → fact_check() created nested scopes")
    print("      → Both inner LLM and tool calls flowed through the NeMo Flow pipeline")
    print("  - Every LLM call went through the NeMo Flow LLM pipeline:")
    print("      → add-tracing-header intercept injected X-NemoOO-Trace header")
    print("  - Every execute_python call went through the NeMo Flow tool pipeline:")
    print("      → code-safety guardrail rejected any shell-execution code")
    print("      → returned_value exposed via BestEffortAnyCodec:")
    print("          str / int / dict / list  → native JSON (inspectable)")
    print("          Pydantic model           → model_dump_json() (inspectable)")
    print("          DataFrame / complex obj  → pickle blob (opaque)")
    print("  - ATIF exporter collected all events into trajectory JSON files")
    print("=" * 70)
