# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 15: NeMo Relay integration — guardrails, intercepts, and trajectory export.

NeMo Relay is a multi-language agent runtime that adds execution scope
management, lifecycle events, and a configurable middleware pipeline
(guardrails/intercepts) to every LLM call and tool execution in NOOA.
It is wired in through ``event_manager.intercept()``.

This example shows how to:
  1. Register an LLM request intercept (inject a system header into every request)
  2. Register a tool conditional-execution guardrail (block dangerous code)
  3. Subscribe to lifecycle events for console logging
  4. Export an ATIF trajectory after the run
  5. Demonstrate nested generation methods (inner LLM+tool calls through NeMo Relay)
  6. Demonstrate how different return types appear in the NeMo Relay pipeline and ATIF

Prerequisites:
  Install NOOA with the nemo-relay extra (public PyPI wheels):

    uv sync --extra nemo-relay

Usage:
  uv run python examples/quickstart/15_nemo_relay.py
"""

import json
from typing import Literal

try:
    import nemo_relay
except ModuleNotFoundError:
    import sys

    print("SKIP: nemo_relay is not installed.\nInstall with: uv sync --extra nemo-relay")
    sys.exit(0)
from pydantic import BaseModel

from nooa.nemo_relay_middleware import nemo_relay_scope
from nooa.util.quickstart import *

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


class ResearchAgent(Agent, llm=llm):
    """An agent that researches topics and summarises findings.

    Demonstrates nested generation: summarize() calls fact_check() which is
    itself a generation method.  Both the outer and inner LLM + tool calls
    flow through the NeMo Relay middleware pipeline, producing a nested ATIF trace.
    """

    def get_current_year(self) -> int:
        """Return the current year (deterministic helper)."""
        import datetime

        return datetime.datetime.now().year

    @strategy(PredictStrategy())
    async def fact_check(self, claim: str) -> Literal["plausible", "dubious"]:
        """Evaluate whether the claim is plausible given your knowledge.

        Return exactly one word: "plausible" or "dubious".
        """
        ...

    async def summarize(self, topic: str) -> str:
        """Research the topic using the available tools.

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
# Return-type demo: shows how different return types appear in NeMo Relay / ATIF
# ---------------------------------------------------------------------------


class Summary(BaseModel):
    """A structured summary with a title and key bullet points."""

    title: str
    points: list[str]


class ReturnTypeDemoAgent(Agent, llm=llm):
    """Agent that demonstrates how different return types are serialized by NeMo Relay.

    BestEffortAnyCodec is applied to the tool's returned_value before handing
    it to the NeMo Relay pipeline.  Simple / structured types appear as readable JSON
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
        """Return a Summary about NOOA with title 'NOOA' and 2 short key points."""
        ...

    async def return_string(self) -> str:
        """Return the string 'Hello from NeMo Relay!' via return_result()."""
        ...


# ---------------------------------------------------------------------------
# 1. LLM request intercept — inject a tracing header into every LLM request
# ---------------------------------------------------------------------------


def _add_tracing_header(
    name: str,
    request: nemo_relay.LLMRequest,
    annotated: nemo_relay.AnnotatedLLMRequest | None,
) -> nemo_relay.LLMRequestInterceptOutcome:
    """Inject a custom header into every outgoing LLM request."""
    request.headers["X-NemoOO-Trace"] = "quickstart-15"
    return nemo_relay.LLMRequestInterceptOutcome(request, annotated)


nemo_relay.intercepts.register_llm_request(
    "add-tracing-header",
    1,
    False,
    _add_tracing_header,
)

# ---------------------------------------------------------------------------
# 2. Tool conditional-execution guardrail — block dangerous code patterns
# ---------------------------------------------------------------------------

_BLOCKED_PATTERNS = ("os.system", "subprocess.run", "subprocess.call", "shell=True")


def _code_safety_check(tool_name: str, args: nemo_relay.Json) -> str | None:
    """Return a rejection reason if the code is dangerous, else None."""
    code = args.get("code", "") if isinstance(args, dict) else ""
    for pattern in _BLOCKED_PATTERNS:
        if pattern in str(code):
            return f"Execution of '{pattern}' is not permitted by policy"
    return None  # Allow


nemo_relay.guardrails.register_tool_conditional_execution(
    "code-safety",
    10,
    _code_safety_check,
)

# ---------------------------------------------------------------------------
# 3. Event subscriber — console logging of all lifecycle events
# ---------------------------------------------------------------------------


def _log_event(event: nemo_relay.Event) -> None:
    """Print a one-line summary of every NeMo Relay lifecycle event."""
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
    print("  NEMO_RELAY " + " | ".join(parts))


nemo_relay.subscribers.register("event-logger", _log_event)

# ---------------------------------------------------------------------------
# Event capture: collect tool outputs at runtime from NeMo Relay events
# ---------------------------------------------------------------------------

_captured_tool_outputs: list[dict] = []


def _capture_tool_output(event: nemo_relay.Event) -> None:
    """Capture tool-call End events and their serialized output for display."""
    if (
        isinstance(event, nemo_relay.ScopeEvent)
        and event.category == "tool"
        and event.scope_category == "end"
        and event.data is not None
    ):
        _captured_tool_outputs.append({"tool": event.name, "output": event.data})


nemo_relay.subscribers.register("tool-output-capture", _capture_tool_output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@autorun
async def main():
    # 4. ATIF exporter — set up here so its lifetime is explicit and scoped to main()
    exporter = nemo_relay.AtifExporter(
        session_id="qs-15-session",
        agent_name="research-agent",
        agent_version="0.1.0",
    )
    exporter.register("atif-exporter")

    print("Running ResearchAgent under NeMo Relay runtime...\n")

    # nemo_relay_scope() installs NeMo Relay middleware (agent_call, llm_call,
    # execute_python) on the agent's event_manager and wraps everything in a
    # root Agent scope.  All LLM calls and code executions inside this block
    # go through the NeMo Relay pipeline.
    agent = ResearchAgent()
    async with nemo_relay_scope(agent, "research-agent"):
        result = await agent.summarize("recent advances in quantum error correction")

    print("\n--- Result ---")
    print(result)

    # Note: steps[] is a flat array, but the scope hierarchy is not lost —
    # each step carries it in extra.ancestry.parent_name (e.g. the nested
    # fact_check turns report parent_name="ResearchAgent.fact_check").
    # Event dispatch is async in nemo_relay; flush before reading the export.
    nemo_relay.subscribers.flush()
    trajectory_json = exporter.export_json()
    trajectory = json.loads(trajectory_json)
    trajectory_path = "/tmp/nemo_relay_trajectory_research.json"
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
        async with nemo_relay_scope(guardrail_agent, "guardrail-demo"):
            await guardrail_agent.run_shell_command()
        print("\n  ERROR: guardrail did not trigger (unexpected)")
    except Exception as exc:
        print(f"\n  BLOCKED by guardrail: {exc}")
        print("  The code-safety guardrail rejected the os.system() call before execution.")

    # -----------------------------------------------------------------------
    # Return-type demo: run each method and inspect what NeMo Relay captures
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Return-type demo: how different return types appear in NeMo Relay / ATIF")
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

        async with nemo_relay_scope(demo_agent, f"demo-{method_name}"):
            value = await method()

        nemo_relay.subscribers.flush()
        traj = json.loads(exporter.export_json())

        # Tool outputs are captured by _capture_tool_output subscriber
        tool_outputs = [o for o in _captured_tool_outputs if o["tool"] == "execute_python"]
        relay_output = tool_outputs[-1]["output"] if tool_outputs else None

        print(f"\n  {method_name}() → Python value: {value!r}")
        if relay_output is not None:
            relay_view = json.dumps(relay_output, default=str)
            if len(relay_view) > 120:
                relay_view = relay_view[:117] + "..."
            if isinstance(relay_output, dict) and "__nv_pickle__" in relay_output:
                tag = "pickle blob (complex type — opaque to guardrails)"
            else:
                tag = "readable JSON"
            print(f"  NeMo Relay sees ({tag}): {relay_view}")
        else:
            print("  NeMo Relay sees: (no tool output captured)")

    demo_traj_path = "/tmp/nemo_relay_trajectory_demo.json"
    with open(demo_traj_path, "w") as f:
        json.dump(traj, f, indent=2, default=str)
    print(f"\n--- Last demo trajectory → {demo_traj_path} ---")

    # Cleanup
    exporter.deregister("atif-exporter")
    nemo_relay.subscribers.deregister("event-logger")
    nemo_relay.subscribers.deregister("tool-output-capture")
    nemo_relay.intercepts.deregister_llm_request("add-tracing-header")
    nemo_relay.guardrails.deregister_tool_conditional_execution("code-safety")

    print("\n" + "=" * 70)
    print("What happened under the hood:")
    print("  - nemo_relay_scope() installed middleware on the agent's event_manager")
    print("  - agent_call middleware pushed a ScopeType.Function scope per traced")
    print("      async method (sync/@no_trace/static/class methods are not wrapped)")
    print("  - Nested generation: summarize() → fact_check() created nested scopes")
    print("      → Both inner LLM and tool calls flowed through the NeMo Relay pipeline")
    print("  - Every LLM call went through the NeMo Relay LLM pipeline:")
    print("      → add-tracing-header intercept injected X-NemoOO-Trace header")
    print("  - Every execute_python call went through the NeMo Relay tool pipeline:")
    print("      → code-safety guardrail rejected any shell-execution code")
    print("      → returned_value exposed via BestEffortAnyCodec:")
    print("          str / int / dict / list  → native JSON (inspectable)")
    print("          Pydantic model           → __nv_pydantic__ envelope (inspectable)")
    print("          DataFrame / complex obj  → pickle blob (opaque)")
    print("  - ATIF exporter collected all events into trajectory JSON files")
    print("=" * 70)
