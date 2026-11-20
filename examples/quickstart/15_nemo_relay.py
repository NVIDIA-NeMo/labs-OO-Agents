# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 15: NeMo Relay integration — guardrails, intercepts, and trajectory export.

Register request intercepts, block selected Python calls, log events, and
export agent runs as ATIF trajectories. The examples include nested agent
calls and structured return values.

Prerequisites:
  Install NOOA with the nemo-relay extra (public PyPI wheels):

    uv sync --extra nemo-relay

Usage:
  uv run python examples/quickstart/15_nemo_relay.py

Optional local diagnostics and CPU/memory measurements:
  NOOA_RELAY_DIAGNOSTICS=1 uv run python examples/quickstart/15_nemo_relay.py
"""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Literal

try:
    import nemo_relay
except ModuleNotFoundError:
    import sys

    print("SKIP: nemo_relay is not installed.\nInstall with: uv sync --extra nemo-relay")
    sys.exit(0)
from pydantic import BaseModel

from nooa import hidden
from nooa.nemo_relay_middleware import export_nemo_relay_atif, nemo_relay_scope
from nooa.snapshot import Limits, capture, render_json, render_python
from nooa.util.quickstart import *

_artifact_directory: Path | None = None


@hidden
def _write_artifact(name: str, value: dict) -> Path:
    """Save an export in a private temporary directory."""
    global _artifact_directory
    if _artifact_directory is None:
        _artifact_directory = Path(tempfile.mkdtemp(prefix="nooa-relay-"))
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"{name}-",
        suffix=".json",
        dir=_artifact_directory,
        delete=False,
    ) as output:
        json.dump(value, output, indent=2, default=str)
        return Path(output.name)


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


class ResearchAgent(Agent, llm=llm):
    """Research a topic and check the summary with a nested agent call."""

    async def get_current_year(self) -> int:
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
        1. Use await self.get_current_year() to anchor your research to the present.
        2. Write a concise 2–3 sentence summary.
        3. Call await self.fact_check(summary) to verify the summary is plausible.
        4. Return the summary.
        """
        ...


# ---------------------------------------------------------------------------
# Agent that intentionally triggers the guardrail
# ---------------------------------------------------------------------------


class GuardrailDemoAgent(Agent, llm=llm):
    """Request a shell command to demonstrate the guardrail blocking it."""

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
    """Compare Python return values with their Relay previews."""

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
# 1. Attach a correlation header to Relay events
# ---------------------------------------------------------------------------


def _add_tracing_header(
    name: str,
    request: nemo_relay.LLMRequest,
    annotated: nemo_relay.AnnotatedLLMRequest | None,
) -> nemo_relay.LLMRequestInterceptOutcome:
    """Attach a correlation header to Relay telemetry, not the HTTP transport."""
    updated = nemo_relay.LLMRequest(
        {**request.headers, "X-NemoOO-Trace": "quickstart-15"}, request.content
    )
    return nemo_relay.LLMRequestInterceptOutcome(updated, annotated)


# Use global registrations so rules also apply to nested and concurrent calls.
# Scope-local rules do not transfer to child method scopes.
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
    return None


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
        preview = render_json(capture(data, Limits(max_chars=120)), max_chars=120)
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


@hidden
async def run_demo():
    exporter = nemo_relay.AtifExporter(
        session_id="qs-15-session",
        agent_name="research-agent",
        agent_version="0.1.0",
    )
    exporter.register("atif-exporter")

    print("Running ResearchAgent under NeMo Relay runtime...\n")

    # Enable Relay for the agent's LLM calls and Python execution.
    agent = ResearchAgent()
    async with nemo_relay_scope(agent, "research-agent"):
        result = await agent.summarize("recent advances in quantum error correction")

    print("\n--- Result ---")
    print(result)

    # Flush pending events and export the trajectory.
    trajectory = await export_nemo_relay_atif(exporter)
    trajectory_path = _write_artifact("research", trajectory)
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
        await nemo_relay.subscribers.flush_async()
        exporter.clear()
        _captured_tool_outputs.clear()

        async with nemo_relay_scope(demo_agent, f"demo-{method_name}"):
            value = await method()

        traj = await export_nemo_relay_atif(exporter)

        tool_outputs = [o for o in _captured_tool_outputs if o["tool"] == "execute_python"]
        relay_output = tool_outputs[-1]["output"] if tool_outputs else None

        value_preview = render_python(capture(value, Limits(max_chars=120)))
        print(f"\n  {method_name}() → Python value: {value_preview}")
        if relay_output is not None:
            relay_view = render_json(capture(relay_output, Limits(max_chars=120)), max_chars=120)
            print(f"  NeMo Relay sees (bounded JSON inspection): {relay_view}")
        else:
            print("  NeMo Relay sees: (no tool output captured)")

    demo_traj_path = _write_artifact("demo", traj)
    print(f"\n--- Last demo trajectory → {demo_traj_path} ---")

    # Cleanup
    exporter.deregister("atif-exporter")
    nemo_relay.subscribers.deregister("event-logger")
    nemo_relay.subscribers.deregister("tool-output-capture")
    nemo_relay.intercepts.deregister_llm_request("add-tracing-header")
    nemo_relay.guardrails.deregister_tool_conditional_execution("code-safety")


@autorun
async def main():
    if os.getenv("NOOA_RELAY_DIAGNOSTICS") != "1":
        await run_demo()
        return

    # Enable optional diagnostics for the whole demo.
    config = nemo_relay.plugin.PluginConfig(
        diagnostics={"enabled": True},
        components=[
            nemo_relay.plugin.ComponentSpec(
                kind="resource_metrics",
                config={
                    "measurement_scope": "application_process",
                    "polling": {"enabled": True, "interval_millis": 1000},
                    "cpu": {"enabled": True},
                    "memory": {"enabled": True},
                    "process": {"enabled": False},
                    "disk": {"enabled": False},
                    "gpu": {"enabled": False},
                    "network": {"enabled": False},
                },
            ),
        ],
    )
    async with nemo_relay.plugin.activate(config):
        await run_demo()
        await nemo_relay.subscribers.flush_async()
        snapshot = nemo_relay.diagnostics.reader().snapshot()
        metrics = asdict(await nemo_relay.resource_metrics.collect())
        output = _write_artifact("diagnostics", {"diagnostics": snapshot, "resources": metrics})
        print(f"\n--- Relay diagnostics and CPU/memory snapshot → {output} ---")
