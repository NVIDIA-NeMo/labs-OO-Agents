# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 06: Tracing — all method types traced with parent-child spans.

Tracing is automatic when `nooa start-dev` is reachable and an application does
not configure its own exporters. This example explicitly writes a durable
journal, which can then be imported into the viewer.

Workflow:
  1. uv run nooa start-dev   # start the trace viewer
  2. uv run python examples/quickstart/06_tracing.py
  3. uv run nooa import-traces traces/quickstart-06-journal
  4. View traces at http://localhost:5001
"""

from nooa import hidden
from nooa.util.quickstart import *

with hidden:
    from pathlib import Path

    from nooa.tracing import enable_tracing, exporters

    TRACE_DIR = Path("traces/quickstart-06-journal")


class MathAgent(Agent, llm=llm):
    """Agent that performs calculations with full tracing."""

    @hidden
    async def run(self, expression: str) -> str:
        """Orchestrator: evaluate the expression, then explain it."""
        value = await self.calculate(expression)
        formatted = await self._format(value)
        explanation = await self.explain(expression, formatted)
        return explanation

    async def calculate(self, expression: str) -> float:
        """Evaluate the mathematical expression and return the numeric result."""
        ...

    async def explain(self, expression: str, result: str) -> str:
        """Explain in one sentence why the expression produces the result."""
        ...

    @hidden
    async def _format(self, value: float) -> str:
        """Private helper — formats the result for display."""
        return f"{value:g}"


@autorun
async def main():
    # The journal file keeps message bodies content-addressed instead of
    # repeating the full LLM conversation on every OTLP span.
    enable_tracing(exporters=[exporters.journal_file(TRACE_DIR)])

    agent = MathAgent()
    result = await agent.run("(10 + 5) * 2")
    print(f"Result: {result}")

    print("\n" + "=" * 80)
    print("Spans captured:")
    print("  - run()        regular Python orchestrator")
    print("  - calculate()  ellipsis/LLM method, child of run()")
    print("  - explain()    ellipsis/LLM method, child of run()")
    print("  - _format()    private helper (also traced)")
    print("=" * 80)
    print(f"\nJournal trace: {TRACE_DIR}")
    print(f"Import with: nooa import-traces {TRACE_DIR}")
    print("=" * 80)
