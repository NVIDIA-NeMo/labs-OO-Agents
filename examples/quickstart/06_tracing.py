# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 06: Tracing — all method types traced with parent-child spans.

Tracing is automatic: if `nooa start-dev` is running, traces appear
in the viewer without any setup code. Just create an agent and go.

Workflow:
  1. nooa start-dev   # start trace viewer and OTLP receiver
  2. uv run python examples/quickstart/06_tracing.py
  3. View traces at http://localhost:5001
"""

from nooa import hidden
from nooa.util.quickstart import *


class MathAgent(Agent, llm=llm):
    """Agent that performs calculations with full tracing."""

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
        """Explain in one sentence why {expression} equals {result}."""
        ...

    @hidden
    async def _format(self, value: float) -> str:
        """Private helper — formats the result for display."""
        return f"{value:g}"


@autorun
async def main():
    # Tracing is auto-enabled when `nooa start-dev` is running.
    # To write JSONL files instead, call enable_tracing(trace_dir="./traces") explicitly.

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
    print("\nTrace Viewer: http://localhost:5001")
    print("=" * 80)
