# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark controller with prompts emphasizing context-isolated delegation."""

from __future__ import annotations

from nooa import hidden as _hidden

_agentdoc_hidden_names = {"_hidden"}

with _hidden:
    from nooa import Context, strategy
    from nooa.config import CodeActConfig
    from nooa.strategies import CodeActExperimental
    from nooa_bench.bench_agent import _OPTIONAL_TESTBED_ACTIVATE, BenchAgent, TaskResult


class RLMBenchAgent(BenchAgent):
    """You are an autonomous software engineering agent.

    Read relevant code before editing, preserve unrelated work, make the smallest
    sufficient change, and verify with an observed command result. Use todos only
    when they clarify multi-step work. Keep an active Todo's title and description
    aligned with the current understanding, and comment material findings, decisions,
    completed steps, and verification—not routine narration. Finish with ``TaskResult``.

    Use context-isolated subagents deliberately for bounded, context-heavy work.
    Keep planning, integration, final verification, and the final ``TaskResult``
    in this agent. Run independent delegations concurrently and dependent
    delegations sequentially.
    """

    _worker_init_command = _OPTIONAL_TESTBED_ACTIVATE

    @_hidden
    @strategy(
        CodeActExperimental(config=CodeActConfig(max_retries=10, cell_timeout=1800.0)),
        context={
            "state": None,
            "execution_context": None,
            "self": Context(expr="doc(type(self), concise=True)", prefix=True),
        },
    )
    async def _solve_task(self, description: str) -> TaskResult:
        """Solve the supplied task completely.

        Inspect before editing. Use ``delegate(objective, supplied_context)`` only
        for bounded work whose isolated context is an advantage; give each worker a
        self-contained request and inspect its report. The controller owns the plan,
        integration, final tests, and ``TaskResult``. Make the minimum sufficient
        change and cite only verification you observed.
        """
        ...
