# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-only BenchAgent variant emphasizing context-isolated delegation.

Capabilities, strategy limits, context settings, and lifecycle are shared with BenchAgent.
"""

from __future__ import annotations

from nooa import hidden as _hidden

_agentdoc_hidden_names = {"_hidden"}

with _hidden:
    from nooa import strategy
    from nooa_bench.bench_agent import _SOLVE_CONTEXT, _SOLVE_STRATEGY, BenchAgent, TaskResult


class RLMBenchAgent(BenchAgent):
    __doc__ = (
        (BenchAgent.__doc__ or "")
        + """

    Use context-isolated subagents deliberately for bounded, context-heavy work.
    Keep planning, integration, final verification, and the final ``TaskResult``
    in this agent. Run independent delegations concurrently and dependent
    delegations sequentially.
    """
    )

    @_hidden
    @strategy(
        _SOLVE_STRATEGY,
        context=_SOLVE_CONTEXT,
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
