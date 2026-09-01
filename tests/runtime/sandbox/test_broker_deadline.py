# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cell deadline must not tick while a brokered ``self.*`` call runs parent-side.

Defect: ``SandboxedExecutor`` computes a fixed ``end = now + cell_timeout + grace``
and bounds brokered parent-side calls by that same end. A slow parent-side tool
(e.g. ``memory.reflect()``'s serial LLM calls) therefore kills the WORKER even
though the cell's own code is idle-waiting — wiping REPL state and swallowing
queued effects (~30 kills across the 20260716 ARC fleet; vc33's 909s incident).

The guardrail exists to kill runaway *cell code*: time spent inside a brokered
call must pause the cell clock. Cell-own CPU time stays bounded, and the worker
must remain fully usable after a genuine kill.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nooa.agent import Agent
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.unifiedllm import FakeLLMClient

pytestmark = pytest.mark.sandbox

# cell_timeout=1.0 + timeout_grace_s=2.0 -> hard deadline 3.0s.
_CELL_TIMEOUT = 1.0
_SLOW_TOOL_S = 4.0  # comfortably past the hard deadline


class _SlowToolAgent(Agent, llm=FakeLLMClient()):
    """Agent with a deliberately slow parent-side tool (mirrors memory.reflect)."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.slow_calls = 0
        self.quick_calls = 0

    async def slow_tool(self) -> str:
        self.slow_calls += 1
        await asyncio.sleep(_SLOW_TOOL_S)
        return "slow-done"

    async def quick_tool(self) -> str:
        self.quick_calls += 1
        return "quick-done"


def _executor(agent: Agent, *, cell_timeout: float | None = _CELL_TIMEOUT) -> SandboxedExecutor:
    return SandboxedExecutor(
        agent,
        SandboxConfig(require=False),
        cell_timeout=cell_timeout,
    )


@pytest.mark.asyncio
async def test_slow_brokered_call_does_not_kill_the_cell():
    """A parent-side self.* call slower than the cell deadline must survive."""
    agent = _SlowToolAgent()
    ex = _executor(agent)
    try:
        result = await ex.run_cell("r = await self.slow_tool()\nprint(r)", execution_count=1)
        assert result.error is None, f"cell was killed during a brokered call: {result.error}"
        assert "slow-done" in result.stdout
        assert agent.slow_calls == 1
    finally:
        await ex.aclose()


@pytest.mark.asyncio
async def test_repl_state_survives_a_slow_brokered_call():
    """The pre-call REPL state must still exist afterwards (no worker restart)."""
    agent = _SlowToolAgent()
    ex = _executor(agent)
    try:
        r1 = await ex.run_cell("marker = 'alive'", execution_count=1)
        assert r1.error is None
        r2 = await ex.run_cell("await self.slow_tool()\nprint(marker)", execution_count=2)
        assert r2.error is None, f"worker restarted mid-brokered-call: {r2.error}"
        assert "alive" in r2.stdout
    finally:
        await ex.aclose()


@pytest.mark.asyncio
async def test_cell_own_busy_time_is_still_bounded():
    """The deadline pause must apply ONLY to brokered time — runaway cell code dies."""
    agent = _SlowToolAgent()
    ex = _executor(agent)
    try:
        result = await ex.run_cell("while True:\n    pass", execution_count=1)
        assert result.error is not None
        assert "deadline" in str(result.error).lower()
    finally:
        await ex.aclose()


@pytest.mark.asyncio
async def test_broker_works_after_a_worker_kill():
    """After a genuine kill+restart, the next cell's brokered call must round-trip."""
    agent = _SlowToolAgent()
    ex = _executor(agent)
    try:
        killed = await ex.run_cell("while True:\n    pass", execution_count=1)
        assert killed.error is not None  # the kill happened
        follow_up = await ex.run_cell("print(await self.quick_tool())", execution_count=2)
        assert follow_up.error is None, f"broker broken after restart: {follow_up.error}"
        assert "quick-done" in follow_up.stdout
        assert agent.quick_calls == 1  # the parent-side effect really happened
    finally:
        await ex.aclose()
