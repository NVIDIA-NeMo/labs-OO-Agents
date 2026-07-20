# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The implicit-return rewrite must not swallow statements on a compound final line.

The REPL-style rewrite prepends ``return `` to the last top-level expression so a
cell's final value is returned (Jupyter semantics). It prepended at the start of
the whole source LINE, so a ``;``-compound final line silently lost every
statement after the first: ``a(); b()`` became ``return a(); b()`` (``b()`` dead
code), and ``x = 1; f(x)`` became ``return x = 1; f(x)`` (bogus SyntaxError).

The fix inserts ``return `` at the last statement's column, keeping every earlier
statement on the line live.
"""

from __future__ import annotations

import pytest

from nooa.agent import Agent
from nooa.runtime.actor import ActorRuntime
from nooa.unifiedllm import FakeLLMClient


class _RecorderAgent(Agent, llm=FakeLLMClient()):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[str] = []

    def record(self, tag: str) -> str:
        self.calls.append(tag)
        return f"recorded:{tag}"


@pytest.mark.asyncio
async def test_compound_final_line_executes_all_statements():
    """wrap_in_function=True is the CodeAct cell path — the rewrite applies there."""
    agent = _RecorderAgent()
    runtime = ActorRuntime(agent)
    result = await runtime.execute_code("self.record('a'); self.record('b')", wrap_in_function=True)
    assert result.error is None, f"cell errored: {result.error}"
    assert agent.calls == ["a", "b"]  # the second call must actually run


@pytest.mark.asyncio
async def test_assignment_then_call_one_liner_is_not_a_syntax_error():
    agent = _RecorderAgent()
    runtime = ActorRuntime(agent)
    result = await runtime.execute_code("n = 3; self.record(str(n))", wrap_in_function=True)
    assert result.error is None, f"bogus SyntaxError: {result.error}"
    assert agent.calls == ["3"]


@pytest.mark.asyncio
async def test_plain_last_expression_still_returns_its_value():
    agent = _RecorderAgent()
    runtime = ActorRuntime(agent)
    result = await runtime.execute_code("1 + 1", wrap_in_function=True)
    assert result.error is None
    assert result.returned_value == 2


@pytest.mark.asyncio
async def test_unwrapped_path_regression():
    """The default (unwrapped) path never rewrote and must stay correct."""
    agent = _RecorderAgent()
    runtime = ActorRuntime(agent)
    result = await runtime.execute_code("self.record('a'); self.record('b')")
    assert result.error is None
    assert agent.calls == ["a", "b"]
