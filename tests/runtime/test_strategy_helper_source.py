# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy helpers retain verifiable source when rebound for later cells."""

from __future__ import annotations

import linecache

import pytest

from nooa import Agent, strategy
from nooa.runtime.execution_namespace import RUNTIME_NAMES, ExecutionNamespace
from nooa.runtime.sandbox.cell_core import run_cell_source
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


@pytest.mark.parametrize("backend", ["actor", "sandbox"])
@pytest.mark.parametrize(
    "prefix", ["offset = 1\n", "from typing import Literal\n\n"], ids=["assignment", "import"]
)
@pytest.mark.parametrize("body", ["...", "pass", ""], ids=["ellipsis", "pass", "docstring-only"])
@pytest.mark.asyncio
async def test_strategy_helper_source_across_cells(backend, prefix, body, monkeypatch):
    """Valid helpers persist and call the LLM; missing ellipses still fail."""
    monkeypatch.setattr(linecache, "cache", linecache.cache.copy())
    fake = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content='{"value": "positive"}',
                tool_calls=[],
                finish_reason="stop",
            )
        ]
    )
    namespace = {"strategy": strategy, "PredictStrategy": PredictStrategy, "fake": fake}
    agent = Agent(llm=fake) if backend == "actor" else None
    owned = ExecutionNamespace(reserved=RUNTIME_NAMES)

    async def execute(code, execution_count):
        """Run a cell with the selected backend; its namespace persists its names."""
        if agent is None:
            return await run_cell_source(code, namespace, execution_count=execution_count)
        with owned.enter({}):
            return await agent.runtime.execute_code(
                code,
                builtins=namespace,
                wrap_in_function=True,
                execution_count=execution_count,
                execution_namespace=owned,
            )

    first = await execute(
        prefix
        + "@strategy(PredictStrategy(), llm=fake)\n"
        + "async def classify(text: str) -> str:\n"
        + '    """Classify the supplied text."""\n'
        + f"    {body}\n",
        801,
    )
    if body != "...":
        assert isinstance(first.error, TypeError)
        assert "source-verifiable" in str(first.error)
        assert fake.call_count == 0
        return

    assert first.error is None, str(first.error)
    second = await execute('await classify("hello")', 802)
    assert second.error is None, str(second.error)
    assert second.returned_value == "positive"
    assert fake.call_count == 1
