# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functions defined in a cell keep seeing that cell's names when called later."""

import pytest

from nooa import Agent
from nooa.runtime.execution_namespace import RUNTIME_NAMES, ExecutionNamespace
from nooa.unifiedllm import FakeLLMClient

_LLM = FakeLLMClient()


class _CellAgent(Agent, llm=_LLM):
    pass


async def _run_cells(agent, cells):
    """Run cells the way CodeAct does: one namespace carries names into the next cell."""
    namespace, results = ExecutionNamespace(reserved=RUNTIME_NAMES), []
    with namespace.enter({}):
        for number, code in enumerate(cells, 1):
            result = await agent.runtime.execute_code(
                code, execution_count=number, wrap_in_function=True, execution_namespace=namespace
            )
            results.append(result)
    return results


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("import json\ndef load(s):\n    return json.loads(s)\nload('1')", "load('2')", 2),
        ("byfile = {'a': 3}\ndef load(k):\n    return byfile[k]\nload('a')", "load('a')", 3),
        (
            "import json\ndef load(self, s):\n    return json.loads(s)\nload(None, '1')",
            "load(None, '4')",
            4,
        ),
        ("byfile = {'a': 5}\ndef load(k, d=byfile):\n    return d[k]\nload('a')", "load('a')", 5),
    ],
    ids=["import", "dict", "self-first-param", "default-arg"],
)
async def test_function_sees_same_cell_names_from_a_later_cell(first, second, expected):
    agent = _CellAgent()
    one, two = await _run_cells(agent, [first, second])
    assert one.error is None, one.error
    assert two.error is None, two.error
    assert two.returned_value == expected


async def test_a_decorator_typo_is_an_ordinary_cell_error_at_its_line():
    """No pre-pass defines helpers: a failing decorator raises where the cell runs it."""
    from nooa.errors.formatting import format_error_for_llm

    agent = _CellAgent()
    code = '@stratgy(PredictStrategy())\nasync def f(x: str) -> str:\n    """Echo {x}."""\n    ...'
    try:
        (result,) = await _run_cells(agent, [code])
        assert isinstance(result.error, NameError)
        formatted = format_error_for_llm(result.error, code)
        assert "Cell In[1], line 1, in <module>" in formatted
        assert "name 'stratgy' is not defined" in formatted
    finally:
        await agent.aclose()


async def test_a_default_from_the_same_cell_resolves_when_the_function_is_defined():
    agent = _CellAgent()
    try:
        first, second = await _run_cells(
            agent, ["KT = 3\ndef scale(x, k=KT):\n    return x * k\nscale(2)", "scale(3)"]
        )
        assert first.error is None and second.error is None, (first.error, second.error)
        assert (first.returned_value, second.returned_value) == (6, 9)
    finally:
        await agent.aclose()
