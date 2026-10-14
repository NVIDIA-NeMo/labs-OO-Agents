# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functions defined in a cell keep seeing that cell's names when called later."""

import pytest

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

_LLM = FakeLLMClient()


class _CellAgent(Agent, llm=_LLM):
    pass


async def _run_cells(agent, cells):
    """Run cells the way CodeAct does: captured locals carry into the next cell."""
    session_locals, results = {}, []
    for number, code in enumerate(cells, 1):
        result = await agent.runtime.execute_code(
            code, execution_count=number, wrap_in_function=True, builtins=dict(session_locals)
        )
        session_locals.update(result.captured_locals or {})
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


def test_helper_predefinition_defers_names_from_the_same_cell():
    """A default argument bound earlier in the cell must not reject the cell."""
    from nooa.strategies.generated_code import HelperFunctionManager

    code = "KT = 3\ndef scale(x, k=KT):\n    return x * k\n"
    result = HelperFunctionManager().apply(code, _CellAgent(), {}, namespace={})
    assert result.errors == []
