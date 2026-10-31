# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``mark_returns``: an owned cell's executed top-level ``return`` is told apart from its end."""

import pytest

from nooa.runtime.execution_namespace import CellReturn, ExecutionNamespace


async def _run(source, *, mark_returns=True):
    ns = ExecutionNamespace(reserved=frozenset())
    with ns.enter({}):
        return await ns.cell(source, "<returns>", mark_returns=mark_returns)


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("return 1", 1),
        ("return None", None),
        ("for i in range(3):\n    if i == 2:\n        return i", 2),
        ("try:\n    return 'try'\nfinally:\n    pass", "try"),
        ("import asyncio\nawait asyncio.sleep(0)\nreturn 'awaited'", "awaited"),
    ],
)
async def test_executed_top_level_return_is_marked(source, value):
    result = await _run(source)
    assert isinstance(result, CellReturn)
    assert result.value == value and not result.bare


async def test_bare_return_is_marked_bare():
    result = await _run("x = 1\nreturn")
    assert isinstance(result, CellReturn) and result.bare and result.value is None


async def test_falling_off_the_end_is_not_a_return():
    assert await _run("if False:\n    return 1\nx = 2") is None


async def test_nested_scopes_keep_their_own_returns():
    source = (
        "def f():\n    return 'f'\n"
        "async def g():\n    return 'g'\n"
        "class K:\n    def m(self):\n        return 'm'\n"
        "h = lambda: 'h'\n"
        "return (f(), await g(), K().m(), h())"
    )
    result = await _run(source)
    assert isinstance(result, CellReturn)
    assert result.value == ("f", "g", "m", "h")


async def test_unmarked_cells_return_plain_values():
    assert await _run("return 1", mark_returns=False) == 1
