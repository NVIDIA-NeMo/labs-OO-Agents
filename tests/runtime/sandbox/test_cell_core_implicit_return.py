# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandbox cell engine's implicit-return rewrite must not drop statements.

``run_cell_source`` prepends ``return `` to the last top-level expression so a
cell's final value is returned (Jupyter semantics). It prepended at the start of
the whole source LINE, so a ``;``-compound final line silently lost every
statement after the first: ``a(); b()`` became ``return a(); b()`` (``b()`` dead
code), and ``x = 1; f(x)`` became ``return x = 1; f(x)`` (bogus SyntaxError).

The fix inserts ``return `` at the last statement's column.
"""

from __future__ import annotations

import pytest

from nooa.runtime.sandbox.cell_core import run_cell_source


@pytest.mark.asyncio
async def test_compound_final_line_executes_all_statements():
    ns: dict = {"x": []}
    result = await run_cell_source("x.append(1); x.append(2)", ns)
    assert result.error is None, f"cell errored: {result.error}"
    assert ns["x"] == [1, 2]  # second statement must NOT become dead code


@pytest.mark.asyncio
async def test_compound_final_line_after_other_lines():
    ns: dict = {"x": []}
    code = "x.append(1)\nx.append(2); x.append(3)"
    result = await run_cell_source(code, ns)
    assert result.error is None, f"cell errored: {result.error}"
    assert ns["x"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_assignment_then_call_one_liner_is_not_a_syntax_error():
    ns: dict = {}
    result = await run_cell_source("y = 5; print(y)", ns)
    assert result.error is None, f"bogus SyntaxError: {result.error}"
    assert "5" in result.stdout
    assert ns.get("y") == 5


@pytest.mark.asyncio
async def test_compound_final_line_still_returns_last_expression():
    ns: dict = {}
    result = await run_cell_source("a = 1; a + 1", ns)
    assert result.error is None, f"cell errored: {result.error}"
    assert result.returned_value == 2
    assert ns.get("a") == 1


@pytest.mark.asyncio
async def test_plain_last_expression_return_still_works():
    ns: dict = {}
    result = await run_cell_source("1 + 1", ns)
    assert result.error is None
    assert result.returned_value == 2


@pytest.mark.asyncio
async def test_multiline_trailing_call_still_works():
    ns: dict = {"total": sum}
    code = "total([\n    1,\n    2,\n])"
    result = await run_cell_source(code, ns)
    assert result.error is None
    assert result.returned_value == 3
