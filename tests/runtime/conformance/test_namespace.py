# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contract: the REPL namespace persists across cells on both backends."""

from __future__ import annotations

from .conftest import cell, finish, resp


async def test_bindings_persist_across_cells(codeact_agent):
    """A name bound in cell 1 is readable in cell 2."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("total = 40", call_id="c1")]),
            resp("", tool_calls=[cell("print(total + 2)", call_id="c2")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    outputs = [e for e in agent.event_manager.values() if e.event_type == "PythonOutput"]
    assert len(outputs) == 2
    assert "42" in outputs[1].stdout


async def test_helper_definitions_persist_across_cells(codeact_agent):
    """A function defined in cell 1 is callable in cell 2."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("def double(v):\n    return v * 2", call_id="c1")]),
            resp("", tool_calls=[cell("print(double(21))", call_id="c2")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    outputs = [e for e in agent.event_manager.values() if e.event_type == "PythonOutput"]
    assert len(outputs) == 2
    assert "42" in outputs[1].stdout


async def test_imports_persist_across_cells(codeact_agent):
    """A module imported in cell 1 is bound in cell 2."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("import math", call_id="c1")]),
            resp("", tool_calls=[cell("print(math.floor(42.9))", call_id="c2")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    outputs = [e for e in agent.event_manager.values() if e.event_type == "PythonOutput"]
    assert len(outputs) == 2
    assert "42" in outputs[1].stdout
