# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contract: a rejected cell is correctable, not fatal.

Validation runs on the parent for both backends, so a rejection must reach the
model as an errored PythonOutput carrying enough context to fix the cell, and the
session must continue rather than abort. The error text here comes from the
validator's issue formatter, which renders cell identity, line and caret — unlike
the SyntaxError path in the same class (see #267).
"""

from __future__ import annotations

from nooa.events import ResultStatus

from .conftest import cell, finish, outputs, resp


async def test_rejected_cell_is_correctable(codeact_agent):
    """The model gets an errored output and its next cell runs normally."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("import subprocess", call_id="c1")]),
            resp("", tool_calls=[cell("x = 1 + 1\nprint(x)", call_id="c2")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    events = outputs(agent)
    assert [e.tool_call_id for e in events] == ["c1", "c2"]
    assert events[0].execution_status is ResultStatus.ERROR
    assert events[1].execution_status is ResultStatus.COMPLETE
    assert "2" in events[1].stdout


async def test_rejection_carries_correctable_context(codeact_agent):
    """A rejection names the cell, the line and the offending source."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("import subprocess", call_id="c1")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    error = outputs(agent)[0].error
    assert "Cell In[1], line 1" in error
    assert "import subprocess" in error
    assert "subprocess" in error


async def test_rejection_does_not_disturb_the_event_sequence(codeact_agent):
    """A rejected cell produces the same event shape as a successful one."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("import subprocess", call_id="c1")]),
            resp("", tool_calls=[cell("x = 1", call_id="c2")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    assert [e.event_type for e in agent.event_manager.values()] == [
        "Task",
        "ToolCallEvent",
        "PythonOutput",
        "ToolCallEvent",
        "PythonOutput",
        "ToolCallEvent",
    ]


async def test_namespace_survives_a_rejected_cell(codeact_agent):
    """A rejection must not reset the session namespace."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("keep = 41", call_id="c1")]),
            resp("", tool_calls=[cell("import subprocess", call_id="c2")]),
            resp("", tool_calls=[cell("print(keep + 1)", call_id="c3")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    events = outputs(agent)
    assert events[1].execution_status is ResultStatus.ERROR
    assert "42" in events[2].stdout
