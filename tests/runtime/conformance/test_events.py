# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contract: event sequence and PythonOutput identity match across backends."""

from __future__ import annotations

from nooa.events import ResultStatus

from .conftest import cell, finish, resp


async def test_event_sequence_is_equivalent(codeact_agent):
    """One cell then a return produces the same ordered event types on both backends."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("x = 1", call_id="c1")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    event_types = [e.event_type for e in agent.event_manager.values()]
    assert event_types == ["Task", "ToolCallEvent", "PythonOutput", "ToolCallEvent"]


async def test_python_output_links_to_its_tool_call(codeact_agent):
    """Each PythonOutput carries the tool_call_id of the cell that produced it."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("a = 1", call_id="c1")]),
            resp("", tool_calls=[cell("b = 2", call_id="c2")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    outputs = [e for e in agent.event_manager.values() if e.event_type == "PythonOutput"]
    assert [o.tool_call_id for o in outputs] == ["c1", "c2"]


async def test_execution_counts_increment_in_order(codeact_agent):
    """Successive cells receive increasing execution counts."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("a = 1", call_id="c1")]),
            resp("", tool_calls=[cell("b = 2", call_id="c2")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    outputs = [e for e in agent.event_manager.values() if e.event_type == "PythonOutput"]
    counts = [o.execution_count for o in outputs]
    assert counts == list(range(counts[0], counts[0] + len(counts)))
    assert len(set(counts)) == len(counts)


async def test_successful_cells_report_complete(codeact_agent):
    """A cell that runs without error reports COMPLETE on both backends."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("x = 1", call_id="c1")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    outputs = [e for e in agent.event_manager.values() if e.event_type == "PythonOutput"]
    assert len(outputs) == 1
    assert outputs[0].execution_status is ResultStatus.COMPLETE
    assert outputs[0].error == ""
