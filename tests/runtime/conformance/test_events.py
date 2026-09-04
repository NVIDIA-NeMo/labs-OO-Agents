# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contract: event sequence and PythonOutput identity match across backends."""

from __future__ import annotations

from nooa.events import ResultStatus

from .conftest import cell, finish, outputs, resp


async def test_event_sequence_is_equivalent(codeact_agent):
    """IPC must not introduce extra events or reorder the ones the in-process path emits."""
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
    """Out-of-order delivery would break Out[n] lookup, so the id must round-trip."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("a = 1", call_id="c1")]),
            resp("", tool_calls=[cell("b = 2", call_id="c2")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    assert [o.tool_call_id for o in outputs(agent)] == ["c1", "c2"]


async def test_execution_counts_increment_in_order(codeact_agent):
    """Counts must be consecutive; the start value is left unpinned pending #189."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("a = 1", call_id="c1")]),
            resp("", tool_calls=[cell("b = 2", call_id="c2")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    counts = [o.execution_count for o in outputs(agent)]
    assert counts == list(range(counts[0], counts[0] + len(counts)))


async def test_successful_cells_report_complete(codeact_agent):
    """A clean cell must not carry residual error text from the transport layer."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("x = 1", call_id="c1")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    events = outputs(agent)
    assert len(events) == 1
    assert events[0].execution_status is ResultStatus.COMPLETE
    assert events[0].error == ""
