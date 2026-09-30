# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contract: calls back into the agent are observationally identical.

In-process, ``self.helper(...)`` is an ordinary method call. On the sandbox path
the worker suspends, the parent executes the method, and the result crosses back
through the broker. These pin that the machinery stays invisible: same value,
same stdout, same event sequence, no broker-specific event.
"""

from __future__ import annotations

from nooa.events import ResultStatus

from .conftest import cell, finish, outputs, resp


async def test_agent_method_call_returns_the_same_value(codeact_agent):
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("print(self.helper(20))", call_id="c1")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    events = outputs(agent)
    assert len(events) == 1
    assert events[0].stdout.strip() == "40"
    assert events[0].execution_status is ResultStatus.COMPLETE


async def test_brokered_call_adds_no_events(codeact_agent):
    """A call back into the agent emits nothing of its own on either backend."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("v = self.helper(20)", call_id="c1")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    assert [e.event_type for e in agent.event_manager.values()] == [
        "Task",
        "ToolCallEvent",
        "PythonOutput",
        "ToolCallEvent",
    ]


async def test_brokered_result_persists_in_the_namespace(codeact_agent):
    """A value obtained through the broker is an ordinary binding afterwards."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("v = self.helper(20)", call_id="c1")]),
            resp("", tool_calls=[cell("print(v + 2)", call_id="c2")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    events = outputs(agent)
    assert len(events) == 2
    assert events[1].stdout.strip() == "42"


async def test_repeated_brokered_calls_all_resolve(codeact_agent):
    """Several calls in one cell each cross and return independently."""
    agent = codeact_agent(
        [
            resp(
                "", tool_calls=[cell("print(sum(self.helper(i) for i in range(4)))", call_id="c1")]
            ),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    assert outputs(agent)[0].stdout.strip() == "12"
