# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contract: cell return values and failure status match across backends."""

from __future__ import annotations

from nooa.events import ResultStatus

from .conftest import cell, finish, outputs, resp


async def test_trailing_expression_is_captured_as_value(codeact_agent):
    """The value is pickled across the boundary on the sandbox path, not recomputed."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("6 * 7", call_id="c1")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    events = outputs(agent)
    assert len(events) == 1
    assert events[0].value == 42
    assert events[0].explicit_return is False


async def test_statement_only_cell_has_no_value(codeact_agent):
    """Absent must stay distinguishable from a returned None."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("x = 6 * 7", call_id="c1")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    events = outputs(agent)
    assert len(events) == 1
    assert events[0].value is None
    assert events[0].explicit_return is False


async def test_explicit_none_return_sets_the_discriminator(codeact_agent):
    """An explicit `return None` carries the opposite discriminator to an absent value.

    An explicit return also auto-completes the task, so no further cell runs.
    """
    agent = codeact_agent([resp("", tool_calls=[cell("return None", call_id="c1")])])
    assert await agent.run() is None

    events = outputs(agent)
    assert len(events) == 1
    assert events[0].value is None
    assert events[0].explicit_return is True


async def test_runtime_error_reports_error_status(codeact_agent):
    """The exception type must survive reconstruction, and the session must continue."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("raise ValueError('deliberate')", call_id="c1")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    events = outputs(agent)
    assert len(events) == 1
    assert events[0].execution_status is ResultStatus.ERROR
    assert "ValueError" in events[0].error


async def test_blocked_import_reports_error_status(codeact_agent):
    """Rejection must surface as an errored event, not be swallowed before emission."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("import subprocess", call_id="c1")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    events = outputs(agent)
    assert len(events) == 1
    assert events[0].execution_status is ResultStatus.ERROR
    assert "subprocess" in events[0].error
