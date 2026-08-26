# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contract: cell return values and failure status match across backends."""

from __future__ import annotations

from nooa.events import ResultStatus

from .conftest import cell, finish, resp


def _outputs(agent):
    return [e for e in agent.event_manager.values() if e.event_type == "PythonOutput"]


async def test_trailing_expression_is_captured_as_value(codeact_agent):
    """A bare trailing expression lands on PythonOutput.value."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("6 * 7", call_id="c1")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    outputs = _outputs(agent)
    assert len(outputs) == 1
    assert outputs[0].value == 42
    assert outputs[0].explicit_return is False


async def test_statement_only_cell_has_no_value(codeact_agent):
    """A cell ending in a statement reports no value."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("x = 6 * 7", call_id="c1")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    outputs = _outputs(agent)
    assert len(outputs) == 1
    assert outputs[0].value is None


async def test_runtime_error_reports_error_status(codeact_agent):
    """A raising cell reports ERROR and the session still completes."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("raise ValueError('deliberate')", call_id="c1")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    outputs = _outputs(agent)
    assert len(outputs) == 1
    assert outputs[0].execution_status is ResultStatus.ERROR
    assert "ValueError" in outputs[0].error


async def test_blocked_import_reports_error_status(codeact_agent):
    """RestrictionsConfig defaults apply to both backends."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("import subprocess", call_id="c1")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    outputs = _outputs(agent)
    assert len(outputs) == 1
    assert outputs[0].execution_status is ResultStatus.ERROR
