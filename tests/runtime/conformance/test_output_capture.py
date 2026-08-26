# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contract: stdout and stderr capture is equivalent across backends."""

from __future__ import annotations

from nooa.events import PythonOutput

from .conftest import cell, finish, resp


def _outputs(agent) -> list[PythonOutput]:
    """Every PythonOutput event the session emitted, in order."""
    return [e for e in agent.event_manager.values() if e.event_type == "PythonOutput"]


async def test_stdout_is_captured(codeact_agent):
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("print('hello from the cell')")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    outputs = _outputs(agent)
    assert len(outputs) == 1
    assert "hello from the cell" in outputs[0].stdout
    assert outputs[0].stderr == ""


async def test_stderr_is_captured(codeact_agent):
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("import sys\nprint('warned', file=sys.stderr)")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    outputs = _outputs(agent)
    assert len(outputs) == 1
    assert "warned" in outputs[0].stderr
    assert "warned" not in outputs[0].stdout
