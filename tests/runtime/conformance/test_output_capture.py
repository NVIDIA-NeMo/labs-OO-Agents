# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contract: stdout and stderr capture is equivalent across backends."""

from __future__ import annotations

from .conftest import cell, finish, outputs, resp


async def test_stdout_is_captured(codeact_agent):
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("print('hello from the cell')")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    events = outputs(agent)
    assert len(events) == 1
    assert "hello from the cell" in events[0].stdout
    assert events[0].stderr == ""


async def test_stderr_is_captured(codeact_agent):
    """Sandbox stderr crosses a worker pipe; the streams must stay separated."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("import sys\nprint('warned', file=sys.stderr)")]),
            resp("", tool_calls=[finish(result=1)]),
        ]
    )
    assert await agent.run() == 1

    events = outputs(agent)
    assert len(events) == 1
    assert "warned" in events[0].stderr
    assert "warned" not in events[0].stdout
