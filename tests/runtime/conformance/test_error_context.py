# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contract: agent-facing errors carry equivalent source context.

The sandbox path reconstructs diagnostics across a process boundary. It
previously dropped every frame, leaving the agent with only the exception line
(#191, fixed by #189). These pin the rendered guidance as a parity contract.
"""

from __future__ import annotations

from .conftest import cell, finish, outputs, resp

_FAILING_CELL = "def inner():\n    return None.strip()\n\ninner()"


async def test_single_frame_error_carries_source_context(codeact_agent):
    """Cell identity, line number and source line reach the agent on both backends."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("raise ValueError('deliberate')", call_id="c1")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    error = outputs(agent)[0].error
    assert "Cell In[1], line 1, in <module>" in error
    assert "raise ValueError('deliberate')" in error
    assert "ValueError: deliberate" in error


async def test_multi_frame_error_carries_every_frame(codeact_agent):
    """Both frames, their source lines and their carets survive on both backends."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell(_FAILING_CELL, call_id="c1")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )
    assert await agent.run() == 7

    error = outputs(agent)[0].error
    assert "Cell In[1], line 4, in <module>" in error
    assert "Cell In[1], line 2, in inner" in error
    assert "return None.strip()" in error
    assert "AttributeError: 'NoneType' object has no attribute 'strip'" in error

    lines = error.splitlines()
    source_index = next(i for i, line in enumerate(lines) if "return None.strip()" in line)
    caret_line = lines[source_index + 1]
    assert caret_line.strip() == "^" * 10
    assert caret_line.index("^") == lines[source_index].index("None.strip")
