# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards on the conformance harness itself.

These verify the fixture builds a working agent and that each parameterisation
actually engages the backend it names — a false green here would invalidate
every conformance assertion built on top.
"""

from __future__ import annotations

import os

from .conftest import cell, finish, resp


async def test_fixture_builds_a_working_agent(codeact_agent):
    agent = codeact_agent(
        [resp("", tool_calls=[cell("x = 1 + 1")]), resp("", tool_calls=[finish(result=2)])]
    )
    assert await agent.run() == 2


async def test_fixture_selects_the_requested_backend(codeact_agent, backend):
    """The cell's pid differs from the parent's only on the sandbox backend."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("import os\nself.note_pid(os.getpid())")]),
            resp("", tool_calls=[finish(result=1)]),
        ],
    )
    assert await agent.run() == 1
    if backend == "sandbox":
        assert agent.seen_pid != os.getpid()
    else:
        assert agent.seen_pid == os.getpid()
