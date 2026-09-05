# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""return_result(): success is shared, failure transport is backend-specific.

A picklable payload behaves identically on both backends. A non-picklable one
does not, and deliberately so: the sandbox returns the value across a process
boundary, so it must be picklable, while the in-process backend hands back the
live object. These pin that divergence as an intended contract rather than
leaving it undocumented — an agent developed against the default backend can
otherwise return a live object and fail only once the sandbox is enabled.
"""

from __future__ import annotations

from typing import Any

import pytest

from nooa.events import ResultStatus

from .conftest import cell, finish, outputs, resp

_MAKE_LOCK = "import threading\nbad = threading.Lock()"


async def test_picklable_payload_returns_on_both_backends(codeact_agent):
    """The shared half of the contract: a JSON-safe payload round-trips identically."""
    agent = codeact_agent(
        [
            resp("", tool_calls=[cell("payload = {'n': 42, 'xs': [1, 2, 3]}", call_id="c1")]),
            resp("", tool_calls=[finish(result={"n": 42, "xs": [1, 2, 3]})]),
        ]
    )
    assert await agent.run() == {"n": 42, "xs": [1, 2, 3]}
    assert all(e.execution_status is ResultStatus.COMPLETE for e in outputs(agent))


@pytest.mark.sandbox
async def test_unpicklable_payload_is_rejected_on_sandbox():
    """Backend-specific: the payload must cross a process boundary, so it must pickle."""
    from nooa import Agent, strategy  # noqa: PLC0415
    from nooa.config import CodeActConfig  # noqa: PLC0415
    from nooa.strategies.codeact import CodeActStrategy  # noqa: PLC0415
    from nooa.unifiedllm import FakeLLMClient  # noqa: PLC0415

    from .conftest import _SANDBOX  # noqa: PLC0415

    llm = FakeLLMClient(
        scripted_responses=[
            resp("", tool_calls=[cell(_MAKE_LOCK, call_id="c1")]),
            resp("", tool_calls=[cell("return_result(bad)", call_id="c2")]),
            resp("", tool_calls=[finish(result=7)]),
        ]
    )

    class _SandboxAgent(Agent, llm=FakeLLMClient()):
        @strategy(
            CodeActStrategy(
                config=CodeActConfig(
                    execution_backend="sandbox", cell_timeout=15.0, sandbox=_SANDBOX
                )
            )
        )
        async def run(self) -> Any:
            """Return an unpicklable value."""
            ...

    agent = _SandboxAgent(llm=llm)
    assert await agent.run() == 7

    events = outputs(agent)
    assert events[-1].execution_status is ResultStatus.ERROR
    assert "CellSerializationError" in events[-1].error
    assert "picklable" in events[-1].error
