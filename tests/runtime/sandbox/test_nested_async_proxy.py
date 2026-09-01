# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cell must be able to ``await`` an async callable reached via a nested proxy.

Regression for the broker-contract gap: ``_NestedProxy.__call__`` ran a
synchronous brokered RPC and returned the parent-resolved value verbatim, so
``await self.tool.amethod(...)`` on an async callable reached through the
``local is None`` path (e.g. a non-picklable object added to the live agent
after the worker forked) awaited a plain value and raised ``TypeError``.

The parent now reports ``was_async`` and the nested proxy re-wraps the result in
an awaitable.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from nooa import Agent
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.unifiedllm import FakeLLMClient

pytestmark = pytest.mark.sandbox


class _AsyncTool:
    """Non-picklable (holds a lock) object with async + sync methods."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # makes the instance unpicklable -> nested proxy

    async def double(self, n: int) -> int:
        return n * 2

    def triple(self, n: int) -> int:
        return n * 3


class _Agent(Agent, llm=FakeLLMClient()):
    pass


def _rr_builtins() -> dict[str, Any]:
    from nooa.strategies.codeact import _ReturnResultSignal

    def return_result(*a: Any, **k: Any):
        raise _ReturnResultSignal(result={"result": a[0] if a else k})

    return {"return_result": return_result}


async def _run(ex: SandboxedExecutor, code: str, n: int):
    return await ex.run_cell(code, execution_count=n)


@pytest.mark.asyncio
async def test_await_async_callable_via_nested_proxy():
    agent = _Agent()
    ex = SandboxedExecutor(
        agent, SandboxConfig(require=False), cell_timeout=10.0, framework_builtins=_rr_builtins()
    )
    try:
        # Warm-up cell forces the worker to fork with a snapshot that lacks `tool`.
        await _run(ex, "1 + 1", 1)
        # Add a non-picklable async tool to the LIVE parent only -> the worker's
        # copy lacks it -> self.tool resolves via the local=None nested-proxy path.
        agent.tool = _AsyncTool()

        r = await _run(ex, "x = await self.tool.double(5)\nprint('async', x)", 2)
        assert r.error is None, f"await on nested async callable broke: {r.error}"
        assert "async 10" in r.stdout

        # sync method through the same nested-proxy path still works
        r2 = await _run(ex, "print('sync', self.tool.triple(4))", 3)
        assert r2.error is None, r2.error
        assert "sync 12" in r2.stdout
    finally:
        await ex.aclose()
