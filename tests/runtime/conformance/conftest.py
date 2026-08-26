# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for CodeAct backend-conformance tests.

Builds an equivalent CodeAct agent for each execution backend so the same
observable-contract assertions run against both. See README.md for the matrix
of shared versus backend-specific contracts.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.guards import probe_capabilities
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

CAPS = probe_capabilities()

# require=False keeps portable semantic tests runnable where a guard is
# unavailable; network stays on so the parent-side FakeLLM is irrelevant to
# the worker. Matches tests/runtime/sandbox/test_codeact_sandbox.py.
_SANDBOX = SandboxConfig(require=False, network=True, filesystem=False)


BACKENDS = [
    pytest.param("inprocess", id="inprocess"),
    pytest.param("sandbox", id="sandbox", marks=pytest.mark.sandbox),
]

Backend = Literal["inprocess", "sandbox"]


def resp(content: str = "", tool_calls: list | None = None) -> LLMResponse:
    """Build a scripted LLM response."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
        assistant_message={"role": "assistant", "content": content},
    )


def cell(code: str, call_id: str = "c1") -> ToolCall:
    """Build an execute_python tool call."""
    return ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))


def finish(result: Any = None, call_id: str = "cret") -> ToolCall:
    """Build a return_result tool call."""
    return ToolCall(id=call_id, name="return_result", arguments=json.dumps({"result": result}))


def _sandbox_skip_reason() -> str | None:
    """Why the sandbox backend cannot be exercised here, or None if it can.

    ``SandboxConfig(require=False)`` lets a cell run with enforcement missing,
    so a green test on a host without these guards would say nothing about the
    behaviour being asserted. Skip explicitly instead.
    """
    if not CAPS.linux:
        return "sandbox backend requires Linux"
    missing = [
        name
        for name, present in (
            ("landlock", CAPS.landlock_abi >= 1),
            ("seccomp", CAPS.seccomp),
            ("rlimit", CAPS.rlimit),
        )
        if not present
    ]
    if missing:
        return f"host cannot enforce: {', '.join(missing)}"
    return None


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> Backend:
    """The execution backend under test; puts the name in the node ID."""
    name: Backend = request.param
    if name == "sandbox":
        reason = _sandbox_skip_reason()
        if reason is not None:
            pytest.skip(reason)
    return name


@pytest.fixture
def codeact_agent(backend: Backend):
    """Build an equivalent CodeAct agent for the backend under test.

    The execution backend is fixed when ``@strategy(...)`` evaluates in the
    class body, so it cannot be an instance argument; each call defines a
    fresh agent class closing over ``backend``.
    """

    def _make(scripted_responses: list[LLMResponse], **attrs: Any) -> Agent:
        config = CodeActConfig(
            execution_backend=backend,
            cell_timeout=15.0,
            sandbox=_SANDBOX,
        )

        class _ConformanceAgent(Agent, llm=FakeLLMClient()):
            def __init__(self, **kwargs: Any):
                super().__init__(**kwargs)
                for key, value in attrs.items():
                    setattr(self, key, value)

            @strategy(CodeActStrategy(config=config))
            async def run(self) -> Any:
                """Run the scripted cells and finish via return_result."""
                ...

            def note_pid(self, pid: int) -> int:
                """Record the pid of the process that executed the cell."""
                self.seen_pid = pid
                return pid

        return _ConformanceAgent(llm=FakeLLMClient(scripted_responses=scripted_responses))

    return _make


def outputs(agent) -> list:
    """Every PythonOutput event the session emitted, in order."""
    return [e for e in agent.event_manager.values() if e.event_type == "PythonOutput"]
