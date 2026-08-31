# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CodeAct strategy driven end-to-end with the sandbox backend enabled.

Uses a scripted ``FakeLLMClient`` (no network) to run a real CodeAct session
whose cells execute in a guarded worker process: broker ``self.*`` to the live
agent, persist a namespace across cells, and finish via ``return_result``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.events import PythonOutput, ResultStatus
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.guards import probe_capabilities
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

pytestmark = pytest.mark.sandbox

CAPS = probe_capabilities()


def _resp(content: str, tool_calls: list | None = None) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
        assistant_message={"role": "assistant", "content": content},
    )


def _exec(code: str, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))


def _ret(result: Any = None, call_id: str = "cret") -> ToolCall:
    return ToolCall(id=call_id, name="return_result", arguments=json.dumps({"result": result}))


def _failed_output(agent: Agent, call_id: str | None = None) -> PythonOutput:
    """Return the first failed Python output, optionally for one tool call."""
    return next(
        event
        for event in agent.event_manager.values()
        if isinstance(event, PythonOutput)
        and event.execution_status is ResultStatus.ERROR
        and (call_id is None or event.tool_call_id == call_id)
    )


# A sandbox that keeps network on (so FakeLLM's parent-side calls are irrelevant
# to the worker) and does not fail closed if a guard is unavailable in CI.
_SANDBOX = SandboxConfig(require=False, network=True, filesystem=False)


class _SumAgent(Agent, llm=FakeLLMClient()):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.data = [1, 2, 3, 4, 5]

    def scale(self, x: int) -> int:
        return x * 10

    @strategy(
        CodeActStrategy(
            config=CodeActConfig(execution_backend="sandbox", cell_timeout=15.0, sandbox=_SANDBOX)
        )
    )
    async def compute(self) -> int:
        """Compute a value from self.data using self.scale()."""
        ...


async def test_sandboxed_codeact_session_end_to_end():
    llm = FakeLLMClient(
        scripted_responses=[
            # Cell 1: read live agent state + define a helper in the worker namespace.
            _resp("", tool_calls=[_exec("total = sum(self.data)\ndef bump(v):\n    return v + 1")]),
            # Cell 2: use the persisted helper + broker a method call to the parent.
            _resp("", tool_calls=[_exec("answer = self.scale(bump(total))\nprint(answer)")]),
            # Finish.
            _resp("", tool_calls=[_ret(result=160)]),
        ]
    )
    agent = _SumAgent(llm=llm)
    result = await agent.compute()
    assert result == 160


class _PidAgent(Agent, llm=FakeLLMClient()):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._worker_pid: int | None = None

    def note_worker_pid(self, pid: int) -> int:
        self._worker_pid = pid
        return pid

    @strategy(
        CodeActStrategy(
            config=CodeActConfig(execution_backend="sandbox", cell_timeout=15.0, sandbox=_SANDBOX)
        )
    )
    async def go(self) -> int:
        """Report the executing process id."""
        ...


async def test_sandbox_cell_runs_in_a_separate_process():
    """Definitive proof the sandbox backend is engaged: os.getpid() in the cell
    (brokered back to the parent) differs from the parent's pid."""
    llm = FakeLLMClient(
        scripted_responses=[
            _resp("", tool_calls=[_exec("import os\nself.note_worker_pid(os.getpid())")]),
            _resp("", tool_calls=[_ret(result=1)]),
        ]
    )
    agent = _PidAgent(llm=llm)
    result = await agent.go()
    assert result == 1
    assert agent._worker_pid is not None
    assert agent._worker_pid != os.getpid()  # the cell ran in the worker, not here


async def test_sandbox_path_still_enforces_restrictions():
    """Routing through execute_code keeps restrictions validation (a blocked import
    is rejected) — parity with the in-process backend."""
    llm = FakeLLMClient(
        scripted_responses=[
            _resp("", tool_calls=[_exec("import subprocess\nsubprocess.run(['echo'])")]),
            _resp("", tool_calls=[_ret(result=1)]),
        ]
    )
    agent = _SumAgent(llm=llm)
    result = await agent.compute()
    assert result == 1

    output = _failed_output(agent, "c1")
    assert output.stdout == ""
    assert output.stderr == ""
    assert "subprocess" in output.error
    assert "not allowed" in output.error.lower() or "restricted" in output.error.lower()
    assert "Traceback" not in output.error


async def test_sandboxed_codeact_reports_rich_cell_error_and_continues():
    llm = FakeLLMClient(
        scripted_responses=[
            _resp(
                "",
                tool_calls=[
                    _exec(
                        "import sys\n"
                        "print('before failure')\n"
                        "print('warning', file=sys.stderr)\n"
                        "text = 'abc'\n"
                        "start = text.index('missing')"
                    )
                ],
            ),
            _resp("", tool_calls=[_ret(result=7)]),
        ]
    )
    agent = _SumAgent(llm=llm)
    result = await agent.compute()

    assert result == 7
    output = _failed_output(agent, "c1")
    assert output.stdout == "before failure\n"
    assert output.stderr == "warning\n"
    assert "Cell In[1], line 5" in output.error
    assert "start = text.index('missing')" in output.error
    assert "ValueError: substring not found" in output.error


async def test_sandboxed_codeact_converts_indirect_system_exit():
    llm = FakeLLMClient(
        scripted_responses=[
            _resp("", tool_calls=[_exec("exit_type = SystemExit\nraise exit_type")]),
            _resp("", tool_calls=[_ret(result=8)]),
        ]
    )
    agent = _SumAgent(llm=llm)

    assert await agent.compute() == 8
    output = _failed_output(agent, "c1")
    assert "RuntimeError: SystemExit raised inside generated code" in output.error


async def test_custom_formatter_receives_worker_rendered_diagnostic():
    """Custom formatters run parent-side and can consume the worker traceback."""

    class TransportFormatter:
        def format(
            self,
            error,
            code=None,
            *,
            line_offset=0,
            max_error=None,
            tail_chars=None,
        ):
            from nooa.runtime.sandbox.errors import SandboxExecutionError

            assert isinstance(error, SandboxExecutionError)
            assert isinstance(error.original_error, ValueError)
            assert line_offset > 0
            assert max_error is not None
            return "CUSTOM\n" + error.diagnostic

    class CustomAgent(Agent, llm=FakeLLMClient()):
        @strategy(
            CodeActStrategy(
                config=CodeActConfig(
                    execution_backend="sandbox", cell_timeout=15.0, sandbox=_SANDBOX
                ),
                error_formatter=TransportFormatter(),
            )
        )
        async def compute(self) -> int:
            """Recover after a failed sandbox cell."""
            ...

    code = "text = 'abc'\ntext.index('missing')"
    llm = FakeLLMClient(
        scripted_responses=[
            _resp("", tool_calls=[_exec(code, call_id="custom-failure")]),
            _resp("", tool_calls=[_ret(result=9)]),
        ]
    )
    agent = CustomAgent(llm=llm)

    assert await agent.compute() == 9
    output = _failed_output(agent, "custom-failure")
    assert output.error.startswith("CUSTOM\nTraceback (most recent call last):")
    assert "Cell In[1], line 2" in output.error
    assert "text.index('missing')" in output.error
    assert output.error.endswith("ValueError: substring not found")
