# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge cases for sandbox.

Focus on:
- REPL command that modifies agent state during generation
- REPL command that calls tool during generation
- Generated code that calls REPL command (should fail)

"REPL command" means a CodeAct ``execute_python`` cell. With
``execution_backend="sandbox"``, cells run in a worker process and broker
``self.*`` to the live parent agent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _resp(content: str = "", tool_calls: list | None = None) -> LLMResponse:
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


# Fail-open for CI hosts without Landlock/seccomp; network stays on so parent-side
# FakeLLM traffic is unrelated to worker filesystem/network policy.
_SANDBOX = SandboxConfig(require=False, network=True, filesystem=False)
_CODEACT = CodeActStrategy(
    config=CodeActConfig(
        execution_backend="sandbox",
        cell_timeout=15.0,
        sandbox=_SANDBOX,
        prefill=None,
    )
)


class _StateAgent(Agent, llm=FakeLLMClient()):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.counter = 0

    @strategy(_CODEACT)
    async def work(self) -> int:
        """Bump counter via a sandboxed cell assignment."""
        ...


class _ToolAgent(Agent, llm=FakeLLMClient()):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.hits = 0

    def bump(self) -> int:
        """Deterministic tool callable from a sandboxed cell."""
        self.hits += 1
        return self.hits

    @strategy(_CODEACT)
    async def work(self) -> int:
        """Call bump() from a sandboxed cell."""
        ...


class _ReplCallAgent(Agent, llm=FakeLLMClient()):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.saw_execute_python_error = False

    def note_error(self, msg: str) -> None:
        if "execute_python" in msg and "not defined" in msg:
            self.saw_execute_python_error = True

    @strategy(_CODEACT)
    async def work(self) -> int:
        """Attempt to invoke execute_python from inside a cell (should fail)."""
        ...


@pytest.mark.asyncio
async def test_repl_command_modifies_agent_state_during_generation():
    """Cell assignment ``self.counter = …`` lands on the live parent agent."""
    llm = FakeLLMClient(
        scripted_responses=[
            _resp("", [_exec("self.counter = 99\nprint(self.counter)")]),
            _resp("", [_ret(99)]),
        ]
    )
    agent = _StateAgent(llm=llm)
    assert await agent.work() == 99
    assert agent.counter == 99


@pytest.mark.asyncio
async def test_repl_command_calls_tool_during_generation():
    """Cell call ``self.bump()`` brokers to the live parent and records side effects."""
    llm = FakeLLMClient(
        scripted_responses=[
            _resp("", [_exec("x = self.bump()\nprint(x)")]),
            _resp("", [_ret(1)]),
        ]
    )
    agent = _ToolAgent(llm=llm)
    assert await agent.work() == 1
    assert agent.hits == 1


@pytest.mark.asyncio
async def test_generated_code_calling_repl_command_fails():
    """``execute_python`` is an LLM tool, not a REPL builtin — cells NameError it.

    The strategy recovers from the cell error and continues to ``return_result``.
    """
    llm = FakeLLMClient(
        scripted_responses=[
            _resp(
                "",
                [
                    _exec(
                        "try:\n"
                        "    execute_python('print(1)')\n"
                        "except NameError as e:\n"
                        "    self.note_error(str(e))\n"
                        "    raise"
                    )
                ],
            ),
            _resp("", [_ret(0)]),
        ]
    )
    agent = _ReplCallAgent(llm=llm)
    assert await agent.work() == 0
    assert agent.saw_execute_python_error is True
