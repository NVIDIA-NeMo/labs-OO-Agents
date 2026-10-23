# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CodeAct must reject cells that replace its completion helper before execution."""

import json

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

BACKENDS = ["inprocess", pytest.param("sandbox", marks=pytest.mark.sandbox)]


def _client(code: str, tool_name: str) -> FakeLLMClient:
    """Script one attempted binding followed by a valid completion."""
    return FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id=f"cell_{index}",
                        name=tool_name,
                        arguments=json.dumps({"code": cell}),
                    )
                ],
                finish_reason="tool_calls",
            )
            for index, cell in enumerate([code, "return_result(42)"])
        ],
        strict_exhaustion=True,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
@pytest.mark.parametrize(
    "binding",
    [
        "return_result = 0",
        "return_result: int = 0",
        "return_result += 1",
        "answer, *return_result = [1, 2]",
        "if False:\n    return_result = 0",
        "for return_result in []:\n    pass",
        "(return_result := 0)",
        "[(return_result := value) for value in [1]]",
        "import math as return_result",
        "from math import sqrt as return_result",
        "def return_result(value):\n    return value",
        "async def return_result(value):\n    return value",
        "@self.compile_effect\ndef return_result(value):\n    return value",
        "del return_result",
        "def helper():\n    global return_result\n    return_result = 0\nhelper()",
    ],
)
async def test_rejected_cell_does_not_poison_completion(backend, strategy_type, binding):
    """An invalid cell has no effects and the next scripted cell can still submit."""
    tool_name = "python_cell" if strategy_type is CodeActV2 else "execute_python"
    llm = _client("self.effects.append('executed')\n" + binding, tool_name)

    class TestAgent(Agent, llm=llm):
        """Track any work performed before a bad binding is rejected."""

        def compile_effect(self, function):
            """Expose premature helper compilation through an observable side effect."""
            self.effects.append("compiled")
            return function

        @strategy(
            strategy_type(
                config=CodeActConfig(
                    prefill=None,
                    execution_backend=backend,
                    sandbox=SandboxConfig(require=False, filesystem=False, network=True),
                )
            )
        )
        async def answer(self) -> int:
            """Return the computed integer."""
            ...

    agent = TestAgent()
    agent.effects = []
    try:
        assert await agent.answer() == 42
        assert agent.effects == []
        outputs = list(agent.events.query(type="PythonOutput"))
        assert len(outputs) == 2
        assert "return_result" in outputs[0].error
        assert "reserved" in outputs[0].error
        assert "result" in outputs[0].error
        assert not outputs[1].error
        assert len(llm.calls) == 2
    finally:
        await agent.aclose()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
@pytest.mark.parametrize(
    "deep_code",
    [
        pytest.param("answer = " + "+".join(["1"] * 1500), id="expression-traversal"),
        pytest.param("answer = " + "+".join(["1"] * 10000), id="parser"),
        pytest.param(
            "def helper():\n    answer = " + "+".join(["1"] * 1500),
            id="scope-discovery",
        ),
    ],
)
async def test_deep_cell_is_rejected_and_next_cell_can_complete(backend, strategy_type, deep_code):
    """Depth failures return correctable feedback without executing or poisoning the session."""
    tool_name = "python_cell" if strategy_type is CodeActV2 else "execute_python"
    code = "self.effects.append('executed')\n" + deep_code + "\nreturn_result = 0"
    llm = _client(code, tool_name)

    class TestAgent(Agent, llm=llm):
        """Expose any premature execution of a rejected deep cell."""

        @strategy(
            strategy_type(
                config=CodeActConfig(
                    prefill=None,
                    execution_backend=backend,
                    sandbox=SandboxConfig(require=False, filesystem=False, network=True),
                )
            )
        )
        async def answer(self) -> int:
            """Return the computed integer."""
            ...

    agent = TestAgent()
    agent.effects = []
    try:
        assert await agent.answer() == 42
        assert agent.effects == []
        outputs = list(agent.events.query(type="PythonOutput"))
        assert len(outputs) == 2
        assert "too deeply" in outputs[0].error
        assert "split" in outputs[0].error
        assert "retry" in outputs[0].error
        assert not outputs[1].error
        assert len(llm.calls) == 2
    finally:
        await agent.aclose()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
@pytest.mark.parametrize(
    "code",
    [
        "return_result = 0\nglobal return_result",
        "return_result(1)\nglobal return_result\nreturn_result = 0",
        "return_result: int = 0\nglobal return_result",
        "def helper(return_result):\n    global return_result\n    return_result = 0",
        "@self.compile_effect\ndef return_result(value):\n    return value\nglobal return_result",
        "return_result = 0\nnonlocal return_result",
        "await self.work()\nreturn_result = 0\nglobal return_result",
        "return_result = 0\nreturn 42\nglobal return_result",
    ],
)
async def test_semantic_syntax_error_precedes_helper_compilation(backend, strategy_type, code):
    """Invalid declarations report SyntaxError without running code or helper decorators."""
    tool_name = "python_cell" if strategy_type is CodeActV2 else "execute_python"
    llm = _client("self.effects.append('executed')\n" + code, tool_name)

    class TestAgent(Agent, llm=llm):
        """Expose any work performed before a semantic syntax error is reported."""

        def compile_effect(self, function):
            """Record premature helper compilation without changing the function."""
            self.effects.append("compiled")
            return function

        async def work(self):
            """Record execution of a statement that requires the async cell context."""
            self.effects.append("awaited")

        @strategy(
            strategy_type(
                config=CodeActConfig(
                    prefill=None,
                    execution_backend=backend,
                    sandbox=SandboxConfig(require=False, filesystem=False, network=True),
                )
            )
        )
        async def answer(self) -> int:
            """Return the computed integer."""
            ...

    agent = TestAgent()
    agent.effects = []
    try:
        assert await agent.answer() == 42
        assert agent.effects == []
        outputs = list(agent.events.query(type="PythonOutput"))
        assert len(outputs) == 2
        assert "SyntaxError:" in outputs[0].error
        assert "return_result" in outputs[0].error
        assert "Cell In[1], line" in outputs[0].error
        assert "Code validation failed" not in outputs[0].error
        assert not outputs[1].error
        assert len(llm.calls) == 2
    finally:
        await agent.aclose()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
@pytest.mark.parametrize(
    "code",
    [
        "answer = 42",
        "answer = [return_result for return_result in [1, 2]]",
        "answer = {return_result: return_result for return_result in [1, 2]}",
        "answer = list(return_result for return_result in [1, 2])",
        "answer = (lambda return_result: return_result)(42)",
        "def helper(return_result):\n    return return_result\nanswer = helper(42)",
        "def helper():\n    return_result = 42\n    return return_result\nanswer = helper()",
    ],
)
async def test_independent_local_bindings_remain_valid(backend, strategy_type, code):
    """Normal locals and names confined to nested scopes do not shadow completion."""
    tool_name = "python_cell" if strategy_type is CodeActV2 else "execute_python"
    llm = _client(code, tool_name)

    class TestAgent(Agent, llm=llm):
        """Exercise ordinary Python scoping during generation."""

        @strategy(
            strategy_type(
                config=CodeActConfig(
                    prefill=None,
                    execution_backend=backend,
                    sandbox=SandboxConfig(require=False, filesystem=False, network=True),
                )
            )
        )
        async def answer(self) -> int:
            """Return the computed integer."""
            ...

    agent = TestAgent()
    try:
        assert await agent.answer() == 42
        outputs = list(agent.events.query(type="PythonOutput"))
        assert len(outputs) == 2
        assert all(not output.error for output in outputs)
        assert len(llm.calls) == 2
    finally:
        await agent.aclose()
