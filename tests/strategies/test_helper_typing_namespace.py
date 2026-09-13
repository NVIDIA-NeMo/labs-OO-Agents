# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precompiled helpers must see the typing defaults already supplied by ActorRuntime."""

import json
import typing as _typing

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy
from nooa.strategies.generated_code import ExecutionNamespaceBuilder, HelperFunctionManager
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


@pytest.mark.parametrize("name", ["Annotated", "Any", "Literal", "Optional", "Union"])
def test_precompiler_has_runtime_typing_default(name):
    # Use an object whose defining module cannot incidentally supply typing imports.
    namespace = ExecutionNamespaceBuilder.build(object())
    assert namespace[name] is getattr(_typing, name)


@pytest.mark.parametrize("name", ["Annotated", "Any", "Literal", "Optional", "Union"])
def test_explicit_namespace_extras_keep_precedence(name):
    sentinel = object()
    namespace = ExecutionNamespaceBuilder.build(object(), extra={name: sentinel})
    assert namespace[name] is sentinel


@pytest.mark.parametrize(
    "annotation",
    ["Annotated[int, 'tag']", "Any", "Literal[3]", "Optional[int]", "Union[str, int]"],
)
def test_plain_helper_compiles_without_module_global_typing_import(annotation):
    agent = object()
    namespace = ExecutionNamespaceBuilder.build(agent)
    session_locals = {}
    result = HelperFunctionManager().apply(
        f"def identity(value: {annotation}):\n    return value\n",
        agent,
        session_locals,
        namespace=namespace,
    )
    assert result.errors == []
    assert result.installed == ["identity"]
    assert session_locals["identity"](3) == 3
    assert not hasattr(agent, "identity")


def test_precompiler_does_not_execute_arbitrary_cell_imports():
    agent = object()
    namespace = ExecutionNamespaceBuilder.build(agent)
    result = HelperFunctionManager().apply(
        "import nonexistent_module_with_side_effects\ndef identity(value: int):\n    return value\n",
        agent,
        {},
        namespace=namespace,
    )
    assert result.errors == []
    assert "nonexistent_module_with_side_effects" not in namespace


@pytest.mark.asyncio
@pytest.mark.parametrize("with_import", [False, True])
@pytest.mark.parametrize("later_cell", [False, True])
@pytest.mark.parametrize(
    ("name", "annotation"),
    [
        ("Annotated", "Annotated[int, 'tag']"),
        ("Any", "Any"),
        ("Literal", "Literal[37]"),
        ("Optional", "Optional[int]"),
        ("Union", "Union[str, int]"),
    ],
)
async def test_typing_helper_in_actual_codeact_cell(
    with_import, later_cell, name, annotation, tmp_path, monkeypatch
):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    code = (
        (f"from typing import {name}\n" if with_import else "")
        + f"def parse_value(value: {annotation}) -> int:\n"
        + "    return int(value)\n"
        + ("print('helper ready')\n" if later_cell else "return_result(parse_value(37))\n")
    )
    cells = [code]
    if later_cell:
        cells.append("return_result(parse_value(37))\n")
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"typing_test_{i}",
                        name="execute_python",
                        arguments=json.dumps({"code": cell}),
                    )
                ],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": ""},
            )
            for i, cell in enumerate(cells)
        ]
    )

    class ParserAgent(Agent, llm=llm):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2)))
        async def solve(self) -> int:
            """Return the parsed integer."""
            ...

    agent = ParserAgent()
    assert await agent.solve() == 37
    assert llm.call_count == len(cells)
    assert not hasattr(agent, "parse_value")
