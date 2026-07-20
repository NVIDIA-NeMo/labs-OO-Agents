# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-module inheritance exec_globals visibility (issue 259).

A child Agent whose parent lives in a different module must be able to use the
parent module's module-level functions, constants, and types inside
``execute_python()`` without NameError. Name collisions resolve leaf-wins, and
hidden parent symbols stay hidden.
"""

from __future__ import annotations

import json

import pytest

from nooa.agentdoc.visibility import (
    filter_mro_module_globals,
    iter_agent_mro_modules,
)
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tests.helpers import cross_module_child, cross_module_parent
from tests.helpers.cross_module_child import ChildAgent
from tests.helpers.cross_module_parent import ParentAgent


def _exec_python_resp(code: str) -> LLMResponse:
    """Build a CodeAct LLM response that calls execute_python(code=...)."""
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id="c1", name="execute_python", arguments=json.dumps({"code": code}))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
    )


# --------------------------------------------------------------------------- #
# Unit tests on the merged-namespace helpers.
# --------------------------------------------------------------------------- #


def test_mro_modules_include_parent_and_child():
    """Both the parent and child module names appear in ChildAgent's MRO module list."""
    modules = iter_agent_mro_modules(ChildAgent)
    names = {m.__name__ for m in modules}
    assert cross_module_parent.__name__ in names
    assert cross_module_child.__name__ in names


def test_mro_modules_exclude_framework_and_builtins():
    """Framework (nooa*) and builtins modules are excluded from the MRO list."""
    names = {m.__name__ for m in iter_agent_mro_modules(ChildAgent)}
    # The framework Agent base module and builtins must be filtered out.
    assert "builtins" not in names
    assert not any(n == "nooa" or n.startswith("nooa.") for n in names)


def test_parent_module_symbols_present():
    """Parent-module function, constant, and type all surface in the child's merged globals."""
    g = filter_mro_module_globals(ChildAgent)
    # Parent-module function, constant, and type all surface in the child.
    assert "shared_util" in g
    assert g["shared_util"] is cross_module_parent.shared_util
    assert g.get("SHARED_CONSTANT") == 7
    assert "ParentModel" in g
    assert g["ParentModel"] is cross_module_parent.ParentModel


def test_leaf_wins_on_collision():
    """A name defined in both modules resolves to the child (leaf) module's value."""
    g = filter_mro_module_globals(ChildAgent)
    # COLLIDING_NAME is defined in both modules; the child (leaf) wins.
    assert g["COLLIDING_NAME"] == "from-child"
    # Child-only symbol is present too.
    assert g.get("CHILD_CONSTANT") == 99


def test_hidden_parent_symbols_stay_hidden():
    """Parent-module symbols hidden via Annotated/@hidden/with-hidden do not leak."""
    g = filter_mro_module_globals(ChildAgent)
    assert "API_KEY" not in g  # Annotated[str, hidden]
    assert "hidden_parent_util" not in g  # @hidden
    assert "HIDDEN_PARENT_SECRET" not in g  # with hidden:


def test_single_module_parent_still_works():
    """The parent on its own (single module) is unaffected by the MRO merge."""
    # Regression guard: the parent on its own (single module) is unaffected.
    g = filter_mro_module_globals(ParentAgent)
    assert g.get("SHARED_CONSTANT") == 7
    assert "shared_util" in g
    assert g["COLLIDING_NAME"] == "from-parent"


# --------------------------------------------------------------------------- #
# Execution-context prompt block reflects the merged namespace.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_execution_context_lists_parent_symbols():
    """The <execution_context> block lists parent-module types and functions for the child."""
    agent = ChildAgent(llm=FakeLLMClient())
    strat = CodeActStrategy()

    class _Runtime:
        def __init__(self, a):
            self.agent = a

    block = await strat.execution_context(_Runtime(agent))
    assert "ParentModel" in block  # parent-module type listed as available
    assert "shared_util" in block  # parent-module function listed as available


# --------------------------------------------------------------------------- #
# End-to-end: child CodeAct generated code uses parent-module symbols, no NameError.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_child_can_call_parent_module_symbols():
    """A CodeAct child runs execute_python() calling the parent module's
    shared_util(SHARED_CONSTANT) and returns the result without NameError."""
    llm = FakeLLMClient(
        scripted_responses=[_exec_python_resp("return_result(shared_util(SHARED_CONSTANT))")]
    )
    agent = ChildAgent(llm=llm)
    result = await agent.child_task()
    assert result == 14  # shared_util(7) == 14, no NameError
