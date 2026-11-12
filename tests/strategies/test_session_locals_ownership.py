# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A call's seeds and write-back touch only what the call owns (#583 review round 2)."""

from typing import Any

import pytest

from nooa import Agent, strategy
from nooa.errors import GenerationError
from nooa.runtime.execution_namespace import RUNTIME_NAMES, ExecutionNamespace
from nooa.strategies.pure_python import _HTTPX_TIMEOUT_EXCEPTIONS, PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient


class _ScriptedStrategy(PurePythonStrategy):
    """Each turn runs ``steps[n](stack)``: it returns the turn's code or raises an LLM error."""

    def __init__(self, steps, **kwargs):
        super().__init__(**kwargs)
        self.steps = list(steps)

    async def _generate_code(self, runtime, session):
        return self.steps.pop(0)(), None


def _agent(steps, *, max_retries: int = 2, namespace: ExecutionNamespace | None = None):
    class ScriptedAgent(Agent, llm=FakeLLMClient()):
        @strategy(_ScriptedStrategy(steps, max_retries=max_retries))
        async def go(self, x: int) -> int:
            """Work on {x}."""
            ...

        def execution_namespace_for(self, call):
            return namespace if namespace is not None else super().execution_namespace_for(call)

    return ScriptedAgent(llm=FakeLLMClient())


@pytest.mark.asyncio
async def test_an_occupied_shared_namespace_is_never_seeded():
    """A second overlapping call fails before it changes the first call's names."""
    shared = ExecutionNamespace(reserved=RUNTIME_NAMES)
    shared.globals["x"] = shared.globals["y"] = "first call"
    shared.user_names.update({"x", "y"})
    agent = _agent([lambda: "return x"], namespace=shared)
    with shared.enter({}):  # the first call is running in it
        with pytest.raises(RuntimeError, match="Concurrent/reentrant"):
            await agent.go(x=2, _session_locals={"y": "second call", "z": 1})
    assert shared.user_values() == {"x": "first call", "y": "first call"}


def _raise(exc: BaseException):
    def step():
        raise exc

    return step


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [_HTTPX_TIMEOUT_EXCEPTIONS[0]("slow"), ConnectionError("down")],
    ids=["timeout", "api-error"],
)
async def test_retry_exhaustion_writes_session_locals_back(error):
    stack: dict[str, Any] = {"kept": 1}
    agent = _agent([lambda: "made = x + kept", _raise(error), _raise(error)])
    with pytest.raises(GenerationError, match="retries"):
        await agent.go(x=2, _session_locals=stack)
    assert stack == {"kept": 1, "made": 3}


@pytest.mark.asyncio
async def test_write_back_keeps_what_others_changed_while_the_call_waited():
    stack: dict[str, Any] = {"seed": 1, "gone": 2, "touched": 3, "dropped": 4}

    def meanwhile():
        stack["fresh"] = "added by someone else"
        stack["seed"] = "updated by someone else"
        del stack["dropped"]
        return "del gone\ntouched = 30\nreturn x"

    agent = _agent([lambda: "made = 1", meanwhile])
    assert await agent.go(x=5, _session_locals=stack) == 5
    assert stack == {
        "seed": "updated by someone else",
        "touched": 30,
        "made": 1,
        "fresh": "added by someone else",
    }
