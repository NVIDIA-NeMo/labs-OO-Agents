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


@pytest.mark.asyncio
async def test_write_back_skips_names_an_earlier_call_bound_in_a_shared_namespace():
    """A reused namespace's older bindings never reach a later call's caller (round 3)."""
    shared = ExecutionNamespace(reserved=RUNTIME_NAMES)
    first: dict[str, Any] = {}
    assert (
        await _agent([lambda: "stale = 'first call'\nreturn x"], namespace=shared).go(
            x=1, _session_locals=first
        )
        == 1
    )
    assert first == {"stale": "first call"}
    stack: dict[str, Any] = {}

    def meanwhile():
        stack["stale"] = "set by someone else"
        # An assignment that never runs is not a change either.
        return "if False:\n    stale = 'unused'\nmade = 2\nreturn x"

    assert await _agent([meanwhile], namespace=shared).go(x=3, _session_locals=stack) == 3
    assert stack == {"stale": "set by someone else", "made": 2}


@pytest.mark.asyncio
async def test_deleting_a_none_seed_the_caller_already_removed_does_not_raise():
    stack: dict[str, Any] = {"empty": None}

    def meanwhile():
        del stack["empty"]
        return "del empty\nreturn x"

    assert await _agent([meanwhile]).go(x=4, _session_locals=stack) == 4
    assert stack == {}


@pytest.mark.asyncio
async def test_a_helper_rebinding_a_seed_via_global_writes_back():
    """A ``global`` write inside a helper is a change of the call (round 4)."""
    stack: dict[str, Any] = {"counter": 0}
    code = "def bump():\n    global counter\n    counter = 1\n\nbump()\nreturn x"
    assert await _agent([lambda: code]).go(x=1, _session_locals=stack) == 1
    assert stack["counter"] == 1 and set(stack) == {"counter", "bump"}


@pytest.mark.asyncio
async def test_a_helper_from_an_earlier_call_rebinding_a_seed_writes_back():
    """A helper defined by an earlier call in a shared namespace writes this call's seed."""
    shared = ExecutionNamespace(reserved=RUNTIME_NAMES)
    first: dict[str, Any] = {}
    helper = "def bump():\n    global counter, made\n    counter = counter + 1\n    made = 'helper'\nreturn x"
    assert await _agent([lambda: helper], namespace=shared).go(x=1, _session_locals=first) == 1
    assert set(first) == {"bump"}
    stack: dict[str, Any] = {"counter": 10, "other": 5}
    assert (
        await _agent([lambda: "bump()\nreturn x"], namespace=shared).go(x=2, _session_locals=stack)
        == 2
    )
    assert stack == {"counter": 11, "made": "helper", "other": 5}


@pytest.mark.asyncio
async def test_a_write_without_binding_syntax_writes_back():
    """A namespace write no cell spells out (``globals()[...]``, exec) is still a change.

    Generated code may not call ``globals()``/``exec``, so the step writes the
    namespace's globals directly while the call runs, as such a write would.
    """
    shared = ExecutionNamespace(reserved=RUNTIME_NAMES)
    stack: dict[str, Any] = {"seed": 1, "kept": 2}

    def meanwhile():
        shared.globals["seed"] = 2
        shared.globals["fresh"] = 3
        del shared.globals["kept"]
        return "return x"

    assert await _agent([meanwhile], namespace=shared).go(x=1, _session_locals=stack) == 1
    assert stack == {"seed": 2, "fresh": 3}


@pytest.mark.asyncio
async def test_a_helper_deleting_a_seed_via_global_deletes_it():
    stack: dict[str, Any] = {"gone": 1, "kept": 2}
    code = "def drop():\n    global gone\n    del gone\n\ndrop()\nreturn x"
    assert await _agent([lambda: code]).go(x=1, _session_locals=stack) == 1
    assert set(stack) == {"kept", "drop"} and stack["kept"] == 2
