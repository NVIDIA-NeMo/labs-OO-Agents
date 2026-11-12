# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A call's seeds and write-back touch only what the call owns.

The rules are in ``src/nooa/runtime/README.md`` ("session_locals seeding and
write-back"). A shared namespace comes from ``Agent.execution_namespace_for``,
which only CodeAct consults: those tests run CodeAct; PurePython always runs in
a fresh namespace per call.
"""

import json
from typing import Any

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.errors import GenerationError
from nooa.runtime.execution_namespace import RUNTIME_NAMES, ExecutionNamespace
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.pure_python import _HTTPX_TIMEOUT_EXCEPTIONS, PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class _ScriptedStrategy(PurePythonStrategy):
    """Each turn runs ``steps[n](stack)``: it returns the turn's code or raises an LLM error."""

    def __init__(self, steps, **kwargs):
        super().__init__(**kwargs)
        self.steps = list(steps)

    async def _generate_code(self, runtime, session):
        return self.steps.pop(0)(), None


def _agent(steps, *, max_retries: int = 2):
    class ScriptedAgent(Agent, llm=FakeLLMClient()):
        @strategy(_ScriptedStrategy(steps, max_retries=max_retries))
        async def go(self, x: int) -> int:
            """Work on {x}."""
            ...

    return ScriptedAgent(llm=FakeLLMClient())


class _StepLLM(FakeLLMClient):
    """Each LLM call runs ``steps[n]()`` and answers with its code as one ``execute_python`` call."""

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)

    def _scripted_call(self, messages, tools, output_model, kwargs):
        code = self.steps.pop(0)()
        call = ToolCall(id="call_1", name="execute_python", arguments=json.dumps({"code": code}))
        self._response_queue.append(
            LLMResponse(
                raw_response=None, content="", tool_calls=[call], finish_reason="tool_calls"
            )
        )
        return super()._scripted_call(messages, tools, output_model, kwargs)


def _shared() -> ExecutionNamespace:
    """A namespace several CodeAct calls share (``x`` is the argument: reserved)."""
    return ExecutionNamespace(reserved=RUNTIME_NAMES | {"x"})


def _codeact_agent(steps, *, namespace: ExecutionNamespace):
    class SharedAgent(Agent, llm=FakeLLMClient()):
        @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
        async def go(self, x: int) -> int:
            """Work on {x}."""
            ...

        def execution_namespace_for(self, call):
            return namespace

    return SharedAgent(llm=_StepLLM(steps))


@pytest.mark.asyncio
async def test_an_occupied_shared_namespace_is_never_seeded():
    """A second overlapping call fails before it changes the first call's names."""
    shared = _shared()
    shared.globals["x2"] = shared.globals["y"] = "first call"
    shared.user_names.update({"x2", "y"})
    agent = _codeact_agent([lambda: "return_result(x)"], namespace=shared)
    with shared.enter({}):  # the first call is running in it
        with pytest.raises(RuntimeError, match="Concurrent/reentrant"):
            await agent.go(x=2, _session_locals={"y": "second call", "z": 1})
    assert {n: shared.globals.get(n) for n in ("x2", "y", "z")} == {
        "x2": "first call",
        "y": "first call",
        "z": None,
    }


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
    """A reused namespace's older bindings never reach a later call's caller."""
    shared = _shared()
    first: dict[str, Any] = {}
    agent = _codeact_agent([lambda: "stale = 'first call'\nreturn_result(x)"], namespace=shared)
    assert await agent.go(x=1, _session_locals=first) == 1
    assert first == {"stale": "first call"}
    stack: dict[str, Any] = {}

    def meanwhile():
        stack["stale"] = "set by someone else"
        # An assignment that never runs is not a change either.
        return "if False:\n    stale = 'unused'\nmade = 2\nreturn_result(x)"

    assert await _codeact_agent([meanwhile], namespace=shared).go(x=3, _session_locals=stack) == 3
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
    """A ``global`` write inside a helper is a change of the call."""
    stack: dict[str, Any] = {"counter": 0}
    code = "def bump():\n    global counter\n    counter = 1\n\nbump()\nreturn x"
    assert await _agent([lambda: code]).go(x=1, _session_locals=stack) == 1
    assert stack["counter"] == 1 and set(stack) == {"counter", "bump"}


@pytest.mark.asyncio
async def test_a_helper_from_an_earlier_call_rebinding_a_seed_writes_back():
    """A helper defined by an earlier call in a shared namespace writes this call's seed."""
    shared = _shared()
    first: dict[str, Any] = {}
    helper = (
        "def bump():\n    global counter, made\n    counter = counter + 1\n"
        "    made = 'helper'\nreturn_result(x)"
    )
    assert (
        await _codeact_agent([lambda: helper], namespace=shared).go(x=1, _session_locals=first) == 1
    )
    assert set(first) == {"bump"}
    stack: dict[str, Any] = {"counter": 10, "other": 5}
    agent = _codeact_agent([lambda: "bump()\nreturn_result(x)"], namespace=shared)
    assert await agent.go(x=2, _session_locals=stack) == 2
    assert stack == {"counter": 11, "made": "helper", "other": 5}


@pytest.mark.asyncio
async def test_names_a_helper_binds_via_global_join_the_inventory():
    """What write-back finds changed is a user name too: ``inventory()`` lists it."""
    shared = _shared()
    code = "def bump():\n    global q\n    q = 1\n\nbump()\nreturn_result(x)"
    assert await _codeact_agent([lambda: code], namespace=shared).go(x=1) == 1
    assert {"bump", "q"} <= set(shared.inventory()["names"])
    assert "q" in shared.user_names


@pytest.mark.asyncio
async def test_a_write_without_binding_syntax_writes_back():
    """A namespace write no cell spells out (``globals()[...]``, exec) is still a change.

    Generated code may not call ``globals()``/``exec``, so the step writes the
    namespace's globals directly while the call runs, as such a write would.
    """
    shared = _shared()
    stack: dict[str, Any] = {"seed": 1, "kept": 2}

    def meanwhile():
        shared.globals["seed"] = 2
        shared.globals["fresh"] = 3
        del shared.globals["kept"]
        return "return_result(x)"

    assert await _codeact_agent([meanwhile], namespace=shared).go(x=1, _session_locals=stack) == 1
    assert stack == {"seed": 2, "fresh": 3}


@pytest.mark.asyncio
async def test_a_helper_deleting_a_seed_via_global_deletes_it():
    stack: dict[str, Any] = {"gone": 1, "kept": 2}
    code = "def drop():\n    global gone\n    del gone\n\ndrop()\nreturn x"
    assert await _agent([lambda: code]).go(x=1, _session_locals=stack) == 1
    assert set(stack) == {"kept", "drop"} and stack["kept"] == 2


@pytest.mark.asyncio
async def test_a_rebound_seeded_dunder_writes_back():
    """A seeded dunder (``__version__``) is the caller's name: rebinding it is a change."""
    stack: dict[str, Any] = {"__version__": "1"}
    code = "__version__ = '2'\nx2: int = 3\nreturn x"
    assert await _agent([lambda: code]).go(x=1, _session_locals=stack) == 1
    assert stack["__version__"] == "2"
    assert "__annotations__" not in stack


def _nesting_agent(outer_steps, other_steps):
    """PurePython methods whose hook returns one namespace for every call."""
    shared = ExecutionNamespace(reserved=RUNTIME_NAMES)

    class NestingAgent(Agent, llm=FakeLLMClient()):
        @strategy(_ScriptedStrategy(outer_steps, max_retries=1))
        async def outer(self, w: int) -> str:
            """Outer."""
            ...

        @strategy(_ScriptedStrategy(other_steps, max_retries=1))
        async def other(self, z: int) -> str:
            """Other with {z}."""
            ...

        def execution_namespace_for(self, call):
            return shared

    return NestingAgent(llm=FakeLLMClient()), shared


@pytest.mark.asyncio
async def test_pure_python_ignores_the_namespace_hook():
    """PurePython runs each call in a fresh namespace; the hook is CodeAct's.

    So a PurePython cell can await another PurePython method of the same agent,
    the arguments of one call are not visible to another, and the namespace
    the hook returns is never touched.
    """
    inner = "try:\n    w\n    seen = '+w'\nexcept NameError:\n    seen = ''\nreturn 'inner' + seen"
    agent, shared = _nesting_agent(
        [lambda: "v = await self.other(z=1)\nreturn v + repr(w)"], [lambda: inner]
    )
    before = dict(shared.globals)
    assert await agent.outer(w=5) == "inner5"
    assert shared.globals == before and not shared.user_names
