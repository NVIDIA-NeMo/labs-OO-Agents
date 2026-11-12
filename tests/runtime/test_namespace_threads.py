# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime names in threads and executors that do not inherit the cell's context."""

import asyncio

import pytest

from nooa import Agent
from nooa.runtime.execution_namespace import RUNTIME_NAMES, ExecutionNamespace
from nooa.unifiedllm import FakeLLMClient


class CellAgent(Agent, llm=FakeLLMClient()):
    pass


async def run(agent, ns, code, **values):
    return await agent.runtime.execute_code(
        code, execution_namespace=ns, wrap_in_function=True, builtins=values, timeout=5
    )


EXECUTOR = """
def helper():
    print("hi from executor")
    return self is expected
loop = asyncio.get_running_loop()
return await loop.run_in_executor(None, helper)
"""

THREAD = """
import threading
out = []
def helper():
    try:
        print("hi from thread")
        out.append("ok")
    except BaseException as e:
        out.append(repr(e))
t = threading.Thread(target=helper)
t.start()
while t.is_alive():
    await asyncio.sleep(0.01)
return out
"""


async def test_run_in_executor_helpers_see_print_and_self_and_output_is_captured():
    agent, ns = CellAgent(), ExecutionNamespace(reserved=RUNTIME_NAMES)
    try:
        with ns.enter({}):
            result = await run(agent, ns, EXECUTOR, self=agent, expected=agent)
        assert result.error is None, result.error
        assert result.returned_value is True
        assert result.stdout == "hi from executor\n"
    finally:
        await agent.aclose()


async def test_thread_helpers_print_into_the_cell():
    agent, ns = CellAgent(), ExecutionNamespace(reserved=RUNTIME_NAMES)
    try:
        with ns.enter({}):
            result = await run(agent, ns, THREAD, self=agent)
        assert result.error is None, result.error
        assert result.returned_value == ["ok"]
        assert result.stdout == "hi from thread\n"
    finally:
        await agent.aclose()


async def test_a_thread_that_prints_after_its_cell_ended_prints_to_process_stdout(capfd):
    agent, ns = CellAgent(), ExecutionNamespace(reserved=RUNTIME_NAMES)
    start = """
import threading
gate = threading.Event()
out = []
def late():
    gate.wait(5)
    try:
        print("too late")
        out.append("printed")
    except RuntimeError as e:
        out.append(str(e))
worker = threading.Thread(target=late)
worker.start()
"""
    try:
        with ns.enter({}):
            first = await run(agent, ns, start, self=agent)
            assert first.error is None, first.error
            ns.globals["gate"].set()  # the cell has ended: no cell is running now
            await asyncio.to_thread(ns.globals["worker"].join, 5)
            second = await run(agent, ns, "return out", self=agent)
        assert second.error is None
        assert second.returned_value == ["printed"]
        assert "too late" not in (first.stdout + second.stdout)
        assert "too late" in capfd.readouterr().out
    finally:
        await agent.aclose()


HELPER = """
def helper(x):
    print(f"helper {x}")
    return self
def finish():
    return_result("too late")
"""


async def test_a_helper_from_an_ended_call_prints_into_the_cell_that_calls_it():
    agent = CellAgent()
    ns_a, ns_b = (
        ExecutionNamespace(reserved=RUNTIME_NAMES),
        ExecutionNamespace(reserved=RUNTIME_NAMES),
    )
    try:
        with ns_a.enter({}):
            defined = await run(agent, ns_a, HELPER, self=agent)
        assert defined.error is None, defined.error
        ns_b.globals["helper"] = ns_a.globals["helper"]
        with ns_b.enter({}):
            used = await run(agent, ns_b, "return helper(1)", self=agent)
        assert used.error is None, used.error
        assert used.stdout == "helper 1\n"
        assert used.returned_value is agent
    finally:
        await agent.aclose()


async def test_an_ended_cell_keeps_print_and_self_but_not_return_result(capfd):
    agent, ns = CellAgent(), ExecutionNamespace(reserved=RUNTIME_NAMES)
    try:
        with ns.enter({}):
            defined = await run(agent, ns, HELPER, self=agent)
        assert defined.error is None, defined.error
        assert ns.globals["helper"](2) is agent  # no cell running anywhere
        assert "helper 2" in capfd.readouterr().out
        with pytest.raises(RuntimeError, match="'return_result' belongs to a cell that has ended"):
            ns.globals["finish"]()
    finally:
        await agent.aclose()


async def test_asyncio_to_thread_still_uses_the_inherited_context():
    agent, ns = CellAgent(), ExecutionNamespace(reserved=RUNTIME_NAMES)
    try:
        with ns.enter({}):
            result = await run(
                agent,
                ns,
                "def helper():\n    print('to_thread')\n    return 1\n"
                "return await asyncio.to_thread(helper)",
                self=agent,
            )
        assert result.error is None and result.returned_value == 1
        assert result.stdout == "to_thread\n"
    finally:
        await agent.aclose()


async def test_a_helper_called_from_another_namespaces_cell_prints_into_the_caller_cell():
    """While the helper's own namespace runs a cell, a caller with its own call keeps the output."""
    a1, a2 = CellAgent(), CellAgent()
    ns1, ns2 = (
        ExecutionNamespace(reserved=RUNTIME_NAMES),
        ExecutionNamespace(reserved=RUNTIME_NAMES),
    )
    try:
        with ns1.enter({}):
            defined = await run(
                a1, ns1, "def helper():\n    print('helper output')\n    return 1", self=a1
            )
        assert defined.error is None, defined.error
        helper = ns1.globals["helper"]
        started = asyncio.Event()
        ns1.globals["started"] = started  # builtins seed a namespace only on its first cell

        async def owner_cell():
            with ns1.enter({}):
                return await run(
                    a1,
                    ns1,
                    "started.set()\nawait asyncio.sleep(0.2)\nreturn 'done'",
                    self=a1,
                )

        async def caller_cell():
            await asyncio.wait_for(started.wait(), 5)
            with ns2.enter({}):
                return await run(a2, ns2, "return h()", self=a2, h=helper)

        owner, caller = await asyncio.gather(owner_cell(), caller_cell())
        assert owner.error is None and caller.error is None, (owner.error, caller.error)
        assert caller.stdout == "helper output\n"
        assert owner.stdout == ""
    finally:
        await a1.aclose()
        await a2.aclose()


async def test_after_a_call_the_namespace_keeps_only_the_output_names_and_self():
    """Arguments, ``_call`` and ``return_result`` are released when the call ends."""
    import gc
    import weakref

    class Big:
        pass

    agent, ns = CellAgent(), ExecutionNamespace(reserved=RUNTIME_NAMES | {"data"})
    data = Big()
    alive = weakref.ref(data)
    try:
        with ns.enter({}):
            # No deadline: its cancelled timer would keep the call's context
            # (and so the call's bindings) until asyncio drops it.
            result = await agent.runtime.execute_code(
                "def helper():\n    print('x')\nreturn 1",
                execution_namespace=ns,
                wrap_in_function=True,
                timeout=None,
                builtins={
                    "self": agent,
                    "data": data,
                    "return_result": lambda value: value,
                    "_call": object(),
                },
            )
        assert result.error is None, result.error
        assert set(ns._last_values) <= {"print", "pprint", "show", "self"}
        del data
        gc.collect()
        assert alive() is None
        assert ns.globals["helper"]() is None  # print keeps working after the call
    finally:
        await agent.aclose()


def test_session_locals_never_overwrite_the_namespace_internals():
    from types import SimpleNamespace

    from nooa.runtime.execution_namespace import enter_call_namespace

    call = SimpleNamespace(
        kwargs={},
        session_locals={"__builtins__": None, "_nooa_ns_delete": "x", "kept": 1},
    )
    with enter_call_namespace(object(), call) as ns:
        assert ns.globals["__builtins__"] is not None
        assert ns.globals["_nooa_ns_delete"] == ns.delete_global
        assert ns.user_values() == {"kept": 1}
    assert call.seeded_locals == {"kept": 1}
