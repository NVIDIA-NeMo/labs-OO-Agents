# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime names in threads and executors that do not inherit the cell's context."""

import asyncio

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


async def test_a_thread_that_prints_after_its_cell_ended_gets_a_clear_error():
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
        [message] = second.returned_value
        assert "after its cell ended" in message
        assert "output would be lost" in message
        assert "too late" not in (first.stdout + second.stdout)
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
