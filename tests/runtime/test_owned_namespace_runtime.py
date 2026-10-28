# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The opt-in engine through real actor validation/capture, no live model."""

import asyncio

import pytest

from nooa import Agent
from nooa.runtime.execution_namespace import ExecutionNamespace
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.current_call import CurrentCall
from nooa.unifiedllm import FakeLLMClient


class CellAgent(Agent, llm=FakeLLMClient()):
    pass


def namespace():
    return ExecutionNamespace(
        reserved=frozenset(
            {
                "self",
                "notification",
                "print",
                "pprint",
                "show",
                "return_result",
                "python_cell_state",
                "Out",
            }
        )
    )


async def run(agent, ns, code, **values):
    return await agent.runtime.execute_code(
        code,
        execution_namespace=ns,
        wrap_in_function=True,
        builtins=values,
        timeout=5,
    )


async def test_actor_globals_rebind_del_and_partial_error():
    agent, ns = CellAgent(), namespace()
    with ns.enter({}):
        first = await run(
            agent,
            ns,
            "counter=0\ndef bump():\n global counter\n counter+=1\n return counter\nprint(bump())",
        )
        second = await run(agent, ns, "print(bump())")
        third = await run(agent, ns, "print(counter)")
        assert [result.error for result in (first, second, third)] == [None] * 3
        assert [r.stdout for r in (first, second, third)] == ["1\n", "2\n", "2\n"]
        error = await run(agent, ns, "del counter\npartial=8\n1/0")
        assert isinstance(error.error, ZeroDivisionError)
        assert "counter" not in ns.globals and ns.globals["partial"] == 8
        result = await run(agent, ns, "counter=9\nreturn bump()")
        assert result.returned_value == 10


async def test_completion_actual_wait_for_task_old_alias_and_background_rejected():
    agent, ns = CellAgent(), namespace()
    call = CurrentCall(id="one", method_name="answer", decorator="agent", return_type=int)
    builtins = CodeActStrategy()._build_builtins(agent.runtime, call)
    with ns.enter({}):
        first = await run(agent, ns, "saved = return_result\nreturn_result(1)", **builtins)
        assert first.error is None and first.signal.result == {"result": 1}
        second = await run(agent, ns, "saved(2)", **builtins)
        assert second.error is None and second.signal.result == {"result": 2}
        child = await run(
            agent,
            ns,
            "async def child():\n return_result(5)\ntask=asyncio.create_task(child())\nawait task",
            **builtins,
        )
        assert isinstance(child.error, RuntimeError)
        assert child.signal is None
    with ns.enter({}):
        stale = await run(agent, ns, "saved(3)", **builtins)
        assert isinstance(stale.error, RuntimeError) and stale.signal is None
        current = await run(agent, ns, "def helper():\n return_result(4)\nhelper()", **builtins)
        assert current.error is None and current.signal.result == {"result": 4}


async def test_runtime_scope_reader_class_local_self():
    agent, ns = CellAgent(), namespace()
    with ns.enter({}):
        first = await run(
            agent,
            ns,
            "class Reader:\n notification=notification\n def read(self):\n  return notification\nreader=Reader()",
            notification="A",
        )
        assert first.error is None
    with ns.enter({}):
        second = await run(agent, ns, "return Reader.notification, reader.read()", notification="B")
        assert second.error is None
        assert second.returned_value == ("A", "B")


@pytest.mark.parametrize("code", ["eval('1')", "exec('x=1')", "globals()", "input()"])
async def test_existing_restrictions_remain(code):
    agent, ns = CellAgent(), namespace()
    with ns.enter({}):
        result = await run(agent, ns, code)
        assert result.error is not None


async def test_unowned_descendants_cannot_write_later_or_finished_cell():
    agent, ns = CellAgent(), namespace()
    gate = asyncio.Event()
    ns.globals["gate"] = gate
    with ns.enter({}):
        first = await run(
            agent,
            ns,
            "async def late():\n await gate.wait()\n print('late')\n return notification\ntask=asyncio.create_task(late())\nprint('A')",
            notification="A",
        )
        assert first.error is None and first.stdout == "A\n"
    with ns.enter({}):
        second = await run(agent, ns, "gate.set()\nvalue=await task\nprint('B')", notification="B")
        assert second.error is None
        assert second.stdout == "B\n" and first.stdout == "A\n"
        assert ns.globals["value"] == "A"


async def test_annotated_strategy_helper_source_is_ready_before_decorating():
    agent, ns = CellAgent(), namespace()
    with ns.enter({}):
        result = await run(
            agent,
            ns,
            """
seen = []
def observe(fn):
    seen.append(fn._generated_source)
    return fn
@observe
@strategy(PredictStrategy())
async def classify(text: str) -> str:
    ...
return len(seen)
""",
        )
        assert result.error is None, result.error
        assert result.returned_value == 1
        assert "async def classify" in ns.globals["seen"][0]


async def test_cancelled_capture_and_partial_writes_release_executor():
    agent, ns = CellAgent(), namespace()
    ns.globals["gate"] = asyncio.Event()
    with ns.enter({}):
        task = asyncio.create_task(
            run(
                agent,
                ns,
                "value=1\nprint('before')\ntry:\n await gate.wait()\nfinally:\n value=2\n print('finally')",
            )
        )
        for _ in range(100):
            if ns.globals.get("value") == 1:
                break
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await task
        assert ns.globals["value"] == 2
        assert cancelled.value.execution_results[-1].stdout == "before\nfinally\n"
        assert (await run(agent, ns, "return value")).returned_value == 2


@pytest.mark.parametrize("file_backed", [False, True])
async def test_open_capture_preserves_gather_and_thread_stdout_stderr_media(file_backed):
    from nooa.config.truncation_config import CaptureConfig, TruncationConfig
    from nooa.media import Image

    agent, ns = CellAgent(), namespace()
    agent._truncation = TruncationConfig(capture=CaptureConfig(file_backed=file_backed))
    ns.globals["image"] = Image.from_bytes(b"image", media_type="image/png")
    with ns.enter({}):
        result = await run(
            agent,
            ns,
            r"""
import sys
def threaded():
    print('thread stdout')
    sys.stderr.write('thread stderr\n')
    show(image)
async def child():
    print('child stdout')
    sys.stderr.write('child stderr\n')
    show(image)
await asyncio.gather(child())
await asyncio.to_thread(threaded)
print('parent')
""",
        )
    assert result.error is None, result.error
    assert "child stdout\n" in result.stdout and "thread stdout\n" in result.stdout
    assert result.stdout.endswith("parent\n")
    assert result.stderr == "child stderr\nthread stderr\n"
    assert len(result.images) == 2


@pytest.mark.parametrize("file_backed", [False, True])
async def test_closed_capture_rejects_late_descendant_and_thread_media(file_backed):
    from nooa.config.truncation_config import CaptureConfig, TruncationConfig
    from nooa.media import Image

    agent, ns = CellAgent(), namespace()
    agent._truncation = TruncationConfig(capture=CaptureConfig(file_backed=file_backed))
    ns.globals["image"] = Image.from_bytes(b"image", media_type="image/png")
    ns.globals["gate"] = asyncio.Event()
    with ns.enter({}):
        first = await run(
            agent,
            ns,
            r"""
import sys
def thread_output():
    print('late thread')
    sys.stderr.write('late thread stderr')
    show(image)
async def late():
    await gate.wait()
    print('late child')
    sys.stderr.write('late child stderr')
    show(image)
    await asyncio.to_thread(thread_output)
task = asyncio.create_task(late())
print('A')
""",
        )
        assert first.error is None
    with ns.enter({}):
        second = await run(agent, ns, "gate.set()\nawait task\nprint('B')")
        assert second.error is None
    assert first.stdout == "A\n" and second.stdout == "B\n"
    assert first.stderr == second.stderr == ""
    assert first.images == second.images == []


async def test_owned_namespace_middleware_modification_and_identity():
    agent, ns = CellAgent(), namespace()
    trail = []

    async def middleware(ctx, nxt):
        assert ctx.params["execution_namespace"] is ns
        trail.append("before")
        ctx.code = "injected=3\n" + ctx.code
        ctx = await nxt(ctx)
        trail.append("after")
        return ctx

    agent.event_manager.intercept("execute_python", middleware)
    with ns.enter({}):
        result = await run(agent, ns, "return injected")
    assert result.error is None and result.returned_value == 3
    assert trail == ["before", "after"] and ns.globals["injected"] == 3


async def test_real_strategy_helper_invocation_and_once_only_defaults():
    from nooa.unifiedllm import LLMResponse

    agent = CellAgent(
        llm=FakeLLMClient(
            [
                LLMResponse(content='{"value": "classified"}', finish_reason="stop"),
            ],
            strict_exhaustion=True,
        )
    )
    ns = namespace()
    with ns.enter({}):
        result = await run(
            agent,
            ns,
            """
seen=[]
def default():
    seen.append('default')
    return 'text'
@strategy(PredictStrategy())
async def classify(text: str = default()) -> str:
    ...
value = await classify()
return value, seen
""",
        )
    assert result.error is None, result.error
    assert result.returned_value == ("classified", ["default"])
    assert len(agent.llm.calls) == 1


async def test_retained_helper_traceback_keeps_original_source_across_handle_counters():
    import traceback

    agent, ns = CellAgent(), namespace()
    with ns.enter({}):
        first = await run(agent, ns, "def broken():\n return 1 / 0  # original helper source")
        assert first.error is None
        filename = ns.globals["broken"].__code__.co_filename
    with ns.enter({}):
        second = await run(agent, ns, "other = 'later cell reusing execution_count=1'\nbroken()")
        assert isinstance(second.error, ZeroDivisionError)
    frame = traceback.extract_tb(second.error.__traceback__)[-1]
    assert frame.filename == filename
    assert frame.lineno == 2 and "original helper source" in frame.line


async def test_same_named_class_method_does_not_overwrite_helper_source_metadata():
    agent, ns = CellAgent(), namespace()
    with ns.enter({}):
        result = await run(
            agent,
            ns,
            """
def helper():
    return 'outer definition'
class C:
    def helper(self):
        return 'inner definition'
""",
        )
        assert result.error is None
    outer = ns.globals["helper"]._generated_source
    inner = ns.globals["C"].helper._generated_source
    assert "outer definition" in outer and "inner definition" not in outer
    assert "inner definition" in inner and "outer definition" not in inner
