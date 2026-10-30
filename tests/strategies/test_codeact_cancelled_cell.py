# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cancel during a CodeAct cell is recorded for the model.

When the task running a CodeAct turn is cancelled while an ``execute_python``
cell is awaiting, the strategy appends a ``PythonOutput`` with
``ResultStatus.CANCELLED`` and the output the cell printed before the cancel,
leaves the cell's ``ToolCallEvent`` untouched (events are appended, never
rewritten), and re-raises. A cancel that
lands during the model call (no cell running) emits no ``PythonOutput``.
"""

import asyncio
import json
from typing import Any

import pytest

from nooa import Agent, strategy
from nooa.config.strategy_config import CodeActConfig
from nooa.context_blocks.events import ToolCallEvent
from nooa.events import PythonOutput, ResultStatus
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# Module globals are visible to generated cells (execute_code builds its
# globals from the agent's module), so the test can hand the cell events.
CELL_STARTED: asyncio.Event | None = None
CELL_BLOCKER: asyncio.Event | None = None

_CELL = """\
import sys
print("partial stdout")
print("partial stderr", file=sys.stderr)
CELL_STARTED.set()
await CELL_BLOCKER.wait()
print("never printed")
"""


def _cell_response(code: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))
        ],
        finish_reason="tool_calls",
    )


class Worker(Agent, llm=FakeLLMClient()):
    @strategy(CodeActStrategy())
    async def work(self) -> str:
        """Do the work."""
        ...


@pytest.mark.asyncio
async def test_cancel_during_cell_records_cancelled_output():
    global CELL_STARTED, CELL_BLOCKER
    CELL_STARTED = asyncio.Event()
    CELL_BLOCKER = asyncio.Event()
    agent = Worker(llm=FakeLLMClient([_cell_response(_CELL, "call_cell")]))

    task = asyncio.create_task(agent.work())
    await asyncio.wait_for(CELL_STARTED.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    events = agent.event_manager.values()
    outputs = [e for e in events if isinstance(e, PythonOutput)]
    assert len(outputs) == 1
    output = outputs[0]
    assert output.tool_call_id == "call_cell"
    assert output.execution_status is ResultStatus.CANCELLED
    assert "partial stdout" in output.stdout
    assert "never printed" not in output.stdout
    assert "partial stderr" in output.stderr
    assert "prefill" not in output.metadata

    calls = [e for e in events if isinstance(e, ToolCallEvent) and e.tool_call_id == "call_cell"]
    assert len(calls) == 1
    # The tool-call event is not rewritten on cancel: it keeps the RUNNING
    # receipt it had when the cell started. The appended PythonOutput is the record.
    assert calls[0].result is not None
    assert calls[0].result.result_status is ResultStatus.RUNNING


@pytest.mark.asyncio
async def test_a_failing_record_does_not_replace_the_cancellation(monkeypatch, caplog):
    """If storing the cancelled cell fails, the cancel still propagates as a cancel.

    ``_record_cancelled_cell`` runs inside ``except CancelledError``; an error
    from ``event_manager.add`` (a SQLite or serialization failure) must be
    logged, not raised in place of the ``CancelledError``.
    """
    global CELL_STARTED, CELL_BLOCKER
    CELL_STARTED = asyncio.Event()
    CELL_BLOCKER = asyncio.Event()
    agent = Worker(llm=FakeLLMClient([_cell_response(_CELL, "call_cell")]))
    real_add = agent.event_manager.add

    def add(event, *args, **kwargs):
        if isinstance(event, PythonOutput) and event.execution_status is ResultStatus.CANCELLED:
            raise RuntimeError("store is broken")
        return real_add(event, *args, **kwargs)

    monkeypatch.setattr(agent.event_manager, "add", add)

    task = asyncio.create_task(agent.work())
    await asyncio.wait_for(CELL_STARTED.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "store is broken" in caplog.text


class _BlockingLLM(FakeLLMClient):
    """A model call that never returns until cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    async def acall(self, *args: Any, **kwargs: Any) -> LLMResponse:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_cancel_during_model_call_emits_no_python_output():
    llm = _BlockingLLM()
    agent = Worker(llm=llm)

    task = asyncio.create_task(agent.work())
    await asyncio.wait_for(llm.entered.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    events = agent.event_manager.values()
    assert not [e for e in events if isinstance(e, PythonOutput)]
    assert not [e for e in events if isinstance(e, ToolCallEvent)]


class _BlockingPrefill:
    """Prefill plugin whose step prints, signals, then blocks until cancelled."""

    def get_code(self, call, config=None) -> str:  # noqa: ANN001 - Prefill protocol
        return _CELL


class PrefillWorker(Agent, llm=FakeLLMClient()):
    @strategy(CodeActStrategy(config=CodeActConfig(prefill=_BlockingPrefill())))
    async def work(self) -> str:
        """Do the work."""
        ...


@pytest.mark.asyncio
async def test_cancel_during_prefill_cell_records_cancelled_output():
    """A prefill step runs before the model loop; a cancel there is recorded the
    same way as a cancel in a model-written cell."""
    global CELL_STARTED, CELL_BLOCKER
    CELL_STARTED = asyncio.Event()
    CELL_BLOCKER = asyncio.Event()
    agent = PrefillWorker(llm=FakeLLMClient([]))

    task = asyncio.create_task(agent.work())
    await asyncio.wait_for(CELL_STARTED.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    events = agent.event_manager.values()
    outputs = [e for e in events if isinstance(e, PythonOutput)]
    assert len(outputs) == 1
    output = outputs[0]
    assert output.tool_call_id.startswith("prefill_")
    assert output.execution_status is ResultStatus.CANCELLED
    assert "partial stdout" in output.stdout
    assert "partial stderr" in output.stderr
    calls = [
        e for e in events if isinstance(e, ToolCallEvent) and e.tool_call_id == output.tool_call_id
    ]
    assert len(calls) == 1
    assert calls[0].result is not None
    assert calls[0].result.result_status is ResultStatus.RUNNING
    # Tagged like a completed prefill output, so counts that skip prefill skip it too.
    assert output.metadata["prefill"] is True
    assert output.metadata["prefill_type"] == calls[0].metadata["prefill_type"]


_OUTER_CELL = """\
print("outer before the nested call")
await self.inner()
"""

_INNER_CELL = """\
print("inner partial")
CELL_STARTED.set()
await CELL_BLOCKER.wait()
"""


class Nesting(Agent, llm=FakeLLMClient()):
    @strategy(CodeActStrategy())
    async def work(self) -> str:
        """Do the work."""
        ...

    @strategy(CodeActStrategy())
    async def inner(self) -> str:
        """Do the inner work."""
        ...


@pytest.mark.asyncio
async def test_nested_cancel_records_each_cell_with_its_own_output():
    """A cancel through a nested CodeAct call records both cells, each with its own output.

    The same CancelledError passes through both cells. Each cell's capture
    appends its own output, and each recorder takes the last entry, which is
    the one its own cell just appended.
    """
    global CELL_STARTED, CELL_BLOCKER
    CELL_STARTED = asyncio.Event()
    CELL_BLOCKER = asyncio.Event()
    agent = Nesting(
        llm=FakeLLMClient(
            [_cell_response(_OUTER_CELL, "call_outer"), _cell_response(_INNER_CELL, "call_inner")]
        )
    )

    task = asyncio.create_task(agent.work())
    await asyncio.wait_for(CELL_STARTED.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    outputs = {
        e.tool_call_id: e for e in agent.event_manager.values() if isinstance(e, PythonOutput)
    }
    assert set(outputs) == {"call_outer", "call_inner"}
    assert all(o.execution_status is ResultStatus.CANCELLED for o in outputs.values())
    assert "inner partial" in outputs["call_inner"].stdout
    assert "outer before the nested call" in outputs["call_outer"].stdout
    assert "inner partial" not in outputs["call_outer"].stdout


_STEP_CELL = """\
print("step 1")
await self.inner()
"""

_PURE_PYTHON_CODE = """\
print("pure python partial")
CELL_STARTED.set()
await CELL_BLOCKER.wait()
"""


class MixedNesting(Agent, llm=FakeLLMClient()):
    @strategy(CodeActStrategy())
    async def work(self) -> str:
        """Do the work."""
        ...

    @strategy(PurePythonStrategy())
    async def inner(self) -> str:
        """Do the inner work."""
        ...


@pytest.mark.asyncio
async def test_nested_cancel_through_a_non_codeact_cell_keeps_the_outer_output():
    """An inner cell that records nothing must not lend its output to the outer cell.

    The PurePython strategy runs its code through ``execute_code`` but does not
    record a cancelled cell, so its partial result stays on the exception. The
    enclosing CodeAct cell must still record its own output.
    """
    global CELL_STARTED, CELL_BLOCKER
    CELL_STARTED = asyncio.Event()
    CELL_BLOCKER = asyncio.Event()
    agent = MixedNesting(
        llm=FakeLLMClient(
            [
                _cell_response(_STEP_CELL, "call_outer"),
                LLMResponse(raw_response=None, content=_PURE_PYTHON_CODE, finish_reason="stop"),
            ]
        )
    )

    task = asyncio.create_task(agent.work())
    await asyncio.wait_for(CELL_STARTED.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    outputs = [e for e in agent.event_manager.values() if isinstance(e, PythonOutput)]
    outer = [o for o in outputs if o.tool_call_id == "call_outer"]
    assert len(outer) == 1
    assert outer[0].execution_status is ResultStatus.CANCELLED
    assert "step 1" in outer[0].stdout
    assert "pure python partial" not in outer[0].stdout
