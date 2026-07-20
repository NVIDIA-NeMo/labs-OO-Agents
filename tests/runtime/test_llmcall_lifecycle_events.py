# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLMCallStart / LLMCallEnd lifecycle events.

These RUNTIME_EVENTs bracket the ``acall()`` round-trip inside
``runtime.generate()``. Observers subscribe with
``event_manager.on("LLMCallStart"/"LLMCallEnd", fn)`` to surface
"waiting on model" state without inferring it from cell boundaries.
"""

import pytest

from nooa import Agent, strategy
from nooa.context_blocks.models import Role
from nooa.errors import GenerationError
from nooa.events import LLMCallEnd, LLMCallStart
from nooa.runtime.event_manager import EventManager
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient


def test_llmcall_events_are_runtime_events_never_recorded():
    """RUNTIME_EVENT role → emitted to on() handlers but not stored as context."""
    em = EventManager()
    seen = []
    em.on("LLMCallStart", lambda e: seen.append(("start", e.method_name, e.turn_number)))
    em.on("LLMCallEnd", lambda e: seen.append(("end", e.success, e.exception_type)))

    em.add(
        LLMCallStart(method_name="m", strategy="S", generation_id="g", turn_number=2),
        record=False,
    )
    em.add(
        LLMCallEnd(
            method_name="m",
            strategy="S",
            generation_id="g",
            turn_number=2,
            success=False,
            exception_type="TimeoutError",
        ),
        record=False,
    )

    assert LLMCallStart._role == Role.RUNTIME_EVENT
    assert LLMCallEnd._role == Role.RUNTIME_EVENT
    assert seen == [("start", "m", 2), ("end", False, "TimeoutError")]
    # Never recorded into the conversation event store.
    assert list(em.keys()) == []


def test_llmcall_end_defaults_to_success():
    e = LLMCallEnd(method_name="m", strategy="S", generation_id="g", turn_number=1)
    assert e.success is True
    assert e.exception_type is None


@pytest.mark.asyncio
async def test_generate_emits_start_then_end_around_acall():
    """A real generation turn fires LLMCallStart then LLMCallEnd via on().

    Scripts a valid code response so codegen succeeds on the first turn,
    making the start→end ordering deterministic.
    """
    code = """
result = x
return result
"""
    llm = FakeLLMClient.with_code_responses([code])

    class A(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def echo(self, x: int) -> int:
            """Return {x} unchanged."""
            ...

    agent = A()
    order: list[str] = []
    agent.event_manager.on("LLMCallStart", lambda e: order.append("start"))
    agent.event_manager.on("LLMCallEnd", lambda e: order.append(f"end:{e.success}"))

    result = await agent.echo(7)
    assert result == 7

    assert "start" in order, order
    first_end = next((o for o in order if o.startswith("end:")), None)
    assert first_end is not None, order
    assert order.index("start") < order.index(first_end)
    assert first_end == "end:True", order


@pytest.mark.asyncio
async def test_generate_emits_end_with_success_false_when_acall_raises():
    """If the LLM round-trip raises, LLMCallEnd still fires with success=False
    and the exception type, so observers can clear "waiting on model" state.

    PurePython wraps the raised error in a GenerationError after exhausting
    retries; each retry is its own generate() turn, so we expect one
    start/end pair per retry, every end reporting failure.
    """

    class Boom(RuntimeError):
        pass

    class _RaisingLLM(FakeLLMClient):
        async def acall(self, *a, **k):  # type: ignore[override]
            raise Boom("kaboom")

    class A(Agent, llm=_RaisingLLM()):
        @strategy(PurePythonStrategy())
        async def echo(self, x: int) -> int:
            """Return {x} unchanged."""
            ...

    agent = A()
    ends: list[tuple[object, str | None]] = []
    agent.event_manager.on("LLMCallStart", lambda e: ends.append(("start", None)))
    agent.event_manager.on("LLMCallEnd", lambda e: ends.append((e.success, e.exception_type)))

    with pytest.raises(GenerationError):
        await agent.echo(1)

    assert ("start", None) in ends, ends
    end_events = [e for e in ends if e != ("start", None)]
    assert end_events, ends
    # Every emitted end reports failure with the underlying exception type.
    assert all(success is False and exc == "Boom" for success, exc in end_events), ends
