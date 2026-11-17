# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``call.id`` names one generation-method call; tracing does not change it."""

import json

from nooa import Agent, no_trace, strategy
from nooa.context_blocks.events import UserEvent
from nooa.default_context_view import PredictView
from nooa.events import Task
from nooa.llm_types import LLMResponse, ToolCall
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import FakeLLMClient


class SpyView(PredictView):
    def __init__(self):
        self.calls = []

    async def assemble(self, owner, call):
        self.calls.append(call)
        async for item in super().assemble(owner, call):
            yield item


def answers(count: int) -> FakeLLMClient:
    return FakeLLMClient([LLMResponse(content=str(i)) for i in range(count)])


def task_call_ids(agent: Agent) -> list[str | None]:
    return [e.call_id for e in agent.event_manager.values() if isinstance(e, Task)]


async def test_no_trace_generation_calls_get_their_own_ids():
    view = SpyView()

    class Example(Agent, llm=answers(3)):
        @no_trace
        @strategy(PredictStrategy(), context_view=view)
        async def step(self, x: int) -> int:
            """Return a number."""
            ...

        async def parent(self) -> None:
            await self.step(1)
            await self.step(2)

    agent = Example()
    await agent.parent()
    await agent.step(3)

    ids = [call.id for call in view.calls]
    assert len(set(ids)) == 3
    assert task_call_ids(agent) == ids


def _cell(code: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))
        ],
        finish_reason="tool_calls",
    )


async def test_helper_events_belong_to_the_generation_call():
    llm = FakeLLMClient([_cell("self.note('from helper')", "c1"), _cell("return_result(1)", "c2")])

    class Example(Agent, llm=llm):
        def note(self, text: str) -> None:
            """Record a note."""
            self.event_manager.add(UserEvent(content=text))

        async def solve(self) -> int:
            """Return a number."""
            ...

    agent = Example()
    await agent.solve()

    (solve_id,) = task_call_ids(agent)
    note = next(e for e in agent.event_manager.values() if isinstance(e, UserEvent))
    assert note.call_id == solve_id


async def test_parent_id_is_the_enclosing_generation_call():
    view = SpyView()
    llm = FakeLLMClient(
        [
            _cell("return_result(await self.inner())", "c1"),
            LLMResponse(content="7"),
            LLMResponse(content="8"),
        ]
    )

    class Example(Agent, llm=llm):
        @strategy(PredictStrategy(), context_view=view)
        async def inner(self) -> int:
            """Return a number."""
            ...

        async def solve(self) -> int:
            """Return a number."""
            ...

        async def plain(self) -> int:
            return await self.inner()

    agent = Example()
    assert await agent.solve() == 7
    assert await agent.plain() == 8

    solve_id = next(
        e.call_id
        for e in agent.event_manager.values()
        if isinstance(e, Task) and e.call_id != view.calls[0].id
    )
    assert view.calls[0].parent_id == solve_id
    assert view.calls[1].parent_id is None
