# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic agent-call lifecycle events (``BeforeAgentCall`` / ``AfterAgentCall``).

These events are the pub-sub complement to the ``agent_call`` middleware and
are symmetric with ``BeforeTurn`` / ``AfterTurn`` for generation turns. They
fire from the agent method wrapper for EVERY agent-level method call —
generation methods, pure-Python orchestrators, and sync helpers — carrying
``is_top_level`` so subscribers can filter to the outermost run. ATIF uses them
to arm/disarm its standalone cascade; this module pins the generic
mechanism independently of ATIF.
"""

from __future__ import annotations

import pytest

from nooa import Agent, strategy
from nooa.context_blocks.roles import Role
from nooa.events import AfterAgentCall, BeforeAgentCall
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse

_DEFAULT = FakeLLMClient()


def _predict_resp(content: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
        usage={"prompt_tokens": 5, "completion_tokens": 1},
    )


class _Agent(Agent, llm=_DEFAULT):
    async def run(self, x: int) -> str:
        # Pure-Python orchestrator (real body, no generation turn).
        doubled = self._double(x)  # sync helper
        return await self.describe(doubled)  # generation method

    def _double(self, x: int) -> int:
        return x * 2

    @strategy(PredictStrategy())
    async def describe(self, value: int) -> str:
        """Describe {value} in one word."""
        ...

    async def boom(self) -> str:
        raise RuntimeError("kaboom")


def _capture(agent: Agent) -> list:
    events: list = []
    agent.event_manager.on("BeforeAgentCall", events.append)
    agent.event_manager.on("AfterAgentCall", events.append)
    return events


@pytest.mark.asyncio
async def test_events_fire_for_pure_python_generation_and_sync_methods() -> None:
    fake = FakeLLMClient(scripted_responses=[_predict_resp('{"value": "four"}')])
    agent = _Agent(llm=fake)
    events = _capture(agent)

    assert await agent.run(2) == "four"

    befores = [e for e in events if isinstance(e, BeforeAgentCall)]
    afters = [e for e in events if isinstance(e, AfterAgentCall)]
    names_before = [e.method_name for e in befores]

    # All three agent-level methods produced a Before AND an After.
    assert names_before == ["run", "_double", "describe"], names_before
    assert [e.method_name for e in afters] == ["_double", "describe", "run"], (
        "After events should complete in LIFO order (inner methods finish first)"
    )

    by_name = {e.method_name: e for e in befores}
    # Only the outermost call is top-level; nested calls (sync + generation)
    # are not.
    assert by_name["run"].is_top_level is True
    assert by_name["_double"].is_top_level is False
    assert by_name["describe"].is_top_level is False

    # needs_generation reflects ellipsis (LLM) vs pure-Python/sync.
    assert by_name["run"].needs_generation is False
    assert by_name["_double"].needs_generation is False
    assert by_name["describe"].needs_generation is True

    # All successful.
    assert all(e.success and e.exception_type is None for e in afters)


@pytest.mark.asyncio
async def test_after_event_carries_exception_on_failure() -> None:
    agent = _Agent(llm=_DEFAULT)
    events = _capture(agent)

    with pytest.raises(RuntimeError, match="kaboom"):
        await agent.boom()

    after = next(e for e in events if isinstance(e, AfterAgentCall) and e.method_name == "boom")
    assert after.success is False
    assert after.exception_type == "RuntimeError"
    assert after.is_top_level is True


@pytest.mark.asyncio
async def test_agent_call_events_are_runtime_events_and_not_recorded() -> None:
    """RUNTIME_EVENT ⇒ emitted to subscribers but never stored in the backend
    (so they don't bloat the event store or the LLM context)."""
    assert BeforeAgentCall._role is Role.RUNTIME_EVENT
    assert AfterAgentCall._role is Role.RUNTIME_EVENT

    fake = FakeLLMClient(scripted_responses=[_predict_resp('{"value": "four"}')])
    agent = _Agent(llm=fake)
    await agent.run(2)

    stored_types = {type(e).__name__ for _, e in agent.event_manager.items()}
    assert "BeforeAgentCall" not in stored_types
    assert "AfterAgentCall" not in stored_types
