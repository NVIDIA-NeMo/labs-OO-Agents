# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CodeAct loop guard: warn once on an identical repeat, stop if it continues."""

import json

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig, LoopGuardConfig
from nooa.context_blocks import ToolCallEvent
from nooa.errors import GenerationError, LoopDetectedError
from nooa.events import LoopGuardTriggered
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.strategies.loop_guard import action_fingerprint, normalize_code
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _call(name: str, arguments: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        parts=(ToolCall(id=call_id, name=name, arguments=arguments),),
        finish_reason="tool_calls",
    )


def _cell(code: str, call_id: str, tool: str = "python_cell") -> LLMResponse:
    return _call(tool, json.dumps({"code": code}), call_id)


_SHORT_GUARD = LoopGuardConfig(repeat_threshold=3, window=8)


def _agent(responses, *, guard=_SHORT_GUARD, strategy_type=CodeActV2):
    llm = FakeLLMClient(scripted_responses=responses)
    config = CodeActConfig(prefill=None, max_retries=20, loop_guard=guard)

    class LoopAgent(Agent, llm=llm):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.runs = 0
            self.ticks = 0

        def fail(self) -> None:
            """Fail every time."""
            self.runs += 1
            raise ValueError("mirror unavailable")

        def probe(self) -> str:
            """Return the same answer every time."""
            self.runs += 1
            return "404 Not Found"

        def tick(self) -> int:
            """Return a new value on each call."""
            self.ticks += 1
            return self.ticks

        @strategy(strategy_type(config=config))
        async def answer(self) -> str:
            """Return an answer."""
            ...

    return LoopAgent()


def _guard_events(agent) -> list[LoopGuardTriggered]:
    return [e for e in agent.event_manager.values() if isinstance(e, LoopGuardTriggered)]


@pytest.mark.parametrize(
    "strategy_type,tool", [(CodeActV2, "python_cell"), (CodeActStrategy, "execute_python")]
)
async def test_identical_failing_call_is_blocked_then_model_recovers(strategy_type, tool):
    agent = _agent(
        [
            _cell("self.fail()", "1", tool),
            _cell("self.fail()  # retry", "2", tool),
            _cell("self.fail()", "3", tool),
            _cell("return_result('recovered')", "4", tool),
        ],
        strategy_type=strategy_type,
    )
    try:
        assert await agent.answer() == "recovered"
        assert agent.runs == 2  # the third identical call was not executed
        (event,) = _guard_events(agent)
        assert event.action == "blocked"
        assert event.repeats == 3
        assert event.window == 8
        assert f"this {tool} call was not executed" in event.content
        assert "mirror unavailable" in event.content
        blocked = next(
            e
            for e in agent.event_manager.values()
            if isinstance(e, ToolCallEvent) and e.tool_call_id == "3"
        )
        assert blocked.result.content == "Not executed: blocked by the loop guard."
    finally:
        await agent.aclose()


async def test_repeating_a_blocked_call_stops_generation():
    agent = _agent([_cell("self.fail()", str(i)) for i in range(5)])
    try:
        with pytest.raises(LoopDetectedError, match="Loop guard stopped generation"):
            await agent.answer()
        assert agent.runs == 2
        assert [e.action for e in _guard_events(agent)] == ["blocked", "stopped"]
        assert issubclass(LoopDetectedError, GenerationError)
    finally:
        await agent.aclose()


async def test_identical_successful_output_warns_on_third_and_stops_on_fourth():
    agent = _agent([_cell("print(self.probe())", str(i)) for i in range(6)])
    try:
        with pytest.raises(LoopDetectedError):
            await agent.answer()
        assert agent.runs == 4  # successful repeats always run
        nudge, stop = _guard_events(agent)
        assert (nudge.action, nudge.repeats) == ("nudged", 3)
        assert "identical output" in nudge.content
        assert "wait inside the cell" in nudge.content
        assert (stop.action, stop.repeats) == ("stopped", 4)
    finally:
        await agent.aclose()


async def test_changing_output_is_not_a_loop():
    agent = _agent(
        [_cell("print(self.tick())", str(i)) for i in range(6)]
        + [_cell("return_result('done')", "end")]
    )
    try:
        assert await agent.answer() == "done"
        assert agent.ticks == 6
        assert _guard_events(agent) == []
    finally:
        await agent.aclose()


async def test_different_call_after_warning_continues():
    agent = _agent(
        [_cell("print(self.probe())", str(i)) for i in range(3)]
        + [_cell("print('something else')", "x"), _cell("return_result('ok')", "end")]
    )
    try:
        assert await agent.answer() == "ok"
        assert [e.action for e in _guard_events(agent)] == ["nudged"]
    finally:
        await agent.aclose()


async def test_repeated_invalid_tool_name_is_blocked_then_stopped():
    agent = _agent([_call("todo", '{"title": "x"}', str(i)) for i in range(5)])
    try:
        with pytest.raises(LoopDetectedError):
            await agent.answer()
        blocked, stopped = _guard_events(agent)
        assert blocked.action == "blocked"
        assert "Unknown tool `todo`" in blocked.content
        assert stopped.action == "stopped"
    finally:
        await agent.aclose()


async def test_repeated_invalid_json_arguments_are_blocked():
    agent = _agent(
        [_call("python_cell", "print(1)", str(i)) for i in range(3)]
        + [_cell("return_result('ok')", "end")]
    )
    try:
        assert await agent.answer() == "ok"
        (event,) = _guard_events(agent)
        assert event.action == "blocked"
        assert "Invalid arguments for tool" in event.content
    finally:
        await agent.aclose()


async def test_failing_completion_gets_return_variable_hint():
    bad = "return_result(int('x'))"
    agent = _agent([_cell(bad, str(i)) for i in range(3)] + [_cell("return_result('ok')", "e")])
    try:
        assert await agent.answer() == "ok"
        (event,) = _guard_events(agent)
        assert "build the return value in a variable first" in event.content
    finally:
        await agent.aclose()


async def test_repeats_outside_the_window_are_forgotten():
    other = [_cell(f"print({i})", f"o{i}") for i in range(4)]
    agent = _agent(
        [_cell("self.fail()", "a"), _cell("self.fail()", "b"), *other, _cell("self.fail()", "c")]
        + [_cell("return_result('ok')", "end")],
        guard=LoopGuardConfig(repeat_threshold=3, window=4),
    )
    try:
        assert await agent.answer() == "ok"
        assert agent.runs == 3
        assert _guard_events(agent) == []
    finally:
        await agent.aclose()


async def test_disabled_by_default():
    agent = _agent(
        [_cell("self.fail()", str(i)) for i in range(5)] + [_cell("return_result('ok')", "end")],
        guard=None,
    )
    try:
        assert await agent.answer() == "ok"
        assert agent.runs == 5
        assert _guard_events(agent) == []
        assert CodeActConfig().loop_guard is None
    finally:
        await agent.aclose()


def test_fingerprint_ignores_comments_and_formatting():
    a = action_fingerprint("python_cell", {"code": "x = f( 1 )  # try"}, "python_cell")
    b = action_fingerprint("python_cell", {"code": "\nx = f(1)\n"}, "python_cell")
    c = action_fingerprint("python_cell", {"code": "x = f(2)"}, "python_cell")
    assert a == b != c
    assert normalize_code("if x:\n  y(") == "if x:\ny("


def test_fingerprint_distinguishes_tool_names_and_arguments():
    base = action_fingerprint("todo", {"title": "x"}, "python_cell")
    assert base == action_fingerprint("todo", {"title": "x"}, "python_cell")
    assert base != action_fingerprint("todo", {"title": "y"}, "python_cell")
    assert base != action_fingerprint("notes", {"title": "x"}, "python_cell")
    assert action_fingerprint("python_cell", "not json", "python_cell")


def test_loop_guard_event_renders_only_the_message():
    event = LoopGuardTriggered(content="warning text", action="nudged", fingerprint="abc")
    assert "warning text" in repr(event)
    assert "abc" not in repr(event)


def test_missing_outcome_never_matches_another_call():
    """If a call's events were summarized away, its outcome is unique, not empty."""
    from types import SimpleNamespace

    from nooa.runtime.event_manager import EventManager

    runtime = SimpleNamespace(event_manager=EventManager())
    first = CodeActStrategy._tool_call_outcome(runtime, "a", 0)
    second = CodeActStrategy._tool_call_outcome(runtime, "b", 0)
    assert first[0] is False and second[0] is False
    assert first[1] != second[1]


@pytest.mark.parametrize(
    "strategy_type,tool", [(CodeActV2, "python_cell"), (CodeActStrategy, "execute_python")]
)
async def test_default_guard_warns_on_seventh_success_and_stops_on_eighth(strategy_type, tool):
    guard = LoopGuardConfig()
    assert (guard.repeat_threshold, guard.window) == (7, 15)
    agent = _agent(
        [_cell("print(self.probe())", str(i), tool) for i in range(10)],
        guard=guard,
        strategy_type=strategy_type,
    )
    try:
        with pytest.raises(LoopDetectedError):
            await agent.answer()
        assert agent.runs == 8
        assert [(e.action, e.repeats, e.window) for e in _guard_events(agent)] == [
            ("nudged", 7, 15),
            ("stopped", 8, 15),
        ]
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "strategy_type,tool", [(CodeActV2, "python_cell"), (CodeActStrategy, "execute_python")]
)
async def test_default_guard_blocks_seventh_failure_and_stops_on_eighth(strategy_type, tool):
    agent = _agent(
        [_cell("self.fail()", str(i), tool) for i in range(10)],
        guard=LoopGuardConfig(),
        strategy_type=strategy_type,
    )
    try:
        with pytest.raises(LoopDetectedError):
            await agent.answer()
        assert agent.runs == 6
        assert [(e.action, e.repeats, e.window) for e in _guard_events(agent)] == [
            ("blocked", 7, 15),
            ("stopped", 8, 15),
        ]
    finally:
        await agent.aclose()


async def test_default_guard_tolerates_cleanup_between_four_changing_trials():
    calls = []
    for i in range(4):
        calls += [
            _cell("print(self.probe())", f"cleanup-{i}"),
            _cell("print(self.tick())", f"trial-{i}"),
        ]
    agent = _agent(calls + [_cell("return_result('done')", "end")], guard=LoopGuardConfig())
    try:
        assert await agent.answer() == "done"
        assert agent.runs == agent.ticks == 4
        assert _guard_events(agent) == []
    finally:
        await agent.aclose()


async def test_default_guard_counts_interleaved_identical_successes_within_window():
    calls = []
    for i in range(8):
        calls += [
            _cell("print(self.probe())", f"probe-{i}"),
            _cell("print(self.tick())", f"tick-{i}"),
        ]
    agent = _agent(calls, guard=LoopGuardConfig())
    try:
        with pytest.raises(LoopDetectedError):
            await agent.answer()
        assert agent.runs == 8
        assert agent.ticks == 7
        assert [(e.action, e.repeats, e.window) for e in _guard_events(agent)] == [
            ("nudged", 7, 15),
            ("stopped", 8, 15),
        ]
    finally:
        await agent.aclose()


async def test_default_guard_ignores_identical_successes_outside_fifteen_call_window():
    calls = [_cell("print(self.probe())", f"old-{i}") for i in range(6)]
    calls += [_cell("print(self.tick())", f"progress-{i}") for i in range(15)]
    calls += [_cell("print(self.probe())", "new"), _cell("return_result('done')", "end")]
    agent = _agent(calls, guard=LoopGuardConfig())
    try:
        assert await agent.answer() == "done"
        assert agent.runs == 7
        assert _guard_events(agent) == []
    finally:
        await agent.aclose()
