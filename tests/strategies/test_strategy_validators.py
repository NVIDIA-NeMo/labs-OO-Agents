# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Method-local strategy validators (ported from the !444 design).

Preconditions/postconditions live on the strategy *config* (method-local, not an
agent-global hook). A postcondition raises ``InvariantError`` to reject a
type-valid result — routed through the existing return_result validation-retry
channel; any other exception surfaces as an infrastructure error. Preconditions
run before generation and fail fast.
"""

from __future__ import annotations

import json

import pytest

from nooa import Agent, InvariantError, strategy
from nooa.config import CodeActConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategy_validation import normalize_conditions, run_postconditions, run_preconditions
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _return_result_call(result, call_id: str = "call_ret") -> ToolCall:
    return ToolCall(id=call_id, name="return_result", arguments=json.dumps({"result": result}))


def _resp(tool_calls: list) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
    )


_DUMMY_LLM = FakeLLMClient()

# --------------------------------------------------------------------------- #
# unit: strategy_validation helpers
# --------------------------------------------------------------------------- #


def test_normalize_conditions_accepts_callables():
    f = lambda *a: None  # noqa: E731
    assert normalize_conditions("postconditions", [f]) == (f,)
    assert normalize_conditions("postconditions", None) == ()


def test_normalize_conditions_rejects_non_callable():
    with pytest.raises(ValueError):
        normalize_conditions("postconditions", [123])
    with pytest.raises(ValueError):
        normalize_conditions("postconditions", "not-iterable-of-callables")


def test_invariant_error_is_valueerror():
    assert issubclass(InvariantError, ValueError)


def test_run_postconditions_reraises_invariant_wraps_others():
    def raises_invariant(agent, result, call):
        raise InvariantError("bad")

    with pytest.raises(InvariantError):
        run_postconditions(object(), 1, object(), [raises_invariant])

    def raises_other(agent, result, call):
        raise KeyError("boom")

    with pytest.raises(RuntimeError):  # non-InvariantError -> infra error
        run_postconditions(object(), 1, object(), [raises_other])


def test_run_preconditions_passes_agent_and_call():
    seen = {}

    def pre(agent, call):
        seen["agent"], seen["call"] = agent, call

    a, c = object(), object()
    run_preconditions(a, c, [pre])
    assert seen == {"agent": a, "call": c}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_config_normalizes_conditions():
    f = lambda *a: None  # noqa: E731
    cfg = CodeActConfig(preconditions=[f], postconditions=[f])
    assert cfg.preconditions == (f,) and cfg.postconditions == (f,)
    with pytest.raises(ValueError):
        CodeActConfig(postconditions=[1])


# --------------------------------------------------------------------------- #
# end-to-end through CodeActStrategy
# --------------------------------------------------------------------------- #


def _post_reject_bad(agent, result, call):
    if result == "bad":
        raise InvariantError("'bad' is not acceptable; return 'good'.")


class _ValidatedAgent(Agent, llm=_DUMMY_LLM):
    @strategy(CodeActStrategy(config=CodeActConfig(postconditions=[_post_reject_bad])))
    async def solve(self, question: str) -> str:
        """Answer {question}."""
        ...


@pytest.mark.asyncio
async def test_postcondition_rejects_then_accepts_on_retry():
    fake = FakeLLMClient(
        scripted_responses=[
            _resp([_return_result_call("bad", "c1")]),  # InvariantError -> retry
            _resp([_return_result_call("good", "c2")]),
        ]
    )
    agent = _ValidatedAgent(llm=fake)
    assert await agent.solve("q") == "good"
    assert fake.call_count == 2  # one retry for the rejected result


@pytest.mark.asyncio
async def test_no_violation_passes_first_try():
    fake = FakeLLMClient(scripted_responses=[_resp([_return_result_call("good")])])
    agent = _ValidatedAgent(llm=fake)
    assert await agent.solve("q") == "good"
    assert fake.call_count == 1


class _PlainAgent(Agent, llm=_DUMMY_LLM):
    @strategy(CodeActStrategy(config=CodeActConfig()))
    async def solve(self, question: str) -> str:
        """Answer {question}."""
        ...


@pytest.mark.asyncio
async def test_no_postconditions_configured_is_unaffected():
    fake = FakeLLMClient(scripted_responses=[_resp([_return_result_call("bad")])])
    agent = _PlainAgent(llm=fake)
    assert await agent.solve("q") == "bad"  # no postcondition -> no rejection
    assert fake.call_count == 1


def _post_buggy(agent, result, call):
    raise TypeError("buggy postcondition")


class _BuggyAgent(Agent, llm=_DUMMY_LLM):
    @strategy(CodeActStrategy(config=CodeActConfig(postconditions=[_post_buggy])))
    async def solve(self, question: str) -> str:
        """Answer {question}."""
        ...


@pytest.mark.asyncio
async def test_buggy_postcondition_surfaces_as_infra_error_not_retry():
    """A postcondition that raises a non-InvariantError must not be swallowed as
    model-correctable feedback (it's an infra/programming error)."""
    fake = FakeLLMClient(scripted_responses=[_resp([_return_result_call("anything")])])
    agent = _BuggyAgent(llm=fake)
    with pytest.raises(RuntimeError, match="postcondition"):
        await agent.solve("q")
    assert fake.call_count == 1  # no spurious retry loop
