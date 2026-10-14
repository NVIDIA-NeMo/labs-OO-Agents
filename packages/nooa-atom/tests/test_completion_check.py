# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The completion check: one tool-free judge call before a turn's Done is accepted."""

import json

import pytest
from nooa_atom.agent import AtomAgent
from nooa_atom.agent.completion import (
    COMPLETION_CHECK_ENV,
    FEEDBACK_SOURCE,
    CompletionCheck,
    CompletionCheckConfig,
    JudgedCodeActV2,
    parse_verdict,
)

from nooa.events import Notification
from nooa.interactive import Done, NeedInput, Waiting
from nooa.llm_types import LLMResponse
from nooa.unifiedllm import FakeLLMClient, ToolCall

_ids = iter(range(10_000))

RESULT = (
    "TaskResult(solution_description='patched', evidence='pytest: 3 passed', "
    "how_to_verify='pytest', report='patched the parser')"
)
DONE = f"return_result(Done(explanation='finished', result={RESULT}))"


def python_cell(code: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=f"call_{next(_ids)}", name="python_cell", arguments=json.dumps({"code": code})
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
    )


def verdict(text: str) -> LLMResponse:
    return LLMResponse(raw_response=None, content=text, tool_calls=[], finish_reason="stop")


class FailingJudgeLLM(FakeLLMClient):
    """Scripted turn cells; any call without tools (the judge) raises."""

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        if not tools:
            raise RuntimeError("judge model is down")
        return await super().acall(messages, tools, output_model, **kwargs)


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(COMPLETION_CHECK_ENV, raising=False)


async def run(method, responses, tmp_path, *, llm=None, **agent_kwargs):
    llm = llm or FakeLLMClient(responses, strict_exhaustion=True)
    agent = AtomAgent(llm=llm, cwd=tmp_path, **agent_kwargs)
    try:
        outcome = await getattr(agent, method)({"user_messages": ["fix the parser bug"]})
        checks = [e for e in agent.event_manager.values() if isinstance(e, CompletionCheck)]
        notes = [
            e
            for e in agent.event_manager.values()
            if isinstance(e, Notification) and e.source == FEEDBACK_SOURCE
        ]
        return outcome, llm, checks, notes
    finally:
        await agent.aclose()


def judge_calls(llm):
    return [call for call in llm.calls if not call.tools]


# ---- verdicts -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("STOP: all done", ("stop", "all done", True)),
        ("  continue: run the tests", ("continue", "run the tests", True)),
        ("CONTINUE:", ("stop", "", False)),
        ("Looks complete to me. STOP: yes", ("stop", "", False)),
        ("", ("stop", "", False)),
    ],
)
def test_only_the_continue_and_stop_prefixes_count(text, expected):
    assert parse_verdict(text) == expected


async def test_stop_accepts_the_done(tmp_path):
    outcome, llm, checks, notes = await run(
        "handle_batch", [python_cell(DONE), verdict("STOP: the fix is tested")], tmp_path
    )
    assert isinstance(outcome, Done) and outcome.explanation == "finished"
    assert len(llm.calls) == 2 and len(judge_calls(llm)) == 1
    assert notes == []
    [check] = checks
    assert (check.method_name, check.decision, check.reason, check.attempt) == (
        "handle_batch",
        "stop",
        "judge",
        1,
    )
    assert check.feedback == "the fix is tested"


async def test_continue_injects_feedback_and_the_same_turn_keeps_going(tmp_path):
    responses = [
        python_cell(f"x = 41\n{DONE}"),
        verdict("CONTINUE: you never ran the parser tests"),
        python_cell(f"print('x is', x + 1)\n{DONE}"),
        verdict("STOP: tests ran"),
    ]
    outcome, llm, checks, notes = await run("handle_batch", responses, tmp_path)
    assert isinstance(outcome, Done)
    assert len(llm.calls) == 4
    # The feedback reaches the model in the same turn, as a user-visible message.
    after_feedback = str(llm.calls[2].messages)
    assert "Completion check: you never ran the parser tests" in after_feedback
    [note] = notes
    assert note.description.startswith("Completion check: you never ran the parser tests")
    assert [(c.decision, c.attempt, c.feedback) for c in checks] == [
        ("continue", 1, "you never ran the parser tests"),
        ("stop", 2, "tests ran"),
    ]
    # Same CodeAct session: the first cell's local is still there.
    assert "x is 42" in str(llm.calls[3].messages)


async def test_the_cap_of_three_continues_then_accepts_unjudged(tmp_path):
    responses = []
    for n in range(3):
        responses += [python_cell(DONE), verdict(f"CONTINUE: still missing {n}")]
    responses.append(python_cell(DONE))  # the fourth Done is not judged
    outcome, llm, checks, notes = await run("handle_batch", responses, tmp_path)
    assert isinstance(outcome, Done)
    assert len(llm.calls) == 7 and len(judge_calls(llm)) == 3
    assert len(notes) == 3
    assert [(c.decision, c.reason, c.attempt) for c in checks] == [
        ("continue", "judge", 1),
        ("continue", "judge", 2),
        ("continue", "judge", 3),
        ("stop", "cap", 4),
    ]


async def test_a_configured_cap_is_respected(tmp_path):
    config = CompletionCheckConfig(max_continues=1)
    responses = [python_cell(DONE), verdict("CONTINUE: more"), python_cell(DONE)]
    outcome, llm, checks, _ = await run(
        "handle_batch", responses, tmp_path, completion_check=config
    )
    assert isinstance(outcome, Done) and len(llm.calls) == 3
    assert [c.reason for c in checks] == ["judge", "cap"]


async def test_unclear_output_accepts_the_done(tmp_path):
    outcome, llm, checks, notes = await run(
        "handle_batch", [python_cell(DONE), verdict("I think it is fine.")], tmp_path
    )
    assert isinstance(outcome, Done) and len(llm.calls) == 2 and notes == []
    [check] = checks
    assert (check.decision, check.reason, check.raw) == ("stop", "unclear", "I think it is fine.")


async def test_a_judge_exception_accepts_the_done(tmp_path):
    llm = FailingJudgeLLM([python_cell(DONE)], strict_exhaustion=True)
    outcome, _, checks, notes = await run("handle_batch", [], tmp_path, llm=llm)
    assert isinstance(outcome, Done) and notes == []
    [check] = checks
    assert (check.decision, check.reason) == ("stop", "error")
    assert "judge model is down" in check.raw


# ---- what is judged -------------------------------------------------------


async def test_need_input_and_waiting_are_not_judged(tmp_path):
    outcome, llm, checks, _ = await run(
        "handle",
        [python_cell("return_result(NeedInput(question='Which branch?'))")],
        tmp_path,
        completion_check="all",
    )
    assert isinstance(outcome, NeedInput) and len(llm.calls) == 1 and checks == []
    outcome, llm, checks, _ = await run(
        "handle_batch",
        [python_cell("return_result(Waiting(explanation='tests running', on=['jobs']))")],
        tmp_path,
    )
    assert isinstance(outcome, Waiting) and len(llm.calls) == 1 and checks == []


async def test_a_done_failing_validation_is_not_judged(tmp_path):
    """require_result runs first: only a Done the turn would really end with is judged."""
    responses = [
        python_cell("return_result(Done(explanation='finished'))"),
        python_cell(DONE),
        verdict("STOP: ok"),
    ]
    outcome, llm, checks, _ = await run("handle_batch", responses, tmp_path)
    assert isinstance(outcome, Done) and len(llm.calls) == 3
    assert [c.attempt for c in checks] == [1]


# ---- configuration --------------------------------------------------------


async def test_default_checks_batch_turns_but_not_interactive_ones(tmp_path):
    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        assert agent.completion_check.applies("handle_batch")
        assert not agent.completion_check.applies("handle")
    finally:
        await agent.aclose()
    outcome, llm, checks, _ = await run(
        "handle", [python_cell("return_result(Done(explanation='hi'))")], tmp_path
    )
    assert isinstance(outcome, Done) and len(llm.calls) == 1 and checks == []


async def test_the_switch_turns_the_check_off_and_on(tmp_path, monkeypatch):
    outcome, llm, checks, _ = await run(
        "handle_batch", [python_cell(DONE)], tmp_path, completion_check="off"
    )
    assert isinstance(outcome, Done) and len(llm.calls) == 1 and checks == []

    outcome, llm, checks, _ = await run(
        "handle",
        [python_cell("return_result(Done(explanation='hi'))"), verdict("STOP: answered")],
        tmp_path,
        completion_check=CompletionCheckConfig.from_mode("interactive"),
    )
    assert len(llm.calls) == 2 and [c.method_name for c in checks] == ["handle"]

    monkeypatch.setenv(COMPLETION_CHECK_ENV, "off")
    outcome, llm, checks, _ = await run("handle_batch", [python_cell(DONE)], tmp_path)
    assert len(llm.calls) == 1 and checks == []

    monkeypatch.setenv(COMPLETION_CHECK_ENV, "all")
    outcome, llm, checks, _ = await run(
        "handle",
        [python_cell("return_result(Done(explanation='hi'))"), verdict("STOP: ok")],
        tmp_path,
    )
    assert len(llm.calls) == 2 and len(checks) == 1


def test_modes_are_validated(monkeypatch):
    with pytest.raises(ValueError, match="completion check mode"):
        CompletionCheckConfig.from_mode("sometimes")
    monkeypatch.setenv(COMPLETION_CHECK_ENV, "sometimes")
    assert CompletionCheckConfig.from_env() == CompletionCheckConfig()


@pytest.mark.parametrize("method", ["handle", "handle_batch"])
def test_both_turn_methods_of_both_agents_use_the_judged_strategy(method):
    from nooa_atom.agent import ExperimentalAtomAgent

    for cls in (AtomAgent, ExperimentalAtomAgent):
        assert isinstance(getattr(cls, method)._plan_strategy, JudgedCodeActV2)


# ---- what the judge sees ---------------------------------------------------


async def test_the_judge_gets_a_bounded_tool_free_view(tmp_path):
    noisy = "print('N' * 50_000)"
    responses = [python_cell(noisy)] + [python_cell(f"print('step {n}')") for n in range(15)]
    responses += [python_cell(DONE), verdict("STOP: ok")]
    _, llm, _, _ = await run("handle_batch", responses, tmp_path)
    [judge] = judge_calls(llm)
    assert not judge.tools
    assert [m["role"] for m in judge.messages] == ["system", "user"]
    system, user = (m["content"] for m in judge.messages)
    assert "CONTINUE:" in system and "STOP:" in system
    assert "fix the parser bug" in user  # the original request
    assert "finished" in user and "pytest: 3 passed" in user  # the Done and its TaskResult
    assert "print('step 14')" in user and "step 14" in user  # recent cells and outputs
    assert "earlier log entries not shown" in user  # only the tail
    assert "N" * 5_000 not in user  # outputs are clipped
    assert len(user) < 40_000
    # Nothing the agent was not shown: no system prompt or context blocks of the turn.
    assert "python_cell_tools" not in user and "repository_instructions" not in user


async def test_the_decision_is_stored_in_the_session_trace(tmp_path, sessions_dir):
    """A batch session (as delegated children and headless runs use) persists the decision."""
    import asyncio

    from nooa_atom.agent.factory import create_session_agent
    from nooa_atom.session.options import SessionOptions
    from nooa_atom.session.registry import SessionRegistry
    from nooa_atom.session.store import SessionStore

    llm = FakeLLMClient(
        [
            python_cell(DONE),
            verdict("CONTINUE: run the tests"),
            python_cell(DONE),
            verdict("STOP: ok"),
        ],
        strict_exhaustion=True,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        root = await registry.create(
            SessionOptions(
                workspace=workspace,
                agent_spec="nooa_atom.agent:AtomAgent",
                llm=llm,
                turn_method="handle_batch",
                sessions_dir=sessions_dir,
            )
        )
        outcome = await asyncio.wait_for(root.prompt("fix the parser bug"), 30)
        assert isinstance(outcome, Done)
        session_id = root.id
    finally:
        await registry.close_all()

    handle = SessionStore(sessions_dir).open(session_id)
    try:
        stored = [e for e in handle.events.all_events() if isinstance(e, CompletionCheck)]
    finally:
        handle.close()
    assert [(c.decision, c.attempt, c.feedback) for c in stored] == [
        ("continue", 1, "run the tests"),
        ("stop", 2, "ok"),
    ]
