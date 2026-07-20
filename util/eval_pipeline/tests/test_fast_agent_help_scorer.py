# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for deterministic fast_agent help/no-help scoring."""

from eval_pipeline.models import ScoringContext
from eval_pipeline.scoring import FastAgentHelpScorer, ModeSelectionScorer, _get_code
from nooa.trace_explorer import TraceExplorer
from nooa.trace_explorer.explorer import AgentSession, ExecutionTurn, LLMTurn


def _trace(*turns):
    session = AgentSession(
        session_id="test",
        agent_name="FastHelpAgent",
        method_name="reply",
        parent_session_id=None,
    )
    session.turns.extend(turns)
    return TraceExplorer(sessions=[session], trace_file="test://fast-agent-help")


def _execution(code: str, *, tool_call_id: str = "call_test") -> ExecutionTurn:
    return ExecutionTurn(
        code=code,
        stdout="",
        error=None,
        returned_value=None,
        status="OK",
        tool_call_id=tool_call_id,
    )


def _llm_response(text: str) -> LLMTurn:
    return LLMTurn(
        session_id="test",
        messages=[],
        response=text,
        model="test-model",
    )


def _ctx(trace, *, expected="NO_HELP", actual="The capital of France is Paris."):
    return ScoringContext(
        task_id="t1",
        input="What is the capital of France?",
        expected=expected,
        actual=actual,
        trace=trace,
    )


def test_direct_no_help_answer_passes_despite_prefill():
    trace = _trace(
        _execution("print('prefill/introspection')", tool_call_id="prefill_123"),
        _llm_response("The capital of France is Paris."),
    )

    # Direct response text is not code, and prefill is ignored.
    assert _get_code(trace, skip_prefill=True) is None

    mode_result = ModeSelectionScorer(expected="internal").score(_ctx(trace))
    assert mode_result.score == 1.0

    result = FastAgentHelpScorer(expect_help=False).score(_ctx(trace))
    assert result.score == 1.0
    assert result.metadata["called_help"] is False


def test_call_for_help_help_case_passes():
    trace = _trace(
        _execution("print('prefill/introspection')", tool_call_id="prefill_123"),
        _execution("result = self.call_for_help(user_message)\nreturn_result(result)"),
    )

    result = FastAgentHelpScorer(expect_help=True).score(
        _ctx(trace, expected="HELP_CALLED", actual="[HELP_CALLED: weather]")
    )
    assert result.score == 1.0
    assert result.metadata["called_help"] is True
    assert result.metadata["help_execution_count"] == 1


def test_missing_help_call_fails_help_case():
    trace = _trace(_llm_response("I think it is sunny in Paris."))

    result = FastAgentHelpScorer(expect_help=True).score(
        _ctx(trace, expected="HELP_CALLED", actual="I think it is sunny in Paris.")
    )
    assert result.score == 0.0
    assert result.metadata["called_help"] is False


def test_unnecessary_help_call_fails_no_help_case():
    trace = _trace(_execution("result = self.call_for_help(user_message)\nreturn_result(result)"))

    result = FastAgentHelpScorer(expect_help=False).score(
        _ctx(trace, expected="NO_HELP", actual="[HELP_CALLED: What is the capital of France?]")
    )
    assert result.score == 0.0
    assert result.metadata["called_help"] is True
