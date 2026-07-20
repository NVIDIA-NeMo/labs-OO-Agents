# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical-fix tests: WAIT-guard, helper dedent, background-reflect wiring, LLM timeout.

WAIT-guard defect: an agent can end its turn with ``return_result(kind=WAIT)``
while nothing was submitted this turn (13 occurrences in the 20260716 fleet —
harness and agent then idled the full 900s nudge timer). A WAIT that is not
backed by an actions.jsonl entry for the current turn must be rejected through
the standard result-validation channel so the session continues and the agent
submits for real.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import solver_agent as sa  # noqa: E402

STATE = {"turn": 0, "state": "NOT_FINISHED", "available_actions": ["UP", "DOWN"]}


def _tool_response(name: str, args: dict):
    from nooa.unifiedllm.unifiedllm import LLMResponse, ToolCall

    args_json = json.dumps(args)
    call_id = f"call_{abs(hash(args_json)) & 0xFFFF}"
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args_json)],
        finish_reason="tool_calls",
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_json},
                }
            ],
        },
        reasoning=None,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _cell(code: str):
    return _tool_response("execute_python", {"code": code})


def _build_agent(tmp_path, scripted_responses):
    from nooa.unifiedllm import FakeLLMClient

    llm = FakeLLMClient(scripted_responses=scripted_responses)
    agent = sa.MdArcSolverAgent(
        llm=llm,
        run_dir=tmp_path,
        game_id="g1",
        alias="g1",
        skill_path=None,
    )
    state = dict(STATE, grid_rows=["0" * 64] * 64)
    agent._states_path.write_text(json.dumps(state) + "\n")
    return agent, llm, state


_SUBMIT_CELL = "self.submit_actions(['UP', 'DOWN', 'UP'], 'recover: submit for real')"


# --------------------------------------------------------------------------- #
# WAIT-guard
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_wait_via_direct_tool_call_without_submission_is_rejected(tmp_path):
    """A bare return_result(WAIT) tool call with no submission must bounce."""
    responses = [
        _tool_response(
            "return_result",
            {"result": {"kind": "WAIT", "explanation": "submitted actions; waiting"}},
        ),
        _cell(_SUBMIT_CELL),
    ]
    agent, llm, state = _build_agent(tmp_path, responses)

    await agent.handle({"game_states": [json.dumps(state)]})

    entries = [json.loads(x) for x in agent._actions_path.read_text().splitlines() if x]
    assert len(entries) == 1, "the WAIT lie was accepted — no submission ever reached the harness"
    assert entries[0]["actions"] == ["UP", "DOWN", "UP"]
    assert llm.call_count == 2  # bounce + real submit, all within the same turn


@pytest.mark.asyncio
async def test_wait_via_inline_return_result_without_submission_is_rejected(tmp_path):
    """return_result(kind='WAIT') inside a cell without a submission must bounce."""
    responses = [
        _cell("return_result(kind='WAIT', explanation='submitted actions; waiting')"),
        _cell(_SUBMIT_CELL),
    ]
    agent, llm, state = _build_agent(tmp_path, responses)

    await agent.handle({"game_states": [json.dumps(state)]})

    entries = [json.loads(x) for x in agent._actions_path.read_text().splitlines() if x]
    assert len(entries) == 1, "the inline WAIT lie was accepted — nothing was submitted"
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_legitimate_wait_from_submit_actions_still_passes(tmp_path):
    """The guard must NOT reject the WAIT produced by a real submission."""
    agent, llm, state = _build_agent(tmp_path, [_cell(_SUBMIT_CELL)])

    result = await agent.handle({"game_states": [json.dumps(state)]})

    entries = [json.loads(x) for x in agent._actions_path.read_text().splitlines() if x]
    assert len(entries) == 1
    assert llm.call_count == 1  # no bounce for a backed WAIT
    kind = getattr(result, "kind", None) or (
        result.get("kind") if isinstance(result, dict) else None
    )
    assert str(kind).lower().endswith("wait")


# --------------------------------------------------------------------------- #
# helper-source validation must dedent before parsing
# --------------------------------------------------------------------------- #


def test_check_helper_ast_accepts_indented_source():
    source = """
        def predict(z, action):
            return z
    """
    assert sa._check_helper_ast(source) is None, "valid-but-indented helper source was rejected"


def test_check_helper_ast_still_rejects_real_syntax_errors():
    err = sa._check_helper_ast("def broken(:\n    pass")
    assert err is not None and "syntax" in err.lower()


def test_check_helper_ast_still_rejects_banned_imports_after_dedent():
    source = textwrap.indent("import os\n", "    ")
    err = sa._check_helper_ast(source)
    assert err is not None and "not allowed" in err


# --------------------------------------------------------------------------- #
# launcher wiring: memory consolidation must run off the critical path
# --------------------------------------------------------------------------- #


def test_launcher_memory_config_uses_background_reflection():
    import launcher

    build = getattr(launcher, "build_memory_config", None)
    assert build is not None, "launcher must expose build_memory_config() for testable wiring"
    cfg = build(store_path="/tmp/x.sqlite")
    assert cfg.reflection.background is True, (
        "manual reflect() consolidation must not run serial LLM calls on the "
        "turn's critical path (600-1500s/game in the 20260716 fleet)"
    )
    assert cfg.reflection.trigger == "manual"


# --------------------------------------------------------------------------- #
# LLM client hard timeout (pin — silent >180s provider stalls burned ~12 min)
# --------------------------------------------------------------------------- #


def test_build_llm_sets_hard_request_timeout(monkeypatch):
    import arc_llm

    monkeypatch.setenv("ARC_LLM_MODEL", "test-model")
    monkeypatch.setenv("ARC_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("ARC_LLM_API_KEY", "test-key")
    monkeypatch.setenv("ARC_LLM_USE_RESPONSES", "0")
    client = arc_llm.build_llm()
    assert client.config.get("timeout") == 180.0
