# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-cap action batches: ``max_actions_per_turn: 0`` disables BOTH enforcement layers.

The per-turn cap is enforced twice — agent-side (``submit_actions`` returns a
``REJECTED`` string) and harness-side (``apply_action_cap`` truncation backstop).
The ``*_nocap`` configs set ``max_actions_per_turn: 0``, which run_multi forwards
to run_solver, which forwards it to BOTH the launcher (agent ctor) and the
harness. These tests pin each layer, and the fake-LLM smoke drives the real
CodeAct pipeline: a scripted model emits a cell submitting 25 actions and the
whole batch lands in ``ipc/actions.jsonl``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import harness as hz  # noqa: E402
import solver_agent as sa  # noqa: E402

from nooa.strategies.codeact import _ReturnResultSignal  # noqa: E402

STATE = {"turn": 0, "state": "NOT_FINISHED", "available_actions": ["UP", "DOWN"]}
BIG_BATCH = ["UP", "DOWN"] * 13  # 26 actions — over the default cap of 20


# --------------------------------------------------------------------------- #
# agent-side cap semantics (submit_actions)
# --------------------------------------------------------------------------- #
def _fake_agent(tmp_path, state, cap=None):
    """Minimal duck-typed object exposing exactly what submit_actions touches."""
    actions_path = tmp_path / "actions.jsonl"
    actions_path.touch()
    fake = SimpleNamespace(
        _latest_state=lambda: state,
        _last_submitted_turn=lambda: -1,
        _actions_path=actions_path,
        _turn_started_at=0.0,
        _effort_ladder=[],
        _ladder_active_effort=None,
    )
    if cap is not None:
        fake._max_actions_per_turn = cap
    return fake


def test_default_cap_truncates_oversized_batch_and_warns(tmp_path):
    """An oversized batch executes its first 20 actions; the rest are dropped LOUDLY.

    Truncate-and-warn (not REJECTED): the useful prefix plays this turn, and the
    yield explanation + the persisted entry both say what was not executed.
    """
    fake = _fake_agent(tmp_path, STATE)
    with pytest.raises(_ReturnResultSignal) as excinfo:
        sa.ArcSolverBase.submit_actions(fake, BIG_BATCH, "r")
    entry = json.loads((tmp_path / "actions.jsonl").read_text().strip())
    assert entry["actions"] == BIG_BATCH[:20]  # first 20 submitted, in order
    assert entry["truncated_from"] == len(BIG_BATCH)  # harness relays this as a state note
    explanation = str(excinfo.value.result.get("explanation", ""))
    assert "NOT executed" in explanation
    assert "20" in explanation and str(len(BIG_BATCH)) in explanation


def test_cap_zero_accepts_oversized_batch(tmp_path):
    fake = _fake_agent(tmp_path, STATE, cap=0)
    with pytest.raises(_ReturnResultSignal):
        sa.ArcSolverBase.submit_actions(fake, BIG_BATCH, "r")
    entry = json.loads((tmp_path / "actions.jsonl").read_text().strip())
    assert len(entry["actions"]) == len(BIG_BATCH)
    assert "truncated_from" not in entry


def test_custom_cap_is_respected(tmp_path):
    fake = _fake_agent(tmp_path, STATE, cap=30)
    with pytest.raises(_ReturnResultSignal):
        sa.ArcSolverBase.submit_actions(fake, BIG_BATCH, "r")
    entry = json.loads((tmp_path / "actions.jsonl").read_text().strip())
    assert len(entry["actions"]) == len(BIG_BATCH)  # 26 <= 30: untouched

    fake2 = _fake_agent(tmp_path, STATE, cap=25)
    with pytest.raises(_ReturnResultSignal):
        sa.ArcSolverBase.submit_actions(fake2, BIG_BATCH, "r")
    entry2 = json.loads((tmp_path / "actions.jsonl").read_text().strip().splitlines()[-1])
    assert entry2["actions"] == BIG_BATCH[:25]
    assert entry2["truncated_from"] == len(BIG_BATCH)


def test_invalid_action_anywhere_still_rejects_whole_batch(tmp_path):
    """Validation covers the FULL batch (even the to-be-dropped tail) — a typo'd
    action signals agent confusion and must bounce, not half-execute."""
    fake = _fake_agent(tmp_path, STATE)
    batch = BIG_BATCH[:22] + ["FLY"]  # invalid action beyond the cap boundary
    out = sa.ArcSolverBase.submit_actions(fake, batch, "r")
    assert isinstance(out, str) and out.startswith("REJECTED")
    assert (tmp_path / "actions.jsonl").read_text().strip() == ""  # nothing submitted


def test_cap_zero_keeps_other_validation(tmp_path):
    """No-cap disables only the LENGTH check — action validity is still enforced."""
    fake = _fake_agent(tmp_path, STATE, cap=0)
    out = sa.ArcSolverBase.submit_actions(fake, ["FLY"] * 25, "r")
    assert isinstance(out, str) and out.startswith("REJECTED")
    assert "not in available" in out


# --------------------------------------------------------------------------- #
# harness-side backstop (apply_action_cap)
# --------------------------------------------------------------------------- #
def test_harness_cap_truncates_by_default():
    assert hz.apply_action_cap(BIG_BATCH, 20) == BIG_BATCH[:20]


@pytest.mark.parametrize("cap", [0, -1])
def test_harness_cap_disabled_passes_everything(cap):
    assert hz.apply_action_cap(BIG_BATCH, cap) == BIG_BATCH


def test_harness_truncation_note_tells_the_agent_what_was_dropped():
    """The next state's note must say the tail was not executed."""
    note = hz.truncation_note(executed=20, requested=26)
    assert "20" in note and "26" in note
    assert "not executed" in note.lower()


def test_harness_truncation_note_empty_when_nothing_dropped():
    assert hz.truncation_note(executed=5, requested=5) == ""


# --------------------------------------------------------------------------- #
# fake-LLM smoke: the real agent + CodeAct pipeline submits >20 actions
# --------------------------------------------------------------------------- #
def _codeact_response(code: str):
    """One scripted assistant turn: an execute_python tool call with *code*."""
    from nooa.unifiedllm.unifiedllm import LLMResponse, ToolCall

    args = json.dumps({"code": code})
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(id=f"call_{abs(hash(code)) & 0xFFFF}", name="execute_python", arguments=args)
        ],
        finish_reason="tool_calls",
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{abs(hash(code)) & 0xFFFF}",
                    "type": "function",
                    "function": {"name": "execute_python", "arguments": args},
                }
            ],
        },
        reasoning=None,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _build_agent(tmp_path, *, cap: int, scripted_cells: list[str]):
    from nooa.unifiedllm import FakeLLMClient

    llm = FakeLLMClient(scripted_responses=[_codeact_response(c) for c in scripted_cells])
    agent = sa.MdArcSolverAgent(
        llm=llm,
        run_dir=tmp_path,
        game_id="g1",
        alias="g1",
        skill_path=None,
        max_actions_per_turn=cap,
    )
    state = dict(STATE, grid_rows=["0" * 64] * 64)
    agent._states_path.write_text(json.dumps(state) + "\n")
    return agent, llm, state


@pytest.mark.asyncio
async def test_fake_llm_submits_26_actions_with_no_cap(tmp_path):
    """End-to-end (agent side): scripted LLM cell submits 26 actions, cap off."""
    cell = f"self.submit_actions({BIG_BATCH!r}, 'smoke: 26 actions, no cap')"
    agent, llm, state = _build_agent(tmp_path, cap=0, scripted_cells=[cell])

    result = await agent.handle({"game_states": [json.dumps(state)]})

    entries = [json.loads(line) for line in agent._actions_path.read_text().splitlines() if line]
    assert len(entries) == 1
    assert entries[0]["actions"] == BIG_BATCH  # all 26, uncapped, in order
    assert entries[0]["turn"] == 0
    # the submit auto-yielded the turn (WAIT), consuming exactly one LLM call
    assert llm.call_count == 1
    kind = getattr(result, "kind", None) or (
        result.get("kind") if isinstance(result, dict) else None
    )
    assert str(kind).lower().endswith("wait")


@pytest.mark.asyncio
async def test_fake_llm_oversized_batch_executes_first_20_with_warning(tmp_path):
    """Control: with the default cap the same 26-action cell submits its first
    20 actions immediately (turn ends, no extra model call); the entry carries
    truncated_from so the harness can tell the agent what was NOT executed."""
    cell_26 = f"self.submit_actions({BIG_BATCH!r}, 'try 26')"
    agent, llm, state = _build_agent(tmp_path, cap=20, scripted_cells=[cell_26])

    await agent.handle({"game_states": [json.dumps(state)]})

    entries = [json.loads(line) for line in agent._actions_path.read_text().splitlines() if line]
    assert len(entries) == 1
    assert entries[0]["actions"] == BIG_BATCH[:20]  # useful prefix plays this turn
    assert entries[0]["truncated_from"] == len(BIG_BATCH)
    assert llm.call_count == 1  # no rejection round-trip — the turn is not wasted
