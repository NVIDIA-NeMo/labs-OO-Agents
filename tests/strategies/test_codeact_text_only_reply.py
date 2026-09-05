# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CodeAct text-only-reply capture + PredictStrategy-style recovery.

Covers the fix for the "model returns text instead of a tool call" failure mode:
- a TextOnlyReply event is recorded (but never shown to the model),
- a model-visible Error correction is added so the model self-corrects,
- recovered=True is set when a real tool call follows,
- the consecutive-text-only backstop still aborts on repeated non-compliance.
"""

import json

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.errors import GenerationError
from nooa.events import PythonOutput
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

_TEST_LLM = FakeLLMClient()


def _resp(content="", tool_calls=None, finish_reason=None):
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _ret(val, cid="c_ret"):
    return ToolCall(id=cid, name="return_result", arguments=json.dumps({"result": val}))


def _drifts(agent):
    return [e for e in agent.event_manager.values() if e.event_type == "TextOnlyReply"]


def _errors(agent):
    return [e for e in agent.event_manager.values() if e.event_type == "Error"]


@pytest.mark.asyncio
async def test_text_only_drift_recorded_and_recovers():
    """A text-only stop (Route A, non-str return) records a TextOnlyReply, adds a
    model-visible Error correction, and recovers when a real tool call follows."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(CodeActStrategy(config=CodeActConfig(max_retries=5, max_iterations=10)))
        async def my_task(self) -> dict:
            """Return a dict — a bare string won't validate, forcing the correction path."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("I think the answer is ready."),  # text-only drift (won't validate as dict)
            _resp(tool_calls=[_ret({"ok": True})]),  # real tool call → recovery
        ]
    )
    agent = TestAgent(llm=fake_llm)
    result = await agent.my_task()
    assert result == {"ok": True}

    drifts = _drifts(agent)
    assert len(drifts) == 1, f"expected one TextOnlyReply, got {len(drifts)}"
    d = drifts[0]
    assert d.route == "return_result"
    assert d.finish_reason == "stop"
    assert d.content == "I think the answer is ready."
    assert d.recovered is True, "drift should be marked recovered after the real tool call"

    # The correction is a model-visible Error event.
    errs = [e for e in _errors(agent) if "no tool call" in e.content]
    assert errs, "expected a model-visible Error correction after the drift"


@pytest.mark.asyncio
async def test_text_only_drift_not_shown_to_model():
    """TextOnlyReply is Role.METADATA — recorded but never rendered into the prompt."""
    from nooa.context_blocks.roles import Role
    from nooa.events import TextOnlyReply

    assert TextOnlyReply._role == Role.METADATA


@pytest.mark.asyncio
async def test_text_only_backstop_aborts_after_threshold():
    """Repeated text-only drift (no recovery) still aborts via the backstop."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(
            CodeActStrategy(
                config=CodeActConfig(
                    max_retries=10,
                    max_iterations=10,
                    max_consecutive_text_only=3,
                )
            )
        )
        async def my_task(self) -> dict:
            """Return a dict."""
            ...

    fake_llm = FakeLLMClient(scripted_responses=[_resp("still chatting") for _ in range(6)])
    agent = TestAgent(llm=fake_llm)
    with pytest.raises(GenerationError, match="plain text without a tool call"):
        await agent.my_task()

    # Three drifts recorded, none recovered.
    drifts = _drifts(agent)
    assert len(drifts) == 3, f"expected 3 drifts before abort, got {len(drifts)}"
    assert all(d.recovered is False for d in drifts)
    # A correction was offered on each drift.
    assert len([e for e in _errors(agent) if "no tool call" in e.content]) == 3


@pytest.mark.asyncio
async def test_multiple_drifts_all_marked_recovered():
    """Several consecutive drifts before a real tool call: all flip recovered=True."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(CodeActStrategy(config=CodeActConfig(max_retries=8, max_iterations=12)))
        async def my_task(self) -> dict:
            """Return a dict."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("thinking 1"),  # drift 1 (Route A, non-str → correction)
            _resp("thinking 2"),  # drift 2
            _resp(tool_calls=[_ret({"ok": True})]),  # recovery
        ]
    )
    agent = TestAgent(llm=fake_llm)
    result = await agent.my_task()
    assert result == {"ok": True}

    drifts = _drifts(agent)
    assert len(drifts) == 2
    assert all(d.recovered is True for d in drifts), (
        "all drifts before the recovering tool call must be marked recovered"
    )


@pytest.mark.asyncio
async def test_route_b_does_not_add_contradictory_correction():
    """Route B (synthetic_comment) injects its own synthetic tool result, so it must
    NOT also add a 'no tool call' Error — that would contradict the synthetic call."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(
            CodeActStrategy(
                config=CodeActConfig(
                    max_retries=8,
                    max_iterations=12,
                    text_only_stop_behavior="synthetic_comment",
                )
            )
        )
        async def my_task(self) -> str:
            """Return a string."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("some prose, no tool call"),  # Route B drift
            _resp(tool_calls=[_ret("done")]),  # recovery
        ]
    )
    agent = TestAgent(llm=fake_llm)
    result = await agent.my_task()
    assert result == "done"

    # Drift recorded and recovered.
    drifts = _drifts(agent)
    assert len(drifts) == 1
    assert drifts[0].route == "synthetic_comment"
    assert drifts[0].recovered is True
    # No contradictory "no tool call" Error correction in Route B.
    assert not [e for e in _errors(agent) if "no tool call" in e.content], (
        "Route B must not add a corrective Error (it has its own synthetic result)"
    )


@pytest.mark.asyncio
async def test_route_b_synthetic_comment_does_not_reuse_next_cell_number():
    """A synthetic comment and the next real cell have distinct execution counts."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(
            CodeActStrategy(
                config=CodeActConfig(
                    max_retries=8,
                    max_iterations=12,
                    text_only_stop_behavior="synthetic_comment",
                )
            )
        )
        async def my_task(self) -> str:
            """Return a string."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("some prose, no tool call"),
            _resp(
                tool_calls=[
                    ToolCall(
                        id="real_cell",
                        name="execute_python",
                        arguments=json.dumps({"code": "'computed'"}),
                    )
                ]
            ),
            _resp(tool_calls=[_ret("done")]),
        ]
    )
    agent = TestAgent(llm=fake_llm)

    assert await agent.my_task() == "done"
    outputs = [event for event in agent.event_manager.values() if isinstance(event, PythonOutput)]
    assert [event.execution_count for event in outputs] == [1, 2]
    assert outputs[1].tool_call_id == "real_cell"
    assert outputs[1].value == "computed"


@pytest.mark.asyncio
async def test_append_only_preserves_provider_llm_output_route_a():
    """Route A append-only mode: the provider LLMOutput (with reasoning_items)
    is preserved rather than deleted, and its reasoning replays on the retry."""
    from nooa.events import LLMOutput

    reasoning_item = {
        "id": "rs_a",
        "type": "reasoning",
        "encrypted_content": "enc",
        "summary": [],
    }

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(
            CodeActStrategy(
                config=CodeActConfig(
                    max_retries=6,
                    max_iterations=10,
                    text_only_correction="return",
                )
            )
        )
        async def my_task(self) -> dict:
            """Return a dict — a bare string won't validate, forcing the retry path."""
            ...

    def _resp_with_reasoning(content):
        return LLMResponse(
            raw_response=None,
            content=content,
            tool_calls=[],
            finish_reason="stop",
            assistant_message={
                "role": "assistant",
                "content": content,
                "reasoning_items": [reasoning_item],
            },
        )

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp_with_reasoning("I think the answer is ready."),  # text-only drift
            _resp(tool_calls=[_ret({"ok": True})]),  # recovery
        ]
    )
    agent = TestAgent(llm=fake_llm)
    assert await agent.my_task() == {"ok": True}

    # The provider-authored LLMOutput survived append-only recovery, carrying
    # the opaque reasoning state for replay.
    outputs = [e for e in agent.event_manager.values() if isinstance(e, LLMOutput)]
    preserved = [o for o in outputs if o.reasoning_items == [reasoning_item]]
    assert preserved, "append-only mode must preserve the provider LLMOutput with reasoning_items"


@pytest.mark.asyncio
async def test_append_only_preserves_provider_llm_output_route_b():
    """Route B append-only mode: provider LLMOutput survives and the synthetic
    comment is appended after it (and flagged synthetic)."""
    from nooa.context_blocks.events import ToolCallEvent
    from nooa.events import LLMOutput, TextOnlyReply

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(
            CodeActStrategy(
                config=CodeActConfig(
                    max_retries=8,
                    max_iterations=12,
                    text_only_stop_behavior="synthetic_comment",
                    text_only_correction="comment",
                )
            )
        )
        async def my_task(self) -> str:
            """Return a string."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("some prose, no tool call"),  # Route B drift
            _resp(tool_calls=[_ret("done")]),  # recovery
        ]
    )
    agent = TestAgent(llm=fake_llm)
    assert await agent.my_task() == "done"

    # Provider output preserved.
    outputs = [e for e in agent.event_manager.values() if isinstance(e, LLMOutput)]
    assert any(o.content == "some prose, no tool call" for o in outputs), (
        "append-only Route B must preserve the provider LLMOutput"
    )
    # Synthetic comment appended and flagged synthetic (not provider-authored).
    synth = [
        e
        for e in agent.event_manager.values()
        if isinstance(e, ToolCallEvent) and (e.metadata or {}).get("synthetic")
    ]
    assert synth, "append-only Route B must append a synthetic comment tool call"
    # Drift still recorded.
    assert any(isinstance(e, TextOnlyReply) for e in agent.event_manager.values())


@pytest.mark.asyncio
async def test_custom_text_only_correction_appends_custom_message():
    """text_only_correction='custom' appends the callable's message as an Error."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(
            CodeActStrategy(
                config=CodeActConfig(
                    max_retries=8,
                    max_iterations=12,
                    text_only_stop_behavior="synthetic_comment",
                    text_only_correction="custom",
                    text_only_correction_fn=lambda text: f"CUSTOM NUDGE: saw {len(text)} chars",
                )
            )
        )
        async def my_task(self) -> str:
            """Return a string."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("prose here"),  # drift
            _resp(tool_calls=[_ret("done")]),  # recovery
        ]
    )
    agent = TestAgent(llm=fake_llm)
    assert await agent.my_task() == "done"

    assert [e for e in _errors(agent) if e.content.startswith("CUSTOM NUDGE:")], (
        "custom correction callable output must be appended as a model-visible Error"
    )


@pytest.mark.asyncio
async def test_legacy_default_still_deletes_provider_output():
    """Without text_only_correction the legacy delete-then-replace behavior holds
    (no provider LLMOutput left for the dropped text-only turn)."""
    from nooa.events import LLMOutput

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(CodeActStrategy(config=CodeActConfig(max_retries=6, max_iterations=10)))
        async def my_task(self) -> dict:
            """Return a dict."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("I think the answer is ready."),  # text-only drift (legacy)
            _resp(tool_calls=[_ret({"ok": True})]),  # recovery
        ]
    )
    agent = TestAgent(llm=fake_llm)
    assert await agent.my_task() == {"ok": True}

    # Legacy mode removed the provider LLMOutput for the dropped turn.
    outputs = [e for e in agent.event_manager.values() if isinstance(e, LLMOutput)]
    assert not any(o.content == "I think the answer is ready." for o in outputs), (
        "legacy default must still delete the provider output (no opt-in)"
    )
