# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Under the append policy the runtime records the live context as a LiveContextSnapshot event.

It is recorded right before the request, once per cadence, and only when a live
block changed; retries and prompt previews never add one.
"""

import json

import pytest

from nooa import Agent, Context, strategy
from nooa.config import CodeActConfig
from nooa.context_blocks import AppendedLiveContext, DynamicLiveContext, RenderConfig
from nooa.context_blocks.events import LiveContextSnapshot, UserEvent
from nooa.llm_types import LLMResponse
from nooa.runtime.actor import _extract_trailing_context_envelope
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.unifiedllm import FakeLLMClient, ToolCall

ENVELOPE = "<context>\n<state>\ns0\n</state>\n</context>"


def cell(code: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name="python_cell", arguments=json.dumps({"code": code}))],
        finish_reason="tool_calls",
    )


def _agent(llm, *, every_n=1):
    class Counter(Agent, llm=llm):
        @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
        async def count(self) -> str:
            """Set x to 1, then return it as text."""
            ...

    return Counter(
        render_config=RenderConfig(live_context=AppendedLiveContext(every_n_turns=every_n))
    )


def _snapshots(agent):
    return [e for e in agent.event_manager.values() if isinstance(e, LiveContextSnapshot)]


async def test_a_snapshot_is_recorded_before_each_due_request_and_renders_in_it():
    requests: list[list] = []

    class Recording(FakeLLMClient):
        async def acall(self, messages, **kwargs):
            requests.append(list(messages))
            return await super().acall(messages, **kwargs)

    llm = Recording(
        scripted_responses=[cell("x = 1", "call_1"), cell("return_result(str(x))", "call_2")]
    )
    agent = _agent(llm, every_n=1)
    agent.context["plan"] = Context(expr="'turn ' + str(len(self.events))")  # changes every turn
    try:
        assert await agent.count() == "1"
        events = list(agent.event_manager.values())
        snapshots = _snapshots(agent)
        assert len(snapshots) == 2  # every_n=1: one before each of the two calls
        for snapshot, response in zip(
            snapshots, [e for e in events if isinstance(e, LLMResponse)], strict=True
        ):
            assert events.index(snapshot) < events.index(response)
        # The request carried the snapshot's bytes as a user message, before the boundary.
        first_request_texts = [str(m.get("content")) for m in requests[0] if isinstance(m, dict)]
        assert snapshots[0].content in first_request_texts
        # The second request still carries the first snapshot unchanged (history).
        second_request_texts = [str(m.get("content")) for m in requests[1] if isinstance(m, dict)]
        assert snapshots[0].content in second_request_texts
    finally:
        await agent.aclose()


async def test_the_cadence_and_change_rule_skip_snapshots():
    """N=5 with a live block that never changes: the first request snapshots, later ones skip."""
    llm = FakeLLMClient(
        scripted_responses=[cell("x = 1", "call_1"), cell("return_result(str(x))", "call_2")]
    )
    agent = _agent(llm, every_n=5)
    try:
        assert await agent.count() == "1"
        assert len(_snapshots(agent)) == 1
    finally:
        await agent.aclose()


async def test_a_due_snapshot_is_skipped_when_no_live_block_changed():
    """every_n=1 but the live context is identical on the second call: no second snapshot."""
    from nooa.llm_types import AssistantText
    from nooa.strategies import PredictStrategy

    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(parts=(AssistantText(text='{"value": "a"}'),)),
            LLMResponse(parts=(AssistantText(text='{"value": "b"}'),)),
        ]
    )

    class Echo(Agent, llm=llm):
        @strategy(PredictStrategy())
        async def say(self, text: str) -> str:
            """Repeat the text."""
            ...

    agent = Echo(render_config=RenderConfig(live_context=AppendedLiveContext(every_n_turns=1)))
    agent.context["plan"] = Context("a fixed plan")
    try:
        await agent.say("one")
        await agent.say("two")
        snapshots = _snapshots(agent)
        assert len(snapshots) == 1
        # A change to a live block makes the next due request record again.
        agent.context["plan"] = Context("a new plan")
        llm._response_queue.append(LLMResponse(parts=(AssistantText(text='{"value": "c"}'),)))
        await agent.say("three")
        assert len(_snapshots(agent)) == 2
    finally:
        await agent.aclose()


async def test_an_event_recorded_during_the_call_follows_the_snapshot():
    holder: dict = {}

    class Interleaving(FakeLLMClient):
        async def acall(self, messages, **kwargs):
            if len(self.calls) == 1:  # first call pending: a note arrives
                holder["agent"].event_manager.add(UserEvent(content="recorded during the call"))
            return await super().acall(messages, **kwargs)

    llm = Interleaving(
        scripted_responses=[cell("x = 1", "call_1"), cell("return_result(str(x))", "call_2")]
    )
    agent = holder["agent"] = _agent(llm, every_n=1)
    try:
        assert await agent.count() == "1"
        events = list(agent.event_manager.values())
        first_snapshot = _snapshots(agent)[0]
        note = next(e for e in events if getattr(e, "content", "") == "recorded during the call")
        assert events.index(first_snapshot) < events.index(note)
    finally:
        await agent.aclose()


async def test_a_retry_after_a_context_window_error_does_not_record_another_snapshot():
    from litellm.exceptions import ContextWindowExceededError

    class Flaky(FakeLLMClient):
        failed = False

        async def acall(self, messages, **kwargs):
            if not self.failed:
                self.failed = True
                raise ContextWindowExceededError("context window exceeded", "fake", "fake")
            return await super().acall(messages, **kwargs)

    llm = Flaky(
        scripted_responses=[cell("x = 1", "call_1"), cell("return_result(str(x))", "call_2")]
    )
    agent = _agent(llm, every_n=5)
    try:
        assert await agent.count() == "1"
        assert len(_snapshots(agent)) == 1
    finally:
        await agent.aclose()


async def test_prompt_preview_records_nothing():
    llm = FakeLLMClient(scripted_responses=[cell("return_result('1')", "call_1")])
    agent = _agent(llm, every_n=1)
    try:
        await agent.runtime._build_messages(agent.count)
        assert _snapshots(agent) == []
    finally:
        await agent.aclose()


async def test_collapse_over_a_snapshot_archives_it_and_the_next_request_records_one():
    llm = FakeLLMClient(
        scripted_responses=[
            cell("x = 1", "call_1"),
            cell("return_result(str(x))", "call_2"),
            cell("return_result('again')", "call_3"),
        ]
    )
    agent = _agent(llm, every_n=5)
    try:
        assert await agent.count() == "1"
        snapshot = _snapshots(agent)[0]
        tags = [e.tag for e in agent.event_manager.values()]
        agent.event_manager.collapse(tags[0], snapshot.tag, summary_text="earlier work")
        assert _snapshots(agent) == []
        assert await agent.count() == "again"
        assert len(_snapshots(agent)) == 1
    finally:
        await agent.aclose()


async def test_dynamic_policy_records_no_snapshot_events():
    llm = FakeLLMClient(scripted_responses=[cell("return_result('1')", "call_1")])

    class Plain(Agent, llm=llm):
        @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
        async def go(self) -> str:
            """Return 1."""
            ...

    agent = Plain(render_config=RenderConfig(live_context=DynamicLiveContext()))
    try:
        assert await agent.go() == "1"
        assert _snapshots(agent) == []
    finally:
        await agent.aclose()


def test_the_acp_bridge_has_no_handler_for_snapshot_events():
    """The transcript shows cards and messages by event type; the snapshot has none, so it is hidden."""
    pytest.importorskip("nooa_atom.acp.event_bridge")
    import inspect

    from nooa_atom.acp import event_bridge

    source = inspect.getsource(event_bridge)
    assert "LiveContextSnapshot" not in source


def test_the_dynamic_context_stamp_is_the_envelope_as_on_main():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "note\n" + ENVELOPE + "\n"},
    ]
    assert _extract_trailing_context_envelope(messages) == ENVELOPE
    assert _extract_trailing_context_envelope([{"role": "user", "content": "just a task"}]) == ""


PROBE = "probe-secret-" * 300  # ~4k chars: well over a 1-token budget, well under a huge one


@pytest.mark.parametrize("policy", [AppendedLiveContext(every_n_turns=1), DynamicLiveContext()])
@pytest.mark.parametrize("budget", [1, 10_000_000])
async def test_the_snapshot_honours_the_same_token_budget_as_the_request(policy, budget):
    """A live block the request would send as EVICTED is snapshotted as EVICTED (issue #670).

    Under ``append`` the probe must be absent from the recorded snapshot and the
    request; under ``dynamic`` from the trailing <context> message. A large
    budget keeps it in both.
    """
    from nooa.config import TruncationConfig
    from nooa.llm_types import AssistantText
    from nooa.strategies import PredictStrategy

    requests: list[list] = []

    class Recording(FakeLLMClient):
        async def acall(self, messages, **kwargs):
            requests.append(list(messages))
            return await super().acall(messages, **kwargs)

    llm = Recording(scripted_responses=[LLMResponse(parts=(AssistantText(text='{"value": "a"}'),))])

    class Echo(Agent, llm=llm, truncation=TruncationConfig(max_context_tokens=budget)):
        @strategy(PredictStrategy())
        async def say(self, text: str) -> str:
            """Repeat the text."""
            ...

    agent = Echo(render_config=RenderConfig(live_context=policy))
    agent.context["probe"] = Context(expr="self.probe")
    agent.probe = PROBE
    try:
        await agent.say("one")
        request_text = "\n".join(str(m.get("content")) for m in requests[0] if isinstance(m, dict))
        snapshots = _snapshots(agent)
        evicted = budget == 1
        if isinstance(policy, AppendedLiveContext):
            assert len(snapshots) == 1
            assert (PROBE in snapshots[0].content) is not evicted
            assert ("EVICTED: over context budget" in snapshots[0].content) is evicted
        else:
            assert snapshots == []
        assert (PROBE in request_text) is not evicted
        assert ("EVICTED: over context budget" in request_text) is evicted
    finally:
        await agent.aclose()


async def test_the_snapshot_and_the_request_share_one_budget_across_recalibration():
    """A ratio recalibrated while the snapshot prepares its context cannot split the two.

    A concurrent ``generate`` can rewrite the chars->tokens ratio while the
    snapshot awaits ``_prepare_context``. The request must still size its blocks
    with the ratio the snapshot used, so both keep (or both evict) the probe.
    """
    from nooa.config import TruncationConfig
    from nooa.llm_types import AssistantText
    from nooa.strategies import PredictStrategy

    requests: list[list] = []

    class Recording(FakeLLMClient):
        async def acall(self, messages, **kwargs):
            requests.append(list(messages))
            return await super().acall(messages, **kwargs)

    llm = Recording(scripted_responses=[LLMResponse(parts=(AssistantText(text='{"value": "a"}'),))])

    # ~4k chars: fits a 2,000-token budget at the default 0.25 tokens/char, not at 1.0.
    class Echo(Agent, llm=llm, truncation=TruncationConfig(max_context_tokens=2_000)):
        @strategy(PredictStrategy())
        async def say(self, text: str) -> str:
            """Repeat the text."""
            ...

    agent = Echo(render_config=RenderConfig(live_context=AppendedLiveContext(every_n_turns=1)))
    agent.context["probe"] = Context(expr="self.probe")
    agent.probe = PROBE
    runtime = agent.runtime
    prepare = runtime._prepare_context

    async def recalibrating_prepare(*args, **kwargs):
        blocks = await prepare(*args, **kwargs)
        runtime._tokens_per_char = 1.0  # another call's response lands mid-await
        return blocks

    runtime._prepare_context = recalibrating_prepare
    try:
        await agent.say("one")
        request_text = "\n".join(str(m.get("content")) for m in requests[0] if isinstance(m, dict))
        (snapshot,) = _snapshots(agent)
        assert PROBE in snapshot.content
        assert PROBE in request_text
        # The request sized its blocks with the snapshot's ratio: nothing evicted
        # (at 1.0 tokens/char it would have dropped the probe the snapshot kept).
        assert runtime._last_context_stats.context_blocks_dropped == 0
    finally:
        await agent.aclose()
