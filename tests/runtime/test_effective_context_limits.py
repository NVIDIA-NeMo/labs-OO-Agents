# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context consumers must budget against the request's actual reply settings."""

from unittest.mock import AsyncMock

import pytest

from nooa import Agent, Context
from nooa.agents.summarization import context_budget
from nooa.events import Message
from nooa.interactive import SummarizationConfig, apply_model_limits, install_summarizer
from nooa.runtime.actor import _compute_reduced_max_tokens, _current_llm_var
from nooa.runtime.middleware import LLMCallContext
from nooa.unifiedllm import FakeLLMClient, LLMResponse, LLMUsage
from nooa.unifiedllm.reasoning import ReasoningConfig


def client():
    llm = FakeLLMClient()
    llm.config["max_tokens"] = 64_000
    llm._reasoning_config = ReasoningConfig(
        levels={"high": {"max_tokens": 96_000, "reasoning_effort": "high"}}
    )
    return llm


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, 64_000),
        ({"max_tokens": 16_000}, 16_000),
        ({"max_completion_tokens": 16_000}, 16_000),
        ({"max_output_tokens": 16_000}, 16_000),
        ({"extra_body": {"max_tokens": 16_000}}, 16_000),
        ({"reasoning_level": "high"}, 96_000),
    ],
)
def test_limits_share_request_resolution(overrides, expected):
    llm = client()
    limits = llm.get_context_limits(overrides)
    assert limits.context_window == 128_000
    assert limits.reserved_output_tokens == expected
    assert limits.usable_input_tokens == 128_000 - expected
    assert not limits.reserve_is_fallback
    config = llm._prepare_call_config(overrides)
    body = {**config, **config.get("extra_body", {})}
    caps = [
        body[k] for k in ("max_tokens", "max_completion_tokens", "max_output_tokens") if k in body
    ]
    assert caps == [expected]
    assert llm.config["max_tokens"] == 64_000


def test_active_level_not_metadata_default_and_unknown_reserve():
    llm = client()
    llm._reasoning_config = ReasoningConfig(levels=llm._reasoning_config.levels, default="high")
    assert llm.get_context_limits().reserved_output_tokens == 64_000
    llm.reasoning_level = "high"
    assert llm.get_context_limits().reserved_output_tokens == 96_000
    llm.reasoning_level = None
    llm.config.clear()
    limits = llm.get_context_limits(fallback_reserve=4096)
    assert limits.reserved_output_tokens == 4096
    assert limits.reserve_is_fallback


@pytest.mark.parametrize("cap", [32_768, 65_536])
@pytest.mark.parametrize("source", ["default", "override", "alias", "extra_body", "level"])
def test_reply_cap_must_leave_room_for_input(cap, source):
    llm = client()
    llm._context_window = 32_768
    llm.config["max_tokens"] = 8192
    overrides = {}
    if source == "default":
        llm.config["max_tokens"] = cap
    elif source == "override":
        overrides = {"max_tokens": cap}
    elif source == "alias":
        overrides = {"max_output_tokens": cap}
    elif source == "extra_body":
        overrides = {"extra_body": {"max_tokens": cap}}
    else:
        llm._reasoning_config = ReasoningConfig(levels={"high": {"max_tokens": cap}})
        overrides = {"reasoning_level": "high"}

    with pytest.raises(ValueError, match="reply cap.*leaves no room for input"):
        context_budget(llm, request_params=overrides)


@pytest.mark.parametrize("cap", [16_384, 24_576])
def test_large_valid_reply_cap_is_not_replaced_by_a_fallback(cap):
    llm = client()
    llm._context_window = 32_768
    llm.config["max_tokens"] = cap
    limits = llm.get_context_limits(fallback_reserve=4096)
    assert limits.reserved_output_tokens == cap
    assert limits.usable_input_tokens == 32_768 - cap
    assert not limits.reserve_is_fallback
    assert context_budget(llm) == int((32_768 - cap) * 0.8)


def test_reply_cap_guard_also_covers_custom_clients_but_not_unknown_limits():
    from types import SimpleNamespace

    from nooa.unifiedllm.limits import context_limits_for

    custom = SimpleNamespace(context_window=32_768)
    with pytest.raises(ValueError, match="reply cap.*leaves no room for input"):
        context_limits_for(custom, {"max_completion_tokens": 32_768})

    llm = client()
    llm._context_window = None
    assert llm.get_context_limits().usable_input_tokens is None
    # An unknown provider cap is not a configured, invalid request.
    custom.context_window = 2048
    assert context_limits_for(custom, fallback_reserve=4096).reserve_is_fallback


def test_invalid_reply_cap_cannot_install_a_one_token_automatic_summary_budget():
    llm = client()
    llm._context_window = 32_768
    llm.config["max_tokens"] = 32_768
    agent = Agent(llm=llm)
    with pytest.raises(ValueError, match="reply cap.*leaves no room for input"):
        install_summarizer(SummarizationConfig(), agent)
    assert not getattr(agent, "_summarizers", [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides,expected",
    [({}, 64_000), ({"reasoning_level": "high"}, 96_000), ({"max_tokens": 16_000}, 16_000)],
)
async def test_rendered_context_and_stats_use_request_limit(overrides, expected):
    llm = client()

    class A(Agent):
        async def answer(self) -> str:
            """Answer."""
            ...

    agent = A(llm=llm)
    agent.context["context_usage"] = Context(
        expr="self.context_stats.format() if self.context_stats else ''"
    )
    token = _current_llm_var.set(llm)
    try:
        await agent.runtime._build_messages(A.answer)
        agent.runtime._last_context_stats = agent.context_stats.model_copy(
            update={"prompt_tokens": 32_000}
        )
        messages = await agent.runtime._build_messages(A.answer, request_params=overrides)
    finally:
        _current_llm_var.reset(token)
    stats = agent.context_stats.model_copy(update={"prompt_tokens": 32_000})
    assert stats.reserved_output_tokens == expected
    assert stats.overall_utilization == 32_000 / (128_000 - expected)
    assert f"output reserve: {expected:,}" in str(messages)


def test_summarization_uses_usable_window_and_preserves_explicit_threshold():
    llm = client()
    assert context_budget(llm) == 51_200
    agent = Agent(llm=llm)
    install_summarizer(SummarizationConfig(max_tokens=12345, threshold_fraction=0.10), agent)
    apply_model_limits(agent)
    assert agent._summarizers[0].config.max_tokens == 12345
    agent._summarizers[0]._uninstall()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("threshold_fraction", "initial_threshold", "call_threshold"),
    [(0.75, 48_000, 24_000), (0.60, 38_400, 19_200)],
)
async def test_automatic_summary_threshold_tracks_actual_call_cap(
    threshold_fraction, initial_threshold, call_threshold
):
    llm = client()
    agent = Agent(llm=llm)
    install_summarizer(
        SummarizationConfig(preserve_recent=0, threshold_fraction=threshold_fraction), agent
    )
    summarizer = agent._summarizers[0]
    assert summarizer.config.max_tokens == initial_threshold
    agent.event_manager.add(Message(content="remember this"))
    llm.acall = AsyncMock(return_value=LLMResponse(content="summary"))
    ctx = LLMCallContext(
        agent=agent,
        runtime=agent.runtime,
        client=llm,
        messages=[{"role": "user", "content": "question"}],
        params={"reasoning_level": "high"},
    )

    async def core(request):
        request.response = LLMResponse(
            content="answer", usage=LLMUsage(input_tokens=30_000, output_tokens=1)
        )
        return request

    try:
        await agent.event_manager.run_middleware("llm_call", ctx, core)
        assert summarizer.config.max_tokens == call_threshold
        assert summarizer._pending_task is not None
        await summarizer._pending_task
        llm.acall.assert_awaited_once()
    finally:
        await summarizer.aclose()


def test_overflow_recovery_never_increases_reply_limit():
    assert (
        _compute_reduced_max_tokens(ValueError("prompt contains 1000 tokens"), 128_000, 4000)
        <= 4000
    )


@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.parametrize(
    "params", [{}, {"reasoning_level": "high"}, {"max_completion_tokens": 48_000}]
)
async def test_recovery_uses_effective_cap_and_keeps_reasoning(middleware, params):
    from nooa.runtime.actor import _current_method_var

    class ContextWindowExceededError(Exception):
        pass

    llm = client()
    agent = Agent(llm=llm)
    seen = []

    async def answer():
        pass

    async def passthrough(ctx, nxt):
        return await nxt(ctx)

    if middleware:
        agent.event_manager.intercept("llm_call", passthrough)

    async def call(messages, **kwargs):
        seen.append(llm._prepare_call_config(kwargs))
        if len(seen) == 1:
            raise ContextWindowExceededError("context window exceeded")
        return LLMResponse(content="answer", usage=LLMUsage(input_tokens=100, output_tokens=1))

    llm.acall = call
    lt = _current_llm_var.set(llm)
    mt = _current_method_var.set(answer)
    original = llm.get_context_limits(params).reserved_output_tokens
    try:
        await agent.runtime.generate(**params)
    finally:
        _current_llm_var.reset(lt)
        _current_method_var.reset(mt)
    assert len(seen) == 2
    cap_key = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
    assert seen[1][cap_key] == original // 2
    assert agent.context_stats.reserved_output_tokens == original // 2
    if "reasoning_level" in params:
        assert seen[1]["reasoning_effort"] == "high"


def test_model_override_does_not_reuse_original_window_and_fallback_is_labelled():
    from nooa.context_blocks.models import ContextWindowStats

    assert client().get_context_limits({"model": "unknown-other-model"}).context_window is None
    llm = FakeLLMClient()
    limits = llm.get_context_limits(fallback_reserve=4096)
    stats = ContextWindowStats(
        context_blocks_count=0,
        events_count=0,
        prompt_tokens=32000,
        model_context_window=limits.context_window,
        reserved_output_tokens=limits.reserved_output_tokens,
        output_reserve_is_fallback=limits.reserve_is_fallback,
    )
    assert "planning reserve: 4,096" in stats.format()
    assert "output reserve:" not in stats.format()


@pytest.mark.parametrize("override_client", [True, False])
async def test_selected_client_and_final_middleware_settings_drive_stats(override_client):
    from nooa.runtime.actor import _current_method_var

    original = client()
    replacement = client()
    replacement._context_window = 256_000
    agent = Agent(llm=original)
    selected = replacement if override_client else original
    unused = original if override_client else replacement
    unused.acall = AsyncMock(side_effect=AssertionError("Unselected client must not be called"))
    selected.acall = AsyncMock(
        return_value=LLMResponse(
            content="answer", usage=LLMUsage(input_tokens=32_000, output_tokens=1)
        )
    )

    async def route(ctx, nxt):
        ctx.params["max_tokens"] = 32_000
        return await nxt(ctx)

    async def answer():
        pass

    agent.event_manager.intercept("llm_call", route)
    lt = _current_llm_var.set(selected)
    mt = _current_method_var.set(answer)
    try:
        await agent.runtime.generate()
    finally:
        _current_llm_var.reset(lt)
        _current_method_var.reset(mt)
    selected.acall.assert_awaited_once()
    assert selected.acall.call_args.kwargs["max_tokens"] == 32_000
    assert agent.context_stats.model_context_window == selected.context_window
    assert agent.context_stats.reserved_output_tokens == 32_000
    assert agent.context_stats.overall_utilization == 32_000 / (selected.context_window - 32_000)


@pytest.mark.parametrize("key", ["max_completion_tokens", "max_output_tokens"])
def test_selected_level_and_explicit_alias_are_rejected_consistently(key):
    llm = client()
    with pytest.raises(ValueError, match="conflicts"):
        llm.get_context_limits({"reasoning_level": "high", key: 16_000})
    with pytest.raises(ValueError, match="conflicts"):
        llm._prepare_call_config({"reasoning_level": "high", key: 16_000})
