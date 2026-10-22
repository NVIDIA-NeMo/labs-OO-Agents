# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry thinking defaults through real LiteLLM/SDKs and offline HTTP."""

import base64
import json
from copy import deepcopy

import httpx
import pytest

from nooa import Agent, no_trace
from nooa.config import ModelConfig
from nooa.config.strategy_config import CodeActConfig
from nooa.decorators import strategy
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.unifiedllm import registry
from nooa.unifiedllm.connect._session import TOKEN_RESERVATION, session_steps
from tests.unifiedllm.connect.connect_http import mock_http, response_body
from tests.unifiedllm.test_anthropic_native_profile import http_fixture

THINKING = {"type": "enabled", "budget_tokens": 1024}
# Synthetic opaque base64, not a provider credential or a claimed server-valid signature.
SIGNATURE = base64.b64encode(bytes(range(256))).decode()


def entry(style, **settings):
    return {
        "model_name": "anthropic/claude-opus-4-6"
        if style == "anthropic"
        else "openai/azure/anthropic/claude-opus-5-5",
        "api_style": style,
        "api_base": "https://api.test" if style == "anthropic" else "https://api.test/v1",
        "max_tokens": 1536,
        "allowed_openai_params": ["thinking"],
        "cache_breakpoint": None,
        "media_inputs": [],
        "retry_config": False,
        **settings,
    }


def test_typed_config_already_preserves_thinking():
    raw = entry("anthropic", thinking=deepcopy(THINKING))
    before = deepcopy(raw)
    typed = ModelConfig.from_registry("local", raw)
    assert typed.thinking == THINKING
    assert typed.model_dump()["thinking"] == THINKING
    assert raw == before


@pytest.mark.parametrize("style", ["chat", "anthropic"])
@pytest.mark.parametrize("is_async", [False, True])
async def test_registry_top_level_matches_level_default_and_overrides(monkeypatch, style, is_async):
    bodies, paths = http_fixture(monkeypatch, style)
    configs = [
        entry(style, thinking=deepcopy(THINKING)),
        entry(
            style, reasoning_levels={"on": {"thinking": deepcopy(THINKING)}}, reasoning_default="on"
        ),
        entry(style),
    ]
    for index, config in enumerate(configs):
        before = deepcopy(config)
        monkeypatch.setattr(registry, "_loaded", True)
        monkeypatch.setitem(registry.MODELS, "thinking-test", config)
        c = registry.get_llm_client("thinking-test", api_key="fake")
        snapshot = deepcopy(c.config)
        try:
            call = c.acall if is_async else c.call

            async def invoke(call=call, **kwargs):
                result = call([{"role": "user", "content": "hello"}], **kwargs)
                return await result if is_async else result

            await invoke(**({"reasoning_level": "on"} if index == 1 else {}))
            assert bodies[-1].get("thinking") == (THINKING if index < 2 else None)
            assert ("thinking" in bodies[-1]) == (index < 2)
            if index == 1:
                # reasoning_default documents the route; it is not an activation.
                await invoke()
                assert "thinking" not in bodies[-1]
            overrides = {"thinking": {"type": "disabled"}}
            original = deepcopy(overrides)
            await invoke(**overrides)
            assert bodies[-1]["thinking"] == overrides["thinking"]
            assert overrides == original
            await invoke(**({"reasoning_level": "on"} if index == 1 else {}))
            assert bodies[-1].get("thinking") == (THINKING if index < 2 else None)
            assert c.config == snapshot and config == before
            assert all(b["max_tokens"] == 1536 and "extra_body" not in b for b in bodies)
        finally:
            await c.aclose()
    # Explicit constructor override wins over the saved top-level setting as well.
    config = configs[0]
    c = registry.client_from_config("local", config, api_key="fake", thinking={"type": "disabled"})
    try:
        await c.acall([{"role": "user", "content": "hello"}])
        assert bodies[-1]["thinking"] == {"type": "disabled"}
        assert config["thinking"] == THINKING
    finally:
        await c.aclose()
    assert set(paths) == {"/v1/messages" if style == "anthropic" else "/v1/chat/completions"}


@pytest.mark.parametrize("placement", ["top", "level"])
@pytest.mark.parametrize(
    "thought",
    ["Check the original inputs.", "", None],
    ids=["text", "signature-only", "not-returned"],
)
async def test_native_registry_session_settings_and_signed_replay(monkeypatch, placement, thought):
    bodies = []
    import nooa.unifiedllm.connect._session as session

    monkeypatch.setattr(session, "SESSION_CALL_PACING_SECONDS", 0)

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert request.url.path == "/v1/messages"
        assert body["thinking"] == THINKING and body["max_tokens"] == 1536
        data = response_body("anthropic")
        if thought is not None:
            data["content"].insert(
                0, {"type": "thinking", "thinking": thought, "signature": SIGNATURE}
            )
        data["usage"]["cache_read_input_tokens"] = 15 if len(bodies) > 1 else 0
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    settings = {"thinking": deepcopy(THINKING)}
    config = entry(
        "anthropic",
        **(
            settings
            if placement == "top"
            else {"reasoning_levels": {"on": settings}, "reasoning_default": "on"}
        ),
    )
    before = deepcopy(config)
    updates = [
        u
        async for u in session_steps(
            "local", config, api_key="fake", budget_tokens=TOKEN_RESERVATION
        )
    ]
    records = {u.name: u.outcome for u in updates if u.outcome["outcome"] != "running"}
    assert records["session"]["outcome"] == "completed", records
    assert len(bodies) == 3
    assert all(records[f"session:{name}"]["settings_sent"] for name in ("seed", "replay", "repeat"))
    retention = records["reasoning_retention"]
    assert retention["settings_retained"] is True
    assert config == before
    if thought is None:
        assert retention["outcome"] == "not_confirmed"
        assert retention["state_retained"] is None
        assert "No replayable reasoning returned" in retention["reason"]
        assert "does not mean reasoning is off" in retention["reason"]
        assert "not preserved" not in retention["reason"]
    else:
        assert retention["outcome"] == "confirmed", records
        assert retention["state_retained"] is True
        assert retention["reasoning_observed_by_turn"] == [True, True, True]
        for body in bodies[1:]:
            prior = next(m for m in body["messages"] if m["role"] == "assistant")
            assert prior["content"][0] == {
                "type": "thinking",
                "thinking": thought,
                "signature": SIGNATURE,
            }
        assert SIGNATURE not in repr(records)


@pytest.mark.parametrize(
    "strategy_type,tool_name", [(CodeActStrategy, "execute_python"), (CodeActV2, "python_cell")]
)
@pytest.mark.parametrize(
    "thought", ["Check the original inputs.", ""], ids=["text", "signature-only"]
)
async def test_actual_agent_native_tool_continuation_retains_original_blocks(
    monkeypatch, strategy_type, tool_name, thought
):
    bodies, responses = [], []
    original_blocks = [
        {"type": "thinking", "thinking": thought, "signature": SIGNATURE},
        {"type": "text", "text": "I will calculate."},
        {
            "type": "tool_use",
            "id": "compute-one",
            "name": tool_name,
            "input": {"code": "value = 42\nprint(value)"},
        },
    ]

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert body["thinking"] == THINKING and body["max_tokens"] == 1536
        data = response_body("anthropic")
        data["stop_reason"] = "tool_use"
        if len(bodies) == 1:
            data["content"] = deepcopy(original_blocks)
        else:
            prior = next(
                m
                for m in body["messages"]
                if m["role"] == "assistant"
                and any(b.get("type") == "tool_use" for b in m["content"])
            )
            assert prior["content"] == original_blocks
            assert any(
                block.get("type") == "tool_result" and block["tool_use_id"] == "compute-one"
                for message in body["messages"]
                for block in message["content"]
            )
            data["content"] = [
                {
                    "type": "tool_use",
                    "id": "complete-two",
                    "name": tool_name,
                    "input": {"code": "return_result(value)"},
                }
            ]
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    config = entry("anthropic", thinking=deepcopy(THINKING))
    before = deepcopy(config)
    c = registry.client_from_config("local", config, api_key="fake")
    acall = c.acall

    async def record(*args, **kwargs):
        response = await acall(*args, **kwargs)
        responses.append(response)
        return response

    monkeypatch.setattr(c, "acall", record)

    class ThinkingAgent(Agent):
        @no_trace
        @strategy(strategy_type(config=CodeActConfig(prefill=None)))
        async def compute(self) -> int:
            """Compute and return 42 using Python."""
            ...

    agent = ThinkingAgent(llm=c)
    try:
        assert await agent.compute() == 42
        assert len(bodies) == 2
        first = responses[0]
        assert [part.kind for part in first.parts] == ["reasoning", "text", "tool_call"]
        assert first.parts[0].text == thought
        assert SIGNATURE in first.model_dump_json()
        # Runtime event retention uses the original response, not reconstructed text.
        event = next(
            e
            for e in agent.event_manager.values()
            if e.event_type == "LLMResponse" and e.id == first.id
        )
        assert event is first and event.parts == first.parts
        execution = next(
            e
            for e in agent.event_manager.values()
            if e.event_type == "ToolCallEvent" and e.tool_call_id == "compute-one"
        )
        assert execution.llm_response_id == first.id
        assert config == before
    finally:
        await agent.aclose()
        await c.aclose()
