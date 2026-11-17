# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry forwarding of chat_template_kwargs, which Connect-saved reasoning levels use."""

from copy import deepcopy

import pytest

from nooa.config import ModelConfig
from nooa.unifiedllm import CompletionClient, ResponsesClient, registry

MODEL = "openai/offline-route-context"
BASE = "https://gateway.test/v1"


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(CompletionClient, "get_model_info", lambda self: None)
    monkeypatch.setattr(ResponsesClient, "get_model_info", lambda self: None)


def entry(**settings):
    return dict(
        model_name=MODEL,
        api_base=BASE,
        context_window=8192,
        media_inputs=[],
        retry_config=False,
        **settings,
    )


@pytest.mark.parametrize("client_type", ["completion", "responses"])
def test_chat_template_kwargs_defaults_levels_and_overrides(client_type):
    raw = entry(
        client_type=client_type,
        chat_template_kwargs={"enable_thinking": False},
        reasoning_levels={"on": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    before = deepcopy(raw)
    typed = ModelConfig.from_registry("local", raw)
    c = registry.client_from_config("local", typed.model_dump(exclude_unset=True))
    assert c._prepare_call_config({})["chat_template_kwargs"] == {"enable_thinking": False}
    assert c._prepare_call_config({"reasoning_level": "on"})["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    override = registry.client_from_config("local", raw, chat_template_kwargs={"caller": True})
    assert override._prepare_call_config({})["chat_template_kwargs"] == {"caller": True}
    assert raw == before


@pytest.mark.parametrize("is_async", [False, True])
async def test_chat_template_kwargs_reach_the_wire(monkeypatch, is_async):
    # Actual CompletionClient -> LiteLLM -> OpenAI SDK -> httpx.MockTransport.
    # This is NOT the separate Direct SDK transport.
    import json

    import httpx

    from nooa.unifiedllm.unifiedllm import _ClientHttp
    from tests.unifiedllm.connect.connect_http import response_body

    bodies, paths, timeouts = [], [], []

    def respond(request):
        bodies.append(json.loads(request.content))
        paths.append(request.url.path)
        timeouts.append(request.extensions.get("timeout"))
        return httpx.Response(200, json=response_body("chat"))

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": transport, "trust_env": False}),
    )
    raw = entry(
        timeout=12.5,
        chat_template_kwargs={"enable_thinking": False},
        allowed_openai_params=["chat_template_kwargs"],
        cache_breakpoint=None,
        reasoning_levels={"on": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    typed = ModelConfig.from_registry("local", raw)
    registry.MODELS["local"] = typed.model_dump(exclude_unset=True)
    c = registry.get_llm_client("local", api_key="fake")
    original = deepcopy(c.config)
    try:

        async def invoke(**kwargs):
            messages = [{"role": "user", "content": "hello"}]
            return await c.acall(messages, **kwargs) if is_async else c.call(messages, **kwargs)

        await invoke()
        await invoke(reasoning_level="on")
        overrides = {"chat_template_kwargs": {"caller": True}, "timeout": 3}
        await invoke(**overrides)
        assert [b["chat_template_kwargs"] for b in bodies] == [
            {"enable_thinking": False},
            {"enable_thinking": True},
            {"caller": True},
        ]
        assert paths == ["/v1/chat/completions"] * 3
        assert [t["read"] for t in timeouts] == [12.5, 12.5, 3]
        assert c.config == original
        assert overrides == {"chat_template_kwargs": {"caller": True}, "timeout": 3}
    finally:
        await c.aclose()
