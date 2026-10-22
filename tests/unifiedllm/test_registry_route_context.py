# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline real-client regression tests for endpoint-local alias metadata."""

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


@pytest.mark.parametrize(
    "settings",
    [
        {"api_base": "https://other.test/v1"},
        {},
        {"api_base": BASE, "api_key": "other"},
        {"api_base": BASE, "custom_llm_provider": "other"},
        {"api_base": BASE, "client": object()},
    ],
)
def test_direct_client_does_not_borrow_other_routes(settings):
    registry.MODELS["local"] = entry()
    assert CompletionClient(MODEL, **settings).context_window is None


def test_responses_does_not_borrow_completion_window():
    registry.MODELS["local"] = entry()
    assert ResponsesClient(MODEL, api_base=BASE).context_window is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "openai/replacement"},
        {"api_base": "https://other.test/v1"},
        {"base_url": "https://other.test/v1"},
        {"client_type": "responses"},
        {"custom_llm_provider": "other"},
        {"api_key": "replacement"},
        {"client": object()},
    ],
)
def test_factory_override_clears_old_context(overrides):
    registry.MODELS["local"] = entry()
    c = registry.get_llm_client("local", **overrides)
    assert c.context_window is None
    explicit = registry.get_llm_client("local", context_window=16384, **overrides)
    assert explicit.context_window == 16384


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_base": "https://other.test/v1"},
        {"base_url": "https://other.test/v1"},
        {"custom_llm_provider": "other"},
        {"api_key": "replacement"},
        {"client": object()},
    ],
)
def test_per_call_route_does_not_inherit_window(overrides):
    c = CompletionClient(MODEL, api_base=BASE, context_window=8192)
    assert c.get_context_limits(overrides).context_window is None
    assert c.get_context_limits({**overrides, "context_window": 16384}).context_window == 16384


@pytest.mark.parametrize("client_type", ["completion", "responses"])
def test_typed_factory_options_and_explicit_overrides(client_type):
    raw = entry(
        client_type=client_type,
        timeout=12.5,
        chat_template_kwargs={"enable_thinking": False},
        reasoning_levels={"on": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    before = deepcopy(raw)
    typed = ModelConfig.from_registry("local", raw)
    c = registry.client_from_config("local", typed.model_dump(exclude_unset=True))
    assert c.config["timeout"] == 12.5
    assert c._prepare_call_config({})["chat_template_kwargs"] == {"enable_thinking": False}
    assert c._prepare_call_config({"reasoning_level": "on"})["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    override = registry.client_from_config(
        "local", raw, timeout=3, chat_template_kwargs={"caller": True}
    )
    assert override.config["timeout"] == 3
    assert override._prepare_call_config({})["chat_template_kwargs"] == {"caller": True}
    assert raw == before


def test_compatible_routes_normalize_endpoint_spellings():
    registry.MODELS["other"] = entry(api_key_env="OTHER_SCOPE")
    registry.MODELS["local"] = entry()
    c = CompletionClient(MODEL, base_url="HTTPS://GATEWAY.TEST/v1/")
    assert c.context_window == 8192
    registry.MODELS["local"]["context_window"] = 12345
    assert c.context_window == 12345  # No stale per-client cache.
    registry.MODELS["duplicate"] = entry()
    assert c.context_window is None  # Conflicting compatible declarations.


def test_alias_key_collision_does_not_prove_model_identity():
    registry.MODELS[MODEL] = {**entry(), "model_name": "openai/different"}
    assert CompletionClient(MODEL, api_base=BASE).context_window is None


def test_credential_scope_is_not_guessed_from_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_SCOPE", "same-value")
    registry.MODELS["local"] = entry(api_key_env="LOCAL_SCOPE")
    assert CompletionClient(MODEL, api_base=BASE).context_window is None
    assert CompletionClient(MODEL, api_base=BASE, api_key="same-value").context_window is None
    c = registry.get_llm_client("local")
    assert c.context_window == 8192
    assert c.config["api_key"] == "same-value"
    # Explicit credentials change declaration scope, even with equal bytes.
    assert registry.get_llm_client("local", api_key="same-value").context_window is None


def test_metadata_fallback_remains_distinct(monkeypatch):
    registry.MODELS["local"] = entry()
    monkeypatch.setattr(CompletionClient, "get_model_info", lambda self: {"max_input_tokens": 4096})
    assert CompletionClient(MODEL, api_base=BASE).context_window == 8192
    assert CompletionClient(MODEL).context_window == 4096
    assert registry.get_llm_client("local", api_base="https://other.test/v1").context_window == 4096


def test_attached_metadata_tracks_actual_route():
    c = registry.client_from_config("local", entry())
    assert c.context_window == 8192
    c.config["api_base"] = "https://other.test/v1"
    assert c.context_window is None


def test_equivalent_factory_override_retains_context():
    c = registry.client_from_config("local", entry(), api_base=BASE + "/")
    assert c.context_window == 8192
    assert c.get_context_limits({"api_base": BASE}).context_window == 8192


def test_direct_lookup_still_lazily_loads(monkeypatch):
    calls = []

    def load():
        calls.append(True)
        with registry._registry_lock:
            registry.MODELS["local"] = entry()

    monkeypatch.setattr(registry, "ensure_loaded", load)
    assert CompletionClient(MODEL, api_base=BASE).context_window == 8192
    assert calls == [True]


@pytest.mark.parametrize("is_async", [False, True])
async def test_template_defaults_levels_and_caller_overrides_on_mock_wire(monkeypatch, is_async):
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


@pytest.mark.parametrize("field", ["api_style", "transport", "replay_vendor"])
def test_route_metadata_is_not_blindly_forwarded(field):
    raw = entry(**{field: "not-a-request-parameter"})
    typed = ModelConfig.from_registry("local", raw)
    assert typed.model_dump(exclude_unset=True)[field] == "not-a-request-parameter"
    c = registry.client_from_config("local", raw)
    assert field not in c.config


def test_concurrent_direct_context_lookup_during_reload(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    paths = []
    for window in (8192, 16384):
        path = tmp_path / f"{window}.yaml"
        path.write_text(
            f"models:\n  local:\n    model_name: {MODEL}\n"
            f"    api_base: {BASE}\n    context_window: {window}\n"
        )
        paths.append(path)
    registry.reload_registry(paths[0])
    c = CompletionClient(MODEL, api_base=BASE)
    start = threading.Barrier(3)

    def read():
        start.wait(timeout=10)
        return [c.context_window for _ in range(100)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        readers = [pool.submit(read) for _ in range(2)]
        start.wait(timeout=10)
        for index in range(40):
            registry.reload_registry(paths[index % 2])
        for future in readers:
            assert set(future.result(timeout=10)) <= {8192, 16384}
