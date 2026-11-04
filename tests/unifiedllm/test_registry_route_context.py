# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline real-client regression tests for endpoint-local alias metadata."""

from copy import deepcopy

import pytest

from nooa.config import ModelConfig
from nooa.unifiedllm import CompletionClient, ResponsesClient, registry

MODEL = "openai/offline-route-context"
BASE = "https://gateway.test/v1"
CHAT_RESPONSE = {
    "id": "test-response",
    "object": "chat.completion",
    "created": 1,
    "model": "model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
}


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(CompletionClient, "get_model_info", lambda self: None)
    monkeypatch.setattr(ResponsesClient, "get_model_info", lambda self: None)


def entry(**settings):
    return {
        "model_name": MODEL,
        "api_base": BASE,
        "context_window": 8192,
        "media_inputs": [],
        "retry_config": False,
        **settings,
    }


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
        base_url=BASE,
        custom_llm_provider="openai",
    )
    before = deepcopy(raw)
    typed = ModelConfig.from_registry("local", raw)
    c = registry.client_from_config("local", typed.model_dump(exclude_unset=True))
    assert c.config["timeout"] == 12.5
    assert c.config["base_url"] == BASE
    assert c.config["custom_llm_provider"] == "openai"
    assert c.context_window == 8192  # Equivalent spelling of the same route.
    override = registry.client_from_config(
        "local", raw, timeout=3, base_url="https://other.test/v1", custom_llm_provider="other"
    )
    assert override.config["timeout"] == 3
    assert override.config["base_url"] == "https://other.test/v1"
    assert override.config["custom_llm_provider"] == "other"
    assert raw == before


def test_compatible_routes_normalize_endpoint_spellings():
    registry.MODELS["other"] = entry(api_base="https://other.test/v1")
    registry.MODELS["local"] = entry()
    c = CompletionClient(MODEL, base_url="HTTPS://GATEWAY.TEST/v1/")
    assert c.context_window == 8192
    registry.MODELS["local"]["context_window"] = 12345
    assert c.context_window == 12345  # No stale per-client cache.
    registry.MODELS["duplicate"] = entry()
    assert c.context_window is None  # Conflicting compatible declarations.


@pytest.mark.parametrize("client_class", [CompletionClient, ResponsesClient])
def test_direct_lookup_ignores_the_unforwarded_endpoint_key(client_class):
    """A completion/Responses entry's registry-only ``endpoint`` is not part of its route."""
    registry.MODELS["local"] = entry(
        endpoint="https://decisions.test/v1/decisions",
        **({"client_type": "responses"} if client_class is ResponsesClient else {}),
    )
    assert client_class(MODEL, api_base=BASE).context_window == 8192


def test_direct_lookup_needs_every_matching_entry_to_declare_the_window():
    """One matching alias without a window makes the registry value unknown."""
    registry.MODELS["local"] = entry()
    registry.MODELS["undeclared"] = {
        key: value for key, value in entry().items() if key != "context_window"
    }
    assert CompletionClient(MODEL, api_base=BASE).context_window is None
    registry.MODELS["undeclared"]["context_window"] = 8192
    assert CompletionClient(MODEL, api_base=BASE).context_window == 8192


def test_alias_key_collision_does_not_prove_model_identity():
    registry.MODELS[MODEL] = {**entry(), "model_name": "openai/different"}
    assert CompletionClient(MODEL, api_base=BASE).context_window is None


def test_direct_lookup_compares_the_entry_resolved_key(monkeypatch):
    """A direct client's explicit key must equal the entry's resolved key."""
    monkeypatch.setenv("LOCAL_SCOPE", "same-value")
    registry.MODELS["local"] = entry(api_key_env="LOCAL_SCOPE")
    # No client key: credentials do not take part.
    assert CompletionClient(MODEL, api_base=BASE).context_window == 8192
    assert CompletionClient(MODEL, api_base=BASE, api_key="same-value").context_window == 8192
    assert CompletionClient(MODEL, api_base=BASE, api_key="different").context_window is None
    registry.MODELS["literal"] = entry(api_key="literal-value", api_base="https://lit.test/v1")
    lit = "https://lit.test/v1"
    assert CompletionClient(MODEL, api_base=lit, api_key="literal-value").context_window == 8192
    assert CompletionClient(MODEL, api_base=lit, api_key="same-value").context_window is None
    assert CompletionClient(MODEL, api_base=lit).context_window == 8192
    monkeypatch.delenv("LOCAL_SCOPE")
    assert CompletionClient(MODEL, api_base=BASE, api_key="same-value").context_window is None
    # Other route parts still have to match.
    assert CompletionClient(MODEL, api_base=BASE, custom_llm_provider="x").context_window is None


def test_endpoint_whitespace_is_not_a_route_change():
    """Surrounding whitespace in an endpoint does not change the route."""
    from nooa.unifiedllm.route_identity import endpoint_changed, normalize_endpoint

    assert normalize_endpoint(BASE + " ") == normalize_endpoint(BASE)
    assert not endpoint_changed({"api_base": BASE}, {"api_base": BASE + " "})
    assert not endpoint_changed({"api_base": BASE}, {"base_url": "\t" + BASE + "/\n"})
    assert endpoint_changed({"api_base": BASE}, {"api_base": "https://other.test/v1 "})


def test_endpoint_or_provider_changed_is_shared_by_reasoning_and_registry():
    from nooa.unifiedllm import reasoning
    from nooa.unifiedllm.route_identity import endpoint_or_provider_changed

    configured = {"api_base": BASE, "custom_llm_provider": "openai"}
    assert not endpoint_or_provider_changed(configured, {"base_url": BASE + "/"})
    assert not endpoint_or_provider_changed(configured, {"custom_llm_provider": "openai"})
    assert endpoint_or_provider_changed(configured, {"custom_llm_provider": "other"})
    assert endpoint_or_provider_changed(configured, {"api_base": "https://other.test/v1"})
    assert not hasattr(reasoning, "_endpoint_changed")


def test_direct_lookup_tolerates_a_malformed_api_key_env():
    """A non-string api_key_env must not crash the lookup; that entry just never matches a key."""
    registry.MODELS["local"] = entry(api_key_env=["NOT", "A", "NAME"])
    assert CompletionClient(MODEL, api_base=BASE, api_key="k").context_window is None
    assert CompletionClient(MODEL, api_base=BASE).context_window == 8192


def test_direct_lookup_treats_an_empty_env_var_as_unset(monkeypatch):
    """As in the registry, an empty api_key_env value is no key, not the empty string."""
    monkeypatch.setenv("LOCAL_SCOPE", "")
    registry.MODELS["local"] = entry(api_key_env="LOCAL_SCOPE")
    assert CompletionClient(MODEL, api_base=BASE, api_key="").context_window is None


def test_registry_client_keeps_its_own_credential_scope(monkeypatch):
    monkeypatch.setenv("LOCAL_SCOPE", "same-value")
    registry.MODELS["local"] = entry(api_key_env="LOCAL_SCOPE")
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


@pytest.mark.parametrize("spelling", ["api_base", "base_url"])
@pytest.mark.parametrize(
    "same", [BASE + "/", "HTTPS://GATEWAY.TEST/v1", "https://Gateway.test/v1/"]
)
def test_equivalent_per_call_endpoint_keeps_media_levels_and_window(spelling, same):
    """Media, reasoning levels and context windows share one endpoint comparison."""
    levels = {"low": {"reasoning_effort": "low"}}
    c = CompletionClient(
        MODEL,
        api_base=BASE,
        media_inputs=["image"],
        reasoning_levels=levels,
        reasoning_level="low",
        context_window=8192,
    )
    params = c._prepare_call_config({spelling: same})
    assert params["reasoning_effort"] == "low"
    assert c._request_media_inputs(params) == ("image",)
    assert c.get_context_limits({spelling: same}).context_window == 8192
    for other in (BASE + "/other", "https://gateway.test/v2"):
        with pytest.raises(ValueError, match="route-specific"):
            c._prepare_call_config({spelling: other})
        changed = c._prepare_call_config({spelling: other, "reasoning_level": None})
        assert c._request_media_inputs(changed) == ()
        assert (
            c.get_context_limits({spelling: other, "reasoning_level": None}).context_window is None
        )


@pytest.mark.parametrize("spelling", ["api_base", "base_url"])
@pytest.mark.parametrize("same", [BASE, BASE + "/", "HTTPS://GATEWAY.TEST/v1"])
def test_factory_same_endpoint_override_keeps_media_and_levels(spelling, same):
    levels = {"low": {"reasoning_effort": "low"}}
    config = entry(media_inputs=["image"], reasoning_levels=levels, reasoning_default="low")
    c = registry.client_from_config("local", deepcopy(config), **{spelling: same})
    assert c.media_inputs == ("image",)
    assert c.reasoning_levels == ("low",)
    assert c._reasoning_config.default == "low"
    other = registry.client_from_config("local", deepcopy(config), **{spelling: BASE + "/x"})
    assert other.media_inputs == ()
    assert other.reasoning_levels is None


@pytest.mark.parametrize("spelling", ["api_base", "base_url"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_endpoint_override_keeps_the_configured_endpoint(spelling, blank):
    """An empty endpoint must not send the gateway key to the provider default."""
    config = entry(media_inputs=["image"], api_key="gw-key")
    c = registry.client_from_config("local", deepcopy(config), **{spelling: blank})
    assert c.config["api_base"] == BASE
    assert c.config.get("base_url") in (None, BASE)
    assert c.media_inputs == ("image",)
    assert c.context_window == 8192
    params = c._prepare_call_config({spelling: blank})
    assert params["api_base"] == BASE
    assert params.get("base_url") in (None, BASE)
    assert c._request_media_inputs(params) == ("image",)
    assert c.get_context_limits({spelling: blank}).context_window == 8192
    levels = {"low": {"reasoning_effort": "low"}}
    d = CompletionClient(MODEL, base_url=BASE, reasoning_levels=levels, reasoning_level="low")
    params = d._prepare_call_config({spelling: blank})
    assert params["reasoning_effort"] == "low"
    assert params["base_url"] == BASE


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
async def test_timeout_defaults_and_caller_overrides_on_mock_wire(monkeypatch, is_async):
    # Actual CompletionClient -> LiteLLM -> OpenAI SDK -> httpx.MockTransport.
    # This is NOT the separate Direct SDK transport.
    import json

    import httpx

    from nooa.unifiedllm.unifiedllm import _ClientHttp

    bodies, paths, timeouts = [], [], []

    def respond(request):
        bodies.append(json.loads(request.content))
        paths.append(request.url.path)
        timeouts.append(request.extensions.get("timeout"))
        return httpx.Response(200, json=CHAT_RESPONSE)

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": transport, "trust_env": False}),
    )
    raw = entry(timeout=12.5, cache_breakpoint=None)
    typed = ModelConfig.from_registry("local", raw)
    registry.MODELS["local"] = typed.model_dump(exclude_unset=True)
    c = registry.get_llm_client("local", api_key="fake")
    original = deepcopy(c.config)
    try:

        async def invoke(**kwargs):
            messages = [{"role": "user", "content": "hello"}]
            return await c.acall(messages, **kwargs) if is_async else c.call(messages, **kwargs)

        await invoke()
        overrides = {"timeout": 3}
        await invoke(**overrides)
        assert [b["messages"][-1]["content"] for b in bodies] == ["hello", "hello"]
        assert paths == ["/v1/chat/completions"] * 2
        assert [t["read"] for t in timeouts] == [12.5, 3]
        assert c.config == original
        assert overrides == {"timeout": 3}
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
