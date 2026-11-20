# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D1 only: optional load checks, supplied entries and leased settings views.

Native/legacy HTTP assertions use actual SDK parsers over MockTransport. No
runtime factory is migrated by these foundation tests.
"""

import asyncio
import builtins
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields, replace
from threading import Barrier
from types import MappingProxyType

import httpx
import pytest

from nooa.unifiedllm import LLM, Need, Requirements, load_llm, load_llm_from_config, registry
from nooa.unifiedllm._accounting import _accounting_receiver
from nooa.unifiedllm.adapters._binding import _UNSET
from nooa.unifiedllm.adapters._loader import _pools
from nooa.unifiedllm.adapters._view_limits import _adapter_limits
from nooa.unifiedllm.adapters.base import Capabilities
from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.limits import ContextLimits, context_limits_for, reduced_reply_params
from nooa.unifiedllm.request import LLMRequest
from nooa.unifiedllm.requirements import _validate_requirements
from tests.unifiedllm.test_native_load_llm import borrowed, dispose, zero_delay
from tests.unifiedllm.test_native_projection import FAMILIES, payload
from tests.unifiedllm.test_native_settings import block_value

ALL = Need.IMAGES | Need.PDFS | Need.AUDIO | Need.VIDEO | Need.TOOLS
REQUEST = LLMRequest(messages=[{"role": "user", "content": "question"}])


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})


def entry(family="openai-chat", sdk=None, **settings):
    return {
        "adapter": family,
        "model_name": "model",
        "api_base": "https://fixture.invalid/v1",
        "api_key": "fixture-key",
        "client": sdk,
        **({"headers": {"x-private-host": "fixture"}} if sdk is not None else {}),
        "max_tokens": 100,
        "retry_config": zero_delay(),
        **settings,
    }


def config_failure(call):
    with pytest.raises(LLMFailure) as caught:
        call()
    assert caught.value.kind == FailureKind.CONFIG
    return str(caught.value)


def test_requirements_shape_and_one_time_injected_comparison():
    assert [(f.name, f.default) for f in fields(Requirements)] == [
        ("flags", Need.NONE),
        ("min_context", None),
    ]
    assert [n.name for n in Need] == ["IMAGES", "PDFS", "AUDIO", "VIDEO", "TOOLS"]
    assert int(ALL) == 31 and int(Need.NONE) == 0
    requirements = Requirements(ALL, min_context=128000)
    with pytest.raises(FrozenInstanceError):
        requirements.flags = Need.NONE
    facts = Capabilities(
        frozenset({"image", "pdf", "audio", "video"}), tools=True, context_window=128000
    )
    _validate_requirements(facts, requirements, label="worker.model")
    _validate_requirements(None)  # optional checks do not invent unknown facts
    for require in (None, Need.NONE, Requirements()):
        _validate_requirements(Capabilities(tools=None), require)
    assert facts.context_window == 128000  # validation retains no send wrapper


@pytest.mark.parametrize("bad", [True, False, 1, 31, -1, "images", {}, [], object()])
def test_wrong_require_rejected_before_http(bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid requirement allocated HTTP")

    monkeypatch.setattr(httpx.Client, "__init__", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "__init__", forbidden)
    config_failure(lambda: load_llm_from_config("slot", entry(), require=bad))
    config_failure(lambda: load_llm("slot", require=bad))


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "128000", [], {}])
def test_invalid_minimum_and_flags(bad):
    config_failure(lambda: Requirements(min_context=bad))
    config_failure(lambda: Requirements(flags=bad))


@pytest.mark.parametrize("bits", [-1, 32, 33, 64, 1024])
def test_unknown_flag_bits(bits):
    config_failure(lambda: Need(bits))


@pytest.mark.parametrize(
    "need, missing",
    [
        (Need.IMAGES, "image"),
        (Need.PDFS, "pdf"),
        (Need.AUDIO, "audio"),
        (Need.VIDEO, "video"),
        (Need.TOOLS, "tools"),
    ],
)
@pytest.mark.parametrize("tools", [None, False])
def test_missing_unknown_capabilities_safe_labels(need, missing, tools):
    facts = Capabilities(tools=tools)
    text = config_failure(lambda: _validate_requirements(facts, need, label="worker.slot"))
    assert missing in text and "worker.slot" in text
    text = config_failure(
        lambda: _validate_requirements(facts, need, label="https://secret:private@host/path")
    )
    assert missing in text and "secret" not in text and "private" not in text and "host" not in text


@pytest.mark.parametrize("window", [None, 127999, False])
def test_minimum_unknown_or_insufficient(window):
    text = config_failure(
        lambda: _validate_requirements(
            Capabilities(context_window=window), Requirements(min_context=128000)
        )
    )
    assert "context" in text


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("required", [False, True])
async def test_public_requirements_and_optional_unknown(family, required):
    seen = []
    sdk, http = borrowed(
        family, "async", lambda r: seen.append(r) or httpx.Response(200, json=payload(family))
    )
    media = ["image"] if family == "openai-chat" else ["image", "pdf"]
    flags = Need.IMAGES | Need.TOOLS | (Need.NONE if family == "openai-chat" else Need.PDFS)
    require = Requirements(flags, min_context=128000) if required else None
    facts = (
        {"media_inputs": media, "tools_supported": True, "context_window": 128000}
        if required
        else {}
    )
    config_failure(lambda: load_llm_from_config("slot", entry(family, sdk, **facts), require=ALL))
    llm = load_llm_from_config("slot", entry(family, sdk, **facts), require=require)
    assert not seen and llm.adapter.capabilities.max_output_tokens is None
    await llm.acall(REQUEST)
    await llm.aclose()
    assert len(seen) == 1 and not http.is_closed and sdk.max_retries == 9
    await sdk.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("require", [Need.IMAGES, Need.TOOLS, Requirements(min_context=128000)])
def test_public_requirement_failure_before_transport_allocation(family, require, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Missing facts allocated HTTP")

    monkeypatch.setattr(httpx.Client, "__init__", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "__init__", forbidden)
    before = dict(_pools)
    config_failure(lambda: load_llm_from_config("slot", entry(family), require=require))
    assert dict(_pools) == before


@pytest.mark.parametrize("family", FAMILIES)
async def test_supplied_entry_detached_pinned_and_global_registry_untouched(family, monkeypatch):
    seen = []
    sdk, http = borrowed(
        family,
        "async",
        lambda r: seen.append(json.loads(r.content)) or httpx.Response(200, json=payload(family)),
    )
    config = entry(
        family,
        sdk,
        temperature=0.2,
        extra_body={"host_knob": {"items": [1]}},
        reasoning_levels={"low": {"temperature": 0.3}},
    )
    registry.MODELS["slot"] = {"adapter": "decision", "private": [1]}
    original = copy.deepcopy(registry.MODELS)

    def forbidden():
        pytest.fail("Supplied entry touched ensure_loaded")

    monkeypatch.setattr(registry, "ensure_loaded", forbidden)
    first = load_llm_from_config("slot", MappingProxyType(config), temperature=0.4)
    config["model_name"] = "changed"
    config["extra_body"]["host_knob"]["items"].append(9)
    config["reasoning_levels"]["low"]["temperature"] = 0.9
    second = load_llm(first, reasoning_level="low", max_output_tokens=40)
    assert registry.MODELS == original and not seen
    assert first._lease is second._lease and first._lease.leases == 2
    assert first.adapter.capabilities is second.adapter.capabilities
    assert first.adapter._route_calibration_record is second.adapter._route_calibration_record
    assert first.adapter._settings.route is second.adapter._settings.route
    await first.aclose()
    await second.acall(REQUEST)
    assert seen[0]["model"] == "model" and seen[0]["temperature"] == 0.3
    assert seen[0]["host_knob"] == {"items": [1]}
    await second.aclose()
    assert not http.is_closed
    await sdk.close()


def test_supplied_concurrent_success_failure_no_global_mutation():
    registry.MODELS["global"] = {"private": [1]}
    original = copy.deepcopy(registry.MODELS)
    barrier = Barrier(6)

    def worker(i):
        barrier.wait()
        if i % 2:
            config_failure(lambda: load_llm_from_config("slot", entry(), require=Need.IMAGES))
        else:
            llm = load_llm_from_config("slot", entry())
            llm.close()
        assert registry.MODELS == original

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(worker, range(6)))
    assert registry.MODELS == original


@pytest.mark.parametrize("adapter", ["decision", "made-up"])
def test_supplied_rejects_non_llm_family(adapter):
    with pytest.raises(ValueError):
        load_llm_from_config("slot", entry(adapter))
    with pytest.raises(TypeError):
        load_llm_from_config("slot", entry(client_type="decision"))


@pytest.mark.parametrize("family", FAMILIES)
async def test_selector_override_drops_stale_facts_and_preserves_selected_defaults(family):
    sdk, http = borrowed(family, "async", lambda r: httpx.Response(200, json=payload(family)))
    declared = entry(
        "litellm",
        sdk,
        context_window=128000,
        media_inputs=["image"],
        tools_supported=True,
        temperature=0.2,
    )
    config_failure(
        lambda: load_llm_from_config("slot", declared, adapter=family, require=Need.IMAGES)
    )
    llm = load_llm_from_config("slot", declared, adapter=family, temperature=0.4)
    assert llm.adapter._settings.selected["temperature"] == 0.4
    assert llm.adapter.capabilities.context_window is None
    assert not llm.adapter.capabilities.media_kinds
    assert llm.adapter.capabilities.tools is None
    await llm.aclose()
    assert not http.is_closed
    await sdk.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_settings_public_views_actual_sdk_caps_timeout_and_borrowed_ownership(family, mode):
    seen, snapshots = [], []

    def handler(request):
        seen.append((json.loads(request.content), request.extensions["timeout"]))
        return httpx.Response(200, json=payload(family))

    sdk, http = borrowed(family, mode, handler)
    source = load_llm_from_config(
        "slot", entry(family, sdk, temperature=0.2, timeout=4, context_window=1000)
    )
    registry.MODELS["slot"] = entry("anthropic-messages", model_name="changed")
    derived = load_llm(
        source,
        temperature=0,
        top_p=None,
        max_output_tokens=50,
        timeout=2,
        require=Requirements(min_context=1000),
    )
    inherited = load_llm(source, temperature=_UNSET, timeout=_UNSET)
    assert source._lease is derived._lease is inherited._lease
    assert source._lease.leases == 3 and not seen
    assert source.adapter.capabilities is derived.adapter.capabilities
    assert source.adapter._route_calibration_record is derived.adapter._route_calibration_record
    assert source._retry_config is derived._retry_config
    assert source._media_policy == derived._media_policy
    clone = derived.adapter._transport._sdk(mode)
    assert clone.max_retries == 0 and clone._client is http and sdk.max_retries == 9
    with _accounting_receiver(snapshots.append):
        derived.call(REQUEST) if mode == "sync" else await derived.acall(REQUEST)
        inherited.call(REQUEST) if mode == "sync" else await inherited.acall(REQUEST)
    key = {
        "openai-chat": "max_completion_tokens",
        "openai-responses": "max_output_tokens",
        "anthropic-messages": "max_tokens",
    }[family]
    assert seen[0][0][key] == 50 and seen[0][0]["temperature"] == 0
    assert "top_p" not in seen[0][0] and seen[0][1]["read"] == 2
    assert seen[1][0][key] == 100 and seen[1][0]["temperature"] == 0.2
    assert seen[1][1]["read"] == 4
    assert len(snapshots) == 2 and all(s.attempt_count == 1 for s in snapshots)
    assert context_limits_for(derived) == ContextLimits(1000, 50)
    await source.aclose()
    await inherited.aclose()
    assert not derived._lease.retired
    await derived.aclose()
    assert derived._lease.retired and not http.is_closed
    await dispose(sdk, mode)


@pytest.mark.parametrize("family", FAMILIES)
async def test_reasoning_selected_limits_blocks_and_none(family):
    seen = []
    sdk, http = borrowed(
        family,
        "async",
        lambda r: seen.append(json.loads(r.content)) or httpx.Response(200, json=payload(family)),
    )
    block = {
        "openai-chat": "prediction",
        "openai-responses": "reasoning",
        "anthropic-messages": "thinking",
    }[family]
    config = entry(
        family,
        sdk,
        context_window=1000,
        **{
            block: block_value(family, "medium"),
            "reasoning_levels": {
                "low": {
                    block: block_value(family, "low"),
                    "max_tokens": 30,
                    **({"top_p": 0.2} if family != "anthropic-messages" else {}),
                },
                "high": {block: block_value(family, "high"), "max_tokens": 60},
            },
            "reasoning_level": "low",
            "temperature": 1 if family == "anthropic-messages" else 0.1,
        },
    )
    first = load_llm_from_config("slot", config)
    second = load_llm(first, reasoning_level="high")
    disabled = load_llm(second, reasoning_level=None, temperature=None, timeout=None)
    assert context_limits_for(first).reserved_output_tokens == 30
    assert context_limits_for(second).reserved_output_tokens == 60
    assert context_limits_for(disabled).reserved_output_tokens == 100
    config_failure(lambda: load_llm(first, reasoning_level="high", **{block: None}))
    for llm in (first, second, disabled):
        await llm.acall(REQUEST)
        await llm.aclose()
    assert seen[0][block] == block_value(family, "low")
    assert seen[1][block] == block_value(family, "high") and "top_p" not in seen[1]
    assert seen[2][block] == block_value(family, "medium") and "temperature" not in seen[2]
    assert not http.is_closed
    await sdk.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens", "max_output_tokens"])
@pytest.mark.parametrize("nested", [False, True])
async def test_cap_aliases_limits_and_known_window_validation(family, key, nested):
    cap = {"extra_body": {key: 50}} if nested else {key: 50}
    config = entry(family, context_window=1000)
    config.pop("max_tokens")
    if nested:
        config_failure(lambda: load_llm_from_config("slot", {**config, **cap}))
        return  # Native recognized controls must remain top-level (C contract).
    llm = load_llm_from_config("slot", {**config, **cap})
    assert context_limits_for(llm) == ContextLimits(1000, 50)
    assert llm.adapter.capabilities.max_output_tokens is None
    assert context_limits_for(llm).usable_input_tokens == 950
    config_failure(lambda: load_llm(llm, max_output_tokens=1000))
    config_failure(lambda: load_llm(llm, max_output_tokens=1001))
    assert llm._lease.leases == 1 and not llm._lease.retired
    await llm.aclose()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cap", [1000, 1001])
def test_known_window_bad_cap_early_no_http(family, cap, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid budget allocated HTTP")

    monkeypatch.setattr(httpx.Client, "__init__", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "__init__", forbidden)
    config_failure(
        lambda: load_llm_from_config("slot", entry(family, context_window=1000, max_tokens=cap))
    )


@pytest.mark.parametrize("family", FAMILIES)
async def test_unknown_window_no_false_estimate_clamp_or_selected_ceiling(family):
    llm = load_llm_from_config("slot", entry(family, max_tokens=1000000))
    assert context_limits_for(llm) == ContextLimits(None, 1000000)
    assert context_limits_for(llm).usable_input_tokens is None
    # C's projection estimate remains partial; it is not a context fact.
    assert llm.adapter._prepare(REQUEST, None, None)[0]
    assert llm.adapter.capabilities.context_window is None
    assert llm.adapter.capabilities.max_output_tokens is None
    config_failure(lambda: load_llm(llm, require=Requirements(min_context=1)))
    await llm.aclose()


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses"])
async def test_none_cap_labeled_fallback_not_advertised_ceiling(family):
    llm = load_llm_from_config("slot", entry(family, context_window=1000))
    derived = load_llm(llm, max_output_tokens=None)
    assert context_limits_for(derived, fallback_reserve=2000) == ContextLimits(1000, 2000, True)
    assert derived.adapter.capabilities.max_output_tokens is None
    config_failure(lambda: context_limits_for(derived, {"max_tokens": 2}))
    config_failure(lambda: context_limits_for(derived, {"api_key": "private"}))
    config_failure(lambda: context_limits_for(derived, fallback_reserve=True))
    assert context_limits_for(derived, {}) == ContextLimits(1000, 0, True)
    await llm.aclose()
    await derived.aclose()


@pytest.mark.parametrize(
    "override",
    [
        {"model": "private"},
        {"model_name": "private"},
        {"adapter": "private"},
        {"client_type": "private"},
        {"client": object()},
        {"base_url": "private"},
        {"api_key": "private"},
        {"context_window": 20},
        {"retry_config": None},
        {"extra_body": {"Authorization": "private"}},
        {"extra_headers": {"x-api-key": "private"}},
        {"timeout": 0},
    ],
)
async def test_derivation_identity_invalid_settings_safe_and_live_pool_reusable(override):
    source = load_llm_from_config("slot", entry())
    before = dict(_pools)
    text = config_failure(lambda: load_llm(source, **override))
    assert "private" not in text and "fixture-key" not in text
    assert source._lease.leases == 1 and not source._closed and not source._lease.retired
    assert dict(_pools) == before
    valid = load_llm(source, temperature=0)
    await source.aclose()
    await valid.aclose()


async def test_wrong_direct_closed_custom_views_rejected_without_bypass(monkeypatch):
    source = load_llm_from_config("slot", entry())
    direct = LLM(source.adapter)
    config_failure(lambda: load_llm(direct))
    config_failure(lambda: context_limits_for(direct))

    class Custom(LLM):
        pass

    custom = Custom(source.adapter)
    config_failure(lambda: load_llm(custom))
    config_failure(lambda: context_limits_for(custom))
    for object_, name in ((source, "acall"), (source.adapter, "send")):
        with monkeypatch.context() as patch:
            patch.setattr(object_, name, lambda *args: None)
            config_failure(lambda: load_llm(source))
    await source.aclose()
    config_failure(lambda: load_llm(source))
    config_failure(lambda: context_limits_for(source))


async def test_same_source_close_vs_derive_thread_race():
    for _ in range(12):
        source = load_llm_from_config("slot", entry())
        barrier = Barrier(2)

        def derive(source=source, barrier=barrier):
            barrier.wait()
            try:
                return load_llm(source, temperature=0)
            except LLMFailure as failure:
                assert failure.kind == FailureKind.CONFIG
                return None

        def close(source=source, barrier=barrier):
            barrier.wait()
            source.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            new = executor.submit(derive)
            old = executor.submit(close)
            view = new.result()
            old.result()
        if view is not None:
            assert view._lease.leases == 1 and not view._lease.retired
            await view.aclose()
        await source.aclose()
        assert source._lease.retired and source._lease.leases == 0


async def test_derivation_and_limits_native_no_legacy_imports_or_calls(monkeypatch):
    original = builtins.__import__
    calls = []

    def guard(name, *args, **kwargs):
        if (
            name == "litellm"
            or name.startswith("litellm.")
            or name
            in {
                "nooa.unifiedllm.unifiedllm",
                "nooa.unifiedllm.adapters.legacy",
                "nooa.unifiedllm.adapters.litellm_chat",
                "nooa.unifiedllm.adapters.litellm_responses",
            }
        ):
            pytest.fail("Native foundation imported legacy")
        return original(name, *args, **kwargs)

    def profile(frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename.replace("\\", "/")
            if "/litellm/" in filename or filename.endswith(
                (
                    "/unifiedllm/unifiedllm.py",
                    "/adapters/legacy.py",
                    "/adapters/litellm_chat.py",
                    "/adapters/litellm_responses.py",
                )
            ):
                calls.append((filename, frame.f_code.co_name))

    monkeypatch.setattr(builtins, "__import__", guard)
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        source = load_llm_from_config("slot", entry())
        view = load_llm(source, max_output_tokens=50)
        assert context_limits_for(view) == ContextLimits(None, 50)
        await source.aclose()
        await view.aclose()
    finally:
        sys.setprofile(previous)
    assert not calls


def test_custom_legacy_limits_compatibility_unchanged():
    class Custom:
        context_window = 1000

    custom = Custom()
    assert context_limits_for(custom, {"max_tokens": 50}) == ContextLimits(1000, 50)
    assert reduced_reply_params(custom, {"max_tokens": 50, "temperature": 0}, 20) == {
        "max_tokens": 20,
        "temperature": 0,
    }


@pytest.mark.parametrize("client_type", ["completion", "responses"])
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_legacy_loaded_views_actual_sdk_effective_caps_and_pinned_route(
    client_type, mode, monkeypatch
):
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    seen = []
    family = "openai-chat" if client_type == "completion" else "openai-responses"
    transport = httpx.MockTransport(
        lambda r: seen.append(json.loads(r.content)) or httpx.Response(200, json=payload(family))
    )
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    config = entry(
        "litellm",
        client_type=client_type,
        model_name="openai/gpt-4o-mini",
        context_window=1000,
        temperature=0.2,
        reasoning_levels={"low": {"max_tokens": 30}},
    )
    source = load_llm_from_config(
        "slot", config, api_key="fixture-key", context_window=1000, retry_config=zero_delay()
    )
    config["api_base"] = "https://changed.invalid"
    registry.MODELS["slot"] = config
    view = load_llm(source, reasoning_level="low", temperature=0, timeout=2)
    assert view._lease is source._lease and view._lease.leases == 2
    assert view.adapter.capabilities is source.adapter.capabilities
    assert context_limits_for(view) == ContextLimits(1000, 30)
    optimistic = load_llm(source, require=Need.TOOLS)  # Advertised True, not certification.
    assert optimistic.capabilities.tools is True and optimistic._lease is source._lease
    await optimistic.aclose()
    assert source._lease.leases == 2
    await source.aclose()
    view.call(REQUEST) if mode == "sync" else await view.acall(REQUEST)
    key = "max_tokens" if client_type == "completion" else "max_output_tokens"
    assert seen[0][key] == 30 and seen[0]["temperature"] == 0
    assert not view._lease.retired
    await view.aclose()
    assert view._lease.client._http.httpx_sync.is_closed
    assert view._lease.client._http.httpx_async.is_closed


@pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens", "max_output_tokens"])
async def test_legacy_nested_cap_effective_limits(key, monkeypatch):
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(
            lambda: {
                "transport": httpx.MockTransport(
                    lambda r: httpx.Response(200, json=payload("openai-chat"))
                )
            }
        ),
    )
    config = entry(
        "litellm", model_name="openai/gpt-4o-mini", context_window=1000, extra_body={key: 40}
    )
    config.pop("max_tokens")
    source = load_llm_from_config("slot", config)
    assert context_limits_for(source) == ContextLimits(1000, 40)
    await source.aclose()


@pytest.mark.parametrize("cancel_failure", [False, True])
async def test_legacy_requirement_failure_new_pool_disposed_once_without_publication(
    cancel_failure, monkeypatch
):
    from nooa.unifiedllm import requirements
    from nooa.unifiedllm.adapters import _loader
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    seen, pools, closes = [], [], []
    original_pool = _loader._Pool

    class TrackedPool(original_pool):
        def __init__(self, client):
            super().__init__(client)
            pools.append(self)

    monkeypatch.setattr(_loader, "_Pool", TrackedPool)
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(
            lambda: {
                "transport": httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(500))
            }
        ),
    )
    original_close = _ClientHttp.aclose

    async def tracked_close(http):
        closes.append(http)
        await original_close(http)

    monkeypatch.setattr(_ClientHttp, "aclose", tracked_close)
    before = dict(_pools)
    cancellation = asyncio.CancelledError("sentinel")
    if cancel_failure:

        def fail(*args, **kwargs):
            raise cancellation

        monkeypatch.setattr(requirements, "_validate_requirements", fail)
        with pytest.raises(asyncio.CancelledError) as caught:
            load_llm_from_config("slot", entry("litellm", model_name="openai/gpt-4o-mini"))
        assert caught.value is cancellation
    else:
        config_failure(
            lambda: load_llm_from_config(
                "slot", entry("litellm", model_name="openai/gpt-4o-mini"), require=Need.VIDEO
            )
        )
    assert dict(_pools) == before and len(pools) == 1 and not seen
    await pools[0].join_disposal()
    await pools[0].join_disposal()
    assert len(closes) == 1 and pools[0].retired and pools[0].leases == 0
    assert pools[0].client._http.httpx_sync.is_closed
    assert pools[0].client._http.httpx_async.is_closed


@pytest.mark.parametrize("family", FAMILIES)
async def test_derived_actual_provider_lease_and_cancelled_close_joins_existing_disposal(
    family, monkeypatch
):
    from nooa.unifiedllm.adapters import _native_transport

    entered, release, provider_done, closing, dispose_release = (asyncio.Event() for _ in range(5))
    wire_tasks, closes = [], []

    async def handler(request):
        wire_tasks.append(asyncio.current_task())
        entered.set()
        await release.wait()
        return httpx.Response(200, json=payload(family))

    sdk, http = borrowed(family, "async", handler)
    original_close = _native_transport._NativeTransport.aclose

    async def tracked_close(transport):
        closes.append(transport)
        closing.set()
        await dispose_release.wait()
        await original_close(transport)

    monkeypatch.setattr(_native_transport._NativeTransport, "aclose", tracked_close)
    source = load_llm_from_config("slot", entry(family, sdk))
    view = load_llm(source, max_output_tokens=50)
    pool = view._lease
    task = asyncio.create_task(view.acall(REQUEST))
    await asyncio.wait_for(entered.wait(), 2)
    assert pool.leases == 4  # two views, logical caller and actual SDK task
    await source.aclose()
    await view.aclose()
    assert pool.leases == 2 and not pool.retired
    task.cancel("sentinel")
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("sentinel",)
    assert pool.leases == 1 and not pool.retired and not wire_tasks[0].done()
    wire_tasks[0].add_done_callback(lambda task: provider_done.set())
    release.set()
    await asyncio.wait_for(provider_done.wait(), 2)
    await asyncio.wait_for(closing.wait(), 2)
    assert pool.retired and pool.leases == 0 and len(closes) == 1
    joiner = asyncio.create_task(view.aclose())
    await asyncio.sleep(0)
    joiner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await joiner
    assert not pool._disposal.cancelled()
    dispose_release.set()
    await asyncio.gather(source.aclose(), view.aclose(), view.aclose())
    assert len(closes) == 1 and not http.is_closed and sdk.max_retries == 9
    await sdk.close()


async def test_known_advertised_ceiling_comparison_without_invented_native_schema():
    source = load_llm_from_config("slot", entry(context_window=1000))
    facts = replace(source.adapter.capabilities, max_output_tokens=80)
    config_failure(lambda: _adapter_limits(source.adapter, facts))
    view = load_llm(source, max_output_tokens=80)
    assert _adapter_limits(view.adapter, facts) == ContextLimits(1000, 80)
    await view.aclose()
    await source.aclose()


@pytest.mark.parametrize("require", [None, Need.NONE, Requirements()])
async def test_empty_requirements_public_unknown_text_only(require):
    source = load_llm_from_config("slot", entry(), require=require)
    view = load_llm(source, require=require)
    assert source.adapter.capabilities.context_window is None
    await source.aclose()
    await view.aclose()


async def test_requirements_checked_once_per_public_load_not_per_send(monkeypatch):
    from nooa.unifiedllm import requirements

    checked, sent = [], []
    original = requirements._validate_requirements

    def counted(*args, **kwargs):
        checked.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(requirements, "_validate_requirements", counted)
    sdk, http = borrowed(
        "openai-chat",
        "async",
        lambda r: sent.append(r) or httpx.Response(200, json=payload("openai-chat")),
    )
    source = load_llm_from_config(
        "slot", entry("openai-chat", sdk, media_inputs=["image"]), require=Need.IMAGES
    )
    view = load_llm(source, require=Need.IMAGES, temperature=0)
    assert checked == [source._capabilities, source._capabilities]
    await source.acall(REQUEST)
    await view.acall(REQUEST)
    assert len(sent) == 2 and len(checked) == 2
    await source.aclose()
    await view.aclose()
    assert not http.is_closed
    await sdk.close()


@pytest.mark.parametrize("broken", ["settings", "facts"])
async def test_malformed_loaded_view_errors_safe_and_does_not_acquire(broken):
    source = load_llm_from_config("slot", entry())
    if broken == "settings":
        source.adapter._settings = None
    else:
        source._capabilities = None
    pool = source._lease
    message = config_failure(lambda: load_llm(source, temperature=0))
    assert "fixture" not in message and pool.leases == 1
    config_failure(lambda: context_limits_for(source))
    assert pool.leases == 1
    await source.aclose()


@pytest.mark.parametrize("client_type", ["completion", "responses"])
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_legacy_derived_view_borrowed_sdk_preparation_strict_clone_and_ownership(
    client_type, mode
):
    family = "openai-chat" if client_type == "completion" else "openai-responses"
    seen = []
    sdk, http = borrowed(
        family,
        mode,
        lambda r: seen.append(json.loads(r.content)) or httpx.Response(200, json=payload(family)),
    )
    source = load_llm_from_config(
        "slot",
        entry("litellm", sdk, client_type=client_type, model_name="openai/gpt-4o-mini"),
        api_key="fixture-key",
        client=sdk,
        retry_config=zero_delay(),
    )
    view = load_llm(source, max_output_tokens=25)
    await source.aclose()
    assert view.adapter._client.config["client"] is sdk
    strict_config = (
        view.adapter._settings(REQUEST)[0]
        if client_type == "completion"
        else view.adapter._call_config(REQUEST)
    )
    assert strict_config["client"] is sdk
    prepared = view.adapter.prepare(
        REQUEST,
        strict_config,
        **({"native_output": False} if client_type == "completion" else {}),
        is_async=mode == "async",
    )
    assert prepared.api_params["client"]._client is http
    # Borrowed legacy SDK coverage stops at the real prepared SDK clone. Real
    # legacy wire bodies are exercised separately with loader-owned SDK clients;
    # native borrowed clients exercise complete sync/async provider dispatch.
    assert prepared.api_params["client"].max_retries == 0
    assert prepared.api_params["max_output_tokens"] == 25
    assert not seen and sdk.max_retries == 9
    await view.aclose()
    assert not http.is_closed
    if mode == "sync":
        sdk.close()
    else:
        await sdk.close()


@pytest.mark.parametrize("bad", [True, False, 0.0, 1.0, 2.0, 16.0, 31.0, "1", None])
def test_need_constructor_rejects_cached_noninteger_values(bad):
    config_failure(lambda: Need(bad))


def test_need_integer_members_combinations_and_inversion():
    for value in range(32):
        assert int(Need(value)) == value
        assert Need(Need(value)) is Need(value)
    # Existing strict negative-bit behavior is retained: Python IntFlag's
    # inversion constructs a negative integer and is therefore not accepted.
    for value in (Need.NONE, Need.IMAGES, ALL):
        config_failure(lambda value=value: ~value)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("tools", [None, False])
async def test_native_explicit_tools_unknown_false_denied_without_http(family, tools):
    sdk, http = borrowed(family, "async", lambda r: pytest.fail("No HTTP allowed"))
    source = load_llm_from_config("slot", entry(family, sdk, tools_supported=tools))
    before = dict(_pools)
    try:
        config_failure(
            lambda: load_llm_from_config(
                "slot", entry(family, sdk, tools_supported=tools), require=Need.TOOLS
            )
        )
        leases = source._lease.leases
        config_failure(lambda: load_llm(source, require=Need.TOOLS))
        assert dict(_pools) == before and source._lease.leases == leases
    finally:
        await source.aclose()
        assert not http.is_closed
        await sdk.close()


@pytest.mark.parametrize("client_type", ["completion", "responses"])
async def test_legacy_advertised_tools_alias_supplied_and_derived(client_type, monkeypatch):
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    seen = []
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(
            lambda: {
                "transport": httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(500))
            }
        ),
    )
    config = entry("litellm", client_type=client_type, model_name="openai/gpt-4o-mini")
    registry.MODELS["slot"] = config
    supplied = load_llm_from_config("slot", config, require=Need.TOOLS)
    alias = load_llm("slot", require=Need.TOOLS)
    derived = load_llm(supplied, require=Need.TOOLS, temperature=0)
    try:
        assert all(view.capabilities.tools is True for view in (supplied, alias, derived))
        assert supplied._lease is derived._lease
        assert supplied.capabilities is derived.capabilities
        assert not seen
    finally:
        for view in (supplied, alias, derived):
            await view.aclose()


@pytest.mark.parametrize("route", ["alias", "supplied"])
@pytest.mark.parametrize(
    "invalid", ["fixture-secret-MUST-NOT-PRINT", ["fixture-secret-MUST-NOT-PRINT"]]
)
async def test_public_legacy_invalid_reasoning_safe_and_live_pool_preserved(
    route, invalid, monkeypatch
):
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    seen = []
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(
            lambda: {
                "transport": httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(500))
            }
        ),
    )
    config = entry(
        "litellm", model_name="openai/gpt-4o-mini", reasoning_levels={"low": {"temperature": 0.1}}
    )
    registry.MODELS["slot"] = config
    source = load_llm_from_config("slot", config)
    before, leases = dict(_pools), source._lease.leases
    try:
        call = (
            (lambda: load_llm("slot", reasoning_level=invalid))
            if route == "alias"
            else (lambda: load_llm_from_config("slot", config, reasoning_level=invalid))
        )
        text = config_failure(call)
        assert "fixture-secret" not in text and "low" not in text
        assert dict(_pools) == before and source._lease.leases == leases and not seen
        valid = load_llm(source, reasoning_level="low")
        await valid.aclose()
        assert not source._lease.retired
    finally:
        await source.aclose()


@pytest.mark.parametrize(
    "target,hook",
    [
        ("shell", "_prepare"),
        ("shell", "adapter"),
        ("shell", "capabilities"),
        ("shell", "call"),
        ("shell", "acall"),
        ("shell", "_acquire"),
        ("adapter", "_prepare"),
        ("adapter", "_received"),
        ("adapter", "_send"),
        ("adapter", "_send_sync"),
        ("adapter", "_numeric_participation"),
        ("adapter", "capabilities"),
        ("adapter", "send"),
        ("adapter", "send_sync"),
    ],
)
@pytest.mark.parametrize("scope", ["instance", "class"])
@pytest.mark.parametrize("replacement", [None, "callable"])
@pytest.mark.parametrize("family", FAMILIES)
async def test_complete_builtin_behavior_guard_before_new_lease(
    target, hook, scope, replacement, family, monkeypatch
):
    sdk, http = borrowed(family, "async", lambda r: pytest.fail("No HTTP allowed"))
    source = load_llm_from_config("slot", entry(family, sdk))
    adapter, pool = source.adapter, source._lease
    before, leases = dict(_pools), pool.leases
    obj = source if target == "shell" else adapter
    value = (lambda *args, **kwargs: None) if replacement == "callable" else None
    try:
        with monkeypatch.context() as patch:
            if scope == "instance":
                # Properties are data descriptors: explicit dict injection must
                # still be rejected rather than ignored during reconstruction.
                patch.setitem(vars(obj), hook, value)
            else:
                patch.setattr(type(obj), hook, value)
            config_failure(lambda: load_llm(source, temperature=0.2))
            assert dict(_pools) == before and pool.leases == leases and not pool.retired
        normal = load_llm(source, temperature=0)
        assert normal.capabilities is source.capabilities and normal._lease is pool
        await normal.aclose()
    finally:
        await source.aclose()
        assert not http.is_closed
        await sdk.close()


async def test_actual_sdk_shell_custom_refusal_preserved_not_reconstructed(monkeypatch):
    seen = []
    sdk, http = borrowed(
        "openai-chat",
        "async",
        lambda r: seen.append(r) or httpx.Response(200, json=payload("openai-chat")),
    )
    source = load_llm_from_config("slot", entry("openai-chat", sdk))
    refusal = LLMFailure(FailureKind.MEDIA_REJECTED, "custom policy refusal")

    def refuse(request):
        raise refusal

    try:
        with monkeypatch.context() as patch:
            patch.setitem(vars(source), "_prepare", refuse)
            with pytest.raises(LLMFailure) as caught:
                await source.acall(REQUEST)
            assert caught.value is refusal and not seen
            leases = source._lease.leases
            config_failure(lambda: load_llm(source, temperature=0))
            assert source._lease.leases == leases and not seen
        normal = load_llm(source, temperature=0)
        await normal.acall(REQUEST)
        assert len(seen) == 1
        await normal.aclose()
    finally:
        await source.aclose()
        assert not http.is_closed
        await sdk.close()


@pytest.mark.parametrize("client_type", ["completion", "responses"])
@pytest.mark.parametrize(
    "target,hook",
    [
        ("adapter", "prepare"),
        ("adapter", "parse"),
        ("adapter", "transmit"),
        ("adapter", "transmit_sync"),
        ("adapter", "_send"),
        ("adapter", "_send_sync"),
        ("adapter", "capabilities"),
        ("adapter", "_numeric_participation"),
        ("client", "_prepare_call_config"),
        ("client", "_validate_request_config"),
        ("client", "_convert_tool_to_schema"),
        ("client", "context_window"),
        ("client", "reasoning_levels"),
        ("client", "_prepare_cache_boundary"),
    ],
)
@pytest.mark.parametrize("scope", ["instance", "class"])
@pytest.mark.parametrize("replacement", [None, "callable"])
async def test_legacy_complete_behavior_guard(
    client_type, target, hook, scope, replacement, monkeypatch
):
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    seen = []
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(
            lambda: {
                "transport": httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(500))
            }
        ),
    )
    source = load_llm_from_config(
        "slot", entry("litellm", client_type=client_type, model_name="openai/gpt-4o-mini")
    )
    adapter, pool = source.adapter, source._lease
    before, leases = dict(_pools), pool.leases
    obj = adapter if target == "adapter" else adapter._client
    value = (lambda *args, **kwargs: None) if replacement == "callable" else None
    try:
        with monkeypatch.context() as patch:
            if scope == "instance":
                patch.setitem(vars(obj), hook, value)
            else:
                patch.setattr(type(obj), hook, value)
            config_failure(lambda: load_llm(source, temperature=0.2))
            assert dict(_pools) == before and pool.leases == leases and not seen
        normal = load_llm(source, temperature=0)
        assert normal.capabilities is source.capabilities and normal._lease is pool
        await normal.aclose()
    finally:
        await source.aclose()


@pytest.mark.parametrize("route", ["alias", "supplied"])
async def test_public_legacy_malformed_declaration_safe_before_http(route, monkeypatch):
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    def forbidden(*args, **kwargs):
        pytest.fail("Malformed declaration allocated HTTP")

    monkeypatch.setattr(_ClientHttp, "__init__", forbidden)
    config = entry(
        "litellm",
        model_name="openai/gpt-4o-mini",
        reasoning_levels={"fixture-secret-MUST-NOT-PRINT": {"api_key": "private-key-value"}},
    )
    registry.MODELS["slot"] = config
    before = dict(_pools)
    call = (
        (lambda: load_llm("slot"))
        if route == "alias"
        else (lambda: load_llm_from_config("slot", config))
    )
    text = config_failure(call)
    assert "fixture-secret" not in text and "private-key" not in text and dict(_pools) == before


@pytest.mark.parametrize("family", [*FAMILIES, "litellm"])
@pytest.mark.parametrize(
    "target,slot",
    [
        ("shell", "_adapter"),
        ("shell", "_view_lock"),
        ("adapter", "_settings"),
        ("adapter", "_transport"),
        ("adapter", "_client"),
    ],
)
async def test_replaced_private_state_property_not_executed_before_guard(
    family, target, slot, monkeypatch
):
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    sdk = http = None
    if family == "litellm":
        monkeypatch.setattr(
            _ClientHttp,
            "_httpx_hardening",
            staticmethod(
                lambda: {"transport": httpx.MockTransport(lambda r: pytest.fail("No HTTP allowed"))}
            ),
        )
        config = entry(family, model_name="openai/gpt-4o-mini")
    else:
        sdk, http = borrowed(family, "async", lambda r: pytest.fail("No HTTP allowed"))
        config = entry(family, sdk)
    source = load_llm_from_config("slot", config)
    obj, pool = (source if target == "shell" else source.adapter), source._lease

    def forbidden(instance):
        pytest.fail("Overridden property executed before derivation guard")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(type(obj), slot, property(forbidden), raising=False)
            config_failure(lambda: load_llm(source, temperature=0))
            assert pool.leases == 1 and not pool.retired
    finally:
        await source.aclose()
        if sdk is not None:
            assert not http.is_closed
            await sdk.close()


@pytest.mark.parametrize("route", ["alias", "supplied", "derived"])
@pytest.mark.parametrize("category", ["selection", "conflict", "positive", "ceiling"])
async def test_public_legacy_private_validation_bridge(route, category, monkeypatch):
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    secret = "fixture-secret-MUST-NOT-PRINT"
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(
            lambda: {
                "transport": httpx.MockTransport(
                    lambda r: pytest.fail("Validation must not send HTTP")
                )
            }
        ),
    )
    config = entry(
        "litellm",
        model_name="openai/gpt-4o-mini",
        context_window=1000,
        reasoning_levels={secret: {"reasoning_effort": "low"}},
    )
    registry.MODELS["slot"] = config
    source = load_llm_from_config("slot", config)
    kwargs = {
        "selection": {"reasoning_level": [secret]},
        "conflict": {"reasoning_level": secret, "reasoning_effort": secret},
        "positive": {"max_tokens": secret},
        "ceiling": {"max_tokens": 1000},
    }[category]
    call = {
        "alias": lambda: load_llm("slot", **kwargs),
        "supplied": lambda: load_llm_from_config("slot", config, **kwargs),
        "derived": lambda: load_llm(source, **kwargs),
    }[route]
    before, leases = dict(_pools), source._lease.leases
    try:
        with pytest.raises(LLMFailure) as caught:
            call()
        error = caught.value
        assert isinstance(error, ValueError) and error.kind == FailureKind.CONFIG
        assert error.__cause__ is None
        assert secret not in str(error) and len(str(error)) < 160
        assert dict(_pools) == before and source._lease.leases == leases
        view = load_llm(source, temperature=0)
        await view.aclose()
    finally:
        await source.aclose()


@pytest.mark.parametrize("route", ["alias", "supplied"])
async def test_public_legacy_declaration_bridge_without_allocation(route, monkeypatch):
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    monkeypatch.setattr(_ClientHttp, "__init__", lambda *a, **k: pytest.fail("No allocation"))
    secret = "fixture-secret-MUST-NOT-PRINT"
    config = entry("litellm", reasoning_levels={secret: {"api_key": secret}})
    registry.MODELS["slot"] = config
    call = (
        (lambda: load_llm("slot"))
        if route == "alias"
        else (lambda: load_llm_from_config("slot", config))
    )
    with pytest.raises(LLMFailure) as caught:
        call()
    assert caught.value.kind == FailureKind.CONFIG and isinstance(caught.value, ValueError)
    assert secret not in str(caught.value) and caught.value.__cause__ is None


@pytest.mark.parametrize("stage", ["capabilities", "transport"])
@pytest.mark.parametrize(
    "exception",
    [
        ValueError("sentinel-value"),
        TypeError("sentinel-type"),
        asyncio.CancelledError("sentinel-control"),
    ],
)
async def test_public_legacy_unrelated_failure_identity_and_cleanup(stage, exception, monkeypatch):
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(
            lambda: {"transport": httpx.MockTransport(lambda r: pytest.fail("No HTTP allowed"))}
        ),
    )
    pools = []
    initialize = _ClientHttp.__init__

    def initialize_spy(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        pools.append(self)

    if stage == "transport":
        # Existing constructor cleanup is exercised, not replaced by a fake root.
        def broken(self):
            pools.append(self)
            raise exception

        monkeypatch.setattr(_ClientHttp, "_build_handler_wrappers", broken)
    else:
        monkeypatch.setattr(_ClientHttp, "__init__", initialize_spy)
        construct = registry._construct_llm_client

        def construct_spy(*args):
            client = construct(*args)

            def broken(self):
                raise exception

            monkeypatch.setattr(type(client._adapter), "capabilities", property(broken))
            return client

        monkeypatch.setattr(registry, "_construct_llm_client", construct_spy)
    before = dict(_pools)
    with pytest.raises(type(exception)) as caught:
        load_llm_from_config(
            "slot",
            entry(
                "litellm",
                client_type="responses",
                model_name="openai/gpt-4o-mini",
                api_base=f"https://fixture.invalid/{stage}/{type(exception).__name__}",
            ),
        )
    assert caught.value is exception and dict(_pools) == before
    await asyncio.sleep(0)
    assert len(pools) == 1
    assert pools[0].httpx_sync.is_closed and pools[0].httpx_async.is_closed
