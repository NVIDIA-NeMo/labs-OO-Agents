# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public loader, shared-route pools and immutable settings (offline)."""

import asyncio
import gc
import math
import textwrap
import weakref
from concurrent.futures import ThreadPoolExecutor

import litellm
import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from nooa.unifiedllm import (
    LLM,
    LLMRequest,
    ResponsesClient,
    RetryConfig,
    get_llm_client,
    load_llm,
    registry,
    reload_registry,
)
from nooa.unifiedllm.adapters import _loader
from nooa.unifiedllm.adapters._binding import _UNSET
from nooa.unifiedllm.errors import LLMFailure
from nooa.unifiedllm.unifiedllm import _ClientHttp


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """A clean registry with no user or project entries."""
    for name in ("user", "project"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    monkeypatch.setenv("LOADER_TEST_KEY", "private-test-key")
    monkeypatch.chdir(tmp_path)
    reload_registry()
    yield
    reload_registry()


def _registry(tmp_path, body):
    path = tmp_path / "llm_config.yaml"
    path.write_text(textwrap.dedent(body))
    reload_registry(path)


@pytest.mark.parametrize("adapter", [None, "litellm"])
def test_loads_public_llm(tmp_path, adapter):
    line = f"adapter: {adapter}" if adapter else ""
    _registry(
        tmp_path,
        f"""
        models:
          chat:
            model_name: openai/chat-model
            api_base: https://gw.example.com/v1
            api_key: test
            media_inputs: []
            {line}
          resp:
            model_name: openai/resp-model
            client_type: responses
            api_key: test
            media_inputs: []
            {line}
        """,
    )
    chat = load_llm("chat", temperature=0.1)
    resp = load_llm("resp")
    assert type(chat) is type(resp) is LLM
    assert chat.adapter._client.model == "openai/chat-model"
    assert chat.adapter._bound_settings.config["temperature"] == 0.1
    assert resp.adapter._client.model == "openai/resp-model"
    assert not hasattr(chat, "config") and not hasattr(chat, "model")
    # The adapter field selects the adapter; it is never sent to the provider.
    assert (
        "adapter" not in chat.adapter._client.config
        and "adapter" not in resp.adapter._client.config
    )
    chat.close()
    resp.close()


def test_unknown_adapter_is_rejected(tmp_path):
    _registry(
        tmp_path,
        """
        models:
          odd:
            model_name: openai/x
            adapter: unsupported-family
        """,
    )
    with pytest.raises(ValueError, match="unsupported adapter 'unsupported-family'.*litellm"):
        load_llm("odd")


def test_plain_model_name_passes_through():
    client = load_llm("openai/gpt-4o-mini", api_key="test")
    assert type(client) is LLM and client.adapter._client.model == "openai/gpt-4o-mini"
    client.close()


def test_adapter_override_is_validated_and_never_sent(monkeypatch):
    with pytest.raises(ValueError, match="unsupported adapter 'bogus'.*litellm"):
        load_llm("openai/gpt-4o-mini", api_key="test", adapter="bogus")
    client = load_llm("openai/gpt-4o-mini", api_key="test", adapter="litellm")
    assert type(client) is LLM and "adapter" not in client.adapter._client.config
    client.close()


def test_adapter_override_wins_over_the_registry(tmp_path):
    _registry(
        tmp_path,
        """
        models:
          odd:
            model_name: openai/x
            api_key: test
            media_inputs: []
            adapter: anthropic-messages
          plain:
            model_name: openai/y
            api_key: test
            media_inputs: []
            adapter: litellm
        """,
    )
    client = load_llm("odd", adapter="litellm")
    assert "adapter" not in client.adapter._client.config
    client.close()
    with pytest.raises(ValueError, match="unsupported adapter 'other'"):
        load_llm("plain", adapter="other")


FAMILIES = pytest.mark.parametrize("family", ["completion", "responses"])
MODES = pytest.mark.parametrize("is_async", [False, True])
MESSAGES = [{"role": "system", "content": "instructions"}, {"role": "user", "content": "hi"}]


def _entry(tmp_path, family, **defaults):
    config = {
        "model_name": "openai/gpt-4o-mini",
        "client_type": family,
        "api_base": "https://endpoint.test/v1",
        "api_key_env": "LOADER_TEST_KEY",
        "media_inputs": [],
        "context_window": 100,
        **defaults,
    }
    # Use the actual shared resolver and a one-entry registry fixture.
    registry.MODELS["alias"] = config
    return config


def _reply(family, actual=20):
    if family == "completion":
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "ok"}}],
            usage={"prompt_tokens": actual, "completion_tokens": 1, "total_tokens": actual + 1},
        )
    return litellm.ResponsesAPIResponse(
        id="r1",
        object="response",
        created_at=0,
        status="completed",
        model="gpt-4o-mini",
        output=[
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage={"input_tokens": actual, "output_tokens": 1, "total_tokens": actual + 1},
    )


@pytest.fixture
def wires(monkeypatch):
    """Mock ALL provider entrypoints; unsupported fallback is an assertion, not HTTP."""
    seen = []
    for family in ("completion", "responses"):

        def send(_family=family, **kwargs):
            seen.append(kwargs)
            return _reply(_family)

        async def asend(_family=family, **kwargs):
            await asyncio.sleep(0)
            seen.append(kwargs)
            return _reply(_family)

        monkeypatch.setattr(litellm, family, send)
        monkeypatch.setattr(litellm, "a" + family, asend)
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: 20)
    return seen


async def _call(llm, is_async):
    request = LLMRequest(messages=MESSAGES)
    return await llm.acall(request) if is_async else llm.call(request)


@pytest.mark.parametrize(
    "fields",
    [
        {"reasoning_level": "low"},
        {"max_output_tokens": 50},
        {"temperature": 0.1},
        {"max_tokens": 50},
        {"tool_choice": {"type": "function"}},
    ],
)
def test_request_rejects_removed_unknown_and_native_fields(fields):
    with pytest.raises(ValidationError):
        LLMRequest(**fields)


@FAMILIES
@MODES
async def test_pool_facts_ema_settings_and_final_release(
    tmp_path, family, is_async, monkeypatch, wires
):
    _entry(
        tmp_path,
        family,
        reasoning_levels={
            "low": {"reasoning_effort": "low"},
            "high": {"reasoning_effort": "high"},
        },
    )
    constructions = []
    construct = registry._construct_llm_client

    def spy(*args):
        result = construct(*args)
        constructions.append(result)
        return result

    monkeypatch.setattr(registry, "_construct_llm_client", spy)
    a = load_llm("alias", temperature=0.1, reasoning_level="low", max_output_tokens=90, timeout=3)
    pool = a._lease
    closed = []
    real_close = pool.client.aclose

    async def close():
        closed.append(1)
        await real_close()

    monkeypatch.setattr(pool.client, "aclose", close)

    # The second view must not even consult provider detection or cached facts.
    def forbidden(*args, **kwargs):
        raise AssertionError("view construction must reuse cached facts/pool")

    monkeypatch.setattr(litellm, "get_llm_provider", forbidden)
    b = load_llm("alias", temperature=0.8, reasoning_level="high", max_output_tokens=30, timeout=7)
    assert a._lease is b._lease and len(constructions) == 1
    assert a.capabilities is b.capabilities
    assert a.adapter.capabilities is b.adapter.capabilities
    assert a.adapter._client._http is b.adapter._client._http
    assert a.adapter._route_calibration_record is b.adapter._route_calibration_record
    record = a.adapter._route_calibration_record
    initial_ratio = record.ratio()
    await _call(a, is_async)
    await a.aclose()
    assert closed == [] and not pool.retired
    await _call(b, is_async)
    assert (wires[0]["temperature"], wires[1]["temperature"]) == (0.1, 0.8)
    assert (wires[0]["timeout"], wires[1]["timeout"]) == (3, 7)
    assert (wires[0]["reasoning_effort"], wires[1]["reasoning_effort"]) == ("low", "high")
    # The route is seeded from legacy calibration, which earlier SDK tests may
    # have updated. Assert the conservative budget from that actual baseline.
    room = 100 - math.ceil(20 * max(1.0, initial_ratio))
    assert wires[0]["max_output_tokens"] == min(90, room)
    assert wires[1]["max_output_tokens"] == 30
    assert a.adapter._bound_settings.config["max_output_tokens"] == 90
    weight = (1 - record._alpha) ** 2
    assert record.ratio() == pytest.approx(initial_ratio * weight + (1 - weight))
    assert all(wire["max_retries"] == wire["num_retries"] == 0 for wire in wires)
    if family == "completion":
        assert all(wire["client"].max_retries == 0 for wire in wires)
        assert wires[0]["client"]._client is wires[1]["client"]._client
    await b.aclose()
    await b.aclose()
    assert closed == [1] and pool.retired
    assert pool.client._http.httpx_async.is_closed and pool.client._http.httpx_sync.is_closed
    with pytest.raises(LLMFailure, match="closed"):
        await _call(a, is_async)
    # Closed roots are never resurrected.
    c = load_llm("alias", max_output_tokens=30)
    assert c._lease is not pool and len(constructions) == 2
    await c.aclose()


@FAMILIES
@MODES
@pytest.mark.parametrize("estimate", [100, 101, 10**400])
async def test_no_estimated_reply_room_is_typed_before_transmit(
    tmp_path, family, is_async, estimate, monkeypatch, wires
):
    _entry(tmp_path, family)
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: estimate)
    llm = load_llm("alias", max_output_tokens=20)
    try:
        with pytest.raises(LLMFailure) as caught:
            await _call(llm, is_async)
        assert caught.value.kind == "context_window"
        assert wires == []
    finally:
        await llm.aclose()


@FAMILIES
@MODES
@pytest.mark.parametrize(
    "case", ["unknown_window", "absent_cap", "none_cap", "fallback", "unavailable"]
)
async def test_no_invented_fit_or_default_reply_cap(
    tmp_path, family, is_async, case, monkeypatch, wires
):
    _entry(tmp_path, family, context_window=None if case == "unknown_window" else 100)
    settings = (
        {} if case == "absent_cap" else {"max_output_tokens": None if case == "none_cap" else 90}
    )
    if case in {"fallback", "unavailable"}:

        def counter(**kw):
            if "messages" in kw or case == "unavailable":
                raise ValueError("full projection unavailable")
            return 150

        monkeypatch.setattr(litellm, "token_counter", counter)
    llm = load_llm("alias", **settings)
    try:
        await _call(llm, is_async)
        assert len(wires) == 1
        cap = wires[0].get("max_output_tokens")
        assert cap == (90 if case in {"unknown_window", "fallback", "unavailable"} else None)
        if case == "absent_cap":
            assert "max_output_tokens" not in wires[0]
    finally:
        await llm.aclose()


@FAMILIES
@MODES
async def test_raw_defaults_none_unset_and_frozen_selection(tmp_path, family, is_async, wires):
    defaults = _entry(
        tmp_path,
        family,
        temperature=0.6,
        max_tokens=40,
        reasoning_effort="native",
        reasoning_level="high",
        reasoning_levels={
            "low": {"thinking": {"type": "disabled"}},
            "high": {"reasoning_effort": "high"},
        },
    )
    inherited = load_llm("alias", temperature=_UNSET)
    reset = load_llm("alias", temperature=None, reasoning_level=None, max_output_tokens=None)
    low = load_llm("alias", reasoning_level="low")
    assert inherited._lease is reset._lease is low._lease
    defaults["reasoning_levels"]["high"]["reasoning_effort"] = "mutated"
    try:
        await _call(inherited, is_async)
        await _call(reset, is_async)
        await _call(low, is_async)
        assert wires[0]["reasoning_effort"] == "high" and wires[0]["temperature"] == 0.6
        assert wires[1]["reasoning_effort"] == "native" and wires[1]["temperature"] is None
        assert wires[1]["max_output_tokens"] is None
        assert wires[2]["reasoning_effort"] == "native"
        thinking = wires[2].get("thinking", wires[2].get("extra_body", {}).get("thinking"))
        assert thinking == {"type": "disabled"}
        assert (
            inherited.adapter._bound_settings.reasoning_levels["high"]["reasoning_effort"] == "high"
        )
    finally:
        await inherited.aclose()
        await reset.aclose()
        await low.aclose()


@FAMILIES
async def test_failed_load_closes_created_pool_but_not_live_view(
    tmp_path, family, monkeypatch, wires
):
    _entry(tmp_path, family, reasoning_levels={})
    constructed = []
    construct = registry._construct_llm_client

    def spy(*args):
        client = construct(*args)
        constructed.append(client)
        return client

    monkeypatch.setattr(registry, "_construct_llm_client", spy)
    # Async final disposal from a sync failed load is scheduled in this loop.
    with pytest.raises(ValueError, match="not supported|unknown"):
        load_llm("alias", reasoning_level="low")
    await asyncio.sleep(0)
    assert constructed[0]._http.httpx_sync.is_closed
    assert constructed[0]._http.httpx_async.is_closed
    live = load_llm("alias", max_output_tokens=20)
    with pytest.raises(ValueError, match="positive integer"):
        load_llm("alias", max_output_tokens=-1)
    assert not live._lease.retired
    await _call(live, True)
    await live.aclose()


@FAMILIES
@pytest.mark.parametrize("sdk_type", [OpenAI, AsyncOpenAI])
async def test_supplied_sdk_is_borrowed_and_identifies_pool(tmp_path, family, sdk_type, wires):
    _entry(tmp_path, family)
    sdk = sdk_type(api_key="sdk-private-key", base_url="https://sdk.test/v1", max_retries=9)
    a = load_llm("alias", client=sdk, temperature=0.1)
    b = load_llm("alias", client=sdk, temperature=0.2)
    assert a._lease is b._lease
    await a.aclose()
    await b.aclose()
    assert not sdk.is_closed()
    assert sdk.max_retries == 9
    if sdk_type is OpenAI:
        sdk.close()
    else:
        await sdk.close()


def test_single_entry_snapshot_is_used_for_validation_and_construction(monkeypatch, tmp_path):
    _entry(tmp_path, "completion")
    original = registry._resolve_client_config
    seen = []

    def resolve(name, config, **kwargs):
        seen.append(config)
        # Registry changed after the one snapshot: never lookup this replacement.
        registry.MODELS[name] = {"adapter": "bogus", "model_name": "wrong-model"}
        return original(name, config, **kwargs)

    monkeypatch.setattr(registry, "_resolve_client_config", resolve)
    with load_llm("alias") as llm:
        assert llm.adapter._client.model == "openai/gpt-4o-mini"
        assert len(seen) == 1 and seen[0]["api_key_env"] == "LOADER_TEST_KEY"


def test_legacy_factories_and_decision_rejection_are_unchanged(tmp_path):
    _entry(tmp_path, "responses")
    legacy = get_llm_client("alias", max_retries=7)
    assert type(legacy) is ResponsesClient and legacy.config["max_retries"] == 7
    legacy.close()
    registry.MODELS["dec"] = {"client_type": "decision", "model_name": "x"}
    with pytest.raises(TypeError, match="Decision"):
        load_llm("dec")


def test_concurrent_loads_use_one_pool_and_protected_cache_key(tmp_path, monkeypatch):
    _entry(tmp_path, "completion")
    constructed = []
    construct = registry._construct_llm_client

    def spy(*args):
        client = construct(*args)
        constructed.append(client)
        return client

    monkeypatch.setattr(registry, "_construct_llm_client", spy)
    with ThreadPoolExecutor(max_workers=8) as executor:
        views = list(executor.map(lambda i: load_llm("alias", temperature=i / 10), range(8)))
    assert len(constructed) == 1 and all(v._lease is views[0]._lease for v in views)
    assert all(type(key) is bytes for key in _loader._pools.keys())
    assert "private-test-key" not in repr(_loader._pools)
    assert "endpoint.test" not in repr(_loader._pools)
    pool_ref = weakref.ref(views[0]._lease)
    for view in views:
        view.close()
    del view, views
    gc.collect()
    assert pool_ref() is None  # No permanent global route-root retention.


@FAMILIES
async def test_inflight_call_lease_survives_close_and_cancellation(tmp_path, family, monkeypatch):
    _entry(tmp_path, family)
    started = asyncio.Event()
    release = asyncio.Event()

    async def provider(**kw):
        started.set()
        await release.wait()
        return _reply(family)

    monkeypatch.setattr(litellm, "a" + family, provider)
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: 20)
    llm = load_llm("alias", max_output_tokens=20)
    pool = llm._lease
    task = asyncio.create_task(_call(llm, True))
    await started.wait()
    await llm.aclose()
    assert not pool.retired and not pool.client._http.httpx_async.is_closed
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not pool.retired and not pool.client._http.httpx_async.is_closed
    assert not pool.client._http.httpx_sync.is_closed
    release.set()
    # The shielded provider owns the last lease until its real completion.
    for _ in range(10):
        await asyncio.sleep(0)
        if pool.retired:
            break
    await llm.aclose()  # Join the pool-owned disposal, without a second release.
    assert pool.retired and pool.leases == 0
    assert pool.client._http.httpx_async.is_closed
    assert pool.client._http.httpx_sync.is_closed


@FAMILIES
async def test_failed_capability_read_releases_created_pool(tmp_path, family, monkeypatch, wires):
    _entry(tmp_path, family)
    constructed = []
    construct = registry._construct_llm_client

    def spy(*args):
        client = construct(*args)
        constructed.append(client)

        def broken(self):
            raise ValueError("bad cached route facts")

        monkeypatch.setattr(type(client._adapter), "capabilities", property(broken))
        return client

    monkeypatch.setattr(registry, "_construct_llm_client", spy)
    with pytest.raises(ValueError, match="bad cached route facts"):
        load_llm("alias")
    await asyncio.sleep(0)
    assert constructed[0]._http.httpx_sync.is_closed
    assert constructed[0]._http.httpx_async.is_closed
    assert wires == []


async def test_failed_transport_construction_disposes_partial_pool(tmp_path, monkeypatch, wires):
    _entry(tmp_path, "responses")
    pools = []
    initialize = _ClientHttp.__init__

    def spy(self, *args):
        initialize(self, *args)
        pools.append(self)

    def broken(self):
        raise ValueError("wrapper construction failed")

    monkeypatch.setattr(_ClientHttp, "__init__", spy)
    monkeypatch.setattr(_ClientHttp, "_build_handler_wrappers", broken)
    with pytest.raises(ValueError, match="wrapper construction failed"):
        load_llm("alias")
    await asyncio.sleep(0)
    assert len(pools) == 1
    assert pools[0].httpx_sync.is_closed and pools[0].httpx_async.is_closed
    assert wires == []


@FAMILIES
async def test_retry_policies_are_per_view_not_first_cached_constructor(tmp_path, family, wires):
    _entry(tmp_path, family)
    override = RetryConfig(max_retries=7)
    a = load_llm("alias", retry_config=override)
    b = load_llm("alias")
    try:
        assert a._lease is b._lease
        assert a._retry_config is override
        assert b._retry_config == RetryConfig()
        assert b._retry_config != override
    finally:
        await a.aclose()
        await b.aclose()


@FAMILIES
@MODES
async def test_single_final_projection_conservative_shared_ema_and_cap_aliases(
    tmp_path, family, is_async, monkeypatch, wires
):
    from dataclasses import replace

    from nooa.unifiedllm.adapters.litellm_chat import LiteLLMChatAdapter
    from nooa.unifiedllm.adapters.litellm_responses import LiteLLMResponsesAdapter

    _entry(tmp_path, family)
    adapter_type = LiteLLMChatAdapter if family == "completion" else LiteLLMResponsesAdapter
    facts = adapter_type.capabilities.fget
    monkeypatch.setattr(
        adapter_type,
        "capabilities",
        property(lambda self: replace(facts(self), max_output_tokens=50)),
    )
    with pytest.raises(ValueError, match="known limit"):
        load_llm("alias", max_output_tokens=51)
    await asyncio.sleep(0)
    llm = load_llm("alias", max_output_tokens=45)
    record = llm.adapter._route_calibration_record
    record._ratio = 3
    prepare = adapter_type.prepare
    prepared_calls = []
    estimates = []

    def final_prepare(self, *args, **kwargs):
        prepared = prepare(self, *args, **kwargs)
        # Exercise conflicting cap aliases introduced by a final wire hook.
        prepared.api_params["max_completion_tokens"] = 45
        prepared.api_params.setdefault("extra_body", {})["max_tokens"] = 45
        prepared_calls.append(prepared)
        return prepared

    def counter(**kwargs):
        prepared = prepared_calls[-1]
        projected = (
            prepared.prepared_messages if family == "completion" else prepared.input_messages
        )
        # Estimation happens after the sole prepare, including lifted instructions.
        expected = projected
        instructions = prepared.api_params.get("instructions")
        if instructions:
            expected = [{"role": "system", "content": instructions}, *projected]
        from nooa.unifiedllm.unifiedllm import _token_counter_messages

        assert kwargs["messages"] == _token_counter_messages(expected)
        estimates.append(kwargs)
        return 20

    monkeypatch.setattr(adapter_type, "prepare", final_prepare)
    monkeypatch.setattr(litellm, "token_counter", counter)
    try:
        await _call(llm, is_async)
        assert len(prepared_calls) == len(wires) == len(estimates) == 1
        # The owned actual SDK entry adds ONLY its private logging carrier and
        # opaque call ID after the sole final projection. Neither belongs to the
        # prepared provider payload or HTTP body; all routing/cap kwargs match.
        sdk_carriers = {"litellm_call_id", "litellm_logging_obj"}
        assert sdk_carriers.isdisjoint(prepared_calls[0].api_params)
        assert sdk_carriers <= wires[0].keys()
        from uuid import UUID

        from litellm.litellm_core_utils.litellm_logging import Logging

        assert isinstance(wires[0]["litellm_logging_obj"], Logging)
        assert UUID(wires[0]["litellm_call_id"])
        assert (
            wires[0]["litellm_logging_obj"].model_call_details["_nooa_logical_call_owned"] is True
        )
        assert {
            key: value for key, value in wires[0].items() if key not in sdk_carriers
        } == prepared_calls[0].api_params
        caps = [
            layer[key]
            for layer in (wires[0], wires[0]["extra_body"])
            for key in ("max_tokens", "max_output_tokens", "max_completion_tokens")
            if key in layer
        ]
        assert caps and set(caps) == {40}  # min(R=45, output ceiling=50, W=100 - ceil(3*20))
        assert llm.adapter._bound_settings.config["max_output_tokens"] == 45
        assert MESSAGES[0]["content"] == "instructions"
    finally:
        await llm.aclose()


@FAMILIES
@MODES
async def test_scalar_sampling_views_share_pool(tmp_path, family, is_async, wires):
    _entry(tmp_path, family)
    a = load_llm("alias", seed=1, frequency_penalty=0.1, presence_penalty=0.2)
    b = load_llm("alias", seed=2, frequency_penalty=0.3, presence_penalty=None)
    try:
        assert a._lease is b._lease
        await _call(a, is_async)
        await _call(b, is_async)
        assert wires[0]["seed"] == 1 and wires[1]["seed"] == 2
        assert wires[0]["frequency_penalty"] == 0.1 and wires[1]["frequency_penalty"] == 0.3
        assert wires[0]["presence_penalty"] == 0.2 and wires[1]["presence_penalty"] is None
    finally:
        await a.aclose()
        await b.aclose()


def test_opaque_sdk_association_is_weak_and_identity_based():
    from nooa.unifiedllm.adapters._calibration import _sdk_identity, _sdk_ids

    class SDK:
        def __eq__(self, other):
            return True

    a, b = SDK(), SDK()
    identity = id(a)
    key = _sdk_identity(a)
    assert _sdk_identity(a) == key
    assert _sdk_identity(b) != key
    reference = weakref.ref(a)
    del a
    gc.collect()
    assert reference() is None and identity not in _sdk_ids


@FAMILIES
@MODES
@pytest.mark.parametrize("native", [False, True])
async def test_nested_cap_alias_views_share_pool_and_preserve_native_defaults(
    tmp_path, family, is_async, native, wires
):
    _entry(tmp_path, family)
    extra = {"native_example": {"enabled": True}} if native else {}
    a = load_llm("alias", extra_body={**extra, "max_tokens": 40})
    b = load_llm("alias", extra_body={**extra, "max_output_tokens": 20})
    c = load_llm("alias", max_tokens=30, **({"extra_body": extra} if extra else {}))
    try:
        assert a._lease is b._lease is c._lease
        assert (
            a.adapter._route_calibration_record
            is b.adapter._route_calibration_record
            is c.adapter._route_calibration_record
        )
        for view in (a, b, c):
            await _call(view, is_async)
        from nooa.unifiedllm.limits import REPLY_CAP_KEYS

        for wire, wanted in zip(wires, (40, 20, 30), strict=True):
            caps = [
                layer[key]
                for layer in (wire, wire.get("extra_body", {}))
                for key in REPLY_CAP_KEYS
                if key in layer
            ]
            assert caps == [wanted]
            if native:
                assert wire["extra_body"]["native_example"] == {"enabled": True}
    finally:
        await a.aclose()
        await b.aclose()
        await c.aclose()


def _reply_cap(config):
    # Both families' existing spelling normalization remains supported.
    return next(
        config[key]
        for key in ("max_tokens", "max_output_tokens", "max_completion_tokens")
        if key in config
    )


@FAMILIES
@MODES
@pytest.mark.parametrize("registry_selection", [False, True])
async def test_inherited_settings_are_not_explicit_level_conflicts(
    tmp_path, family, is_async, registry_selection, wires
):
    _entry(
        tmp_path,
        family,
        max_tokens=50,
        temperature=0.5,
        reasoning_effort="native",
        reasoning_level="low" if registry_selection else None,
        reasoning_levels={
            "low": {"max_tokens": 30, "temperature": 0.1, "reasoning_effort": "low"},
            "high": {"top_p": 0.9},
        },
    )
    llm = load_llm("alias", **({} if registry_selection else {"reasoning_level": "low"}))
    from nooa.unifiedllm.adapters._binding import _bind_adapter

    disabled = LLM(_bind_adapter(llm.adapter, reasoning_level=None))
    high = LLM(_bind_adapter(llm.adapter, reasoning_level="high"))
    try:
        # The legacy config hook and actual wire agree; raw defaults aren't the patch.
        config = llm.adapter._client._prepare_call_config({})
        assert _reply_cap(config) == 30 and config["temperature"] == 0.1
        assert llm.adapter._bound_settings.config["max_tokens"] == 50
        assert llm.adapter._bound_settings.config["temperature"] == 0.5
        await _call(llm, is_async)
        await _call(disabled, is_async)
        await _call(high, is_async)
        assert _reply_cap(wires[0]) == 30 and wires[0]["temperature"] == 0.1
        for wire in wires[1:]:
            assert _reply_cap(wire) == 50 and wire["temperature"] == 0.5
            assert wire["reasoning_effort"] == "native"
        assert wires[2]["top_p"] == 0.9
    finally:
        disabled.close()  # These direct views borrow the owning loaded view's pool.
        high.close()
        await llm.aclose()


@FAMILIES
@pytest.mark.parametrize("registry_selection", [False, True])
@pytest.mark.parametrize(
    "explicit",
    [
        {"max_output_tokens": 50},
        {"max_tokens": None},
        {"temperature": 0.5},
        {"temperature": None},
        {"extra_body": {"max_tokens": 50}},
        {"extra_body": {"reasoning_effort": "native"}},
    ],
)
async def test_genuine_explicit_conflicts_are_rejected_at_load(
    tmp_path, family, registry_selection, explicit, wires
):
    _entry(
        tmp_path,
        family,
        reasoning_level="low" if registry_selection else None,
        reasoning_levels={"low": {"max_tokens": 30, "temperature": 0.1, "reasoning_effort": "low"}},
    )
    with pytest.raises(ValueError, match="conflicts with explicit"):
        load_llm("alias", **explicit, **({} if registry_selection else {"reasoning_level": "low"}))
    assert wires == []
    await asyncio.sleep(0)  # Join the constructor-failure cleanup before loop exit.


@FAMILIES
@MODES
async def test_scalar_raw_defaults_can_differ_in_same_pool(tmp_path, family, is_async, wires):
    entry = _entry(
        tmp_path,
        family,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
        reasoning_levels={"low": {"reasoning_effort": "low"}},
    )
    a = load_llm("alias", reasoning_level="low", temperature=_UNSET)
    entry["temperature"] = 0.6
    entry["max_tokens"] = 40
    b = load_llm("alias", reasoning_level=None)
    assert a._lease is b._lease
    assert a.capabilities is b.capabilities
    try:
        await _call(a, is_async)
        await _call(b, is_async)
        assert wires[0]["temperature"] is None and _reply_cap(wires[0]) is None
        assert wires[0]["reasoning_effort"] == "low"
        assert wires[1]["temperature"] == 0.6 and _reply_cap(wires[1]) == 40
        assert wires[1]["reasoning_effort"] is None
    finally:
        await a.aclose()
        await b.aclose()


@FAMILIES
@pytest.mark.parametrize("sync_first", [False, True])
async def test_final_disposal_is_shielded_joinable_and_exact_once(
    tmp_path, family, sync_first, monkeypatch
):
    _entry(tmp_path, family)
    llm = load_llm("alias")
    pool = llm._lease
    http = pool.client._http
    started, finish = asyncio.Event(), asyncio.Event()
    counts = {"sync": 0, "async": 0}
    real_sync, real_async = http.httpx_sync.close, http.httpx_async.aclose

    def sync_close():
        counts["sync"] += 1
        real_sync()

    async def async_close():
        counts["async"] += 1
        started.set()
        await finish.wait()
        await real_async()

    monkeypatch.setattr(http.httpx_sync, "close", sync_close)
    monkeypatch.setattr(http.httpx_async, "aclose", async_close)
    if sync_first:
        llm.close()
    first = asyncio.create_task(llm.aclose())
    await started.wait()
    disposal = pool._disposal
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not disposal.done() and pool.leases == 0
    second = asyncio.create_task(llm.aclose())
    await asyncio.sleep(0)
    assert not second.done() and pool._disposal is disposal
    finish.set()
    await second
    await llm.aclose()
    llm.close()
    assert counts == {"sync": 1, "async": 1}
    assert http.httpx_sync.is_closed and http.httpx_async.is_closed
    assert pool.leases == 0 and pool.retired


@FAMILIES
async def test_disposal_failure_is_retained_without_background_exception(
    tmp_path, family, monkeypatch
):
    _entry(tmp_path, family)
    llm = load_llm("alias")
    pool = llm._lease
    error = RuntimeError("disposal failed after both halves closed")
    real_close = pool.client.aclose
    count = 0

    async def failing():
        nonlocal count
        count += 1
        await real_close()
        raise error

    monkeypatch.setattr(pool.client, "aclose", failing)
    unhandled = []
    loop = asyncio.get_running_loop()
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: unhandled.append(context))
    try:
        llm.close()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        for _ in range(2):
            with pytest.raises(RuntimeError) as caught:
                await llm.aclose()
            assert caught.value is error
        assert count == 1 and pool.leases == 0
        assert pool.client._http.httpx_async.is_closed and pool.client._http.httpx_sync.is_closed
        assert unhandled == []
    finally:
        loop.set_exception_handler(old_handler)


@FAMILIES
async def test_pending_wire_deadline_after_caller_cancel_keeps_transport(
    tmp_path, family, monkeypatch, wires
):
    from nooa.unifiedllm import HttpConfig

    _entry(tmp_path, family)
    started, cancelled, finish, done = (asyncio.Event() for _ in range(4))

    async def provider(**kw):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await finish.wait()  # Actual provider teardown can itself await.
            raise
        finally:
            done.set()

    monkeypatch.setattr(litellm, "a" + family, provider)
    llm = load_llm("alias", http_config=HttpConfig(request_timeout=0.05), max_output_tokens=20)
    pool = llm._lease
    caller = asyncio.create_task(_call(llm, True))
    await started.wait()
    llm.close()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert pool.leases == 1 and not pool.retired
    await asyncio.wait_for(cancelled.wait(), 1)
    assert not done.is_set() and not pool.retired
    assert (
        not pool.client._http.httpx_async.is_closed and not pool.client._http.httpx_sync.is_closed
    )
    finish.set()
    await done.wait()
    for _ in range(10):
        await asyncio.sleep(0)
        if pool.retired:
            break
    await llm.aclose()
    assert pool.leases == 0 and pool.client._http.httpx_async.is_closed
    assert pool.client._http.httpx_sync.is_closed


@FAMILIES
async def test_queued_admission_cancel_never_dispatches_or_leaks(tmp_path, family, wires):
    from nooa.unifiedllm.admission import _admission_controller_scope

    _entry(tmp_path, family)
    queued = asyncio.Event()

    class Controller:
        async def acquire(self, observer):
            queued.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled queue unexpectedly admitted")

    llm = load_llm("alias", max_output_tokens=20)
    pool = llm._lease
    with _admission_controller_scope(Controller()):
        caller = asyncio.create_task(_call(llm, True))
        await queued.wait()
        assert pool.leases == 2  # View and logical call only: no dispatched wire.
        llm.close()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
    await llm.aclose()
    assert wires == [] and pool.leases == 0
    assert pool.client._http.httpx_async.is_closed and pool.client._http.httpx_sync.is_closed


@FAMILIES
async def test_sync_inflight_call_keeps_both_transports(tmp_path, family, monkeypatch, wires):
    from threading import Event

    _entry(tmp_path, family)
    started, finish = Event(), Event()

    def provider(**kw):
        started.set()
        assert finish.wait(5), "test failed to release sync provider"
        return _reply(family)

    monkeypatch.setattr(litellm, family, provider)
    llm = load_llm("alias", max_output_tokens=20)
    pool = llm._lease
    caller = asyncio.create_task(asyncio.to_thread(llm.call, LLMRequest(messages=MESSAGES)))
    assert await asyncio.to_thread(started.wait, 5)
    llm.close()
    assert pool.leases == 1 and not pool.retired
    assert (
        not pool.client._http.httpx_sync.is_closed and not pool.client._http.httpx_async.is_closed
    )
    finish.set()
    await caller
    await llm.aclose()
    assert pool.leases == 0 and pool.retired
    assert pool.client._http.httpx_sync.is_closed and pool.client._http.httpx_async.is_closed


@FAMILIES
@pytest.mark.parametrize("in_loop", [False, True])
def test_wrapper_construction_failure_preserves_error_and_closes_both(
    tmp_path, family, in_loop, monkeypatch
):
    _entry(tmp_path, family)
    original = RuntimeError("original wrapper construction failure")
    captured = []
    counts = {"sync": 0, "async": 0}

    def fail(http, *args):
        captured.append(http)
        real_sync, real_async = http.httpx_sync.close, http.httpx_async.aclose

        def sync_close():
            counts["sync"] += 1
            real_sync()
            raise RuntimeError("secondary sync cleanup failure")

        async def async_close():
            counts["async"] += 1
            await real_async()
            raise RuntimeError("secondary async cleanup failure")

        monkeypatch.setattr(http.httpx_sync, "close", sync_close)
        monkeypatch.setattr(http.httpx_async, "aclose", async_close)
        raise original

    method = "_build_completion_wrappers" if family == "completion" else "_build_handler_wrappers"
    monkeypatch.setattr(_ClientHttp, method, fail)

    def construct():
        with pytest.raises(RuntimeError) as caught:
            load_llm("alias")
        assert caught.value is original

    async def construct_in_loop():
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: unhandled.append(context)
        )
        construct()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert unhandled == []

    if in_loop:
        asyncio.run(construct_in_loop())
    else:
        construct()
    assert counts == {"sync": 1, "async": 1}
    assert captured[0].httpx_sync.is_closed and captured[0].httpx_async.is_closed


@FAMILIES
async def test_provider_cancel_before_coroutine_entry_releases_actual_wire(
    tmp_path, family, monkeypatch, wires
):
    from nooa.unifiedllm import unifiedllm as provider_module
    from nooa.unifiedllm.admission import _admission_controller_scope

    _entry(tmp_path, family)
    llm = load_llm("alias", max_output_tokens=20)
    pool = llm._lease
    real_create = asyncio.create_task
    releases = []

    class Permit:
        def release(self):
            releases.append(True)

    class Controller:
        async def acquire(self, observer):
            return Permit()

    def create_and_cancel(coro, **kwargs):
        task = real_create(coro, **kwargs)
        task.cancel()  # Owned wrapper's finally never gets to enter.
        return task

    monkeypatch.setattr(provider_module.asyncio, "create_task", create_and_cancel)
    with _admission_controller_scope(Controller()):
        with pytest.raises(asyncio.CancelledError):
            await _call(llm, True)
    assert wires == [] and releases == [True] and pool.leases == 1
    monkeypatch.setattr(provider_module.asyncio, "create_task", real_create)
    await llm.aclose()
    assert pool.leases == 0 and pool.client._http.httpx_sync.is_closed
    assert pool.client._http.httpx_async.is_closed
