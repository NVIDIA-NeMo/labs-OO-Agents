# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline gates for private same-route settings, not a public loader API."""

import asyncio
from dataclasses import FrozenInstanceError

import httpx
import litellm
import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from nooa.unifiedllm import LLM, CompletionClient, LLMRequest, ResponsesClient, RetryConfig, Tool
from nooa.unifiedllm.adapters._binding import _UNSET, _bind_adapter
from nooa.unifiedllm.adapters.legacy import configure_legacy_adapter
from nooa.unifiedllm.unifiedllm import _ClientHttp

FAMILIES = pytest.mark.parametrize("family", [CompletionClient, ResponsesClient])
MODES = pytest.mark.parametrize("is_async", [False, True])


def reply(family):
    if family is CompletionClient:
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "ok"}}])
    return litellm.ResponsesAPIResponse(
        id="resp_1",
        object="response",
        created_at=0,
        status="completed",
        model="gpt-4o-mini",
        output=[
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


@pytest.fixture(autouse=True)
def deny_external_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected external network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)


@pytest.fixture
def capture(monkeypatch):
    seen = []

    def install(family):
        name = "completion" if family is CompletionClient else "responses"

        def send(**kwargs):
            seen.append(kwargs)
            # A provider/SDK may mutate its owned arguments.
            kwargs.get("extra_body", {}).get("nested", []).append("wire mutation")
            return reply(family)

        async def asend(**kwargs):
            await asyncio.sleep(0)
            return send(**kwargs)

        monkeypatch.setattr(litellm, name, send)
        monkeypatch.setattr(litellm, "a" + name, asend)
        return seen

    return install


def make(family, **kwargs):
    return family(
        "openai/gpt-4o-mini", api_key="test", api_base="https://endpoint.test/v1", **kwargs
    )


@FAMILIES
@MODES
async def test_snapshot_isolation_and_no_construction(family, is_async, capture, monkeypatch):
    seen = capture(family)
    inherited = {"nested": ["initial"], "tuple": ({"list": [1]},)}
    caller = {"nested": ["override"]}
    client = make(family, extra_body=inherited, temperature=0.9, max_retries=2)

    def hook(*args):
        return None

    client._validate_cache_breakpoint_model = hook
    original_caps = client._adapter.capabilities

    def forbidden(*args, **kwargs):
        raise AssertionError("binding must not construct or send")

    monkeypatch.setattr(_ClientHttp, "for_completion", forbidden)
    monkeypatch.setattr(_ClientHttp, "for_responses", forbidden)
    get_provider = litellm.get_llm_provider
    monkeypatch.setattr(litellm, "get_llm_provider", forbidden)
    a = _bind_adapter(client._adapter, temperature=0.1, top_p=0.8)
    b = _bind_adapter(a, temperature=0.2, extra_body=caller)
    assert b.capabilities is a.capabilities
    assert a.capabilities == original_caps
    assert a.capabilities.context_window == original_caps.context_window
    assert a.capabilities.max_output_tokens == original_caps.max_output_tokens
    assert a.capabilities.reasoning_levels == original_caps.reasoning_levels
    assert a._client._http is b._client._http is client._http
    assert a._client._http_config is client._http_config
    assert a._client._validate_cache_breakpoint_model is hook
    with pytest.raises(FrozenInstanceError):
        a._bound_settings.config = {}
    with pytest.raises(TypeError):
        a._bound_settings.config["temperature"] = 100
    inherited["nested"].append("caller mutation")
    inherited["tuple"][0]["list"].append(2)
    caller["nested"].append("caller mutation")
    client.config["temperature"] = 7
    # Even an internal compatibility hook editing the derived client's defaults
    # cannot change the bound record used by later sends or settings derivation.
    a._client.config["temperature"] = 8
    a._client.config["extra_body"]["nested"].append("derived mutation")
    c = _bind_adapter(a, top_p=0.6)
    assert c._client.config["temperature"] == 0.1
    assert c._client.config["extra_body"]["nested"] == ["initial"]
    # Restore local provider resolution only for wire preparation, not binding.
    monkeypatch.setattr(litellm, "get_llm_provider", get_provider)
    request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
    try:
        if is_async:
            await asyncio.gather(a.send(request), b.send(request), a.send(request))
        else:
            a.send_sync(request)
            b.send_sync(request)
            a.send_sync(request)
        assert [call["temperature"] for call in seen] == [0.1, 0.2, 0.1]
        assert seen[0]["extra_body"]["tuple"][0]["list"] == [1]
        assert seen[0]["extra_body"]["nested"] == ["initial", "wire mutation"]
        assert seen[1]["extra_body"]["nested"] == ["override", "wire mutation"]
        assert seen[2]["extra_body"]["nested"] == ["initial", "wire mutation"]
        assert all(call["max_retries"] == call["num_retries"] == 0 for call in seen)
    finally:
        await client.aclose()


@FAMILIES
async def test_unset_none_and_reasoning_precedence(family, capture):
    seen = capture(family)
    client = make(
        family,
        temperature=0.7,
        max_tokens=80,
        reasoning_levels={"low": {"reasoning_effort": "low"}, "high": {"reasoning_effort": "high"}},
        reasoning_level="high",
    )
    try:
        inherited = _bind_adapter(client._adapter, temperature=_UNSET)
        reset = _bind_adapter(
            client._adapter, temperature=None, reasoning_level=None, max_output_tokens=None
        )
        inherited.send_sync(LLMRequest())
        reset.send_sync(LLMRequest())
        assert seen[0]["temperature"] == 0.7 and seen[0]["reasoning_effort"] == "high"
        assert seen[1]["temperature"] is None and "reasoning_effort" not in seen[1]
        assert seen[1]["max_output_tokens"] is None
        bound = _bind_adapter(client._adapter, reasoning_level="low")
        client._reasoning_config.levels["high"]["reasoning_effort"] = "mutated"
        _bind_adapter(bound, reasoning_level="high", max_output_tokens=23).send_sync(LLMRequest())
        assert seen[-1]["reasoning_effort"] == "high"
        assert seen[-1]["max_output_tokens"] == 23
        before = len(seen)
        for options in (
            {"reasoning_level": "bogus"},
            {"max_output_tokens": -1},
            {"reasoning_level": "low", "reasoning_effort": "high"},
        ):
            with pytest.raises(ValueError):
                _bind_adapter(client._adapter, **options)
        assert len(seen) == before
    finally:
        await client.aclose()


@FAMILIES
async def test_request_contract_and_rejections(family, capture):
    seen = capture(family)

    class Answer(BaseModel):
        text: str

    def lookup(query: str) -> str:
        return query

    tool = Tool(name="lookup", description="lookup", callable=lookup)
    client = make(
        family,
        tools=[{"wrong": True}],
        tool_choice="none",
        extra_body={"tools": ["bad"], "tool_choice": "none"},
        response_format={"wrong": True},
        text={"format": {"wrong": True}},
    )
    try:
        bound = _bind_adapter(client._adapter)
        # Preparation is sufficient: fake text does not satisfy the output model.
        request = LLMRequest(tools=[tool], tool_choice="lookup", output_type=Answer)
        if family is CompletionClient:
            config, native = bound._settings(request)
            wire = bound.prepare(request, config, native_output=native, is_async=False).api_params
            assert wire["response_format"] is Answer
        else:
            wire = bound.prepare(request, bound._call_config(request), is_async=False).api_params
            assert wire["text_format"] is Answer and "text" not in wire
        assert (
            wire["tools"][0].get("name", wire["tools"][0].get("function", {}).get("name"))
            == "lookup"
        )
        assert wire["tool_choice"]["type"] == "function"
        assert "tools" not in wire["extra_body"]
        bound.send_sync(LLMRequest())
        assert "tools" not in seen[-1]
        for options in (
            {"model": "other"},
            {"api_key": "other"},
            {"base_url": None},
            {"client": object()},
            {"api_style": "chat"},
            {"session_affinity": "x"},
            {"extra_body": {"api_base": "x"}},
        ):
            with pytest.raises(ValueError, match="route/identity"):
                _bind_adapter(client._adapter, **options)
        with pytest.raises(TypeError):
            _bind_adapter(object())
        with pytest.raises(TypeError):
            _bind_adapter(configure_legacy_adapter(client._adapter, {}))
    finally:
        await client.aclose()


@FAMILIES
@MODES
async def test_bound_public_sdk_budget_and_pool(family, is_async, monkeypatch):
    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    sdk = (
        AsyncOpenAI(
            api_key="test",
            base_url="https://endpoint.test/v1",
            http_client=httpx.AsyncClient(transport=transport),
            max_retries=2,
        )
        if is_async
        else OpenAI(
            api_key="test",
            base_url="https://endpoint.test/v1",
            http_client=httpx.Client(transport=transport),
            max_retries=2,
        )
    )

    def forbidden_network(*args, **kwargs):
        raise AssertionError("unexpected external network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden_network)
    client = make(family, client=sdk, max_retries=2)
    if family is ResponsesClient:
        # LiteLLM normally uses its handler, not a supplied OpenAI Responses SDK.
        # Exercise the supplied SDK explicitly to verify the adapter's retry copy.
        def responses(**kwargs):
            return kwargs["client"].responses.create(model="gpt-4o-mini", input="hi")

        async def aresponses(**kwargs):
            return await kwargs["client"].responses.create(model="gpt-4o-mini", input="hi")

        monkeypatch.setattr(litellm, "responses", responses)
        monkeypatch.setattr(litellm, "aresponses", aresponses)
    try:
        bound = _bind_adapter(client._adapter, temperature=0.1)
        assert bound._client.config["client"] is sdk
        request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        if family is CompletionClient:
            config, native = bound._settings(request)
            prepared = bound.prepare(request, config, native_output=native, is_async=is_async)
        else:
            prepared = bound.prepare(request, bound._call_config(request), is_async=is_async)
        assert prepared.api_params["client"]._client is sdk._client
        assert prepared.api_params["client"].max_retries == 0
        assert sdk.max_retries == 2
        llm = LLM(
            bound,
            retry_config=RetryConfig(
                max_retries=1, rate_limit_extra_retries=0, base_delay=0, jitter_factor=0
            ),
        )
        with pytest.raises(Exception, match="(?i)internal"):
            if is_async:
                await llm.acall(request)
            else:
                llm.call(request)
        assert len(attempts) == 2
        assert client.config["max_retries"] == 2
    finally:
        await client.aclose()
        if is_async:
            await sdk.close()
        else:
            sdk.close()


@FAMILIES
async def test_reasoning_nested_defaults_and_unknown_validation(family, capture):
    seen = capture(family)
    client = make(
        family, reasoning_levels={"low": {"reasoning": {"effort": "low", "summary": ["keep"]}}}
    )
    try:
        bound = _bind_adapter(client._adapter, reasoning_level="low")
        client._reasoning_config.levels["low"]["reasoning"]["summary"].append("changed")
        bound.send_sync(LLMRequest())
        assert seen[-1]["reasoning"]["summary"] == ["keep"]
        _bind_adapter(bound, reasoning_level="low").send_sync(LLMRequest())
        assert seen[-1]["reasoning"]["summary"] == ["keep"]
    finally:
        await client.aclose()
    for declaration in (None, {}):
        unknown = make(family, reasoning_levels=declaration)
        try:
            before = len(seen)
            with pytest.raises(ValueError, match="unknown|not supported"):
                _bind_adapter(unknown._adapter, reasoning_level="low")
            assert len(seen) == before
        finally:
            await unknown.aclose()


async def test_native_mode_reset_conflicts_and_custom_derivation(capture):
    capture(CompletionClient)
    client = make(
        CompletionClient, structured_output_mode="anthropic", output_config={"effort": "low"}
    )
    try:
        inherited = _bind_adapter(client._adapter)
        reset = _bind_adapter(client._adapter, structured_output_mode=None)
        assert inherited._settings(LLMRequest())[1] is True
        assert reset._settings(LLMRequest())[1] is False
        inherited._client.structured_output_mode = None
        reset._client.structured_output_mode = "anthropic"
        assert inherited._settings(LLMRequest())[1] is True
        assert reset._settings(LLMRequest())[1] is False
        assert (
            inherited._client._native_output_request_scope.__func__
            is client._native_output_request_scope.__func__
        )
        from dataclasses import replace

        inherited._bound_capabilities = replace(inherited.capabilities, max_output_tokens=100)
        with pytest.raises(ValueError, match="known limit"):
            _bind_adapter(inherited, max_output_tokens=101)
        with pytest.raises(ValueError, match="only one"):
            _bind_adapter(client._adapter, max_tokens=10, max_output_tokens=20)

        class Custom(type(client._adapter)):
            pass

        with pytest.raises(TypeError, match="built-in"):
            _bind_adapter(Custom(client))
    finally:
        await client.aclose()


@MODES
async def test_bound_responses_real_owned_handler(is_async, monkeypatch):
    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected external network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    client = make(ResponsesClient, max_retries=2)
    try:
        bound = _bind_adapter(client._adapter, temperature=0.1)
        assert bound._client._http is client._http
        request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(Exception, match="(?i)internal"):
            if is_async:
                await bound.send(request)
            else:
                bound.send_sync(request)
        assert len(attempts) == 1
    finally:
        await client.aclose()


@FAMILIES
@MODES
@pytest.mark.parametrize("native_defaults", [False, True])
async def test_reasoning_selection_provenance(family, is_async, native_defaults, capture):
    seen = capture(family)
    levels = {
        "low": {"thinking": {"type": "disabled"}},
        "high": {"reasoning_effort": "high"},
    }
    raw = (
        {
            "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            "reasoning_effort": "medium",
        }
        if native_defaults
        else {}
    )
    client = make(family, reasoning_levels=levels, reasoning_level="high", **raw)

    def config(adapter, request):
        return (
            adapter._settings(request)[0]
            if family is CompletionClient
            else adapter._call_config(request)
        )

    async def send(adapter, request):
        if is_async:
            await adapter.send(request)
        else:
            adapter.send_sync(request)

    try:
        high = _bind_adapter(client._adapter)
        low = _bind_adapter(high, reasoning_level="low")
        assert low._bound_settings.reasoning_level == "low"
        assert low._bound_settings.config.get("reasoning_effort") == raw.get("reasoning_effort")
        high._client.reasoning_level = "low"
        high._client.config["thinking"] = {"type": "disabled"}
        low._client.config["thinking"] = {"type": "disabled"}
        low._client._reasoning_config.levels["high"]["reasoning_effort"] = "mutated"
        inherited = _bind_adapter(high, reasoning_level=_UNSET)
        assert inherited._bound_settings.reasoning_level == "high"
        reset = _bind_adapter(high, reasoning_level=None)
        reset_direct = _bind_adapter(client._adapter, reasoning_level=None)
        assert config(reset, LLMRequest()) == config(reset_direct, LLMRequest())
        assert config(reset, LLMRequest()).get("reasoning_effort") == raw.get("reasoning_effort")
        direct_high = _bind_adapter(client._adapter, reasoning_level="high")
        rebound_high = _bind_adapter(low, reasoning_level="high")
        assert config(rebound_high, LLMRequest()) == config(direct_high, LLMRequest())
        assert config(_bind_adapter(low, reasoning_level="high"), LLMRequest()) == config(
            direct_high, LLMRequest()
        )
        assert "thinking" not in config(rebound_high, LLMRequest())
        for adapter, request in (
            (reset, LLMRequest()),
            (reset_direct, LLMRequest()),
            (rebound_high, LLMRequest()),
            (direct_high, LLMRequest()),
            (_bind_adapter(low, reasoning_level="high", max_output_tokens=23), LLMRequest()),
            (inherited, LLMRequest()),
        ):
            await send(adapter, request)
        # Strict Chat creates retry-disabled SDK copies; compare their shared
        # pool separately rather than treating copy identity as a wire field.
        wire = [{key: value for key, value in call.items() if key != "client"} for call in seen]
        assert wire[0] == wire[1]
        assert wire[2] == wire[3] == wire[5]
        if family is CompletionClient:
            assert all(call["client"]._client is seen[0]["client"]._client for call in seen)
        assert seen[4]["reasoning_effort"] == "high"
        assert seen[4].get("extra_body", {}).get("thinking") == raw.get("extra_body", {}).get(
            "thinking"
        )
        assert seen[4]["max_output_tokens"] == 23
        for conflict in (
            {"reasoning_effort": "low"},
            {"extra_body": {"reasoning_effort": "low"}},
        ):
            with pytest.raises(ValueError, match="conflicts"):
                _bind_adapter(low, reasoning_level="high", **conflict)
        # A genuinely native default survives disabling a managed selection.
        native = _bind_adapter(reset, reasoning_effort="native")
        assert config(native, LLMRequest())["reasoning_effort"] == "native"
        assert (
            config(_bind_adapter(native, reasoning_level="high"), LLMRequest())["reasoning_effort"]
            == "high"
        )
        assert (
            config(
                _bind_adapter(_bind_adapter(native, reasoning_level="high"), reasoning_level=None),
                LLMRequest(),
            )["reasoning_effort"]
            == "native"
        )
    finally:
        await client.aclose()


@FAMILIES
@MODES
async def test_identity_carriers_rejected_before_real_wire(family, is_async, monkeypatch):
    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    client = make(
        family, reasoning_levels={"unsafe": {"extra_headers": {"Authorization": "override"}}}
    )
    request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
    try:
        with pytest.raises(ValueError, match="route/identity"):
            _bind_adapter(client._adapter, reasoning_level="unsafe")
        neutral = _bind_adapter(client._adapter)
        with pytest.raises(ValueError, match="route/identity"):
            if is_async:
                await _bind_adapter(neutral, reasoning_level="unsafe").send(request)
            else:
                _bind_adapter(neutral, reasoning_level="unsafe").send_sync(request)
        assert not attempts
        rejected = [
            {"extra_headers": {header: "override"}}
            for header in (
                "Authorization",
                "AUTHORIZATION",
                "proxy-authorization",
                "Host",
                "api-key",
                "X-API-Key",
                "OpenAI-Organization",
                "OpenAI-Project",
                "x-goog-user-project",
                "X-Forwarded-Host",
                "Cookie",
            )
        ]
        rejected += [
            {"auth": object()},
            {"API_KEY": "override"},
            {"HOST": "other.test"},
            {"organization": "other"},
            {"project_id": "other"},
            {"azure_ad_token_provider": lambda: "override"},
            {"extra_query": {"api-key": "override"}},
            {"headers": {"Authorization": "override"}},
            {"extra_headers": object()},
            {"extra_headers": {"benign": {"auth": "override"}}},
        ]
        rejected += [{"extra_body": value} for value in rejected.copy()]
        rejected += [{"extra_body": {"extra_body": {"auth": "override"}}}]
        for options in rejected:
            with pytest.raises(ValueError, match="route/identity"):
                bound = _bind_adapter(client._adapter, **options)
                if is_async:
                    await bound.send(request)
                else:
                    bound.send_sync(request)
            assert not attempts
        # The actual LiteLLM/owned SDK-handler route still works with model
        # settings and an ordinary metadata header, without credential changes.
        bound = _bind_adapter(
            client._adapter, temperature=0.1, extra_headers={"X-Binding-Test": "safe"}
        )
        with pytest.raises(Exception, match="(?i)internal"):
            if is_async:
                await bound.send(request)
            else:
                bound.send_sync(request)
        assert len(attempts) == 1
        assert attempts[0].url.host == "endpoint.test"
        assert attempts[0].headers["authorization"] == "Bearer test"
        assert attempts[0].headers["x-binding-test"] == "safe"
    finally:
        await client.aclose()
