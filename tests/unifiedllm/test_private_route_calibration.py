# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline numeric bridge tests; this is NOT shell observation accounting."""

import asyncio
import weakref
from concurrent.futures import ThreadPoolExecutor

import httpx
import litellm
import pytest

from nooa.unifiedllm import CompletionClient, LLMRequest, ResponsesClient
from nooa.unifiedllm.adapters import _calibration as routes
from nooa.unifiedllm.adapters._binding import _bind_adapter
from nooa.unifiedllm.unifiedllm import TokenCalibration, _ClientHttp

FAMILIES = pytest.mark.parametrize("family", [CompletionClient, ResponsesClient])


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    from nooa.unifiedllm import unifiedllm as core

    legacy = TokenCalibration(alpha=0.3)
    monkeypatch.setattr(core, "_token_calibration", legacy)
    monkeypatch.setattr(routes, "_records", {})
    monkeypatch.setattr(routes, "_sdk_ids", {})

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected external HTTP")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    return legacy


def make(family, **overrides):
    config = {"api_key": "private-key", "api_base": "https://endpoint.test/v1"}
    config.update(overrides)
    return family("openai/gpt-4o-mini", **config)


@FAMILIES
async def test_seed_once_warm_continuity_and_divergence(family, isolated, monkeypatch):
    client = make(family)
    other = make(family, api_base="https://other.test/v1")
    isolated.update(client.model, 100, 230)
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: 100)
    try:
        assert client._adapter._route_calibration_record is None
        a = _bind_adapter(client._adapter, temperature=0.1)
        b = _bind_adapter(a, temperature=0.2)
        record = routes._route_calibration(a)
        assert routes._route_calibration(b) is record
        assert routes._route_calibration(client._adapter) is record
        assert record.calibrate(100) == client.count_tokens("warm") == 229
        isolated.update(client.model, 100, 100)
        assert record.ratio() == pytest.approx(2.3)
        c = _bind_adapter(client._adapter, top_p=0.8)
        assert routes._route_calibration(c) is record
        different = routes._route_calibration(_bind_adapter(other._adapter))
        assert different is not record
        assert different.ratio() == isolated.ratio(client.model)
        legacy_before = isolated.ratio(client.model)
        record.update(100, 400)
        assert record.ratio() == pytest.approx(0.3 * 4 + 0.7 * 2.3)
        assert isolated.ratio(client.model) == legacy_before
        assert different.ratio() == legacy_before
        assert client.count_tokens("still legacy") == int(100 * legacy_before)
        before = record.ratio()
        for estimated, actual in [(0, 100), (100, 0), (-1, 100), (100, -1)]:
            record.update(estimated, actual)
        assert record.ratio() == before
    finally:
        await client.aclose()
        await other.aclose()


@FAMILIES
async def test_same_route_aliases_deep_views_and_no_probe(family, isolated, monkeypatch):
    first = make(family, context_window=10000)
    inherited = {"nested": ["original"]}
    alias = make(family, api_base=None, base_url=" HTTPS://ENDPOINT.TEST/v1/ ")
    caps = first._adapter.capabilities

    def forbidden(*args, **kwargs):
        raise AssertionError("binding constructed, probed or sent")

    for name in ("for_completion", "for_responses"):
        monkeypatch.setattr(_ClientHttp, name, forbidden)
    monkeypatch.setattr(litellm, "get_llm_provider", forbidden)
    monkeypatch.setattr(litellm, "get_model_info", forbidden)
    monkeypatch.setattr(litellm, "token_counter", forbidden)
    try:
        a = _bind_adapter(first._adapter, extra_body=inherited)
        record = routes._route_calibration(a)
        assert routes._route_calibration(_bind_adapter(alias._adapter)) is record
        for index in range(30):
            a = _bind_adapter(a, temperature=index / 30)
            assert routes._route_calibration(a) is record
            assert a._client._http is first._http
            assert a.capabilities is caps or a.capabilities == caps
        inherited["nested"].append("mutated")
        assert a._bound_settings.config["extra_body"]["nested"].items == ("original",)
        assert a.capabilities.context_window == 10000
    finally:
        await first.aclose()
        await alias.aclose()


@FAMILIES
async def test_route_model_provider_credential_and_sdk_separation(family):
    class SDK:
        def __deepcopy__(self, memo):
            raise AssertionError("SDK must never be deepcopied")

    sdk, other_sdk = SDK(), SDK()
    configs = [
        {},
        {"api_key": "different-key"},
        {"api_base": "https://different.test/v1"},
        {"api_base": "https://endpoint.test/other"},
        {"custom_llm_provider": "openai"},
        {"client": sdk},
        {"client": other_sdk},
    ]
    clients = [make(family, **config) for config in configs]
    same_sdk = make(family, client=sdk)
    another_family = make(ResponsesClient if family is CompletionClient else CompletionClient)
    another_model = family(
        "openai/other", api_key="private-key", api_base="https://endpoint.test/v1"
    )
    try:
        records = [routes._route_calibration(_bind_adapter(client._adapter)) for client in clients]
        assert len({id(record) for record in records}) == len(records)
        assert routes._route_calibration(_bind_adapter(same_sdk._adapter)) is records[5]
        assert routes._route_calibration(_bind_adapter(another_family._adapter)) not in records
        assert routes._route_calibration(_bind_adapter(another_model._adapter)) not in records
    finally:
        for client in [*clients, same_sdk, another_family, another_model]:
            await client.aclose()


@FAMILIES
@pytest.mark.parametrize(
    "extra",
    [
        {"api_base": "not-a-url"},
        {"base_url": "https://conflict.test"},
        {"api_base": None},
        {"api_key": None},
        {"organization": "private-org"},
        {"extra_headers": {"Authorization": "secret-header"}},
        {"extra_body": {"default_query": {"api-key": "secret-query"}}},
        {"client": object()},
    ],
)
async def test_ambiguous_identity_is_root_local(family, extra):
    first, second = make(family, **extra), make(family, **extra)
    try:
        a, b = _bind_adapter(first._adapter), _bind_adapter(second._adapter)
        assert routes._route_calibration(a) is not routes._route_calibration(b)
        assert routes._route_calibration(
            _bind_adapter(a, temperature=0.1)
        ) is routes._route_calibration(a)
    finally:
        await first.aclose()
        await second.aclose()


@FAMILIES
async def test_secrets_never_in_cache_keys_or_record_repr(family):
    endpoint = "https://secret-user:secret-password@ENDPOINT.TEST/v1?api-key=secret-query"
    client = make(family, api_base=endpoint)
    try:
        record = routes._route_calibration(_bind_adapter(client._adapter))
        assert all(isinstance(key, bytes) and len(key) == 32 for key in routes._records)
        visible = (
            repr(routes._records) + repr(record) + repr(client._adapter._route_calibration_identity)
        )
        for secret in [
            "private-key",
            "secret-user",
            "secret-password",
            "secret-query",
            "endpoint.test",
        ]:
            assert secret not in visible
        different = make(family, api_base=endpoint.replace("secret-query", "other-query"))
        try:
            assert routes._route_calibration(_bind_adapter(different._adapter)) is not record
        finally:
            await different.aclose()
    finally:
        await client.aclose()


@FAMILIES
async def test_concurrent_publication_has_one_stable_seed(family, isolated):
    clients = [make(family) for _ in range(8)]
    isolated.update(clients[0].model, 100, 200)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            records = list(
                pool.map(lambda client: routes._route_calibration(client._adapter), clients)
            )
        assert all(record is records[0] for record in records)
        assert len(routes._records) == 1
        records[0].update(100, 300)
        isolated.update(clients[0].model, 100, 100)

        async def derive(client):
            await asyncio.sleep(0)
            return routes._route_calibration(_bind_adapter(client._adapter))

        again = await asyncio.gather(*(derive(client) for client in clients))
        assert all(record is records[0] for record in again)
        assert records[0].ratio() == pytest.approx(2.3)
    finally:
        for client in clients:
            await client.aclose()


@FAMILIES
async def test_success_hooks_still_update_only_legacy(family, isolated, monkeypatch):
    # Reuse already-maintained fake replies without recording/replacing goldens.
    from tests.unifiedllm.test_private_settings_binding import reply

    client = make(family)
    response = reply(family)
    response.usage = (
        {"prompt_tokens": 200, "completion_tokens": 1}
        if family is ResponsesClient
        else litellm.Usage(prompt_tokens=200, completion_tokens=1, total_tokens=201)
    )
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: 100)
    monkeypatch.setattr(
        litellm,
        "completion" if family is CompletionClient else "responses",
        lambda **kwargs: response,
    )
    try:
        bound = _bind_adapter(client._adapter)
        record = routes._route_calibration(bound)
        bound.send_sync(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
        assert isolated.ratio(client.model) == 2
        assert record.ratio() == 1
    finally:
        await client.aclose()


@FAMILIES
@pytest.mark.parametrize("unknown", [False, True])
async def test_changed_root_does_not_change_prior_view(family, unknown, isolated):
    extra = {"extra_headers": {"Authorization": "secret"}} if unknown else {}
    client = make(family, **extra)
    try:
        first = _bind_adapter(client._adapter)
        record = routes._route_calibration(first)
        record.update(100, 300)
        assert routes._route_calibration(_bind_adapter(client._adapter)) is record
        client.config["api_key"] = "different-key"
        next_view = _bind_adapter(client._adapter)
        assert routes._route_calibration(next_view) is not record
        assert routes._route_calibration(_bind_adapter(first)) is record
        if unknown:
            client.config["extra_headers"]["Authorization"] = "different-auth"
            assert routes._route_calibration(
                _bind_adapter(client._adapter)
            ) is not routes._route_calibration(next_view)
    finally:
        await client.aclose()


def test_sdk_expired_id_receives_fresh_association_not_old_token():
    class SDK:
        pass

    dead = SDK()
    reference = weakref.ref(dead)
    del dead
    current = SDK()
    routes._sdk_ids[id(current)] = (reference, b"old-token")
    key = routes._sdk_identity(current)
    assert key is not False and key != b"old-token".hex()
    assert routes._sdk_identity(current) == key
