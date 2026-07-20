# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import httpx
import pytest
from pydantic import BaseModel, ValidationError

from nooa.unifiedllm.http_config import HttpConfig


def _configured_keepalive(client) -> int:
    """Read max_keepalive_connections from our stable _ClientHttp limits object."""
    return client._http.limits.max_keepalive_connections


def test_http_config_is_pydantic_model():
    assert issubclass(HttpConfig, BaseModel)


def test_http_config_defaults():
    c = HttpConfig()
    assert c.max_connections == 100
    assert c.max_keepalive_connections == 0
    assert c.keepalive_expiry == 5.0
    assert c.connect_timeout == 10.0
    assert c.read_timeout == 60.0
    assert c.write_timeout == 10.0
    assert c.pool_timeout == 10.0


def test_http_config_frozen():
    c = HttpConfig()
    with pytest.raises(ValidationError):
        c.connect_timeout = 5.0


def test_completion_client_accepts_http_config():
    from nooa.unifiedllm import CompletionClient, HttpConfig

    # Should not raise — just verifies the constructor signature
    client = CompletionClient("gpt-4o-mini", http_config=HttpConfig(connect_timeout=5.0))
    assert client._http_config.connect_timeout == 5.0
    client.close()


def test_to_httpx_limits_and_timeout():
    c = HttpConfig(max_connections=42, max_keepalive_connections=7, read_timeout=33.0)
    limits = c.to_httpx_limits()
    assert limits.max_connections == 42
    assert limits.max_keepalive_connections == 7
    assert limits.keepalive_expiry == 5.0
    timeout = c.to_httpx_timeout()
    assert timeout.read == 33.0


def test_enabling_keepalive_connections_reuses_by_default():
    c = HttpConfig(max_keepalive_connections=9)
    limits = c.to_httpx_limits()
    assert limits.max_keepalive_connections == 9
    assert limits.keepalive_expiry > 0


# ── #329 regression tests: per-client httpx, no global monkey-patch ──────────


def test_importing_unifiedllm_does_not_patch_httpx_asyncclient():
    """Importing unifiedllm must NOT monkey-patch httpx.AsyncClient (GitLab #329)."""
    import nooa.unifiedllm.unifiedllm as u  # noqa: F401  (force import)

    init = httpx.AsyncClient.__init__
    # The stdlib/httpx __init__ — not a closure installed by unifiedllm.
    assert init.__qualname__ == "AsyncClient.__init__"
    assert init.__module__.startswith("httpx")
    # The global monkey-patch machinery must be gone entirely.
    assert not hasattr(u, "_active_http_config")
    assert not hasattr(u, "_set_http_config")
    assert not hasattr(u, "_apply_httpx_no_pool_patch")


@pytest.mark.asyncio
async def test_unrelated_httpx_client_is_unaffected():
    """A plain httpx.AsyncClient created by unrelated code keeps its own limits."""
    import nooa.unifiedllm  # noqa: F401  (ensure the library is imported)

    # No http_config given anywhere; the user's explicit keepalive must survive.
    limits = httpx.Limits(max_keepalive_connections=17)
    user = httpx.AsyncClient(limits=limits)
    try:
        assert limits.max_keepalive_connections == 17
        assert not user.is_closed
    finally:
        await user.aclose()


def test_two_clients_do_not_share_limits():
    """Two clients with different http_config get independent httpx limits (no bleed)."""
    from nooa.unifiedllm import CompletionClient, HttpConfig

    a = CompletionClient("gpt-4o-mini", http_config=HttpConfig(max_keepalive_connections=0))
    b = CompletionClient("gpt-4o-mini", http_config=HttpConfig(max_keepalive_connections=9))
    try:
        assert a._http.limits.max_keepalive_connections == 0
        assert b._http.limits.max_keepalive_connections == 9
        # Distinct httpx client objects, distinct pools.
        assert a._http.httpx_async is not b._http.httpx_async
        assert _configured_keepalive(a) == 0
        assert _configured_keepalive(b) == 9
        # The last-constructed client (b) must not have changed a's limits.
        assert _configured_keepalive(a) == 0
    finally:
        a.close()
        b.close()


def test_no_pooling_client_has_keepalive_zero():
    """A client requesting no pooling produces httpx clients with keepalive=0."""
    from nooa.unifiedllm import CompletionClient, HttpConfig, ResponsesClient

    comp = CompletionClient("anthropic/claude-3-5-sonnet", http_config=HttpConfig())
    resp = ResponsesClient("openai/gpt-5.3-codex", http_config=HttpConfig())
    try:
        assert comp._http_config.max_keepalive_connections == 0
        assert _configured_keepalive(comp) == 0
        assert _configured_keepalive(comp) == 0
        assert _configured_keepalive(resp) == 0
        assert _configured_keepalive(resp) == 0
    finally:
        comp.close()
        resp.close()


def test_responses_client_passes_own_httpx_via_handler():
    """ResponsesClient wraps its own httpx client in litellm's AsyncHTTPHandler/HTTPHandler."""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    from nooa.unifiedllm import HttpConfig, ResponsesClient

    r = ResponsesClient("openai/gpt-5.3-codex", http_config=HttpConfig(max_keepalive_connections=3))
    try:
        assert isinstance(r._http.async_client, AsyncHTTPHandler)
        assert isinstance(r._http.sync_client, HTTPHandler)
        # The handler must use THIS client's httpx client (not litellm's default).
        assert r._http.async_client.client is r._http.httpx_async
        assert r._http.sync_client.client is r._http.httpx_sync
        assert _configured_keepalive(r) == 3
    finally:
        r.close()


def test_handler_wrapper_does_not_create_throwaway_async_client(monkeypatch):
    """Building handler wrappers should not leak a throwaway AsyncHTTPHandler client."""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    from nooa.unifiedllm import ResponsesClient

    def fail_create_client(*args, **kwargs):
        raise AssertionError("AsyncHTTPHandler.__init__ created a throwaway client")

    monkeypatch.setattr(AsyncHTTPHandler, "create_client", fail_create_client)
    c = ResponsesClient("openai/gpt-5.3-codex")
    try:
        assert isinstance(c._http.async_client, AsyncHTTPHandler)
        assert c._http.async_client.client is c._http.httpx_async
    finally:
        c.close()


@pytest.mark.asyncio
async def test_aclose_is_idempotent_after_aclose_and_close(monkeypatch):
    """aclose() should match close() idempotency and not double-close transports."""
    from nooa.unifiedllm import CompletionClient

    c = CompletionClient("anthropic/claude-3-5-sonnet")
    calls = 0
    original_aclose = c._http.httpx_async.aclose

    async def counted_aclose():
        nonlocal calls
        calls += 1
        await original_aclose()

    monkeypatch.setattr(c._http.httpx_async, "aclose", counted_aclose)
    await c.aclose()
    await c.aclose()
    c.close()
    assert calls == 1


@pytest.mark.asyncio
async def test_aclose_releases_async_transport_after_close(monkeypatch):
    """close() must not prevent a later aclose() from releasing async resources."""
    from nooa.unifiedllm import CompletionClient

    c = CompletionClient("anthropic/claude-3-5-sonnet")
    calls = 0
    original_aclose = c._http.httpx_async.aclose

    async def counted_aclose():
        nonlocal calls
        calls += 1
        await original_aclose()

    monkeypatch.setattr(c._http.httpx_async, "aclose", counted_aclose)
    c.close()
    assert c._http.httpx_sync.is_closed
    assert not c._http.httpx_async.is_closed
    await c.aclose()
    await c.aclose()
    assert c._http.httpx_async.is_closed
    assert calls == 1


def test_completion_openai_family_uses_openai_sdk_wrapping_httpx(monkeypatch):
    """OpenAI-family completion clients wrap their httpx client in an AsyncOpenAI/OpenAI."""
    from openai import AsyncOpenAI, OpenAI

    from nooa.unifiedllm import CompletionClient, HttpConfig

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    c = CompletionClient("gpt-4o-mini", http_config=HttpConfig(max_keepalive_connections=0))
    try:
        assert isinstance(c._http.async_client, AsyncOpenAI)
        assert isinstance(c._http.sync_client, OpenAI)
        # The OpenAI SDK client must route through our httpx client.
        assert c._http.async_client._client is c._http.httpx_async
        assert _configured_keepalive(c) == 0
    finally:
        c.close()


def test_completion_non_openai_uses_handler():
    """Anthropic/bedrock completion clients use the AsyncHTTPHandler/HTTPHandler wrappers."""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    from nooa.unifiedllm import CompletionClient

    c = CompletionClient("anthropic/claude-3-5-sonnet")
    try:
        assert isinstance(c._http.async_client, AsyncHTTPHandler)
        assert isinstance(c._http.sync_client, HTTPHandler)
        assert c._http.async_client.client is c._http.httpx_async
    finally:
        c.close()


@pytest.mark.asyncio
async def test_aclose_closes_httpx_clients():
    """aclose() releases the per-client httpx resources."""
    from nooa.unifiedllm import CompletionClient

    c = CompletionClient("anthropic/claude-3-5-sonnet")
    async_client = c._http.httpx_async
    sync_client = c._http.httpx_sync
    await c.aclose()
    assert async_client.is_closed
    assert sync_client.is_closed


def test_httpx_clients_preserve_litellm_transport_hardening():
    """Per-client httpx clients keep litellm's SSL + redirect behaviour."""
    from nooa.unifiedllm import CompletionClient

    c = CompletionClient("anthropic/claude-3-5-sonnet")
    try:
        # follow_redirects must stay True (litellm's default), not httpx's False.
        assert c._http.httpx_async.follow_redirects is True
        assert c._http.httpx_sync.follow_redirects is True
        # ...while still applying this client's connection-pool limits.
        assert _configured_keepalive(c) == 0
    finally:
        c.close()


def test_call_forwards_client_to_litellm(monkeypatch):
    """CompletionClient.call must hand its own client object to litellm.completion."""
    import litellm

    from nooa.unifiedllm import CompletionClient

    captured = {}

    def fake_completion(**kwargs):
        captured["client"] = kwargs.get("client")
        raise RuntimeError("stop here — we only need the client kwarg")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(litellm, "completion", fake_completion)
    c = CompletionClient(
        "gpt-4o-mini",
        retry_config=None,  # no retries — fail fast after the single call
        http_config=HttpConfig(),
    )
    try:
        with pytest.raises(RuntimeError):
            c.call([{"role": "user", "content": "hi"}])
        assert captured["client"] is c._http.sync_client
    finally:
        c.close()


def test_responses_call_forwards_client_to_litellm(monkeypatch):
    """ResponsesClient.call must hand its own client object to litellm.responses."""
    import litellm

    from nooa.unifiedllm import ResponsesClient

    captured = {}

    def fake_responses(**kwargs):
        captured["client"] = kwargs.get("client")
        raise RuntimeError("stop here — we only need the client kwarg")

    monkeypatch.setattr(litellm, "responses", fake_responses)
    c = ResponsesClient("openai/gpt-5.3-codex", retry_config=None)
    try:
        with pytest.raises(RuntimeError):
            c.call([{"role": "user", "content": "hi"}])
        assert captured["client"] is c._http.sync_client
    finally:
        c.close()
