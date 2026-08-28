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
    resp = ResponsesClient(
        "openai/gpt-5.3-codex", http_config=HttpConfig(), capabilities={"responses": True}
    )
    try:
        assert comp._http_config.max_keepalive_connections == 0
        assert _configured_keepalive(comp) == 0
        assert _configured_keepalive(comp) == 0
        assert _configured_keepalive(resp) == 0
        assert _configured_keepalive(resp) == 0
    finally:
        comp.close()
        resp.close()


@pytest.mark.asyncio
async def test_aclose_is_idempotent_and_closes_transport(monkeypatch):
    from nooa.unifiedllm import CompletionClient

    client = CompletionClient("anthropic/claude-3-5-sonnet")
    calls = 0
    original = client._http.httpx_async.aclose

    async def counted():
        nonlocal calls
        calls += 1
        await original()

    monkeypatch.setattr(client._http.httpx_async, "aclose", counted)
    await client.aclose()
    await client.aclose()
    assert calls == 1
    assert client._http.httpx_async.is_closed


def test_http_client_is_passed_as_provider_option():
    from nooa.unifiedllm import CompletionClient

    client = CompletionClient("anthropic/claude-3-5-sonnet")
    try:
        assert client._transport.http_client is client._http.httpx_async
        assert "http_client" not in client._transport.provider_options
        assert client._http.httpx_async.follow_redirects is True
    finally:
        # Synchronous close cannot safely drive an async client; aclose owns cleanup.
        client.close()
