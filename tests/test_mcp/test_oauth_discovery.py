# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RFC 9728 OAuth authorization-server discovery in mcp/oauth.py."""

import httpx
import pytest

from nooa.mcp import oauth


def _client(handler):
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, follow_redirects=True)


@pytest.mark.asyncio
async def test_discovery_follows_www_authenticate_resource_metadata():
    """The 401 challenge's resource_metadata pointer drives discovery.

    Mirrors MaaS: the metadata path is NOT a suffix of the server URL, so the
    only way to find it is the WWW-Authenticate header.
    """
    server_url = "https://maas.prd.example.com/maas/jira/mcp"
    metadata_url = "https://maas.prd.example.com/.well-known/oauth-protected-resource/maas/jira/mcp"
    auth_server = "https://maas.prd.example.com/maas/auth/jira-callback"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "POST" and url == server_url:
            return httpx.Response(
                401,
                headers={
                    "www-authenticate": (
                        f'Bearer error="invalid_token", resource_metadata="{metadata_url}"'
                    )
                },
            )
        if url == metadata_url:
            return httpx.Response(200, json={"authorization_servers": [auth_server]})
        # Server-URL-relative well-known probes 404 (MaaS shape).
        return httpx.Response(404)

    async with _client(handler) as client:
        servers = await oauth._discover_authorization_servers(client, server_url)

    assert servers == [auth_server]


@pytest.mark.asyncio
async def test_discovery_falls_back_to_well_known_paths():
    """Servers that expose metadata at the conventional path still work."""
    server_url = "https://example.com/mcp"
    auth_server = "https://example.com/auth"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "POST" and url == server_url:
            return httpx.Response(401)  # no WWW-Authenticate header
        if url == "https://example.com/.well-known/oauth-protected-resource/mcp":
            return httpx.Response(200, json={"authorization_servers": [auth_server]})
        return httpx.Response(404)

    async with _client(handler) as client:
        servers = await oauth._discover_authorization_servers(client, server_url)

    assert servers == [auth_server]


@pytest.mark.asyncio
async def test_discovery_returns_empty_when_nothing_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with _client(handler) as client:
        servers = await oauth._discover_authorization_servers(client, "https://x.example/mcp")

    assert servers == []


@pytest.mark.asyncio
async def test_fetch_authorization_server_metadata():
    auth_server = "https://example.com/auth"
    meta = {
        "authorization_endpoint": f"{auth_server}/authorize",
        "token_endpoint": f"{auth_server}/token",
        "registration_endpoint": f"{auth_server}/register",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == f"{auth_server}/.well-known/oauth-authorization-server":
            return httpx.Response(200, json=meta)
        return httpx.Response(404)

    async with _client(handler) as client:
        result = await oauth._fetch_authorization_server_metadata(client, auth_server)

    assert result == meta


@pytest.mark.asyncio
async def test_resource_metadata_pointer_parses_header():
    server_url = "https://x.example/mcp"
    pointer = "https://x.example/.well-known/oauth-protected-resource/mcp"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, headers={"www-authenticate": f'Bearer resource_metadata="{pointer}"'}
        )

    async with _client(handler) as client:
        result = await oauth._resource_metadata_pointer(client, server_url)

    assert result == pointer


@pytest.mark.asyncio
async def test_resource_metadata_pointer_ignored_on_non_401():
    """A 200 response carrying a stray WWW-Authenticate header is not trusted."""
    pointer = "https://x.example/.well-known/oauth-protected-resource/mcp"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"www-authenticate": f'Bearer resource_metadata="{pointer}"'}
        )

    async with _client(handler) as client:
        result = await oauth._resource_metadata_pointer(client, "https://x.example/mcp")

    assert result is None


@pytest.mark.asyncio
async def test_dynamic_registration_uses_already_bound_callback_uri(monkeypatch):
    """Dynamic registration receives the exact callback URI owned by HTTPServer."""
    registered: list[str] = []

    original_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        data = dict(request.read() and __import__("json").loads(request.content))
        registered.extend(data["redirect_uris"])
        return httpx.Response(201, json={"client_id": "client-id", "client_secret": "secret"})

    monkeypatch.setattr(
        oauth.httpx,
        "AsyncClient",
        lambda *args, **kwargs: original_async_client(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ),
    )

    config = oauth.OAuthConfig(
        authorization_endpoint="https://example.com/authorize",
        token_endpoint="https://example.com/token",
        client_id=None,
        redirect_uri="http://localhost:0/callback",
        registration_endpoint="https://example.com/register",
        timeout=0.01,
    )
    handler_obj = oauth.OAuthHandler(config)

    with pytest.raises(RuntimeError, match="timed out"):
        await handler_obj._capture_code_via_local_server(open_browser=False)

    assert registered
    registered_uri = registered[0]
    actual_uri = handler_obj._actual_redirect_uri
    assert registered_uri == actual_uri
    parsed = oauth.urlparse(registered_uri)
    assert parsed.hostname == "localhost"
    assert parsed.port not in (None, 0)
    assert parsed.path == "/callback"


@pytest.mark.asyncio
async def test_authorize_fails_fast_when_callback_server_fails(monkeypatch):
    """OAuth must not fall back to input(), which blocks/corrupts the TUI."""
    config = oauth.OAuthConfig(
        authorization_endpoint="https://example.com/authorize",
        token_endpoint="https://example.com/token",
        client_id="client-id",
        redirect_uri="http://127.0.0.1:0/callback",
    )
    handler = oauth.OAuthHandler(config)

    async def fail_callback(open_browser: bool) -> str:
        raise OSError("port unavailable")

    def fail_input(*args, **kwargs):
        raise AssertionError("authorize() must not call input()")

    monkeypatch.setattr(handler, "_capture_code_via_local_server", fail_callback)
    monkeypatch.setattr("builtins.input", fail_input)

    with pytest.raises(RuntimeError, match="OAuth browser callback failed"):
        await handler.authorize(open_browser=False)


def test_token_is_expired_logic():
    fresh = oauth.OAuthToken(access_token="a", expires_in=3600, obtained_at=oauth.time.time())
    assert not fresh.is_expired()
    stale = oauth.OAuthToken(access_token="a", expires_in=10, obtained_at=oauth.time.time() - 100)
    assert stale.is_expired()
    no_exp = oauth.OAuthToken(access_token="a", expires_in=None)
    assert not no_exp.is_expired()


def test_token_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path))
    url = "https://maas.example/mcp"
    assert oauth._load_cached_token(url) is None

    token = oauth.OAuthToken(
        access_token="acc",
        refresh_token="ref",
        expires_in=3600,
        obtained_at=oauth.time.time(),
        client_id="client-id",
        client_secret="client-secret",
    )
    oauth._save_cached_token(url, token)

    loaded = oauth._load_cached_token(url)
    assert loaded is not None
    assert loaded.access_token == "acc"
    assert loaded.refresh_token == "ref"
    assert loaded.client_id == "client-id"
    assert loaded.client_secret == "client-secret"

    cache_file = tmp_path / ".nooa" / "mcp_tokens.json"
    assert cache_file.exists()
    # Owner-only permissions (0o600).
    assert (cache_file.stat().st_mode & 0o777) == 0o600


@pytest.mark.asyncio
async def test_handle_mcp_oauth_returns_cached_token(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path))
    url = "https://maas.example/mcp"
    oauth._save_cached_token(
        url,
        oauth.OAuthToken(access_token="cached", expires_in=3600, obtained_at=oauth.time.time()),
    )

    async def fail_discover(client, server_url):
        return []

    monkeypatch.setattr(oauth, "_discover_authorization_servers", fail_discover)

    token = await oauth.handle_mcp_oauth(url)
    assert token.access_token == "cached"


@pytest.mark.asyncio
async def test_handle_mcp_oauth_refreshes_expired_token(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path))
    url = "https://maas.example/mcp"
    oauth._save_cached_token(
        url,
        oauth.OAuthToken(
            access_token="old",
            refresh_token="ref",
            expires_in=10,
            obtained_at=oauth.time.time() - 100,
            client_id="cached-client",
            client_secret="cached-secret",
        ),
    )

    async def fake_discover(client, server_url):
        return ["https://maas.example/auth"]

    async def fake_metadata(client, auth_server):
        return {
            "authorization_endpoint": f"{auth_server}/authorize",
            "token_endpoint": f"{auth_server}/token",
            "registration_endpoint": f"{auth_server}/register",
        }

    async def fake_refresh(token_endpoint, client_id, refresh_token, client_secret=None):
        assert client_id == "cached-client"
        assert client_secret == "cached-secret"
        assert refresh_token == "ref"
        return oauth.OAuthToken(
            access_token="new", refresh_token="ref2", expires_in=3600, obtained_at=oauth.time.time()
        )

    monkeypatch.setattr(oauth, "_discover_authorization_servers", fake_discover)
    monkeypatch.setattr(oauth, "_fetch_authorization_server_metadata", fake_metadata)
    monkeypatch.setattr(oauth, "_refresh_access_token", fake_refresh)

    token = await oauth.handle_mcp_oauth(url)
    assert token.access_token == "new"
    # Refreshed token is persisted.
    assert oauth._load_cached_token(url).access_token == "new"


@pytest.mark.asyncio
async def test_manual_authorize_uses_oob_and_code_prompt(monkeypatch):
    """Manual mode registers the OOB redirect and reads the code via the prompt."""
    registered: list[str] = []
    original_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        registered.extend(body["redirect_uris"])
        return httpx.Response(201, json={"client_id": "oob-client"})

    monkeypatch.setattr(
        oauth.httpx,
        "AsyncClient",
        lambda *a, **k: original_async_client(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ),
    )

    seen_url: list[str] = []

    async def code_prompt(auth_url: str) -> str:
        seen_url.append(auth_url)
        return "pasted-code"

    config = oauth.OAuthConfig(
        authorization_endpoint="https://maas.example/auth/authorize",
        token_endpoint="https://maas.example/auth/token",
        client_id=None,
        redirect_uri="http://127.0.0.1:0/callback",
        registration_endpoint="https://maas.example/auth/register",
    )
    handler_obj = oauth.OAuthHandler(config, manual=True, code_prompt=code_prompt)

    code = await handler_obj.authorize(open_browser=False)

    assert code == "pasted-code"
    assert registered == ["urn:ietf:wg:oauth:2.0:oob"]
    assert seen_url and "redirect_uri=urn" in seen_url[0]
    assert handler_obj._actual_redirect_uri == "urn:ietf:wg:oauth:2.0:oob"


def test_extract_authorization_code_accepts_oob_callback_url():
    pasted = "urn:ietf:wg:oauth:2.0:oob?code=abc123&state=xyz"

    assert oauth._extract_authorization_code(pasted) == "abc123"


def test_extract_authorization_code_accepts_curl_command_from_maas_page():
    pasted = "curl 'urn:ietf:wg:oauth:2.0:oob?code=abc123&state=xyz'"

    assert oauth._extract_authorization_code(pasted) == "abc123"


def test_extract_authorization_code_preserves_raw_code():
    assert oauth._extract_authorization_code("abc123") == "abc123"


@pytest.mark.asyncio
async def test_handle_mcp_oauth_defaults_scope_from_resource_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path))
    seen_scopes: list[str | None] = []

    async def fake_resource_metadata(client, server_url):
        return {
            "authorization_servers": ["https://maas.example/auth"],
            "scopes_supported": ["READ", "WRITE"],
        }

    async def fake_metadata(client, auth_server):
        return {
            "authorization_endpoint": f"{auth_server}/authorize",
            "token_endpoint": f"{auth_server}/token",
            "registration_endpoint": f"{auth_server}/register",
        }

    async def fake_complete_flow(self, open_browser=True):
        seen_scopes.append(self.config.scope)
        return oauth.OAuthToken(access_token="token")

    monkeypatch.setattr(oauth, "_fetch_protected_resource_metadata", fake_resource_metadata)
    monkeypatch.setattr(oauth, "_fetch_authorization_server_metadata", fake_metadata)
    monkeypatch.setattr(oauth.OAuthHandler, "complete_flow", fake_complete_flow)

    await oauth.handle_mcp_oauth("https://maas.example/mcp", client_id="client-id")

    assert seen_scopes == ["READ WRITE"]


@pytest.mark.asyncio
async def test_authorize_falls_back_to_manual_when_no_browser(monkeypatch):
    """Headless sessions auto-use manual OOB when a code prompt is available."""
    monkeypatch.setattr(oauth, "_system_browser_available", lambda: False)

    config = oauth.OAuthConfig(
        authorization_endpoint="https://maas.example/auth/authorize",
        token_endpoint="https://maas.example/auth/token",
        client_id="client-id",
        redirect_uri="http://127.0.0.1:0/callback",
    )

    seen: list[str] = []

    async def code_prompt(auth_url: str) -> str:
        seen.append(auth_url)
        return "pasted-code"

    handler = oauth.OAuthHandler(config, code_prompt=code_prompt)

    async def fail_local(open_browser):
        raise AssertionError("loopback callback flow must not run headless")

    monkeypatch.setattr(handler, "_capture_code_via_local_server", fail_local)

    code = await handler.authorize(open_browser=True)

    assert code == "pasted-code"
    assert seen and seen[0].startswith("https://maas.example/auth/authorize")
    assert handler._actual_redirect_uri == "urn:ietf:wg:oauth:2.0:oob"


@pytest.mark.asyncio
async def test_authorize_headless_without_prompt_raises_actionable_error(monkeypatch):
    """Headless with no code prompt fails fast with config instructions, not a hang."""
    monkeypatch.setattr(oauth, "_system_browser_available", lambda: False)

    config = oauth.OAuthConfig(
        authorization_endpoint="https://maas.example/auth/authorize",
        token_endpoint="https://maas.example/auth/token",
        client_id="client-id",
        redirect_uri="http://127.0.0.1:0/callback",
    )
    handler = oauth.OAuthHandler(config)

    async def fail_local(open_browser):
        raise AssertionError("loopback callback flow must not run headless")

    monkeypatch.setattr(handler, "_capture_code_via_local_server", fail_local)

    with pytest.raises(RuntimeError, match="oauth_manual = true"):
        await handler.authorize(open_browser=True)


def test_system_browser_available_false_when_no_browser(monkeypatch):
    """Returns False when webbrowser.get() raises AND no launcher executable is on PATH."""

    def raise_error():
        raise oauth.webbrowser.Error("no browser")

    monkeypatch.setattr(oauth.webbrowser, "get", raise_error)
    # No xdg-open / sensible-browser / open / wslview launcher available either.
    monkeypatch.setattr(oauth.shutil, "which", lambda name: None)
    assert oauth._system_browser_available() is False


def test_system_browser_available_true_when_browser_present(monkeypatch):
    """Returns True when webbrowser.get() succeeds without raising."""
    monkeypatch.setattr(oauth.webbrowser, "get", lambda *a, **k: object())
    assert oauth._system_browser_available() is True


@pytest.mark.asyncio
async def test_authorize_uses_browser_open_hook_when_no_system_browser(monkeypatch):
    """With no in-process browser, a browser_open hook drives the loopback flow (not OOB)."""
    monkeypatch.setattr(oauth, "_system_browser_available", lambda: False)

    config = oauth.OAuthConfig(
        authorization_endpoint="https://maas.example/authorize",
        token_endpoint="https://maas.example/token",
        client_id="cid",
        redirect_uri="http://localhost:0/callback",
    )

    opened: list[str] = []

    async def browser_open(url: str) -> bool:
        opened.append(url)
        return True

    handler = oauth.OAuthHandler(config, browser_open=browser_open)

    captured = {}

    async def fake_capture(open_browser):
        captured["open_browser"] = open_browser
        return "the-code"

    monkeypatch.setattr(handler, "_capture_code_via_local_server", fake_capture)

    code = await handler.authorize(open_browser=True)

    # Loopback flow runs (not OOB); the hook is available for it to use.
    assert code == "the-code"
    assert captured["open_browser"] is True


@pytest.mark.asyncio
async def test_capture_routes_through_browser_open_hook(monkeypatch):
    """_capture_code_via_local_server opens the auth URL via the hook, skipping webbrowser."""
    config = oauth.OAuthConfig(
        authorization_endpoint="https://maas.example/authorize",
        token_endpoint="https://maas.example/token",
        client_id="cid",
        redirect_uri="http://localhost:0/callback",
        timeout=0.05,
    )

    opened: list[str] = []

    async def browser_open(url: str) -> bool:
        opened.append(url)
        return True

    async def no_register(redirect_uri):
        return None

    def boom(*a, **k):
        raise AssertionError("webbrowser.open must not be called when the hook succeeds")

    handler = oauth.OAuthHandler(config, browser_open=browser_open)
    monkeypatch.setattr(handler, "_register_dynamic_client", no_register)
    monkeypatch.setattr(oauth.webbrowser, "open", boom)

    # Times out waiting for a callback (no real browser), but the hook must have fired first.
    with pytest.raises(RuntimeError, match="timed out"):
        await handler._capture_code_via_local_server(open_browser=True)

    assert opened and opened[0].startswith("https://maas.example/authorize")


@pytest.mark.asyncio
async def test_authorize_prefers_browser_open_over_manual_oob(monkeypatch):
    """When both a hook and a code prompt exist headless, the hook (loopback) wins over OOB."""
    monkeypatch.setattr(oauth, "_system_browser_available", lambda: False)

    config = oauth.OAuthConfig(
        authorization_endpoint="https://maas.example/authorize",
        token_endpoint="https://maas.example/token",
        client_id="cid",
        redirect_uri="http://localhost:0/callback",
    )

    async def browser_open(url: str) -> bool:
        return True

    async def code_prompt(url: str) -> str:
        raise AssertionError("manual OOB must not run when a browser_open hook is available")

    handler = oauth.OAuthHandler(config, code_prompt=code_prompt, browser_open=browser_open)

    async def fake_capture(open_browser):
        return "loopback-code"

    monkeypatch.setattr(handler, "_capture_code_via_local_server", fake_capture)

    assert await handler.authorize(open_browser=True) == "loopback-code"


def test_default_redirect_uri_uses_localhost():
    """Default loopback redirect must use localhost; some MaaS gateways reject 127.0.0.1."""
    import inspect

    sig = inspect.signature(oauth.handle_mcp_oauth)
    default = sig.parameters["redirect_uri"].default
    assert default == "http://localhost:0/callback"


@pytest.mark.asyncio
async def test_client_credentials_token_success(monkeypatch):
    """client_credentials_token posts the grant and returns the access token."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["data"] = dict(httpx.QueryParams(request.content.decode()))
        return httpx.Response(200, json={"access_token": "cc-token", "expires_in": 3600})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        oauth.httpx,
        "AsyncClient",
        lambda *a, **k: original(transport=httpx.MockTransport(handler)),
    )

    config = oauth.OAuthConfig(
        authorization_endpoint="https://maas.example/authorize",
        token_endpoint="https://maas.example/token",
        client_id="cid",
        client_secret="secret",
        redirect_uri="http://localhost:0/callback",
        scope="a b",
    )
    token = await oauth.OAuthHandler(config).client_credentials_token()

    assert token.access_token == "cc-token"
    assert captured["data"]["grant_type"] == "client_credentials"
    assert captured["data"]["client_id"] == "cid"
    assert captured["data"]["client_secret"] == "secret"
    assert captured["data"]["scope"] == "a b"


@pytest.mark.asyncio
async def test_client_credentials_token_requires_secret():
    """Without a client_secret the grant fails fast with a clear message."""
    config = oauth.OAuthConfig(
        authorization_endpoint="https://maas.example/authorize",
        token_endpoint="https://maas.example/token",
        client_id="cid",
        redirect_uri="http://localhost:0/callback",
    )
    with pytest.raises(RuntimeError, match="requires a client_id and client_secret"):
        await oauth.OAuthHandler(config).client_credentials_token()


@pytest.mark.asyncio
async def test_client_credentials_token_missing_access_token(monkeypatch):
    """A 200 body without access_token raises a descriptive error, not KeyError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"token_type": "Bearer"})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        oauth.httpx,
        "AsyncClient",
        lambda *a, **k: original(transport=httpx.MockTransport(handler)),
    )

    config = oauth.OAuthConfig(
        authorization_endpoint="https://maas.example/authorize",
        token_endpoint="https://maas.example/token",
        client_id="cid",
        client_secret="secret",
        redirect_uri="http://localhost:0/callback",
    )
    with pytest.raises(RuntimeError, match="missing 'access_token'"):
        await oauth.OAuthHandler(config).client_credentials_token()


@pytest.mark.asyncio
async def test_handle_mcp_oauth_prefers_client_credentials(monkeypatch, tmp_path):
    """When the server advertises client_credentials and a secret exists, use it (no browser)."""
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path))
    url = "https://maas.example/mcp"

    async def fake_resource_metadata(client, server_url):
        return {"authorization_servers": ["https://maas.example/auth"]}

    async def fake_metadata(client, auth_server):
        return {
            "authorization_endpoint": f"{auth_server}/authorize",
            "token_endpoint": f"{auth_server}/token",
            "registration_endpoint": f"{auth_server}/register",
            "grant_types_supported": ["authorization_code", "client_credentials"],
        }

    called = {}

    async def fake_cc(self):
        called["cc"] = True
        return oauth.OAuthToken(access_token="cc-token", client_id=self.config.client_id)

    async def fail_complete_flow(self, open_browser=True):
        raise AssertionError("interactive flow must not run when client_credentials is available")

    monkeypatch.setattr(oauth, "_fetch_protected_resource_metadata", fake_resource_metadata)
    monkeypatch.setattr(oauth, "_fetch_authorization_server_metadata", fake_metadata)
    monkeypatch.setattr(oauth.OAuthHandler, "client_credentials_token", fake_cc)
    monkeypatch.setattr(oauth.OAuthHandler, "complete_flow", fail_complete_flow)

    token = await oauth.handle_mcp_oauth(
        url, client_id="cid", client_secret="secret", use_cache=False
    )

    assert token.access_token == "cc-token"
    assert called.get("cc") is True
