# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP sign-in through the MCP SDK's OAuth provider, with a pasted redirect URL."""

import asyncio
import json
import stat
import time
from datetime import timedelta

import httpx
import pytest
from fake_oauth_mcp import FakeOAuthServer
from mcp.client.auth.exceptions import OAuthFlowError
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from nooa_atom.skills.mcp_auth import (
    AuthorizedHTTPClient,
    FileTokenStorage,
    PastedSignIn,
    oauth_provider,
)

_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@pytest.fixture
def server(monkeypatch):
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    fake = FakeOAuthServer().start()
    yield fake
    fake.stop()


@pytest.fixture
def alternate_server(monkeypatch):
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    issuer = FakeOAuthServer().start()
    resource = FakeOAuthServer(authorization_server=issuer)
    resource.valid_tokens = issuer.valid_tokens
    resource.start()
    yield resource, issuer
    resource.stop()
    issuer.stop()


def _client(server, signin, tokens, client_id=None):
    storage = FileTokenStorage(server.url, tokens, client_id=client_id)
    return AuthorizedHTTPClient(server.url, provider=oauth_provider(signin, storage), signin=signin)


async def _tool_names(client) -> list[str]:
    async with client.connect_to_server() as session:
        return [tool.name for tool in (await session.list_tools()).tools]


async def _sign_in(server, tokens, **kwargs) -> PastedSignIn:
    signin = PastedSignIn()
    listing = asyncio.create_task(_tool_names(_client(server, signin, tokens, **kwargs)))
    await asyncio.wait_for(signin.url_ready.wait(), 10)
    pasted = await asyncio.to_thread(server.consent, signin.url)
    signin.complete(pasted)
    assert await asyncio.wait_for(listing, 10) == ["echo"]
    return signin


async def test_signing_in_with_the_pasted_redirect_url_stores_the_tokens(server, tmp_path):
    tokens = tmp_path / "tokens.json"
    signin = await _sign_in(server, tokens)
    assert signin.url.startswith(f"{server.base}/authorize?")
    assert server.grants == ["authorization_code"]
    assert server.registrations == 1
    saved = json.loads(tokens.read_text())[server.url]
    assert saved["tokens"]["access_token"] in server.valid_tokens
    assert saved["client_info"]["client_id"] == "client-1"
    assert stat.S_IMODE(tokens.stat().st_mode) == 0o600


async def test_stored_tokens_are_used_without_a_new_sign_in(server, tmp_path):
    tokens = tmp_path / "tokens.json"
    await _sign_in(server, tokens)
    signin = PastedSignIn()
    assert await asyncio.wait_for(_tool_names(_client(server, signin, tokens)), 10) == ["echo"]
    assert signin.url is None
    assert server.grants == ["authorization_code"]
    assert server.registrations == 1


async def test_an_expired_stored_token_is_refreshed_without_a_new_sign_in(server, tmp_path):
    tokens = tmp_path / "tokens.json"
    await _sign_in(server, tokens)
    data = json.loads(tokens.read_text())
    data[server.url]["expires_at"] = 0
    tokens.write_text(json.dumps(data))
    server.expire_all_tokens()

    signin = PastedSignIn()
    assert await asyncio.wait_for(_tool_names(_client(server, signin, tokens)), 10) == ["echo"]
    assert signin.url is None
    assert server.grants == ["authorization_code", "refresh_token"]


async def test_a_configured_client_id_is_used_without_registering(server, tmp_path):
    signin = await _sign_in(server, tmp_path / "tokens.json", client_id="configured-client")
    assert "client_id=configured-client" in signin.url
    assert server.registrations == 0


@pytest.mark.parametrize(
    "pasted",
    [
        "http://localhost/callback?code=abc&state=xyz",
        "  curl 'http://localhost/callback?code=abc&state=xyz'  ",
    ],
)
async def test_the_pasted_redirect_gives_the_code_and_state(pasted):
    signin = PastedSignIn()
    signin.complete(pasted)
    assert await signin.callback() == ("abc", "xyz")


@pytest.mark.parametrize(
    ("pasted", "message"),
    [
        ("http://localhost/callback?state=xyz", "no authorisation code"),
        ("http://localhost/callback?error=access_denied&error_description=No", "access_denied: No"),
        ("not a url", "no authorisation code"),
    ],
)
async def test_a_pasted_redirect_without_a_code_is_refused(pasted, message):
    signin = PastedSignIn()
    with pytest.raises(ValueError, match=message):
        signin.complete(pasted)


async def test_waiting_for_the_pasted_redirect_times_out():
    signin = PastedSignIn(timeout=0.05)
    with pytest.raises(TimeoutError, match="sign-in"):
        await signin.callback()


async def test_expired_cache_discovers_alternate_issuer_before_refresh(alternate_server, tmp_path):
    resource, issuer = alternate_server
    tokens = tmp_path / "tokens.json"
    await _sign_in(resource, tokens)
    data = json.loads(tokens.read_text())
    data[resource.url]["expires_at"] = 0
    tokens.write_text(json.dumps(data))
    issuer.expire_all_tokens()
    resource.requests.clear()
    issuer.requests.clear()

    signin = PastedSignIn(timeout=0.2)
    assert await asyncio.wait_for(_tool_names(_client(resource, signin, tokens)), 5) == ["echo"]
    assert signin.url is None
    assert issuer.grants == ["authorization_code", "refresh_token"]
    assert ("POST", "/token") not in resource.requests
    assert issuer.requests.index(("GET", "/.well-known/oauth-authorization-server")) < (
        issuer.requests.index(("POST", "/token"))
    )
    assert issuer.refresh_resources == [resource.url]


async def test_loaded_token_keeps_absolute_expiry(server, tmp_path):
    tokens = tmp_path / "tokens.json"
    await _sign_in(server, tokens)
    data = json.loads(tokens.read_text())
    expires_at = time.time() + 120
    data[server.url]["expires_at"] = expires_at
    tokens.write_text(json.dumps(data))
    provider = oauth_provider(PastedSignIn(), FileTokenStorage(server.url, tokens))
    async with httpx.AsyncClient(auth=provider, trust_env=False) as client:
        await client.post(server.url, json={})
    actual_expiry = provider.context.token_expiry_time
    assert actual_expiry == expires_at


async def test_rejected_cached_access_token_refreshes_once(alternate_server, tmp_path):
    resource, issuer = alternate_server
    tokens = tmp_path / "tokens.json"
    await _sign_in(resource, tokens)
    issuer.expire_all_tokens()  # Local expiry is still in the future.
    signin = PastedSignIn(timeout=0.2)
    assert await asyncio.wait_for(_tool_names(_client(resource, signin, tokens)), 5) == ["echo"]
    assert signin.url is None
    assert issuer.grant_attempts == ["authorization_code", "refresh_token"]
    assert ("POST", "/token") not in resource.requests


async def test_revoked_refresh_requires_legitimate_manual_sign_in(alternate_server, tmp_path):
    resource, issuer = alternate_server
    tokens = tmp_path / "tokens.json"
    await _sign_in(resource, tokens)
    data = json.loads(tokens.read_text())
    data[resource.url]["expires_at"] = 0
    tokens.write_text(json.dumps(data))
    issuer.expire_all_tokens()
    issuer.revoke_refresh_tokens()
    await _sign_in(resource, tokens)
    assert issuer.grant_attempts == ["authorization_code", "refresh_token", "authorization_code"]
    assert ("POST", "/token") not in resource.requests


async def test_revoked_refresh_at_alternate_issuer_preserves_human_wait_budget(
    alternate_server, tmp_path
):
    resource, issuer = alternate_server
    tokens = tmp_path / "tokens.json"
    await _sign_in(resource, tokens)
    storage = FileTokenStorage(resource.url, tokens)
    storage._update(expires_at=0)
    issuer.expire_all_tokens()
    issuer.revoke_refresh_tokens()

    signin = PastedSignIn(timeout=5)
    provider = oauth_provider(signin, storage)
    client = AuthorizedHTTPClient(
        resource.url,
        provider=provider,
        signin=signin,
        tool_call_timeout=timedelta(seconds=0.5),
    )
    listing = asyncio.create_task(_tool_names(client))
    try:
        await asyncio.wait_for(signin.url_ready.wait(), 5)
        # The discovering generator falls back to the SDK callback after the
        # refresh is rejected. Only that human wait may pause the MCP budget.
        await asyncio.sleep(1)
        assert not listing.done()
        assert issuer.grant_attempts == ["authorization_code", "refresh_token"]
        pasted = await asyncio.to_thread(resource.consent, signin.url)
        signin.complete(pasted)
        assert await asyncio.wait_for(listing, 5) == ["echo"]
        assert issuer.grant_attempts == [
            "authorization_code",
            "refresh_token",
            "authorization_code",
        ]
        assert ("POST", "/token") not in resource.requests
        assert storage.get_issuer() == f"{issuer.base}/"
        assert storage.get_token_expiry() > time.time()
        assert not provider.context.lock.locked()
        assert signin._budgets == {}
    finally:
        listing.cancel()
        await asyncio.gather(listing, return_exceptions=True)


async def test_expired_cache_can_discover_challenge_only_metadata(alternate_server, tmp_path):
    resource, issuer = alternate_server
    resource.custom_resource_metadata = True
    tokens = tmp_path / "tokens.json"
    await _sign_in(resource, tokens)
    data = json.loads(tokens.read_text())
    data[resource.url]["expires_at"] = 0
    tokens.write_text(json.dumps(data))
    issuer.expire_all_tokens()
    signin = PastedSignIn(timeout=0.2)
    assert await asyncio.wait_for(_tool_names(_client(resource, signin, tokens)), 5) == ["echo"]
    assert signin.url is None
    assert issuer.grants == ["authorization_code", "refresh_token"]
    assert ("POST", "/token") not in resource.requests


async def test_repeated_401_after_refresh_does_not_refresh_again(alternate_server, tmp_path):
    resource, issuer = alternate_server
    tokens = tmp_path / "tokens.json"
    await _sign_in(resource, tokens)
    issuer.expire_all_tokens()
    issuer.reject_refreshed_tokens = True
    await _sign_in(resource, tokens)
    assert issuer.grant_attempts == ["authorization_code", "refresh_token", "authorization_code"]


async def test_legacy_cache_gets_issuer_binding_after_refresh(alternate_server, tmp_path):
    resource, issuer = alternate_server
    tokens = tmp_path / "tokens.json"
    await _sign_in(resource, tokens)
    data = json.loads(tokens.read_text())
    data[resource.url].pop("issuer")
    data[resource.url]["expires_at"] = 0
    tokens.write_text(json.dumps(data))
    issuer.expire_all_tokens()
    signin = PastedSignIn(timeout=0.2)
    assert await asyncio.wait_for(_tool_names(_client(resource, signin, tokens)), 5) == ["echo"]
    saved_issuer = FileTokenStorage(resource.url, tokens).get_issuer()
    assert saved_issuer == f"{issuer.base}/"


@pytest.mark.parametrize("failure", ["resource", "issuer", "bound_issuer"])
async def test_discovery_rejects_unsafe_refresh_without_sending_credentials(tmp_path, failure):
    url = "https://resource.example/mcp"
    issuer = "https://issuer.example/tenant"
    storage = FileTokenStorage(url, tmp_path / "tokens.json")
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            client_secret="test-secret",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="client_secret_post",
        )
    )
    await storage.set_tokens(
        OAuthToken(
            access_token="test-access",
            refresh_token="test-refresh",
            token_type="Bearer",
            expires_in=1,
        )
    )
    storage._update(expires_at=0)
    if failure == "bound_issuer":
        storage._update(issuer="https://previous-issuer.example/tenant")
    requests = []

    def respond(request):
        requests.append((request.method, str(request.url)))
        assert "authorization" not in request.headers
        assert request.method == "GET"  # No token/client credentials may be posted.
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "resource": "https://unrelated.example/mcp" if failure == "resource" else url,
                    "authorization_servers": [issuer],
                },
            )
        return httpx.Response(
            200,
            json={
                "issuer": "https://unrelated.example/tenant" if failure == "issuer" else issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": "https://token.example/exchange",
            },
        )

    signin = PastedSignIn()
    provider = oauth_provider(signin, storage)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(OAuthFlowError):
            await client.get(url)
    assert signin.url is None
    assert requests
    # An exception must close the SDK generator and release its context lock.
    assert not provider.context.lock.locked()


async def test_runtime_expiry_refresh_uses_sdk_client_auth_and_discovered_endpoint(tmp_path):
    url = "https://resource.example/mcp"
    issuer = "https://issuer.example/tenant"
    token_endpoint = "https://token.example/exchange"
    storage = FileTokenStorage(url, tmp_path / "tokens.json")
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            client_secret="test-secret",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="client_secret_basic",
        )
    )
    await storage.set_tokens(
        OAuthToken(
            access_token="test-access",
            refresh_token="test-refresh",
            token_type="Bearer",
            expires_in=3600,
        )
    )
    methods = []

    def respond(request):
        methods.append(request.method)
        if str(request.url) == url:
            return httpx.Response(200)
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(200, json={"resource": url, "authorization_servers": [issuer]})
        if str(request.url) == token_endpoint:
            assert request.headers.get("authorization", "").startswith("Basic ")
            assert b"client_secret=" not in request.content
            assert b"resource=https%3A%2F%2Fresource.example%2Fmcp" in request.content
            return httpx.Response(
                200,
                json={
                    "access_token": "fresh-access",
                    "refresh_token": "fresh-refresh",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        return httpx.Response(
            200,
            json={
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": token_endpoint,
            },
        )

    signin = PastedSignIn()
    provider = oauth_provider(signin, storage)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        await client.get(url)
        provider.context.token_expiry_time = (
            time.time() - 1
        )  # Simulate this provider's clock expiring.
        await client.get(url)
    assert methods.count("POST") == 1
    assert signin.url is None


@pytest.mark.parametrize("missing_metadata", [False, True])
async def test_manual_auth_does_not_reuse_credentials_at_changed_issuer(tmp_path, missing_metadata):
    url = "https://resource.example/mcp"
    issuer = "https://new-issuer.example/tenant"
    storage = FileTokenStorage(url, tmp_path / "tokens.json")
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            client_secret="test-secret",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="client_secret_post",
        )
    )
    await storage.set_tokens(
        OAuthToken(access_token="test-access", token_type="Bearer", expires_in=3600)
    )
    storage._update(issuer="https://old-issuer.example/tenant")
    posts = []

    def respond(request):
        if request.method == "POST":
            posts.append(str(request.url))
            return httpx.Response(400)
        if str(request.url) == url:
            return httpx.Response(401)
        if missing_metadata:
            return httpx.Response(404)
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(200, json={"resource": url, "authorization_servers": [issuer]})
        return httpx.Response(
            200,
            json={
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": f"{issuer}/token",
            },
        )

    signin = PastedSignIn()
    provider = oauth_provider(signin, storage)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(OAuthFlowError):
            await client.get(url)
    assert signin.url is None
    assert posts == []


async def test_cancelled_discovery_releases_lock_and_concurrent_requests_refresh_once(tmp_path):
    url = "https://resource.example/mcp"
    issuer = "https://issuer.example/tenant"
    storage = FileTokenStorage(url, tmp_path / "tokens.json", client_id="test-client")
    await storage.set_tokens(
        OAuthToken(
            access_token="test-access",
            refresh_token="test-refresh",
            token_type="Bearer",
            expires_in=1,
        )
    )
    storage._update(expires_at=0)
    entered = asyncio.Event()
    resume = asyncio.Event()
    refreshes = []

    async def respond(request):
        if str(request.url) == url:
            return httpx.Response(200)
        if "oauth-protected-resource" in request.url.path:
            entered.set()
            await resume.wait()
            return httpx.Response(200, json={"resource": url, "authorization_servers": [issuer]})
        if request.method == "POST":
            refreshes.append(str(request.url))
            return httpx.Response(
                200,
                json={
                    "access_token": "fresh-access",
                    "refresh_token": "fresh-refresh",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        return httpx.Response(
            200,
            json={
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": f"{issuer}/token",
            },
        )

    signin = PastedSignIn()
    provider = oauth_provider(signin, storage)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        task = asyncio.create_task(client.get(url))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not provider.context.lock.locked()
        resume.set()
        results = await asyncio.wait_for(asyncio.gather(client.get(url), client.get(url)), 2)
    assert [response.status_code for response in results] == [200, 200]
    assert len(refreshes) == 1
    assert signin.url is None


@pytest.mark.parametrize("expired", [True, False])
async def test_native_sdk_issuer_binding_blocks_cross_issuer_refresh(tmp_path, expired):
    url = "https://resource.example/mcp"
    issuer = "https://new-issuer.example/tenant"
    storage = FileTokenStorage(url, tmp_path / "tokens.json")
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            client_secret="test-secret",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="client_secret_post",
        )
    )
    # Write a native SDK 1.30 cache verbatim, including when testing SDK 1.28's
    # older client-info model (which would otherwise silently drop this field).
    saved_client = storage._entry()["client_info"]
    storage._update(client_info={**saved_client, "issuer": "https://old-issuer.example/tenant"})
    await storage.set_tokens(
        OAuthToken(
            access_token="test-access",
            refresh_token="test-refresh",
            token_type="Bearer",
            expires_in=3600,
        )
    )
    if expired:
        storage._update(expires_at=0)
    assert storage.get_issuer() is None  # A cache written by the SDK, not Atom.
    posts = []

    def respond(request):
        if request.method == "POST":
            # Capture only endpoint/method; never retain raw credential payloads.
            posts.append(str(request.url))
            return httpx.Response(
                200,
                json={
                    "access_token": "fresh-access",
                    "refresh_token": "fresh-refresh",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        if str(request.url) == url:
            return httpx.Response(200 if posts else 401)
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(200, json={"resource": url, "authorization_servers": [issuer]})
        return httpx.Response(
            200,
            json={
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": f"{issuer}/token",
            },
        )

    signin = PastedSignIn()
    provider = oauth_provider(signin, storage)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(OAuthFlowError):
            await client.get(url)
    assert posts == []
    assert signin.url is None
    assert not provider.context.lock.locked()


@pytest.mark.parametrize(
    ("native_issuer", "token_issuer", "configured_cimd", "refresh", "allowed"),
    [
        (None, None, False, True, True),  # Unbound legacy credentials.
        ("https://issuer.example/tenant", None, False, True, True),
        ("https://old-issuer.example/tenant", None, False, True, False),
        ("https://old-issuer.example/tenant", None, False, False, False),
        ("https://old-issuer.example/tenant", None, True, False, True),  # Portable client identity.
        (
            "https://old-issuer.example/tenant",
            None,
            True,
            True,
            False,
        ),  # Not a portable refresh token.
        (None, "https://old-issuer.example/tenant", True, False, True),
        (None, "https://old-issuer.example/tenant", True, True, False),
        (None, "https://old-issuer.example/tenant", False, False, False),
        ("https://issuer.example/tenant", "https://old-issuer.example/tenant", True, True, False),
    ],
)
async def test_native_binding_and_configured_cimd_matching(
    tmp_path, native_issuer, token_issuer, configured_cimd, refresh, allowed
):
    storage = FileTokenStorage("https://resource.example/mcp", tmp_path / "tokens.json")
    client_id = "https://client.example/metadata.json"  # URL shape alone must not imply CIMD.
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id=client_id,
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="none",
        )
    )
    storage._update(client_info={**storage._entry()["client_info"], "issuer": native_issuer})
    if token_issuer is not None:
        storage._update(issuer=token_issuer)
    provider = oauth_provider(PastedSignIn(), storage)
    if configured_cimd:
        provider.context.client_metadata_url = client_id
    await provider._initialize()
    if allowed:
        provider._validate_cached_issuer("https://issuer.example/tenant", refresh=refresh)
    else:
        with pytest.raises(OAuthFlowError):
            provider._validate_cached_issuer("https://issuer.example/tenant", refresh=refresh)


@pytest.mark.parametrize(
    ("native_issuer", "allowed"),
    [
        ("https://issuer.example", True),
        ("https://issuer.example/", True),
        ("https://issuer.example/tenant", False),
        ("https://ISSUER.example/", False),  # SDK issuer comparison is not URL canonicalization.
    ],
)
async def test_native_binding_uses_sdk_exact_issuer_comparison(tmp_path, native_issuer, allowed):
    storage = FileTokenStorage("https://resource.example/mcp", tmp_path / "tokens.json")
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="none",
        )
    )
    storage._update(client_info={**storage._entry()["client_info"], "issuer": native_issuer})
    provider = oauth_provider(PastedSignIn(), storage)
    await provider._initialize()
    if allowed:
        provider._validate_cached_issuer("https://issuer.example/", refresh=True)
    else:
        with pytest.raises(OAuthFlowError):
            provider._validate_cached_issuer("https://issuer.example/", refresh=True)


async def test_replacing_client_registration_preserves_native_refresh_issuer(tmp_path):
    storage = FileTokenStorage("https://resource.example/mcp", tmp_path / "tokens.json")
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="old-client",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="none",
        )
    )
    storage._update(
        client_info={**storage._entry()["client_info"], "issuer": "https://old-issuer.example/"}
    )
    await storage.set_tokens(
        OAuthToken(
            access_token="old-access",
            refresh_token="old-refresh",
            token_type="Bearer",
            expires_in=3600,
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="new-client",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="none",
        )
    )
    actual_issuer = storage.get_issuer()
    assert actual_issuer == "https://old-issuer.example/"


async def test_cancelled_native_sdk_step_up_preserves_refresh_binding_on_restart(tmp_path):
    if "issuer" not in OAuthClientInformationFull.model_fields:
        pytest.skip("SDK 1.30 supports metadata-discovering scope step-up")
    url = "https://resource.example/mcp"
    issuer = "https://new-issuer.example/tenant"
    storage = FileTokenStorage(url, tmp_path / "tokens.json")
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="old-client",
            issuer="https://old-issuer.example/tenant",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="none",
        )
    )
    await storage.set_tokens(
        OAuthToken(
            access_token="old-access",
            refresh_token="old-refresh",
            token_type="Bearer",
            expires_in=3600,
        )
    )
    posts = []
    step_up = True

    def respond(request):
        if str(request.url) == url:
            if step_up:
                return httpx.Response(
                    403, headers={"WWW-Authenticate": 'Bearer error="insufficient_scope"'}
                )
            return httpx.Response(200)
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(200, json={"resource": url, "authorization_servers": [issuer]})
        if request.method == "POST":
            posts.append(str(request.url))
            if request.url.path.endswith("/register"):
                return httpx.Response(
                    201,
                    json={
                        "client_id": "new-client",
                        "redirect_uris": ["http://localhost/callback"],
                        "token_endpoint_auth_method": "none",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "access_token": "fresh-access",
                    "refresh_token": "fresh-refresh",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        return httpx.Response(
            200,
            json={
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": f"{issuer}/token",
                "registration_endpoint": f"{issuer}/register",
            },
        )

    original = storage.path.read_text()
    provider = oauth_provider(PastedSignIn(timeout=0.05), storage)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        # Fail before the SDK can discard native binding or register at a new
        # issuer. A scope step-up is not permission for automatic migration.
        with pytest.raises(OAuthFlowError):
            await client.get(url)
    assert posts == []
    assert storage.path.read_text() == original
    assert provider.context.client_info.client_id == "old-client"
    step_up = False
    storage._update(expires_at=0)
    restarted = oauth_provider(PastedSignIn(timeout=0.05), storage)
    async with httpx.AsyncClient(auth=restarted, transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(OAuthFlowError):
            await client.get(url)
    assert posts == []
