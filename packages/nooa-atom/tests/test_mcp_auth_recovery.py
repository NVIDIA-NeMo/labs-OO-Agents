# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline regressions for explicit local forgetting and legacy OAuth recovery."""

import asyncio
import json
import stat
from types import SimpleNamespace

import httpx
import pytest
import test_mcp_auth as auth_fixtures
import test_mcp_servers as server_fixtures
from mcp.client.auth.exceptions import OAuthFlowError
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from nooa_atom.skills.mcp_auth import (
    AuthorizedHTTPClient,
    FileTokenStorage,
    PastedSignIn,
    SignInTimeoutError,
    oauth_provider,
)
from nooa_atom.skills.mcp_servers import MCPSignInRequired
from test_mcp_servers import _control, approve

server = auth_fixtures.server
servers = server_fixtures.servers
oauth_server = server_fixtures.oauth_server


async def seed(storage, *, expired=True, issuer="https://resource.example/"):
    """A bound registration containing deliberately unusable cached tokens."""
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="cached-client",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="none",
        )
    )
    storage._update(
        client_info={**storage._entry()["client_info"], "issuer": issuer}, issuer=issuer
    )
    await storage.set_tokens(
        OAuthToken(
            access_token="stale-access",
            refresh_token="stale-refresh",
            token_type="Bearer",
            expires_in=3600,
        )
    )
    if expired:
        storage._update(expires_at=0)


@pytest.mark.parametrize("expired", [True, False])
async def test_absent_prm_same_origin_completes_manual_tool_call(server, tmp_path, expired):
    tokens = tmp_path / "tokens.json"
    await auth_fixtures._sign_in(server, tokens)
    data = json.loads(tokens.read_text())
    if expired:
        data[server.url]["expires_at"] = 0
    tokens.write_text(json.dumps(data))
    server.expire_all_tokens()
    server.missing_resource_metadata = True
    server.requests.clear()
    await auth_fixtures._sign_in(server, tokens)
    signin = PastedSignIn()
    client = auth_fixtures._client(server, signin, tokens)
    async with client.connect_to_server() as session:
        result = await session.call_tool("echo", {"text": "legacy recovery"})
    assert result.content[0].text == "legacy recovery"
    assert server.grant_attempts == ["authorization_code", "authorization_code"]
    assert server.registrations == 1
    assert server.requests.index(
        ("GET", "/.well-known/oauth-authorization-server")
    ) < server.requests.index(("POST", "/token"))


@pytest.mark.parametrize("expired", [True, False])
@pytest.mark.parametrize("absent", [True, False])
async def test_cancelled_manual_persists_token_only_invalidation(tmp_path, expired, absent):
    url = "https://resource.example/mcp"
    storage = FileTokenStorage(url, tmp_path / "tokens.json")
    await seed(storage, expired=expired)
    registration = storage._entry()["client_info"]
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "POST":
            assert b"grant_type=refresh_token" in request.content
            assert not absent  # A guessed refresh must never hit the transport.
            return httpx.Response(400, json={"error": "invalid_grant"})
        if str(request.url) == url:
            return httpx.Response(401)
        assert "authorization" not in request.headers
        if "oauth-protected-resource" in request.url.path:
            return (
                httpx.Response(404)
                if absent
                else httpx.Response(
                    200,
                    json={"resource": url, "authorization_servers": ["https://resource.example/"]},
                )
            )
        return httpx.Response(
            200,
            json={
                "issuer": "https://resource.example/",
                "authorization_endpoint": "https://resource.example/authorize",
                "token_endpoint": "https://resource.example/token",
            },
        )

    signin = PastedSignIn(timeout=0.02)
    provider = oauth_provider(signin, storage)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(SignInTimeoutError):
            await client.get(url)
    assert signin.url is None
    entry = storage._entry()
    assert "tokens" not in entry and "expires_at" not in entry
    assert entry["client_info"] == registration
    assert entry["issuer"] == "https://resource.example/"
    assert not provider.context.lock.locked()
    restarted = oauth_provider(PastedSignIn(), FileTokenStorage(url, storage.path))
    await restarted._initialize()
    assert restarted.context.current_tokens is None
    assert restarted.context.client_info.client_id == "cached-client"
    resource_requests = [r for r in requests if str(r.url) == url]
    if expired:
        assert all("authorization" not in r.headers for r in resource_requests)


@pytest.mark.parametrize("binding", ["issuer", "client_info"])
@pytest.mark.parametrize("expired", [True, False])
async def test_absent_prm_alternate_binding_remains_fail_closed(tmp_path, binding, expired):
    url = "https://resource.example/mcp"
    storage = FileTokenStorage(url, tmp_path / "tokens.json")
    await seed(storage, expired=expired, issuer="https://alternate.example/tenant")
    if binding == "issuer":
        storage._update(
            client_info={k: v for k, v in storage._entry()["client_info"].items() if k != "issuer"}
        )
    else:
        storage._update(issuer=None)
    original = storage.path.read_text()
    requests = []

    def respond(request):
        requests.append(request)
        assert request.method == "GET"
        return httpx.Response(401 if str(request.url) == url else 404)

    signin = PastedSignIn()
    provider = oauth_provider(signin, storage)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(OAuthFlowError):
            await client.get(url)
    assert requests and signin.url is None
    assert storage.path.read_text() == original
    assert not provider.context.lock.locked()


@pytest.mark.parametrize("kind", ["prm", "asm"])
@pytest.mark.parametrize("failure", ["malformed", "invalid", "429", "503", "network"])
async def test_unavailable_or_invalid_discovery_is_not_absence(tmp_path, kind, failure):
    url = "https://resource.example/mcp"
    storage = FileTokenStorage(url, tmp_path / "tokens.json")
    await seed(storage)
    original = storage.path.read_text()
    signin = PastedSignIn()

    def respond(request):
        assert request.method == "GET" and "authorization" not in request.headers
        prm = "oauth-protected-resource" in request.url.path
        if prm != (kind == "prm"):
            return httpx.Response(
                200, json={"resource": url, "authorization_servers": ["https://resource.example/"]}
            )
        if failure == "network":
            raise httpx.ConnectError("offline test", request=request)
        if failure == "malformed":
            return httpx.Response(200, content=b"not JSON")
        if failure == "invalid":
            return httpx.Response(200, json={})
        return httpx.Response(int(failure))

    provider = oauth_provider(signin, storage)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        with pytest.raises((OAuthFlowError, httpx.ConnectError)):
            await client.get(url)
    assert signin.url is None and storage.path.read_text() == original
    assert not provider.context.lock.locked()


async def test_forget_auth_removes_one_url_preserving_approval_aliases_and_client_id(
    servers, tmp_path
):
    url = "https://resource.example/mcp"
    servers.register("remote", url=url, oauth_client_id="configured-client")
    servers.register("alias", url=url)
    servers.register("sibling", url="https://other.example/mcp")
    for name in ("remote", "alias", "sibling"):
        approve(servers, name)
    approval = servers._approval_store.path.read_text()
    storage = FileTokenStorage(url, servers._token_path, client_id="configured-client")
    await seed(storage)
    sibling = FileTokenStorage("https://other.example/mcp", servers._token_path)
    await seed(sibling)
    sibling_entry = sibling._entry()
    closed = []

    class LiveClient:
        async def aclose(self):
            closed.append(True)

    servers._attach("alias", SimpleNamespace(_client=LiveClient()))
    assert await servers.forget_auth("remote") == ["alias", "remote"]
    assert storage._entry() == {} and sibling._entry() == sibling_entry
    assert stat.S_IMODE(storage.path.stat().st_mode) == 0o600
    assert servers._approval_store.path.read_text() == approval
    assert servers.is_approved("remote") and servers.is_approved("alias")
    assert (await storage.get_client_info()).client_id == "configured-client"
    assert servers.connected() == [] and closed == [True]
    assert servers.approval_request("remote").config["oauth_client_id"] == "configured-client"
    result = await _control(servers, tmp_path).run(["forget-auth", "remote"])
    assert result.success and "not remote" in str(result) and "fresh" in str(result)


@pytest.mark.parametrize("name", ["unknown", "*", "stdio"])
async def test_forget_auth_rejects_unknown_glob_and_stdio_without_mutation(servers, name):
    servers.register("remote", url="https://resource.example/mcp")
    servers.register("stdio", command="probe")
    storage = FileTokenStorage("https://resource.example/mcp", servers._token_path)
    await seed(storage)
    original = storage.path.read_text()
    with pytest.raises(ValueError):
        await servers.forget_auth(name)
    assert storage.path.read_text() == original


async def test_forget_auth_cancels_pending_signin_before_deleting_and_registers_fresh(
    servers, oauth_server
):
    servers.register("remote", url=oauth_server.url)
    approve(servers, "remote")
    with pytest.raises(MCPSignInRequired):
        await servers.connect(["remote"])
    pending = servers._signins["remote"]
    assert oauth_server.registrations == 1
    await servers.forget_auth("remote")
    assert pending.task.done() and pending.signin._pasted.cancelled()
    assert FileTokenStorage(oauth_server.url, servers._token_path)._entry() == {}
    with pytest.raises(MCPSignInRequired):
        await servers.connect(["remote"])
    pasted = await asyncio.to_thread(oauth_server.consent, servers.sign_in_url("remote"))
    assert "connected" in await servers.complete_sign_in("remote", pasted)
    assert oauth_server.registrations == 2
    await servers.aclose()


@pytest.mark.parametrize(
    "failure", ["prm_issuer", "prm_absent", "asm_issuer", "prm_429", "asm_malformed"]
)
async def test_full_rediscovery_cannot_clear_or_reuse_retained_binding(tmp_path, failure):
    url = "https://resource.example/mcp"
    issuer = (
        "https://alternate.example/tenant"
        if failure == "prm_absent"
        else "https://resource.example/"
    )
    storage = FileTokenStorage(url, tmp_path / "tokens.json")
    await seed(storage, issuer=issuer)
    registration = storage._entry()["client_info"]
    refresh_failed = False
    posts = []
    signin = PastedSignIn()

    def respond(request):
        nonlocal refresh_failed
        if str(request.url) == url:
            return httpx.Response(401)
        if request.method == "POST":
            posts.append(request)
            assert not refresh_failed and b"grant_type=refresh_token" in request.content
            refresh_failed = True
            return httpx.Response(400, json={"error": "invalid_grant"})
        if "oauth-protected-resource" in request.url.path:
            if refresh_failed and failure == "prm_absent":
                return httpx.Response(404)
            if refresh_failed and failure == "prm_429":
                return httpx.Response(429)
            selected = (
                "https://changed.example/" if refresh_failed and failure == "prm_issuer" else issuer
            )
            return httpx.Response(200, json={"resource": url, "authorization_servers": [selected]})
        if refresh_failed and failure == "asm_malformed":
            return httpx.Response(200, content=b"not JSON")
        selected = (
            "https://changed.example/" if refresh_failed and failure == "asm_issuer" else issuer
        )
        return httpx.Response(
            200,
            json={
                "issuer": selected,
                "authorization_endpoint": f"{issuer.rstrip('/')}/authorize",
                "token_endpoint": f"{issuer.rstrip('/')}/token",
            },
        )

    provider = oauth_provider(signin, storage)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(OAuthFlowError):
            await client.get(url)
    assert len(posts) == 1 and signin.url is None
    assert storage._entry()["client_info"] == registration
    assert storage.get_issuer() == issuer
    assert "tokens" not in storage._entry() and "expires_at" not in storage._entry()
    assert provider.context.client_info.client_id == "cached-client"
    assert not provider.context.lock.locked()


@pytest.mark.parametrize("configuration", ["same", "removed", "changed", "reattached"])
async def test_forget_auth_drains_live_alias_and_disables_old_provider(
    servers, monkeypatch, configuration
):
    from contextlib import asynccontextmanager

    url = "https://resource.example/mcp"
    servers.register("remote", url=url)
    servers.register("alias", url=url)
    storage = FileTokenStorage(url, servers._token_path)
    await seed(storage)
    signin = PastedSignIn()
    provider = oauth_provider(signin, storage)
    client = AuthorizedHTTPClient(url, provider=provider, signin=signin)

    def respond(request):
        assert request.method == "GET"
        if str(request.url) == url:
            return httpx.Response(401)
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(410)
        return httpx.Response(
            200,
            json={
                "issuer": "https://resource.example/",
                "authorization_endpoint": "https://resource.example/authorize",
                "token_endpoint": "https://resource.example/token",
            },
        )

    @asynccontextmanager
    async def transport():
        async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as http:
            await http.get(url)
            yield None

    monkeypatch.setattr(client, "_connect_transport", transport)
    servers._oauth_clients[client] = ("alias", url)
    definitions = {"remote": {"url": url}, "alias": {"url": url}}
    servers.config_path().write_text(json.dumps({"atom": {"mcp_servers": definitions}}))
    servers.refresh_settings()
    servers._attach("alias", SimpleNamespace(_client=client))

    async def call():
        async with client.connect_to_server():
            pass

    calling = asyncio.create_task(call())
    await asyncio.wait_for(signin.url_ready.wait(), 2)
    assert provider.context.lock.locked()
    sibling_url = "https://other.example/mcp"
    sibling = FileTokenStorage(sibling_url, servers._token_path)
    await seed(sibling)
    sibling_entry = sibling._entry()
    new_client = None
    if configuration != "same":
        if configuration == "removed":
            definitions.pop("alias")
        else:
            definitions["alias"] = {"url": sibling_url}
        servers.config_path().write_text(json.dumps({"atom": {"mcp_servers": definitions}}))
        if configuration == "reattached":
            servers.refresh_settings()
            fresh_signin = PastedSignIn()
            new_client = AuthorizedHTTPClient(
                sibling_url, provider=oauth_provider(fresh_signin, sibling), signin=fresh_signin
            )
            servers._oauth_clients[new_client] = ("alias", sibling_url)
            servers._attach("alias", SimpleNamespace(_client=new_client))
    await servers.forget_auth("remote")
    assert sibling._entry() == sibling_entry
    if new_client is not None:
        assert servers.tool("alias")._client is new_client and not new_client._closed
    else:
        assert servers.connected() == []
    assert calling.done() and not calling.cancelled() and calling.cancelling() == 0
    with pytest.raises(RuntimeError, match="closed while opening"):
        await calling
    assert not client._sessions and not provider.context.lock.locked()
    assert signin._pasted.cancelled() and storage._entry() == {}
    with pytest.raises(RuntimeError, match="fresh sign-in"):
        async with client.connect_to_server():
            pass
    await servers.aclose()


async def test_forget_auth_drains_opening_before_signin_url_and_blocks_new_alias(
    servers, monkeypatch
):
    url = "https://resource.example/mcp"
    servers.register("remote", url=url)
    approve(servers, "remote")
    entered, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    storage = FileTokenStorage(url, servers._token_path)
    await seed(storage)

    async def discover(*args):
        entered.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            stopping.set()
            await release.wait()
            await storage.set_tokens(OAuthToken(access_token="late-write", token_type="Bearer"))
            raise

    monkeypatch.setattr(servers, "_discover", discover)
    connecting = asyncio.create_task(servers.connect(["remote"]))
    await asyncio.wait_for(entered.wait(), 2)
    forgetting = asyncio.create_task(servers.forget_auth("remote"))
    await asyncio.wait_for(stopping.wait(), 2)
    servers.register("new_alias", url=url)
    approve(servers, "new_alias")
    with pytest.raises(RuntimeError, match="being forgotten"):
        await servers.connect(["new_alias"])
    release.set()
    assert await asyncio.wait_for(forgetting, 2) == ["remote"]
    await asyncio.gather(connecting, return_exceptions=True)
    assert storage._entry() == {} and servers._connecting == set() and servers._opening == {}


async def test_forget_auth_exact_literal_glob_and_non_oauth_http(servers):
    servers.register("literal*", url="https://resource.example/mcp", headers={"X-Region": "test"})
    servers.register("literal-other", url="https://other.example/mcp")
    storage = FileTokenStorage("https://other.example/mcp", servers._token_path)
    await seed(storage)
    original = storage.path.read_text()
    assert await servers.forget_auth("literal*") == ["literal*"]
    assert storage.path.read_text() == original
    assert servers.approval_request("literal*").config["headers"] == {"X-Region": "test"}


async def test_forget_auth_does_not_attach_already_completed_opening(servers, monkeypatch):
    url = "https://resource.example/mcp"
    servers.register("remote", url=url)
    approve(servers, "remote")
    entered = asyncio.Event()
    closed = []

    class Client:
        async def aclose(self):
            closed.append(True)

    async def discover(*args):
        entered.set()
        return SimpleNamespace(_client=Client())

    monkeypatch.setattr(servers, "_discover", discover)
    connecting = asyncio.create_task(servers.connect(["remote"]))
    await asyncio.wait_for(entered.wait(), 2)
    assert servers._opening["remote"].task.done()
    await servers.forget_auth("remote")
    await asyncio.gather(connecting, return_exceptions=True)
    assert connecting.cancelled() and servers.connected() == []
    assert closed == [True] and servers._opening == {} and servers._connecting == set()


async def test_forget_auth_allows_fresh_registration_at_changed_issuer(servers, oauth_server):
    new_issuer = auth_fixtures.FakeOAuthServer().start()
    try:
        servers.register("remote", url=oauth_server.url)
        approve(servers, "remote")
        with pytest.raises(MCPSignInRequired):
            await servers.connect(["remote"])
        pasted = await asyncio.to_thread(oauth_server.consent, servers.sign_in_url("remote"))
        await servers.complete_sign_in("remote", pasted)
        assert await servers.agent.remote.echo(text="original") == "original"
        oauth_server.authorization_server = new_issuer
        oauth_server.valid_tokens = new_issuer.valid_tokens
        with pytest.raises(RuntimeError, match="OAuthFlowError"):
            await servers.agent.remote.echo(text="changed")
        assert new_issuer.registrations == 0 and new_issuer.grant_attempts == []
        await servers.forget_auth("remote")
        assert servers.is_approved("remote")
        with pytest.raises(MCPSignInRequired):
            await servers.connect(["remote"])
        pasted = await asyncio.to_thread(new_issuer.consent, servers.sign_in_url("remote"))
        await servers.complete_sign_in("remote", pasted)
        assert await servers.agent.remote.echo(text="fresh") == "fresh"
        assert new_issuer.registrations == 1
        assert new_issuer.grant_attempts == ["authorization_code"]
        storage = FileTokenStorage(oauth_server.url, servers._token_path)
        assert storage.get_issuer() == f"{new_issuer.base}/"
    finally:
        await servers.aclose()
        new_issuer.stop()


@pytest.mark.parametrize("expired", [True, False])
async def test_absent_prm_resumes_correct_sdk_yield_and_never_sends_guessed_refresh(
    tmp_path, expired
):
    from urllib.parse import parse_qs, urlsplit

    url = "https://resource.example/mcp"
    storage = FileTokenStorage(url, tmp_path / "tokens.json")
    await seed(storage, expired=expired)
    signin = PastedSignIn()
    requests = []
    authorized = False

    async def redirect(address):
        await signin.redirect(address)
        state = parse_qs(urlsplit(address).query)["state"][0]
        signin.complete(f"http://localhost/callback?code=offline-code&state={state}")

    def respond(request):
        nonlocal authorized
        # The SDK reuses/mutates its resource Request on retry. Capture the
        # headers as actually delivered, not the later in-memory bearer value.
        requests.append(
            httpx.Request(
                request.method, request.url, headers=request.headers, content=request.content
            )
        )
        if request.method == "POST":
            assert request.url.path == "/token"
            assert b"grant_type=authorization_code" in request.content
            assert b"code_verifier=" in request.content
            assert b"stale-refresh" not in request.content
            assert "authorization" not in request.headers
            authorized = True
            return httpx.Response(
                200,
                json={"access_token": "fresh-access", "token_type": "Bearer", "expires_in": 3600},
            )
        if str(request.url) == url:
            return httpx.Response(200 if authorized else 401)
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(410)
        return httpx.Response(
            200,
            json={
                "issuer": "https://resource.example/",
                "authorization_endpoint": "https://resource.example/authorize",
                "token_endpoint": "https://resource.example/token",
            },
        )

    provider = oauth_provider(signin, storage)
    provider.context.redirect_handler = redirect
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        assert (await client.get(url)).status_code == 200
    resource_requests = [r for r in requests if str(r.url) == url]
    assert len(resource_requests) == (3 if expired else 2)
    if expired:
        assert all("authorization" not in r.headers for r in resource_requests[:-1])
    else:
        assert resource_requests[0].headers["authorization"] == "Bearer stale-access"
    assert resource_requests[-1].headers["authorization"] == "Bearer fresh-access"
    assert requests[-2].method == "POST" and requests[-1] is resource_requests[-1]
    assert len([r for r in requests if r.method == "POST"]) == 1
    assert not provider.context.lock.locked()
