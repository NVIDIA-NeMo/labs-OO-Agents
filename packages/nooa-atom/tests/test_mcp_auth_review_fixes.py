# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline review regressions: transient refreshes and session-owned cancellation."""

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import anyio
import httpx
import pytest
import test_mcp_auth as auth_fixtures
from mcp.server.fastmcp import FastMCP
from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata, OAuthToken
from mcp.shared.exceptions import McpError
from nooa_atom.skills import mcp_auth
from nooa_atom.skills.mcp_auth import (
    AuthorizedHTTPClient,
    FileTokenStorage,
    PastedSignIn,
    SignInTimeoutError,
    oauth_provider,
)

from nooa.mcp.tool import MCPTool, _flatten_exceptions


async def seeded_provider(tmp_path):
    storage = FileTokenStorage("https://resource.example/mcp", tmp_path / "tokens.json")
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="synthetic-client",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="none",
        )
    )
    with storage.bind_token_writes("https://issuer.example/"):
        await storage.set_tokens(
            OAuthToken(
                access_token="synthetic-old",
                refresh_token="synthetic-refresh",
                token_type="Bearer",
                expires_in=900,
                scope="read write",
            )
        )
    provider = oauth_provider(PastedSignIn(), storage, scope="read")
    await provider._initialize()
    provider.context.oauth_metadata = OAuthMetadata(
        issuer="https://issuer.example/",
        authorization_endpoint="https://issuer.example/authorize",
        token_endpoint="https://issuer.example/token",
        response_types_supported=["code"],
    )
    return storage, provider


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (429, {"error": "temporarily_unavailable"}),
        (500, {"error": "server_error"}),
        (503, {"error": "temporarily_unavailable"}),
        (400, {"error": "server_error"}),
        (400, {"error": "temporarily_unavailable"}),
        (401, {"error": "temporarily_unavailable"}),
        (200, {}),
        (200, {"access_token": "synthetic-bad", "token_type": "invalid"}),
        (200, {"access_token": "synthetic-bad", "token_type": "Bearer", "scope": 123}),
        (200, {"access_token": "synthetic-bad", "token_type": "Bearer", "refresh_token": {}}),
        (200, "not JSON"),
    ],
)
async def test_transient_refresh_preserves_exact_cache_and_clears_memory(tmp_path, status, payload):
    storage, provider = await seeded_provider(tmp_path)
    before = storage.path.read_bytes()
    response = (
        httpx.Response(status, json=payload)
        if isinstance(payload, dict)
        else httpx.Response(status, text=payload)
    )
    assert not await provider._handle_refresh_response(response)
    assert storage.path.read_bytes() == before
    assert provider.context.current_tokens is None
    assert provider.context.token_expiry_time is None


@pytest.mark.parametrize("during", ["tool", "body", "opening"])
async def test_close_drains_session_without_cancelling_direct_tool_caller(
    tmp_path, monkeypatch, during
):
    storage, provider = await seeded_provider(tmp_path)
    client = AuthorizedHTTPClient(storage.server_url, provider=provider, signin=PastedSignIn())
    entered, blocked, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owners = []

    class Session:
        async def call_tool(self, name, args):
            blocked.set()
            await asyncio.Event().wait()

    @asynccontextmanager
    async def transport():
        owner = asyncio.current_task()
        owners.append(owner)
        try:
            async with anyio.create_task_group():
                entered.set()
                if during == "opening":
                    blocked.set()
                    await asyncio.Event().wait()
                yield Session()
        finally:
            assert asyncio.current_task() is owner
            exited.set()

    monkeypatch.setattr(client, "_connect_transport", transport)
    tool = MCPTool(client, "synthetic")
    release_body = asyncio.Event()

    async def caller():
        error = None
        try:
            if during == "body":
                async with client.connect_to_server():
                    blocked.set()
                    await release_body.wait()
            else:
                await tool._invoke("echo", {})
        except BaseException as exc:
            error = type(exc).__name__
        # A real host turn must remain usable after only its MCP session closes.
        await asyncio.sleep(0)
        return asyncio.current_task().cancelling(), error, "continued"

    calling = asyncio.create_task(caller())
    try:
        await asyncio.wait_for(blocked.wait(), 2)
        await asyncio.wait_for(client.aclose(), 2)
        assert exited.is_set()
        release_body.set()
        cancelling, error, continued = await asyncio.wait_for(calling, 2)
        assert cancelling == 0
        assert error != "CancelledError"
        assert continued == "continued"
        assert not client._sessions
    finally:
        release_body.set()
        calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, "invalid_grant"),
        (401, "invalid_grant"),
        (400, "invalid_client"),
        (401, "invalid_client"),
    ],
)
@pytest.mark.parametrize("winner", [False, True])
async def test_genuine_refresh_rejection_invalidates_only_snapshot(tmp_path, status, error, winner):
    storage, provider = await seeded_provider(tmp_path)
    original = storage._entry()
    if winner:
        with storage.bind_token_writes("https://issuer.example/"):
            await storage.set_tokens(
                OAuthToken(
                    access_token="synthetic-winner",
                    refresh_token="synthetic-rotated",
                    token_type="Bearer",
                    expires_in=3600,
                    scope="read write",
                )
            )
        saved = storage._entry()
    result = await provider._handle_refresh_response(httpx.Response(status, json={"error": error}))
    assert result is winner
    if winner:
        assert storage._entry() == saved
        assert provider.context.current_tokens.access_token == "synthetic-winner"
        assert provider.context.token_expiry_time == saved["expires_at"]
    else:
        assert storage._entry() == {
            k: v for k, v in original.items() if k not in {"tokens", "expires_at"}
        }
        assert provider.context.current_tokens is None


@pytest.mark.parametrize("status", [429, 500, 503, 200])
async def test_transient_refresh_never_rewrites_concurrent_winner(tmp_path, status):
    storage, provider = await seeded_provider(tmp_path)
    storage._update(
        tokens={"access_token": "synthetic-other", "token_type": "Bearer", "scope": "other"},
        expires_at=12345,
    )
    saved = storage.path.read_bytes()
    assert not await provider._handle_refresh_response(httpx.Response(status, json={}))
    assert storage.path.read_bytes() == saved
    assert provider.context.current_tokens is None


@pytest.mark.parametrize("expired", [False, True])
async def test_network_refresh_failure_preserves_disk_and_does_not_retry(tmp_path, expired):
    storage, provider = await seeded_provider(tmp_path)
    before = storage.path.read_bytes()
    if expired:
        provider.context.token_expiry_time = 0
    posts = 0

    def respond(request):
        nonlocal posts
        if str(request.url) == storage.server_url:
            return httpx.Response(401)
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "resource": storage.server_url,
                    "authorization_servers": ["https://issuer.example/"],
                },
            )
        if request.method == "POST":
            posts += 1
            raise httpx.ConnectError("synthetic network failure", request=request)
        return httpx.Response(
            200,
            json={
                "issuer": "https://issuer.example/",
                "authorization_endpoint": "https://issuer.example/authorize",
                "token_endpoint": "https://issuer.example/token",
            },
        )

    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get(storage.server_url)
    assert posts == 1
    assert storage.path.read_bytes() == before
    assert provider.context.current_tokens is None
    assert not provider.context.lock.locked()


async def test_many_session_owners_and_parallel_calls_drain_without_caller_cancel(
    tmp_path, monkeypatch
):
    storage, provider = await seeded_provider(tmp_path)
    client = AuthorizedHTTPClient(storage.server_url, provider=provider, signin=PastedSignIn())
    opened = asyncio.Event()
    start_calls = asyncio.Event()
    entered = 0
    calls = 0
    owner_tasks = []

    class Session:
        async def call_tool(self, name, args):
            nonlocal calls
            calls += 1
            if calls == 6:
                start_calls.set()
            await asyncio.Event().wait()

    @asynccontextmanager
    async def transport():
        nonlocal entered
        owner = asyncio.current_task()
        owner_tasks.append(owner)
        async with anyio.create_task_group():
            entered += 1
            if entered == 3:
                opened.set()
            try:
                yield Session()
            finally:
                assert asyncio.current_task() is owner

    monkeypatch.setattr(client, "_connect_transport", transport)

    async def caller():
        async with client.connect_to_server() as session:
            errors = await asyncio.gather(
                session.call_tool("a", {}), session.call_tool("b", {}), return_exceptions=True
            )
            assert all(isinstance(error, RuntimeError) for error in errors)
            assert asyncio.current_task().cancelling() == 0
        return "continued"

    callers = [asyncio.create_task(caller()) for _ in range(3)]
    try:
        await asyncio.wait_for(opened.wait(), 2)
        await asyncio.wait_for(start_calls.wait(), 2)
        await asyncio.wait_for(asyncio.gather(client.aclose(), client.aclose()), 2)
        assert await asyncio.wait_for(asyncio.gather(*callers), 2) == ["continued"] * 3
        assert all(task.done() for task in owner_tasks)
        assert not client._sessions
        await client.aclose()
    finally:
        for task in callers:
            task.cancel()
        await asyncio.gather(*callers, return_exceptions=True)


@pytest.mark.parametrize(
    "failure", ["before_yield", "caller_cancel", "close_in_body", "immediate_close"]
)
async def test_session_opening_and_caller_cleanup_are_owned(tmp_path, monkeypatch, failure):
    storage, provider = await seeded_provider(tmp_path)
    client = AuthorizedHTTPClient(storage.server_url, provider=provider, signin=PastedSignIn())
    entered, exited = asyncio.Event(), asyncio.Event()
    owners = []

    @asynccontextmanager
    async def transport():
        owner = asyncio.current_task()
        owners.append(owner)
        try:
            async with anyio.create_task_group():
                entered.set()
                if failure == "before_yield":
                    raise ValueError("synthetic opening failure")
                if failure in {"caller_cancel", "immediate_close"}:
                    await asyncio.Event().wait()
                yield SimpleNamespace()
        finally:
            assert asyncio.current_task() is owner
            exited.set()

    monkeypatch.setattr(client, "_connect_transport", transport)

    async def caller():
        async with client.connect_to_server():
            await client.aclose()  # External context-body close cannot self-await.
            assert asyncio.current_task().cancelling() == 0
        return "continued"

    if failure == "close_in_body":
        assert await asyncio.wait_for(caller(), 2) == "continued"
    elif failure == "before_yield":
        with pytest.raises(ExceptionGroup) as error:
            await asyncio.wait_for(caller(), 2)
        assert any(isinstance(e, ValueError) for e in error.value.exceptions)
    else:
        calling = asyncio.create_task(caller())
        await asyncio.wait_for(entered.wait(), 2)
        if failure == "caller_cancel":
            calling.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(calling, 2)
            assert calling.cancelling() == 1
        else:
            await asyncio.wait_for(client.aclose(), 2)
            with pytest.raises(RuntimeError, match="closed while opening"):
                await asyncio.wait_for(calling, 2)
            assert calling.cancelling() == 0
    assert exited.is_set()
    assert all(task.done() for task in owners)
    assert not client._sessions
    await client.aclose()


async def test_cancel_only_one_rpc_keeps_same_session_usable(tmp_path, monkeypatch):
    storage, provider = await seeded_provider(tmp_path)
    client = AuthorizedHTTPClient(storage.server_url, provider=provider, signin=PastedSignIn())
    entered, stopped = asyncio.Event(), asyncio.Event()

    class Session:
        async def call_tool(self, name, args):
            if name == "wait":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()
            return "still usable"

    @asynccontextmanager
    async def transport():
        async with anyio.create_task_group():
            yield Session()

    monkeypatch.setattr(client, "_connect_transport", transport)
    async with client.connect_to_server() as session:
        task = asyncio.create_task(session.call_tool("wait", {}))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(stopped.wait(), 2)
        assert await session.call_tool("normal", {}) == "still usable"
        assert asyncio.current_task().cancelling() == 0
    assert not client._sessions


server = auth_fixtures.server


async def test_real_sdk_transport_tool_close_keeps_host_turn_running(tmp_path, monkeypatch):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    owners = []
    original_call = FastMCP.call_tool
    original_transport = mcp_auth.streamable_http_client

    async def slow_call(self, *args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        while not release.is_set():
            await asyncio.sleep(0.005)
        return await original_call(self, *args, **kwargs)

    @asynccontextmanager
    async def transport(*args, **kwargs):
        owner = asyncio.current_task()
        owners.append(owner)
        try:
            async with original_transport(*args, **kwargs) as streams:
                yield streams
        finally:
            assert asyncio.current_task() is owner

    # FastMCP binds its handler during construction, before the fake starts.
    monkeypatch.setattr(FastMCP, "call_tool", slow_call)
    monkeypatch.setattr(mcp_auth, "streamable_http_client", transport)
    for name in auth_fixtures._PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    server = auth_fixtures.FakeOAuthServer().start()
    client = None
    calling = None
    try:
        path = tmp_path / "tokens.json"
        await auth_fixtures._sign_in(server, path)
        client = auth_fixtures._client(server, PastedSignIn(), path)
        tool = MCPTool(client, "synthetic")

        async def turn():
            with pytest.raises((RuntimeError, McpError)):
                await tool._invoke("echo", {"text": "synthetic"})
            assert asyncio.current_task().cancelling() == 0
            return "host continued"

        calling = asyncio.create_task(turn())
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(client.aclose(), 5)
        assert await asyncio.wait_for(calling, 2) == "host continued"
        assert all(task is not calling and task.done() for task in owners)
        assert not client._sessions
    finally:
        release.set()
        if calling is not None:
            calling.cancel()
            await asyncio.gather(calling, return_exceptions=True)
        if client is not None:
            await client.aclose()
        server.stop()


@pytest.mark.parametrize("stop", ["close", "cancel", "expire"])
async def test_real_sdk_pending_callback_cleanup_does_not_cancel_host(server, tmp_path, stop):
    signin = PastedSignIn(timeout=0.05 if stop == "expire" else 5)
    storage = FileTokenStorage(server.url, tmp_path / "tokens.json")
    provider = oauth_provider(signin, storage)
    client = AuthorizedHTTPClient(server.url, provider=provider, signin=signin)
    tool = MCPTool(client, "synthetic")

    async def turn():
        try:
            await tool._invoke("echo", {"text": "synthetic"})
        except Exception as error:
            if stop == "expire":
                assert any(
                    isinstance(leaf, SignInTimeoutError) for leaf in _flatten_exceptions(error)
                )
            assert asyncio.current_task().cancelling() == 0
            return "host continued"
        raise AssertionError("An incomplete synthetic sign-in cannot succeed")

    calling = asyncio.create_task(turn())
    try:
        await asyncio.wait_for(signin.url_ready.wait(), 5)
        if stop == "cancel":
            calling.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(calling, 2)
            assert calling.cancelling() == 1  # Only explicit caller cancellation does this.
        else:
            if stop == "close":
                await asyncio.wait_for(client.aclose(), 5)
            assert await asyncio.wait_for(calling, 2) == "host continued"
        assert not client._sessions
        assert not provider.context.lock.locked()
        assert signin._pasted.cancelled()
        assert "tokens" not in storage._entry()
        with pytest.raises(ValueError, match="no longer waiting"):
            signin.complete("http://localhost/callback?code=synthetic-late&state=synthetic-late")
    finally:
        calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)
        await client.aclose()


async def test_owner_close_during_sdk_discovery_preserves_cache_and_host(tmp_path, monkeypatch):
    storage, provider = await seeded_provider(tmp_path)
    provider.context.token_expiry_time = 0
    before = storage.path.read_bytes()
    client = AuthorizedHTTPClient(storage.server_url, provider=provider, signin=PastedSignIn())
    entered = asyncio.Event()

    async def respond(request):
        entered.set()
        await asyncio.Event().wait()

    @asynccontextmanager
    async def transport():
        async with anyio.create_task_group():
            async with httpx.AsyncClient(
                auth=provider, transport=httpx.MockTransport(respond)
            ) as http:
                await http.get(storage.server_url)
                yield None

    monkeypatch.setattr(client, "_connect_transport", transport)

    async def turn():
        with pytest.raises(RuntimeError, match="closed while opening"):
            async with client.connect_to_server():
                pass
        return asyncio.current_task().cancelling()

    calling = asyncio.create_task(turn())
    await asyncio.wait_for(entered.wait(), 2)
    assert provider.context.lock.locked()
    await asyncio.wait_for(client.aclose(), 2)
    assert await calling == 0
    assert not provider.context.lock.locked()
    assert storage.path.read_bytes() == before
    assert not client._sessions


async def test_repeated_short_session_lifetimes_leave_no_owner_or_rpc_tasks(tmp_path, monkeypatch):
    storage, provider = await seeded_provider(tmp_path)
    client = AuthorizedHTTPClient(storage.server_url, provider=provider, signin=PastedSignIn())

    class Session:
        async def list_tools(self):
            return "synthetic"

    @asynccontextmanager
    async def transport():
        async with anyio.create_task_group():
            yield Session()

    monkeypatch.setattr(client, "_connect_transport", transport)
    before = asyncio.all_tasks()
    fd_path = Path("/proc/self/fd")
    fd_count = len(os.listdir(fd_path)) if fd_path.is_dir() else None
    for _ in range(100):
        async with client.connect_to_server() as session:
            assert await session.list_tools() == "synthetic"
        assert not client._sessions
    await client.aclose()
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before)
    if fd_count is not None:
        assert len(os.listdir(fd_path)) == fd_count
    assert asyncio.current_task().cancelling() == 0


async def test_cancelling_close_waiter_does_not_recancel_owner_cleanup(tmp_path, monkeypatch):
    storage, provider = await seeded_provider(tmp_path)
    client = AuthorizedHTTPClient(storage.server_url, provider=provider, signin=PastedSignIn())
    opened, cleanup, release, exited = (asyncio.Event() for _ in range(4))

    @asynccontextmanager
    async def transport():
        try:
            async with anyio.create_task_group():
                yield SimpleNamespace()
        finally:
            with anyio.CancelScope(shield=True):
                cleanup.set()
                await release.wait()
            exited.set()

    monkeypatch.setattr(client, "_connect_transport", transport)
    body_release = asyncio.Event()

    async def caller():
        async with client.connect_to_server():
            opened.set()
            await body_release.wait()
        assert asyncio.current_task().cancelling() == 0

    calling = asyncio.create_task(caller())
    closing = None
    try:
        await asyncio.wait_for(opened.wait(), 2)
        closing = asyncio.create_task(client.aclose())
        await asyncio.wait_for(cleanup.wait(), 2)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert closing.cancelling() == 1
        assert client._sessions  # Owner is tracked until shielded cleanup actually drains.
        release.set()
        await asyncio.wait_for(client.aclose(), 2)
        body_release.set()
        await asyncio.wait_for(calling, 2)
        assert exited.is_set() and not client._sessions
    finally:
        release.set()
        body_release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)
        await client.aclose()
