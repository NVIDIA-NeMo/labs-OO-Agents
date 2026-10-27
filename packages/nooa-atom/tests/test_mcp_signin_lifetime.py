# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline OAuth lifetime regressions: human wait is not an MCP response timeout."""

import asyncio
from datetime import timedelta

import httpx
import pytest
import test_mcp_servers as fixtures
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
from nooa_atom.skills import mcp_servers
from nooa_atom.skills.mcp_auth import PastedSignIn, SignInTimeoutError, _AuthorizedSession
from nooa_atom.skills.mcp_servers import MCPSignInRequired, _PendingSignIn
from test_mcp_servers import approve

servers = fixtures.servers
oauth_server = fixtures.oauth_server


async def test_sign_in_survives_the_tool_response_budget(servers, oauth_server):
    servers.register("remote", url=oauth_server.url)
    approve(servers, "remote")
    with pytest.raises(MCPSignInRequired):
        await servers.connect(["remote"], tool_call_timeout=timedelta(seconds=0.5))
    try:
        # connect returned; other host turns can run while the human signs in.
        # This also exceeds the HTTPX read timeout: OAuth callback wait is not I/O.
        await asyncio.sleep(1)
        assert servers.state("remote") == "needs-auth", servers.error("remote")
        pasted = await asyncio.to_thread(oauth_server.consent, servers.sign_in_url("remote"))
        assert "connected" in await servers.complete_sign_in("remote", pasted)
        assert await servers.agent.remote.echo(text="after waiting") == "after waiting"
    finally:
        await servers.aclose()


async def test_human_expiry_with_a_longer_call_budget_is_distinct(
    servers, oauth_server, monkeypatch
):
    # Model 600s human / 1200s call budgets without waiting ten minutes.
    monkeypatch.setattr(mcp_servers, "PastedSignIn", lambda: PastedSignIn(timeout=0.1))
    servers.register("remote", url=oauth_server.url)
    approve(servers, "remote")
    with pytest.raises(MCPSignInRequired):
        await servers.connect(["remote"], tool_call_timeout=timedelta(seconds=2))
    pending = servers._signins["remote"]
    try:
        with pytest.raises(BaseExceptionGroup):
            await pending.task
        await asyncio.sleep(0)  # run the completion callback
        assert servers.state("remote") == "failed"
        assert "SignInTimeoutError" in servers.error("remote")
        assert servers.sign_in_url("remote") is None
        with pytest.raises(ValueError, match="no longer waiting"):
            pending.signin.complete("http://localhost/callback?code=late&state=late")
        with pytest.raises(ValueError, match="No sign-in"):
            await servers.complete_sign_in("remote", "http://localhost/callback?code=late")
        assert oauth_server.grants == []
    finally:
        await servers.aclose()


@pytest.mark.parametrize("operation", ["disconnect", "aclose"])
async def test_stopping_the_host_cancels_and_drains_sign_in(servers, oauth_server, operation):
    servers.register("remote", url=oauth_server.url)
    approve(servers, "remote")
    with pytest.raises(MCPSignInRequired):
        await servers.connect(["remote"])
    pending = servers._signins["remote"]
    if operation == "aclose":
        await servers.aclose()
    else:
        assert await servers.disconnect(["remote"]) == ["remote"]
    assert pending.task.done()
    assert pending.signin._pasted.cancelled()
    assert servers.state("remote") == "available"
    assert not hasattr(servers.agent, "remote")
    with pytest.raises(ValueError, match="no longer waiting"):
        pending.signin.complete("http://localhost/callback?code=late&state=late")
    assert oauth_server.grants == []


async def test_a_stale_done_callback_cannot_attach_or_remove_a_new_sign_in(servers):
    servers.register("remote", url="http://localhost/mcp")
    approve(servers, "remote")
    request = servers.approval_request("remote")
    old = asyncio.create_task(asyncio.sleep(0, result=object()))
    await old
    fresh = asyncio.create_task(asyncio.sleep(10))
    pending = _PendingSignIn(PastedSignIn(), fresh, False)
    servers._signins["remote"] = pending
    try:
        servers._signed_in("remote", request, {}, old)
        assert servers._signins["remote"] is pending
        assert servers.connected() == []
    finally:
        await servers.aclose()


async def test_a_response_without_human_input_still_times_out():
    signin = PastedSignIn()
    with pytest.raises(TimeoutError):
        async with signin.response_budget(0.05):
            await asyncio.sleep(1)
    assert signin._budgets == {}


async def test_the_response_budget_resumes_after_the_pasted_callback():
    signin = PastedSignIn(timeout=1)
    with pytest.raises(TimeoutError):
        async with signin.response_budget(0.1) as budget:
            await asyncio.sleep(0.03)  # pre-sign-in work consumes response allowance
            callback = asyncio.create_task(signin.callback())
            await asyncio.sleep(0.2)  # does not consume the response budget
            assert budget.when() is None
            signin.complete("http://localhost/callback?code=fake&state=fake")
            assert await callback == ("fake", "fake")
            remaining = budget.when() - asyncio.get_running_loop().time()
            assert 0 < remaining < 0.09  # retains the remainder, not a fresh budget
            await asyncio.sleep(1)


async def test_an_expired_callback_is_invalidated():
    signin = PastedSignIn(timeout=0.01)
    with pytest.raises(SignInTimeoutError):
        await signin.callback()
    with pytest.raises(ValueError, match="no longer waiting"):
        signin.complete("http://localhost/callback?code=late")


def test_grouped_diagnostics_show_only_safe_leaf_metadata(servers):
    servers.register("remote", url="http://localhost/mcp")
    secret = "https://auth.example/authorize?code=secret&state=secret"
    request = httpx.Request("GET", secret)
    error = ExceptionGroup(
        secret,
        [
            ExceptionGroup(secret, [McpError(ErrorData(code=408, message=secret))]),
            httpx.HTTPStatusError(
                secret, request=request, response=httpx.Response(401, request=request)
            ),
            httpx.ReadTimeout(secret, request=request),
            SignInTimeoutError(secret),
        ],
    )
    message = mcp_servers._connection_error(servers.approval_request("remote"), error, {})
    assert "McpError (MCP code 408)" in message
    assert "HTTPStatusError (HTTP status 401)" in message
    assert "ReadTimeout" in message
    assert "SignInTimeoutError" in message
    assert "secret" not in message
    assert "https://" not in message


@pytest.mark.parametrize("override", [None, timedelta(seconds=0.02), 0.02])
async def test_sdk_session_response_timeout_and_explicit_override(override):
    import anyio
    from mcp import types

    outgoing, reader = anyio.create_memory_object_stream(1)
    incoming, writer = anyio.create_memory_object_stream(1)
    signin = PastedSignIn()
    session = _AuthorizedSession(reader, incoming, signin, 0.05)
    async with outgoing, reader, incoming, writer:
        with pytest.raises(McpError) as raised:
            await asyncio.wait_for(
                session.send_request(
                    types.ClientRequest(types.PingRequest(method="ping")),
                    types.EmptyResult,
                    override,
                    None,  # preserve positional SDK metadata/progress API
                    None,
                ),
                1,
            )
        assert raised.value.error.code == 408
        assert signin._budgets == {}
        assert session._response_streams == {}


async def test_cancelling_connect_before_url_registration_drains_discovery(servers, monkeypatch):
    signin = PastedSignIn()
    monkeypatch.setattr(mcp_servers, "PastedSignIn", lambda: signin)
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def discover(*args):
        entered.set()
        try:
            await asyncio.sleep(10)
        finally:
            stopped.set()

    monkeypatch.setattr(servers, "_discover", discover)
    servers.register("remote", url="http://localhost/mcp")
    approve(servers, "remote")
    connecting = asyncio.create_task(servers.connect(["remote"]))
    await entered.wait()
    connecting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await connecting
    assert stopped.is_set()
    assert signin._pasted.cancelled()
    assert servers._connecting == set()
    assert servers._signins == {}


async def test_httpx_network_timeout_is_not_paused_with_human_wait(monkeypatch):
    from contextlib import asynccontextmanager

    from nooa_atom.skills import mcp_auth
    from nooa_atom.skills.mcp_auth import AuthorizedHTTPClient

    started = asyncio.Event()
    release = asyncio.Event()

    async def stalled(reader, writer):
        try:
            await reader.read(1)
            started.set()
            await release.wait()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(stalled, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/mcp"
    signin = PastedSignIn(timeout=1)

    @asynccontextmanager
    async def transport(url, *, http_client):
        # Test the real HTTPX client the production transport receives, while
        # an unrelated human callback is holding the response budget paused.
        async with signin.response_budget(0.05) as budget:
            assert budget.when() is None
            await http_client.get(url)
        yield  # unreachable: the stalled socket must hit HTTPX read timeout

    monkeypatch.setattr(mcp_auth, "streamable_http_client", transport)
    callback = asyncio.create_task(signin.callback())
    await asyncio.sleep(0)
    client = AuthorizedHTTPClient(
        url,
        provider=httpx.Auth(),
        signin=signin,
        tool_call_timeout=timedelta(seconds=0.05),
    )
    try:
        with pytest.raises(httpx.ReadTimeout):
            async with client.connect_to_server():
                pass
        assert started.is_set()
        assert not callback.done()
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        callback.cancel()
        await asyncio.gather(callback, return_exceptions=True)


async def test_external_pending_task_cancellation_is_reported(servers, oauth_server):
    servers.register("remote", url=oauth_server.url)
    approve(servers, "remote")
    with pytest.raises(MCPSignInRequired):
        await servers.connect(["remote"])
    pending = servers._signins["remote"]
    pending.task.cancel()
    await asyncio.gather(pending.task, return_exceptions=True)
    assert servers.state("remote") == "failed"
    assert "cancelled" in servers.error("remote")
    assert servers.sign_in_url("remote") is None
    assert pending.signin._pasted.cancelled()
    await servers.aclose()


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("https://auth.example/authorize?state=secret"),
        McpError(ErrorData(code=408, message="https://auth.example/authorize?state=secret")),
        SignInTimeoutError("https://auth.example/authorize?state=secret"),
    ],
)
def test_direct_transport_auth_errors_also_hide_messages(servers, error):
    servers.register("remote", url="http://localhost/mcp")
    message = mcp_servers._connection_error(servers.approval_request("remote"), error, {})
    assert type(error).__name__ in message
    assert "secret" not in message
    assert "https://" not in message
