# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP sign-in through the MCP SDK's OAuth provider, with a pasted redirect URL."""

import asyncio
import json
import stat

import pytest
from fake_oauth_mcp import FakeOAuthServer
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


def _client(server, signin, tokens, client_id=None):
    storage = FileTokenStorage(server.url, tokens, client_id=client_id)
    return AuthorizedHTTPClient(server.url, provider=oauth_provider(signin, storage))


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
