# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP sign-in through the MCP SDK's ``OAuthClientProvider``.

The SDK does the OAuth work: discovery of the authorisation server, dynamic
client registration, PKCE, token refresh, and a new sign-in when the server
answers 401. This module supplies the three pieces it asks for:

- ``FileTokenStorage`` keeps tokens and client registrations in
  ``~/.config/nooa/mcp_oauth.json`` (readable by the user only), one entry
  per server URL.
- ``PastedSignIn`` is the redirect and callback handler pair. It never opens
  a browser or listens on a port, because a sandbox cannot receive a
  loopback callback. It keeps the sign-in link for the host to show; the
  person opens it, signs in, and pastes the address the browser was sent to
  (it fails to load; only its ``code`` and ``state`` matter).
- ``AuthorizedHTTPClient`` is a core ``MCPBaseClient`` whose HTTP requests
  go through the provider.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from nooa.mcp import MCPBaseClient
from nooa.paths import get_user_dir

# Where the browser is sent after sign-in. Nothing listens there: the person
# copies the address from the browser. Some gateways reject 127.0.0.1.
DEFAULT_REDIRECT_URI = "http://localhost/callback"
# How long a sign-in waits for the pasted redirect address.
SIGN_IN_TIMEOUT_SECONDS = 600.0
# Establishing the connection keeps the short budget core's clients use.
_CONNECT_TIMEOUT_SECONDS = 5.0


def token_file() -> Path:
    """The user's MCP token file: ``~/.config/nooa/mcp_oauth.json``."""
    return get_user_dir("mcp_oauth.json")


class FileTokenStorage:
    """The SDK's ``TokenStorage`` for one server, kept in a JSON file.

    Each server URL has an entry with its tokens, the time the access token
    expires and the client registration. An access token that has expired is
    handed to the SDK without its value when a refresh token exists, so the
    SDK refreshes it instead of sending it and signing in again.

    ``client_id`` is a client registered in advance (``oauth_client_id`` in
    the server's settings); it is used when no registration is stored.
    """

    def __init__(
        self,
        server_url: str,
        path: Path | None = None,
        *,
        client_id: str | None = None,
        redirect_uri: str = DEFAULT_REDIRECT_URI,
    ) -> None:
        self.server_url = server_url
        self.path = path or token_file()
        self._client_id = client_id
        self.redirect_uri = redirect_uri

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _entry(self) -> dict[str, Any]:
        entry = self._read().get(self.server_url)
        return entry if isinstance(entry, dict) else {}

    def _update(self, **fields: Any) -> None:
        data = self._read()
        entry = data.get(self.server_url)
        data[self.server_url] = {**(entry if isinstance(entry, dict) else {}), **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".mcp_oauth.")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(data, stream, indent=2)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    async def get_tokens(self) -> OAuthToken | None:
        entry = self._entry()
        try:
            tokens = OAuthToken.model_validate(entry["tokens"])
        except (KeyError, ValueError):
            return None
        expires_at = entry.get("expires_at")
        if isinstance(expires_at, int | float) and time.time() >= expires_at:
            if not tokens.refresh_token:
                return None
            return tokens.model_copy(update={"access_token": ""})
        return tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        expires_at = time.time() + tokens.expires_in if tokens.expires_in else None
        self._update(tokens=tokens.model_dump(mode="json"), expires_at=expires_at)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        try:
            return OAuthClientInformationFull.model_validate(self._entry()["client_info"])
        except (KeyError, ValueError):
            pass
        if self._client_id is None:
            return None
        return OAuthClientInformationFull(
            client_id=self._client_id,
            redirect_uris=[self.redirect_uri],  # type: ignore[list-item]
            token_endpoint_auth_method="none",
        )

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._update(client_info=client_info.model_dump(mode="json"))


class PastedSignIn:
    """Redirect and callback handlers for a sign-in the person completes by pasting.

    ``redirect`` keeps the sign-in link in ``url`` and sets ``url_ready``;
    the host shows the link. ``complete`` takes the address the browser
    ended on and hands its ``code`` and ``state`` to ``callback``, which the
    SDK awaits. ``callback`` gives up after ``timeout`` seconds.
    """

    def __init__(self, *, timeout: float = SIGN_IN_TIMEOUT_SECONDS) -> None:
        self.url: str | None = None
        self.url_ready = asyncio.Event()
        self._timeout = timeout
        self._pasted: asyncio.Future[tuple[str, str | None]] = (
            asyncio.get_running_loop().create_future()
        )

    async def redirect(self, url: str) -> None:
        self.url = url
        self.url_ready.set()

    async def callback(self) -> tuple[str, str | None]:
        try:
            return await asyncio.wait_for(asyncio.shield(self._pasted), self._timeout)
        except TimeoutError:
            raise TimeoutError(
                f"The sign-in was not completed within {self._timeout:g} seconds; "
                "connect again to start a new one."
            ) from None

    def complete(self, pasted: str) -> None:
        """Take the pasted redirect address; ``ValueError`` if it holds no code."""
        value = pasted.strip()
        quoted = re.search(r"curl\s+['\"]([^'\"]+)['\"]", value)
        if quoted:
            value = quoted.group(1)
        query = parse_qs(urlsplit(value).query)
        if "error" in query:
            detail = query.get("error_description", [""])[0]
            raise ValueError(
                "The sign-in was refused: " + query["error"][0] + (f": {detail}" if detail else "")
            )
        code = query.get("code", [""])[0]
        if not code:
            raise ValueError(
                "The pasted address has no authorisation code; paste the full address "
                "the browser was sent to after signing in."
            )
        if not self._pasted.done():
            self._pasted.set_result((code, query.get("state", [None])[0]))


def oauth_provider(
    signin: PastedSignIn, storage: FileTokenStorage, *, scope: str | None = None
) -> OAuthClientProvider:
    """The SDK provider for ``storage.server_url``, signing in through ``signin``."""
    metadata = OAuthClientMetadata(
        client_name="NOOA",
        redirect_uris=[storage.redirect_uri],  # type: ignore[list-item]
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        scope=scope,
    )
    return OAuthClientProvider(
        server_url=storage.server_url,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=signin.redirect,
        callback_handler=signin.callback,
    )


class AuthorizedHTTPClient(MCPBaseClient):
    """A streamable-HTTP or SSE MCP client whose requests carry the provider's tokens."""

    def __init__(
        self,
        url: str,
        *,
        provider: OAuthClientProvider,
        headers: dict[str, str] | None = None,
        transport: Literal["sse", "streamable-http"] = "streamable-http",
        tool_call_timeout: timedelta = timedelta(seconds=60),
    ) -> None:
        super().__init__(tool_call_timeout=tool_call_timeout)
        self._url = url
        self._headers = dict(headers or {})
        self._transport = transport
        self._provider = provider

    @property
    def transport(self) -> Literal["sse", "stdio", "streamable-http"]:
        return self._transport

    @property
    def server_config(self) -> dict[str, Any]:
        return {"url": self._url, "headers": self._headers, "transport": self._transport}

    @asynccontextmanager
    async def connect_to_server(self) -> AsyncGenerator[ClientSession, None]:
        seconds = self.tool_call_timeout.total_seconds()
        if self._transport == "sse":
            async with (
                sse_client(
                    self._url,
                    headers=self._headers or None,
                    sse_read_timeout=seconds,
                    auth=self._provider,
                ) as (read, write),
                ClientSession(read, write, read_timeout_seconds=self.tool_call_timeout) as session,
            ):
                await session.initialize()
                yield session
            return
        http_client = httpx.AsyncClient(
            headers=self._headers or None,
            timeout=httpx.Timeout(seconds, connect=_CONNECT_TIMEOUT_SECONDS),
            auth=self._provider,
        )
        async with (
            http_client,
            streamable_http_client(self._url, http_client=http_client) as streams,
            ClientSession(
                streams[0], streams[1], read_timeout_seconds=self.tool_call_timeout
            ) as session,
        ):
            await session.initialize()
            yield session
