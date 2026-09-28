# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An MCP server behind a small OAuth authorisation server, for the sign-in tests.

``FakeOAuthServer`` runs in a thread on a free loopback port. ``/mcp`` is a
streamable-HTTP MCP server with one tool, ``echo``; it answers 401 without a
valid bearer token. The server publishes protected-resource and
authorisation-server metadata, registers clients dynamically, and issues
authorisation codes (PKCE), access tokens and refresh tokens. It records the
grants it served so tests can tell a refresh from a new sign-in.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import socket
import threading
import time
from urllib.parse import parse_qs, urlencode

import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeOAuthServer:
    """MCP server plus authorisation server on one loopback origin."""

    def __init__(self, *, expires_in: int = 3600) -> None:
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.url = f"{self.base}/mcp"
        self.expires_in = expires_in
        self.grants: list[str] = []
        self.registrations = 0
        self.valid_tokens: set[str] = set()
        self._codes: dict[str, dict[str, str]] = {}
        self._refresh_tokens: set[str] = set()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    # ---- the "person" in the browser --------------------------------

    def consent(self, authorization_url: str) -> str:
        """Open the sign-in link as a browser would; return the redirect it ends on."""
        response = httpx.get(authorization_url, follow_redirects=False, trust_env=False)
        assert response.status_code == 302, response.text
        return response.headers["location"]

    # ---- endpoints ---------------------------------------------------

    def _app(self):
        mcp = FastMCP("fake", stateless_http=True, json_response=True)

        @mcp.tool()
        def echo(text: str) -> str:
            """Return the text unchanged."""
            return text

        app = mcp.streamable_http_app()
        app.router.routes.extend(
            [
                Route("/.well-known/oauth-protected-resource/mcp", self._resource_metadata),
                Route("/.well-known/oauth-protected-resource", self._resource_metadata),
                Route("/.well-known/oauth-authorization-server", self._server_metadata),
                Route("/register", self._register, methods=["POST"]),
                Route("/authorize", self._authorize),
                Route("/token", self._token, methods=["POST"]),
            ]
        )
        server = self

        async def guarded(scope, receive, send):
            if scope["type"] == "http" and scope["path"].startswith("/mcp"):
                headers = dict(scope["headers"])
                token = headers.get(b"authorization", b"").decode().removeprefix("Bearer ")
                if token not in server.valid_tokens:
                    metadata = f"{server.base}/.well-known/oauth-protected-resource/mcp"
                    response = Response(
                        status_code=401,
                        headers={"WWW-Authenticate": f'Bearer resource_metadata="{metadata}"'},
                    )
                    await response(scope, receive, send)
                    return
            await app(scope, receive, send)

        return guarded

    async def _resource_metadata(self, request: Request) -> Response:
        return JSONResponse({"resource": self.url, "authorization_servers": [self.base]})

    async def _server_metadata(self, request: Request) -> Response:
        return JSONResponse(
            {
                "issuer": self.base,
                "authorization_endpoint": f"{self.base}/authorize",
                "token_endpoint": f"{self.base}/token",
                "registration_endpoint": f"{self.base}/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"],
            }
        )

    async def _register(self, request: Request) -> Response:
        self.registrations += 1
        body = await request.json()
        return JSONResponse(
            {
                **body,
                "client_id": f"client-{self.registrations}",
                "token_endpoint_auth_method": "none",
            },
            status_code=201,
        )

    async def _authorize(self, request: Request) -> Response:
        params = request.query_params
        code = secrets.token_urlsafe(8)
        self._codes[code] = {
            "challenge": params["code_challenge"],
            "redirect_uri": params["redirect_uri"],
        }
        query = urlencode({"code": code, "state": params["state"]})
        return RedirectResponse(f"{params['redirect_uri']}?{query}", status_code=302)

    async def _token(self, request: Request) -> Response:
        form = parse_qs((await request.body()).decode())
        grant = form["grant_type"][0]
        if grant == "authorization_code":
            issued = self._codes.pop(form["code"][0], None)
            verifier = form.get("code_verifier", [""])[0]
            digest = hashlib.sha256(verifier.encode()).digest()
            challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            if issued is None or issued["challenge"] != challenge:
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
        elif grant == "refresh_token":
            if form["refresh_token"][0] not in self._refresh_tokens:
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
        else:
            return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)
        self.grants.append(grant)
        access, refresh = secrets.token_urlsafe(8), secrets.token_urlsafe(8)
        self.valid_tokens.add(access)
        self._refresh_tokens.add(refresh)
        return JSONResponse(
            {
                "access_token": access,
                "token_type": "Bearer",
                "expires_in": self.expires_in,
                "refresh_token": refresh,
            }
        )

    # ---- lifecycle ---------------------------------------------------

    def expire_all_tokens(self) -> None:
        """Make every issued access token invalid, as if they had expired."""
        self.valid_tokens.clear()

    def start(self) -> FakeOAuthServer:
        config = uvicorn.Config(
            self._app(), host="127.0.0.1", port=self.port, log_level="warning", lifespan="on"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            assert time.monotonic() < deadline, "fake OAuth server did not start"
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
