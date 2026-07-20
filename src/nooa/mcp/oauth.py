# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# OAuth 2.0 implementation following RFC 8252 "OAuth 2.0 for Native Apps":
#   https://datatracker.ietf.org/doc/html/rfc8252
#
# Key RFC 8252 practices applied here:
#   §4   — Use the system browser (webbrowser.open), not an embedded webview
#   §7.3 — Use loopback redirect URIs (localhost; some gateways reject 127.0.0.1)
#           with a dynamic OS-assigned
#           port (bind to port 0) so no fixed port needs to be pre-registered
#           and port conflicts are impossible
#   §8   — PKCE (RFC 7636, S256 method) is required for all native clients
import asyncio
import base64
import contextlib
import hashlib
import html
import json
import logging
import os
import re
import secrets
import shutil
import stat
import time
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OAuthConfig:
    """Configuration for OAuth 2.0 flow.

    Attributes:
        authorization_endpoint: OAuth authorization URL
        token_endpoint: OAuth token exchange URL
        client_id: OAuth client ID, or None until dynamic registration completes
        redirect_uri: Redirect URI for OAuth callback
        scope: Optional OAuth scopes
        client_secret: Optional client secret (for dynamic registration)
        registration_endpoint: Optional OAuth dynamic client registration endpoint
    """

    authorization_endpoint: str
    token_endpoint: str
    client_id: str | None
    redirect_uri: str
    scope: str | None = None
    client_secret: str | None = None
    registration_endpoint: str | None = None
    timeout: float = 300.0  # 5 minutes


@dataclass
class OAuthToken:
    """OAuth access token with metadata.

    Attributes:
        access_token: The access token
        token_type: Token type (typically "Bearer")
        expires_in: Token expiration time in seconds (if provided)
        refresh_token: Refresh token (if provided)
    """

    access_token: str
    token_type: str = "Bearer"
    expires_in: int | None = None
    refresh_token: str | None = None
    obtained_at: float = 0.0
    client_id: str | None = None
    client_secret: str | None = None

    def is_expired(self, leeway: float = 60.0) -> bool:
        """True if the access token is at/near expiry (with a safety leeway)."""
        if not self.expires_in:
            return False
        return time.time() >= (self.obtained_at + self.expires_in - leeway)


def _system_browser_available() -> bool:
    """True if a real system browser can be launched for the OAuth redirect.

    Headless/remote sessions (docker sandbox, SSH without X) have no browser, so
    the loopback-callback flow can never complete — the user never sees the
    consent page and the local callback is unreachable. Detect that here so the
    caller can fall back to the manual out-of-band flow instead of hanging until
    timeout.

    Best-effort heuristic, tuned for minimal Docker images: ``webbrowser.get()``
    raises only when *no* browser-like executable is found. On a non-minimal host
    that has a text browser (``elinks``/``links``/``lynx``) or ``xdg-open`` on
    PATH it returns True even without a graphical display, so the auto-fallback
    won't trigger there. Such environments should set ``oauth_manual = true`` in
    the server config to force the out-of-band flow.
    """
    try:
        webbrowser.get()
        return True
    except webbrowser.Error:
        pass
    # webbrowser.get() misses common launchers that DO open a host browser
    # (e.g. a sandbox's xdg-open that forwards to the host). If one is on PATH,
    # the loopback-callback flow can still complete — prefer it over OOB paste.
    for launcher in ("xdg-open", "sensible-browser", "open", "wslview"):
        if shutil.which(launcher):
            # Register once — repeated calls must not keep prepending to the
            # process-global browser registry.
            try:
                webbrowser.get(launcher)
            except webbrowser.Error:
                webbrowser.register(
                    launcher, None, webbrowser.GenericBrowser(launcher), preferred=True
                )
            return True
    return False


def _extract_authorization_code(pasted: str) -> str:
    """Extract an OAuth code from either a raw code or a pasted callback URL.

    Docker Sandbox/MaaS shows an OOB callback URI such as
    ``urn:ietf:wg:oauth:2.0:oob?code=...&state=...``. Users naturally paste
    that whole URI, so accept it instead of sending the full URI as the code.
    """
    value = pasted.strip()
    if not value:
        return ""

    # The MaaS helper page also offers ``curl '<callback-url>'``; accept that too.
    curl_match = re.search(r"curl\s+['\"]([^'\"]+)['\"]", value)
    if curl_match:
        value = curl_match.group(1)

    parsed = urlparse(value)
    if parsed.query:
        params = parse_qs(parsed.query)
        code_values = params.get("code")
        if code_values and code_values[0]:
            return code_values[0]

    return value


def _html_page(title: str, body: str) -> str:
    """Return a minimal styled HTML page for the OAuth callback browser tab."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      display: flex; align-items: center; justify-content: center;
      height: 100vh; margin: 0; background: #f5f5f5;
    }}
    .card {{
      background: #fff; border-radius: 12px; padding: 40px 48px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.10); text-align: center; max-width: 420px;
    }}
    h1 {{ margin-top: 0; font-size: 1.5em; }}
    p {{ margin: 1em 0; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    {body}
  </div>
</body>
</html>"""


class OAuthHandler:
    """Handles OAuth 2.0 PKCE flow for MCP authentication."""

    def __init__(
        self,
        config: OAuthConfig,
        manual: bool = False,
        code_prompt: "Callable[[str], Awaitable[str]] | None" = None,
        browser_open: "Callable[[str], Awaitable[bool]] | None" = None,
    ):
        """Initialize OAuth handler.

        Args:
            config: OAuth configuration
            manual: Use the out-of-band flow (show link, paste code) instead of
                binding a local callback server. Required for headless/remote
                sessions (e.g. a docker sandbox with no browser/localhost loop).
            code_prompt: Async callback that takes the authorization URL and
                returns the pasted code. Used in manual mode so the host
                application (TUI) controls input instead of blocking ``input()``.
            browser_open: Async hook that opens a URL in a *reachable* browser and
                returns True on success. Lets a headless/sandboxed host open the
                consent page elsewhere (e.g. on the host machine) while the
                loopback ``localhost`` callback is forwarded back — preferred over
                the OOB fallback when the in-process browser is unavailable.
        """
        self.config = config
        self.manual = manual
        self._code_prompt = code_prompt
        self._browser_open = browser_open
        self._code_verifier: str | None = None
        self._code_challenge: str | None = None
        # Set by _capture_code_via_local_server after dynamic port assignment;
        # used by exchange_code_for_token to send the exact redirect_uri the
        # authorization server received (RFC 8252 §4.1 requirement).
        self._actual_redirect_uri: str | None = None

    def _generate_pkce_pair(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge.

        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        # Generate code verifier (43-128 characters, URL-safe)
        code_verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
        )

        # Generate code challenge (SHA256 hash of verifier)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("utf-8")).digest())
            .decode("utf-8")
            .rstrip("=")
        )

        return code_verifier, code_challenge

    def _build_authorization_url(self, redirect_uri: str | None = None) -> str:
        """Build OAuth authorization URL with PKCE parameters.

        Args:
            redirect_uri: Override redirect URI (e.g. after dynamic port selection).
                          Defaults to self.config.redirect_uri.

        Returns:
            Authorization URL
        """
        self._code_verifier, self._code_challenge = self._generate_pkce_pair()
        if not self.config.client_id:
            raise RuntimeError(
                "OAuth client_id is missing and dynamic client registration did not complete"
            )

        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": redirect_uri or self.config.redirect_uri,
            "code_challenge": self._code_challenge,
            "code_challenge_method": "S256",
        }

        if self.config.scope:
            params["scope"] = self.config.scope

        query_string = urlencode(params)
        return f"{self.config.authorization_endpoint}?{query_string}"

    async def _authorize_manual(self, open_browser: bool = True) -> str:
        """Out-of-band authorization: show the URL, collect a pasted code.

        No local callback server is bound, so this works in headless/remote
        environments (e.g. a docker sandbox). Uses the OOB redirect URI and the
        host-provided ``code_prompt`` callback to read the code, never blocking
        ``input()``.
        """
        oob_redirect = "urn:ietf:wg:oauth:2.0:oob"
        self._actual_redirect_uri = oob_redirect
        await self._register_dynamic_client(oob_redirect)
        auth_url = self._build_authorization_url(redirect_uri=oob_redirect)

        if open_browser:
            with contextlib.suppress(Exception):
                webbrowser.open(auth_url)
        logger.info(f"Please authorize at: {auth_url}")

        if self._code_prompt is None:
            raise RuntimeError("Manual OAuth requires a code prompt callback but none was provided")
        code = _extract_authorization_code(await self._code_prompt(auth_url))
        if not code:
            raise RuntimeError("Authorization code not provided")
        return code

    async def _register_dynamic_client(self, redirect_uri: str) -> None:
        """Register an OAuth client for the already-bound callback URI."""
        if self.config.client_id or not self.config.registration_endpoint:
            return

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.config.registration_endpoint,
                json={"redirect_uris": [redirect_uri]},
                timeout=10.0,
            )
            if response.status_code not in (200, 201):
                raise RuntimeError(
                    f"Client registration failed: {response.status_code} {response.text}"
                )
            data = response.json()
            self.config.client_id = data.get("client_id")
            self.config.client_secret = data.get("client_secret")
            if not self.config.client_id:
                raise RuntimeError("Client registration response did not include client_id")
            logger.info(f"Successfully registered OAuth client: {self.config.client_id}")

    async def authorize(self, open_browser: bool = True) -> str:
        """Perform OAuth authorization flow.

        Implements RFC 8252 §7.3: binds a temporary local HTTP server on an
        OS-assigned port (port 0) so the callback is captured automatically
        without any copy-paste.  Fails fast if the local callback server
        cannot be started; blocking stdin prompts are unsafe inside the TUI.

        Args:
            open_browser: Whether to automatically open browser

        Returns:
            Authorization code

        Raises:
            RuntimeError: If authorization fails
        """
        if self.manual:
            return await self._authorize_manual(open_browser)

        # The loopback-callback flow needs a browser to reach the consent page and
        # call back to localhost. In headless/remote sessions there is no in-process
        # browser. If a browser_open hook is available (e.g. a sandbox that opens the
        # URL on the host and forwards localhost back), use the loopback flow via that
        # hook. Otherwise fall back to manual OOB, or fail with config guidance.
        if open_browser and not _system_browser_available():
            if self._browser_open is not None:
                logger.info("No in-process browser; using browser_open hook for the loopback flow.")
                # open_browser stays True: _capture_code_via_local_server will route
                # the URL through the hook instead of webbrowser.open.
            elif self._code_prompt is not None:
                logger.info("No system browser detected; using manual OAuth (paste the code/URL).")
                return await self._authorize_manual(open_browser=False)
            else:
                raise RuntimeError(
                    "OAuth requires a browser to complete the loopback-callback flow, but no "
                    "system browser is available in this session (headless/remote). Configure "
                    "the MCP server for manual OAuth by adding to its block in "
                    ".nooa/config.toml:\n"
                    "    oauth_manual = true\n"
                    "    oauth_open_browser = false\n"
                    "then reconnect and paste the authorization code or callback URL when prompted."
                )

        try:
            code = await self._capture_code_via_local_server(open_browser)
        except Exception as e:
            raise RuntimeError(
                "OAuth browser callback failed before an authorization code was received. "
                "Retry the connection; if this persists, check that the browser can open "
                "localhost callback URLs from this TUI session."
            ) from e

        if not code:
            raise RuntimeError("Authorization code not provided")

        return code

    async def _capture_code_via_local_server(self, open_browser: bool) -> str:
        """Start a temporary HTTP server on a dynamic OS-assigned port (RFC 8252 §7.3).

        Args:
            open_browser: Whether to automatically open the browser

        Returns:
            Authorization code extracted from the callback request

        Raises:
            RuntimeError: If the server times out or receives an error callback
        """
        parsed = urlparse(self.config.redirect_uri)
        host = parsed.hostname or "localhost"
        callback_path = parsed.path or "/callback"
        scheme = parsed.scheme or "http"
        # Use port from redirect_uri if specified, otherwise bind to port 0 for dynamic assignment
        requested_port = parsed.port if parsed.port is not None and parsed.port != 0 else 0

        received_code: list[str] = []
        error_info: list[str] = []
        done = asyncio.Event()
        loop = asyncio.get_running_loop()

        class CallbackHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass  # Silence request logs

            def do_GET(self) -> None:
                req_parsed = urlparse(self.path)
                if req_parsed.path != callback_path:
                    self.send_response(404)
                    self.end_headers()
                    return

                params = parse_qs(req_parsed.query)
                if "error" in params:
                    error_info.append(params["error"][0])
                    body = _html_page(
                        "Authorization Failed",
                        f"<p style='color:red'>Error: {html.escape(params['error'][0])}</p>"
                        "<p>You can close this tab.</p>",
                    )
                elif "code" in params:
                    received_code.append(params["code"][0])
                    body = _html_page(
                        "Authorization Successful",
                        "<p style='color:green'>Success</p>"
                        "<p>You can close this tab and return to the application.</p>",
                    )
                else:
                    body = _html_page(
                        "Unexpected Response", "<p>No code received. You can close this tab.</p>"
                    )

                encoded = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                # Set the asyncio.Event from the thread using thread-safe call
                loop.call_soon_threadsafe(done.set)

        # Bind to requested port (0 means OS picks a free port, RFC 8252 §7.3)
        server = HTTPServer((host, requested_port), CallbackHandler)
        actual_port = server.server_address[1]
        server.timeout = 1.0  # Wake up every second to check for cancellation

        # Build the redirect URI now that HTTPServer has bound the actual port.
        # Dynamic registration happens while this server still owns the port, so
        # no other process can claim it between port selection and callback bind.
        actual_redirect_uri = f"{scheme}://{host}:{actual_port}{callback_path}"
        self._actual_redirect_uri = actual_redirect_uri
        try:
            await self._register_dynamic_client(actual_redirect_uri)
            auth_url = self._build_authorization_url(redirect_uri=actual_redirect_uri)
        except Exception:
            server.server_close()
            raise

        logger.info(f"OAuth callback server listening on {actual_redirect_uri}")

        def serve() -> None:
            while not done.is_set():
                server.handle_request()
            server.server_close()

        # Run the blocking server loop in a thread (asyncio.to_thread is for one-shot
        # functions, but this is a long-running loop that needs to run until done)
        thread = Thread(target=serve, daemon=True)
        thread.start()

        if open_browser:
            opened = False
            if self._browser_open is not None:
                try:
                    opened = await self._browser_open(auth_url)
                except Exception as e:
                    logger.warning(f"browser_open hook failed: {e}")
                if opened:
                    logger.info("Opened authorization URL via browser_open hook")
            if not opened:
                try:
                    webbrowser.open(auth_url)
                    logger.info("Opened browser for authorization")
                except Exception as e:
                    logger.warning(f"Failed to open browser: {e}")
                    logger.info(f"Please visit: {auth_url}")
        else:
            logger.info(f"Please visit: {auth_url}")

        with contextlib.suppress(asyncio.TimeoutError):
            # Timeout is handled below by checking if received_code is empty
            await asyncio.wait_for(done.wait(), timeout=self.config.timeout)

        thread.join(timeout=2)

        if error_info:
            raise RuntimeError(f"OAuth authorization error: {error_info[0]}")
        if not received_code:
            raise RuntimeError(
                f"OAuth callback timed out — no authorization code received within {self.config.timeout} seconds"
            )

        logger.info("Authorization code captured automatically from callback")
        return received_code[0]

    async def exchange_code_for_token(self, code: str) -> OAuthToken:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from OAuth callback

        Returns:
            OAuth token

        Raises:
            RuntimeError: If token exchange fails
        """
        if not self._code_verifier:
            raise RuntimeError("Code verifier not set. Call authorize() first.")

        async with httpx.AsyncClient() as client:
            # Use the redirect_uri that was actually sent to the authorization
            # server (may differ from config due to dynamic port selection).
            redirect_uri_used = self._actual_redirect_uri or self.config.redirect_uri
            normalized_redirect_uri = redirect_uri_used.rstrip("/")

            data = {
                "grant_type": "authorization_code",
                "code": code.strip(),
                "redirect_uri": normalized_redirect_uri,
                "client_id": self.config.client_id,
                "code_verifier": self._code_verifier,
            }

            if self.config.client_secret:
                data["client_secret"] = self.config.client_secret

            try:
                response = await client.post(self.config.token_endpoint, data=data)
                response.raise_for_status()
                token_data = response.json()

                logger.info("Token exchange successful!")

                return OAuthToken(
                    access_token=token_data["access_token"],
                    token_type=token_data.get("token_type", "Bearer"),
                    expires_in=token_data.get("expires_in"),
                    refresh_token=token_data.get("refresh_token"),
                    obtained_at=time.time(),
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret,
                )
            except httpx.HTTPStatusError as e:
                error_detail = e.response.text if e.response else str(e)
                error_json = None
                try:
                    if e.response:
                        error_json = e.response.json()
                except Exception:
                    pass

                logger.error(
                    f"Token exchange failed: {e.response.status_code}, Request data: {data}, Response: {error_detail}",
                )
                if error_json:
                    logger.error(f"Error details: {error_json}")

                # Provide helpful error message for common issues
                error_msg = f"Failed to exchange authorization code for token (HTTP {e.response.status_code})"
                if error_json and error_json.get("error") == "invalid_grant":
                    error_msg += (
                        "\nThe authorization code may have expired or already been used. "
                        "Authorization codes are typically valid for only a few minutes. "
                        "Please try the OAuth flow again to get a fresh code."
                    )
                else:
                    error_msg += f": {error_detail}"

                raise RuntimeError(error_msg) from e

    async def client_credentials_token(self) -> OAuthToken:
        """Fetch a token via the non-interactive client_credentials grant.

        Machine-to-machine flow (RFC 6749 §4.4): no browser, no redirect, no
        user consent — just client_id/client_secret exchanged for a token. Used
        for headless/agent connects against servers that advertise it. Requires
        a client_secret (from config or dynamic registration).
        """
        if not self.config.client_id or not self.config.client_secret:
            raise RuntimeError(
                "client_credentials grant requires a client_id and client_secret. "
                "Provide oauth_client_id/oauth_client_secret in config, or use a server "
                "that supports dynamic client registration."
            )
        data = {
            "grant_type": "client_credentials",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }
        if self.config.scope:
            data["scope"] = self.config.scope

        async with httpx.AsyncClient() as client:
            response = await client.post(self.config.token_endpoint, data=data, timeout=30.0)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = e.response.text if e.response is not None else str(e)
                raise RuntimeError(
                    f"client_credentials token request failed "
                    f"(HTTP {e.response.status_code}): {detail}"
                ) from e
            td = response.json()
        if "access_token" not in td:
            raise RuntimeError(f"client_credentials token response missing 'access_token': {td}")
        return OAuthToken(
            access_token=td["access_token"],
            token_type=td.get("token_type", "Bearer"),
            expires_in=td.get("expires_in"),
            refresh_token=td.get("refresh_token"),
            obtained_at=time.time(),
            client_id=self.config.client_id,
            client_secret=self.config.client_secret,
        )

    async def complete_flow(self, open_browser: bool = True) -> OAuthToken:
        """Complete full OAuth flow: authorize and exchange code for token.

        Args:
            open_browser: Whether to automatically open browser

        Returns:
            OAuth token

        Raises:
            RuntimeError: If OAuth flow fails
        """
        try:
            logger.info("Starting OAuth authorization...")
            code = await self.authorize(open_browser=open_browser)
            logger.info(f"Authorization code received (length: {len(code)})")
            logger.info("Exchanging authorization code for token...")
            token = await self.exchange_code_for_token(code)
            logger.info("OAuth flow completed successfully")
            return token
        except Exception as e:
            logger.error(f"OAuth flow failed: {e}")
            raise


async def _discover_authorization_servers(client: httpx.AsyncClient, server_url: str) -> list[str]:
    """Discover OAuth authorization servers for an MCP resource.

    Follows the RFC 9728 protected-resource-metadata flow:
      1. Make an unauthenticated request and read the ``resource_metadata``
         pointer from the ``WWW-Authenticate`` header (the canonical source —
         the metadata path is *not* a fixed suffix of the server URL).
      2. Fall back to probing well-known paths relative to the server URL for
         servers that don't emit the header.

    Returns the ``authorization_servers`` list, or ``[]`` if none were found.
    """
    metadata_urls: list[str] = []

    pointer = await _resource_metadata_pointer(client, server_url)
    if pointer:
        metadata_urls.append(pointer)

    # RFC 9728 §2.2: the well-known segment is inserted *before* the resource
    # path, not appended after it — e.g. https://host/.well-known/
    # oauth-protected-resource/mcp for a resource at https://host/mcp.
    parsed = urlparse(server_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    resource_path = parsed.path.rstrip("/")
    metadata_urls.extend(
        [
            f"{origin}/.well-known/oauth-protected-resource{resource_path}",
            f"{origin}/.well-known/oauth-authorization-server{resource_path}",
        ]
    )

    for url in metadata_urls:
        try:
            response = await client.get(url, timeout=5.0)
        except Exception:
            continue
        if response.status_code != 200:
            continue
        try:
            data = response.json()
        except Exception:
            continue
        servers = data.get("authorization_servers") or []
        if servers:
            return servers
        # Some servers return authorization-server metadata directly here.
        if data.get("authorization_endpoint") and data.get("token_endpoint"):
            issuer = data.get("issuer")
            if issuer:
                return [issuer]

    return []


async def _fetch_protected_resource_metadata(
    client: httpx.AsyncClient, server_url: str
) -> dict[str, Any]:
    """Fetch RFC 9728 protected-resource metadata for ``server_url`` if available."""
    metadata_urls: list[str] = []
    pointer = await _resource_metadata_pointer(client, server_url)
    if pointer:
        metadata_urls.append(pointer)

    parsed = urlparse(server_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    resource_path = parsed.path.rstrip("/")
    metadata_urls.append(f"{origin}/.well-known/oauth-protected-resource{resource_path}")

    seen: set[str] = set()
    for url in metadata_urls:
        if url in seen:
            continue
        seen.add(url)
        try:
            response = await client.get(url, timeout=5.0)
        except Exception:
            continue
        if response.status_code != 200:
            continue
        try:
            data = response.json()
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


async def _resource_metadata_pointer(client: httpx.AsyncClient, server_url: str) -> str | None:
    """Read the RFC 9728 ``resource_metadata`` URL from a 401 challenge.

    MCP servers behind an OAuth gateway answer an unauthenticated request with
    ``401`` and a ``WWW-Authenticate: Bearer ... resource_metadata="<url>"``
    header. That URL is the only reliable way to locate the metadata document.
    """
    try:
        response = await client.post(
            server_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "nemo-oo-agents", "version": "0"},
                },
            },
            headers={"Accept": "application/json, text/event-stream"},
            timeout=5.0,
        )
    except Exception:
        return None

    # Only trust the pointer from a genuine auth challenge; a 200 with a stray
    # WWW-Authenticate header (e.g. from a validating proxy) must not redirect
    # discovery to the wrong metadata document.
    if response.status_code != 401:
        return None
    header = response.headers.get("www-authenticate", "")
    match = re.search(r'resource_metadata="([^"]+)"', header)
    return match.group(1) if match else None


async def _fetch_authorization_server_metadata(
    client: httpx.AsyncClient, auth_server: str
) -> dict[str, Any]:
    """Fetch RFC 8414 authorization-server metadata for ``auth_server``."""
    for suffix in (
        "/.well-known/oauth-authorization-server",
        "/.well-known/openid-configuration",
    ):
        try:
            response = await client.get(auth_server + suffix, timeout=5.0)
        except Exception:
            continue
        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                continue
    return {}


def _token_cache_path() -> Path:
    """Return the per-project token cache path (.nooa/mcp_tokens.json)."""
    base = os.environ.get("NEMO_OO_PROJECT_DIR") or os.getcwd()
    return Path(base) / ".nooa" / "mcp_tokens.json"


def _load_cached_token(server_url: str) -> OAuthToken | None:
    """Load a cached OAuth token for ``server_url`` if present and well-formed."""
    path = _token_cache_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    entry = data.get(server_url)
    if not isinstance(entry, dict) or "access_token" not in entry:
        return None
    return OAuthToken(
        access_token=entry["access_token"],
        token_type=entry.get("token_type", "Bearer"),
        expires_in=entry.get("expires_in"),
        refresh_token=entry.get("refresh_token"),
        obtained_at=entry.get("obtained_at", 0.0),
        client_id=entry.get("client_id"),
        client_secret=entry.get("client_secret"),
    )


def _save_cached_token(server_url: str, token: OAuthToken) -> None:
    """Persist ``token`` for ``server_url`` to the project token cache (chmod 600)."""
    path = _token_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        data[server_url] = {
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in": token.expires_in,
            "refresh_token": token.refresh_token,
            "obtained_at": token.obtained_at,
            "client_id": token.client_id,
            "client_secret": token.client_secret,
        }
        path.write_text(json.dumps(data, indent=2))
        with contextlib.suppress(OSError):
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError as e:
        logger.warning(f"Could not persist MCP OAuth token cache: {e}")


async def _refresh_access_token(
    token_endpoint: str,
    client_id: str,
    refresh_token: str,
    client_secret: str | None = None,
) -> OAuthToken | None:
    """Exchange a refresh token for a fresh access token, or None on failure."""
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    if client_secret:
        data["client_secret"] = client_secret
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(token_endpoint, data=data, timeout=10.0)
            response.raise_for_status()
            td = response.json()
        return OAuthToken(
            access_token=td["access_token"],
            token_type=td.get("token_type", "Bearer"),
            expires_in=td.get("expires_in"),
            # Some servers omit a rotated refresh token; keep the existing one.
            refresh_token=td.get("refresh_token", refresh_token),
            obtained_at=time.time(),
            client_id=client_id,
            client_secret=client_secret,
        )
    except Exception as e:
        logger.info(f"Refresh token exchange failed ({e}); falling back to full auth")
        return None


async def handle_mcp_oauth(
    server_url: str,
    redirect_uri: str = "http://localhost:0/callback",
    client_id: str | None = None,
    client_secret: str | None = None,
    scope: str | None = None,
    open_browser: bool = True,
    manual: bool = False,
    code_prompt: "Callable[[str], Awaitable[str]] | None" = None,
    browser_open: "Callable[[str], Awaitable[bool]] | None" = None,
    use_cache: bool = True,
    timeout: float | None = None,
) -> OAuthToken:
    """Handle OAuth flow for MCP server, reusing a cached token when possible.

    Resolution order:
      1. A non-expired cached access token for ``server_url`` is returned as-is.
      2. A cached refresh token is exchanged for a fresh access token.
      3. A full authorization flow runs — either the local-callback browser flow
         or, when ``manual=True``, the out-of-band (link + pasted code) flow for
         headless/remote sessions.

    Args:
        server_url: MCP server URL
        redirect_uri: OAuth redirect URI for the local-callback flow
        client_id: OAuth client ID (if not provided, may be discovered/registered)
        client_secret: OAuth client secret (enables the client_credentials grant)
        scope: OAuth scopes
        open_browser: Whether to automatically open the browser
        manual: Use the out-of-band (link + paste code) flow
        code_prompt: Async callback returning the pasted code (manual mode)
        browser_open: Async hook to open the auth URL in a reachable browser (host handoff)
        use_cache: Read/write the per-project token cache

    Returns:
        OAuthToken object containing the access token and metadata

    Raises:
        RuntimeError: If OAuth flow fails or endpoints cannot be discovered
    """

    auth_endpoint = None
    token_endpoint = None
    registration_endpoint = None
    grant_types: list[str] = []

    async with httpx.AsyncClient(follow_redirects=True) as client:
        resource_metadata = await _fetch_protected_resource_metadata(client, server_url)
        if scope is None:
            scopes = resource_metadata.get("scopes_supported") or []
            if all(isinstance(s, str) for s in scopes):
                scope = " ".join(scopes) or None

        authorization_servers = resource_metadata.get("authorization_servers") or []
        if not authorization_servers:
            authorization_servers = await _discover_authorization_servers(client, server_url)

        if authorization_servers:
            auth_server = authorization_servers[0].rstrip("/")
            metadata = await _fetch_authorization_server_metadata(client, auth_server)
            auth_endpoint = metadata.get("authorization_endpoint") or auth_server + "/authorize"
            token_endpoint = metadata.get("token_endpoint") or auth_server + "/token"
            registration_endpoint = (
                metadata.get("registration_endpoint") or auth_server + "/register"
            )
            grant_types = metadata.get("grant_types_supported") or []

    # 1/2. Reuse a cached token (or refresh it) before prompting for auth again.
    if use_cache:
        cached = _load_cached_token(server_url)
        if cached and not cached.is_expired():
            logger.info("Using cached MCP OAuth token")
            return cached
        refresh_client_id = client_id or cached.client_id if cached else None
        if cached and cached.refresh_token and token_endpoint and refresh_client_id:
            refreshed = await _refresh_access_token(
                token_endpoint, refresh_client_id, cached.refresh_token, cached.client_secret
            )
            if refreshed:
                logger.info("Refreshed MCP OAuth token from cache")
                _save_cached_token(server_url, refreshed)
                return refreshed

    final_client_id = client_id
    final_client_secret = client_secret

    if not final_client_id and not registration_endpoint:
        raise RuntimeError(
            "client_id is required but could not be discovered, and the OAuth server "
            "did not advertise a dynamic client registration endpoint. Please provide "
            "oauth_client_id in .mcp.json or contact the MCP server administrator."
        )

    if auth_endpoint is None or token_endpoint is None:
        raise ValueError(
            "OAuth discovery failed: missing authorization/token endpoint. "
            "The OAuth server did not provide the required endpoints."
        )

    config = OAuthConfig(
        authorization_endpoint=auth_endpoint,
        token_endpoint=token_endpoint,
        client_id=final_client_id,
        client_secret=final_client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
        registration_endpoint=registration_endpoint,
        # Caller-supplied timeout overrides the OAuthConfig default (300s); this
        # is the single authoritative OAuth wait — the local callback server's
        # serve loop polls every 1s and exits on it, so there is no orphaned
        # thread to leak.
        **({"timeout": timeout} if timeout is not None else {}),
    )

    handler = OAuthHandler(
        config, manual=manual, code_prompt=code_prompt, browser_open=browser_open
    )

    # Prefer the non-interactive client_credentials grant when the server
    # advertises it. It needs no browser/redirect/user consent, so it's the
    # right path for headless/agent connects (and for servers like maas-nvbugs
    # that reject the OOB redirect used by the manual authorization_code flow).
    # Dynamic registration (run inside the auth flows) supplies the client_secret.
    if "client_credentials" in grant_types:
        if config.client_secret is None and config.registration_endpoint:
            await handler._register_dynamic_client(redirect_uri)
        if config.client_id and config.client_secret:
            logger.info("Using client_credentials grant (non-interactive)")
            token = await handler.client_credentials_token()
            if use_cache:
                _save_cached_token(server_url, token)
            return token

    token = await handler.complete_flow(open_browser=open_browser)
    if use_cache:
        _save_cached_token(server_url, token)
    return token
