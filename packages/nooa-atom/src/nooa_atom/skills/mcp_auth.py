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
import fcntl
import inspect
import json
import os
import re
import tempfile
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

import anyio
import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.auth import utils as oauth_utils
from mcp.client.auth.exceptions import OAuthFlowError
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_oauth_metadata_request,
    extract_resource_metadata_from_www_auth,
    handle_auth_metadata_response,
    handle_protected_resource_response,
)
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from mcp.types import ErrorData
from pydantic import AnyHttpUrl

from nooa.mcp import MCPBaseClient
from nooa.paths import get_user_dir

try:  # mcp < 2 spells it McpError and constructs it from ErrorData
    from mcp.shared.exceptions import McpError

    MCP_TIMEOUT_CODE = 408  # what the 1.x SDK's own response timeout reports

    def mcp_error(error: ErrorData) -> McpError:
        """The installed SDK's exception for ``error``."""
        return McpError(error)

except ImportError:  # pragma: no cover - depends on the installed mcp
    # mcp >= 2 renamed it and takes (code, message, data); from_error_data keeps the ErrorData shape.
    from mcp.shared.exceptions import MCPError as McpError

    MCP_TIMEOUT_CODE = -32001  # the 2.x SDK's request-timeout code

    def mcp_error(error: ErrorData) -> McpError:
        """The installed SDK's exception for ``error``."""
        return McpError.from_error_data(error)


# Where the browser is sent after sign-in. Nothing listens there: the person
# copies the address from the browser. Some gateways reject 127.0.0.1.
DEFAULT_REDIRECT_URI = "http://localhost/callback"
# How long a sign-in waits for the pasted redirect address.
SIGN_IN_TIMEOUT_SECONDS = 600.0
# Establishing the connection keeps the short budget core's clients use.
_CONNECT_TIMEOUT_SECONDS = 5.0


def _issuers_match(left: str, right: str) -> bool:
    """Use SDK 1.30 issuer matching, with the same root-slash rule on 1.28."""
    match = getattr(oauth_utils, "issuers_match", None)
    if match is not None:
        return match(left, right)
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    parsed = urlsplit(shorter)
    return longer == f"{shorter}/" and shorter == f"{parsed.scheme}://{parsed.netloc}"


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

    Cooperating POSIX writers serialize whole-file read/derive/replace with
    an advisory flock on a stable 0600 sidecar (which must not be removed
    while writers are active). Readers see atomically published snapshots.
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
        self._token_write: ContextVar[tuple[str | None, dict[str, Any]] | None] = ContextVar(
            "mcp_token_write", default=None
        )

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _entry(self) -> dict[str, Any]:
        entry = self._read().get(self.server_url)
        return entry if isinstance(entry, dict) else {}

    def _open_lock(self) -> int:
        # Lock a stable sidecar, not the replaced JSON inode. Separate opens
        # also serialize threads/providers in one process (Linux flock).
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except BaseException:
            os.close(fd)
            raise
        return fd

    @contextmanager
    def _transaction(self):
        fd = self._open_lock()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    @asynccontextmanager
    async def _async_transaction(self):
        # No blocking executor waiter survives cancellation. Locks cover only
        # local read/derive/replace, never network I/O or the human callback.
        fd = self._open_lock()
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.01)
            yield
        finally:
            os.close(fd)

    def _update_locked(self, **fields: Any) -> dict[str, Any]:
        data = self._read()
        entry = data.get(self.server_url)
        data[self.server_url] = {**(entry if isinstance(entry, dict) else {}), **fields}
        self._write(data)
        return data[self.server_url]

    def _update(self, **fields: Any) -> None:
        with self._transaction():
            self._update_locked(**fields)

    async def update(self, **fields: Any) -> dict[str, Any]:
        async with self._async_transaction():
            return self._update_locked(**fields)

    @contextmanager
    def bind_token_writes(self, issuer: str | None):
        """Bind an SDK token save and capture its exact published snapshot.

        This context holds no file lock. ContextVar isolates providers sharing
        storage, and set_tokens takes the short transaction only when saving.
        """
        saved: dict[str, Any] = {}
        token = self._token_write.set((issuer, saved))
        try:
            yield saved
        finally:
            self._token_write.reset(token)

    def _write(self, data: dict[str, Any]) -> None:
        """Replace a transaction's snapshot; callers must hold the sidecar lock."""
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

    def clear_tokens(self) -> None:
        """Forget only tokens/expiry; retain client registration and both issuer bindings."""
        with self._transaction():
            self._clear_tokens_locked()

    def _clear_tokens_locked(self) -> None:
        data = self._read()
        entry = data.get(self.server_url)
        if isinstance(entry, dict):
            data[self.server_url] = {
                key: value for key, value in entry.items() if key not in {"tokens", "expires_at"}
            }
            self._write(data)

    async def invalidate(self, expected: dict[str, Any]) -> dict[str, Any] | None:
        """Clear only the credential snapshot used by this provider.

        Return a concurrent winner without changing it. Comparing the entire
        entry includes token value, absolute expiry and registration/bindings;
        unrelated URL writes do not cause a spurious conflict.
        """
        async with self._async_transaction():
            current = self._entry()
            if current != expected:
                return current
            self._clear_tokens_locked()
            return None

    def forget_auth(self) -> None:
        """Forget this URL's local credentials and binding, not remotely revoke tokens.

        Callers must first drain all providers using this URL. Configuration,
        including a configured client ID, is deliberately not changed.
        """
        # An active same-URL provider in another process can still subsequently
        # publish *its own* credentials. The registry only drains local owners;
        # this transaction prevents unrelated URL writes resurrecting this one.
        with self._transaction():
            data = self._read()
            if self.server_url in data:
                del data[self.server_url]
                self._write(data)

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens_from_entry(self._entry())

    @staticmethod
    def _tokens_from_entry(entry: dict[str, Any]) -> OAuthToken | None:
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

    def get_token_expiry(self) -> float | None:
        """The persisted absolute deadline, not a restarted ``expires_in`` clock."""
        expires_at = self._entry().get("expires_at")
        return expires_at if isinstance(expires_at, int | float) else None

    def get_issuer(self) -> str | None:
        """Issuer binding for credentials saved by Atom; older caches have none."""
        issuer = self._entry().get("issuer")
        return issuer if isinstance(issuer, str) else None

    def get_client_issuer(self) -> str | None:
        """Preserve SDK-native binding even when loading through an older model."""
        client_info = self._entry().get("client_info")
        issuer = client_info.get("issuer") if isinstance(client_info, dict) else None
        return issuer if isinstance(issuer, str) else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        expires_at = time.time() + tokens.expires_in if tokens.expires_in else None
        fields = {"tokens": tokens.model_dump(mode="json"), "expires_at": expires_at}
        binding = self._token_write.get()
        if binding and binding[0] is not None:
            fields["issuer"] = binding[0]
        entry = await self.update(**fields)
        if binding:
            binding[1]["entry"] = entry

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_from_entry(self._entry())

    def _client_from_entry(self, entry: dict[str, Any]) -> OAuthClientInformationFull | None:
        try:
            return OAuthClientInformationFull.model_validate(entry["client_info"])
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
        async with self._async_transaction():
            entry = self._entry()
            fields: dict[str, Any] = {"client_info": client_info.model_dump(mode="json")}
            tokens = entry.get("tokens")
            previous = entry.get("client_info")
            previous_issuer = previous.get("issuer") if isinstance(previous, dict) else None
            if (
                previous_issuer
                and previous.get("client_id") == client_info.client_id
                and not fields["client_info"].get("issuer")
            ):
                # An older SDK model must not erase the native issuer when
                # round-tripping the very same registration.
                fields["client_info"]["issuer"] = previous_issuer
            if isinstance(tokens, dict) and tokens.get("refresh_token") and not entry.get("issuer"):
                # SDK step-up may persist a registration before new tokens.
                # Derive this binding from the same locked snapshot we update.
                if previous_issuer:
                    fields["issuer"] = previous_issuer
            self._update_locked(**fields)


class SignInTimeoutError(TimeoutError):
    """The human callback deadline expired (distinct from a network timeout)."""


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
        self._waiting = False
        self._budgets: dict[asyncio.Timeout, float] = {}
        self._pasted: asyncio.Future[tuple[str, str | None]] = (
            asyncio.get_running_loop().create_future()
        )

    async def redirect(self, url: str) -> None:
        self.url = url
        self.url_ready.set()

    async def callback(self) -> tuple[str, str | None]:
        self._waiting = True
        loop = asyncio.get_running_loop()
        for budget in self._budgets:
            if budget.expired():
                continue
            deadline = budget.when()
            if deadline is not None:
                self._budgets[budget] = max(0.0, deadline - loop.time())
                budget.reschedule(None)
        try:
            return await asyncio.wait_for(asyncio.shield(self._pasted), self._timeout)
        except TimeoutError:
            self.cancel()
            raise SignInTimeoutError(
                f"The sign-in was not completed within {self._timeout:g} seconds; "
                "connect again to start a new one."
            ) from None
        except asyncio.CancelledError:
            self.cancel()
            raise
        finally:
            self._waiting = False
            for budget, remaining in self._budgets.items():
                if not budget.expired():
                    budget.reschedule(loop.time() + remaining)

    @asynccontextmanager
    async def response_budget(self, seconds: float) -> AsyncGenerator[asyncio.Timeout, None]:
        """An MCP response deadline, paused only while awaiting human input.

        HTTPX still enforces network I/O deadlines. The SDK's own response
        deadline cannot be paused, so the authorized session uses this one.
        """
        async with asyncio.timeout(None if self._waiting else seconds) as budget:
            self._budgets[budget] = seconds
            try:
                yield budget
            finally:
                self._budgets.pop(budget, None)

    def cancel(self) -> None:
        """Invalidate a callback belonging to a stopped or expired connection."""
        self._pasted.cancel()
        self.url = None

    def complete(self, pasted: str) -> None:
        """Take the pasted redirect address; ``ValueError`` if it holds no code."""
        if self._pasted.done():
            raise ValueError("This sign-in is no longer waiting; connect again to start a new one.")
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


class RefreshMetadataAbsent(OAuthFlowError):
    """PRM candidates are genuinely absent; validated same-origin manual auth is safe.

    This is not a parser failure or a transient discovery error, and must not
    be used to authorize migration to another issuer.
    """


class DiscoveringOAuthClientProvider(OAuthClientProvider):
    """SDK OAuth with discovery before refresh (SDK 1.28-1.30).

    Drive the SDK generator rather than copying its OAuth flow: its lock,
    PKCE, scope handling, client authentication and token parsing stay intact.
    Only refresh requests and the first resource 401 are intercepted. Discovery
    uses SDK parsers/resource validation and never sends credentials. A refresh
    requires discovered metadata, never the SDK's resource-origin /token guess.
    """

    async def _initialize(self) -> None:
        storage = self.context.storage
        if not isinstance(storage, FileTokenStorage):
            await super()._initialize()
            return
        # All credential components must come from one atomically published
        # snapshot, not tokens from A and registration/expiry from B.
        entry = storage._entry()
        self._credentials_entry = entry
        self.context.current_tokens = storage._tokens_from_entry(entry)
        self.context.client_info = storage._client_from_entry(entry)
        expiry = entry.get("expires_at")
        self.context.token_expiry_time = expiry if isinstance(expiry, int | float) else None
        self._initialized = True

    def _is_cimd_client(self) -> bool:
        # URL-shaped registered IDs are not CIMD: only exact configured identity.
        return bool(
            self.context.client_info
            and self.context.client_metadata_url is not None
            and self.context.client_info.client_id == self.context.client_metadata_url
        )

    def _cached_issuer_bindings(self) -> tuple[str | None, str | None]:
        """Bindings of the loaded credentials, never a concurrent disk winner."""
        storage = self.context.storage
        if not isinstance(storage, FileTokenStorage):
            return None, None
        entry = getattr(self, "_credentials_entry", None)
        if entry is None:
            entry = storage._entry()
        client = entry.get("client_info")
        native = client.get("issuer") if isinstance(client, dict) else None
        issuer = entry.get("issuer")
        return (
            issuer if isinstance(issuer, str) else None,
            native if isinstance(native, str) else None,
        )

    def _validate_cached_issuer(self, expected: str, *, refresh: bool) -> None:
        token_issuer, cached_native_issuer = self._cached_issuer_bindings()
        native_issuer = getattr(self.context.client_info, "issuer", None)
        match = getattr(oauth_utils, "credentials_match_issuer", None)
        if (
            self.context.client_info is not None
            and match is not None
            and not match(self.context.client_info, expected, self.context.client_metadata_url)
        ):
            raise OAuthFlowError("Cached client credentials belong to another issuer")
        # CIMD client identity is portable, but a known refresh-token issuer is not.
        for binding in (native_issuer, cached_native_issuer):
            if binding and (refresh or not self._is_cimd_client()):
                if not _issuers_match(binding, expected):
                    raise OAuthFlowError("Cached credentials belong to another issuer")
        if token_issuer and (refresh or not self._is_cimd_client()):
            if not _issuers_match(token_issuer, expected):
                raise OAuthFlowError(
                    "Authorization server changed; cached credentials cannot be reused"
                )

    async def _validate_resource_match(self, metadata: Any) -> None:
        """Guard full SDK rediscovery before 1.30 can drop a bound registration."""
        await super()._validate_resource_match(metadata)
        self._validate_cached_issuer(str(metadata.authorization_servers[0]), refresh=False)

    def _expected_issuer(self) -> str:
        """Validate before SDK 1.30's automatic issuer-clearing/registration step."""
        expected = super()._expected_issuer()
        self._validate_cached_issuer(expected, refresh=False)
        return expected

    async def _clear_tokens(self, *, reuse_winner: bool = False) -> bool:
        """Invalidate only this provider's snapshot, never a concurrent winner."""
        storage = self.context.storage
        winner = None
        if isinstance(storage, FileTokenStorage):
            winner = await storage.invalidate(self._credentials_entry)
        self.context.clear_tokens()
        if reuse_winner and winner and self._accept_winner(winner):
            return True
        return False

    def _accept_winner(self, entry: dict[str, Any]) -> bool:
        """Reuse only unexpired, scope-sufficient tokens with the same client/AS.

        Resource and AS metadata were validated by refresh discovery. Unknown
        bindings, changed registrations and portable CIMD IDs do not justify
        trusting another issuer's token. A rejected winner is left on disk.
        """
        storage = self.context.storage
        if not isinstance(storage, FileTokenStorage) or self.context.oauth_metadata is None:
            return False
        if entry.get("client_info") != self._credentials_entry.get("client_info"):
            return False
        expected = str(self.context.oauth_metadata.issuer)
        client = entry.get("client_info")
        # Raw None == None (or equal malformed entries) cannot prove who the
        # winner's token was issued to. Do not use configured-client fallback.
        try:
            persisted_client = OAuthClientInformationFull.model_validate(client)
        except ValueError:
            return False
        if (
            not persisted_client.client_id
            or self.context.client_info is None
            or persisted_client.client_id != self.context.client_info.client_id
        ):
            return False
        native = client.get("issuer") if isinstance(client, dict) else None
        bindings = [value for value in (entry.get("issuer"), native) if value is not None]
        if not bindings or any(
            not isinstance(value, str) or not _issuers_match(value, expected) for value in bindings
        ):
            return False
        tokens = storage._tokens_from_entry(entry)
        expiry = entry.get("expires_at")
        if (
            tokens is None
            or not tokens.access_token
            or not isinstance(expiry, int | float)
            or not time.time() < expiry
        ):
            return False
        required = set((self.context.client_metadata.scope or "").split())
        previous_tokens = self._credentials_entry.get("tokens")
        previous_scope = previous_tokens.get("scope") if isinstance(previous_tokens, dict) else None
        if isinstance(previous_scope, str):
            required.update(previous_scope.split())
        if not required.issubset(set((tokens.scope or "").split())):
            return False
        self.context.current_tokens = tokens
        self.context.token_expiry_time = expiry
        self._credentials_entry = entry
        return True

    @staticmethod
    async def _parse_prm(response: httpx.Response) -> Any:
        """SDK-parse PRM, distinguishing genuine 404/410 absence from all failures."""
        if response.status_code in (404, 410):
            return None
        if response.status_code != 200:
            raise OAuthFlowError(
                f"Protected-resource discovery failed: HTTP {response.status_code}"
            )
        metadata = await handle_protected_resource_response(response)
        if metadata is None:
            raise OAuthFlowError("Invalid protected-resource metadata")
        return metadata

    @staticmethod
    async def _parse_asm(response: httpx.Response) -> Any:
        """SDK-parse AS metadata without treating malformed JSON or outages as absence."""
        if response.status_code in (404, 410):
            return None
        if response.status_code != 200:
            raise OAuthFlowError(
                f"Authorization-server discovery failed: HTTP {response.status_code}"
            )
        _, metadata = await handle_auth_metadata_response(response)
        if metadata is None:
            raise OAuthFlowError("Invalid authorization-server metadata")
        return metadata

    def _validate_issuer(self) -> None:
        metadata = self.context.oauth_metadata
        expected = self.context.auth_server_url or self.context.get_authorization_base_url(
            self.context.server_url
        )
        if metadata is None or metadata.issuer != AnyHttpUrl(expected):
            raise OAuthFlowError("Authorization metadata issuer does not match discovery")
        self._validate_cached_issuer(expected, refresh=False)

    async def _perform_authorization(self) -> httpx.Request:
        # The SDK 1.28 resource check is retained; also bind AS metadata to PRM.
        if self.context.oauth_metadata is not None or any(self._cached_issuer_bindings()):
            self._validate_issuer()
        return await super()._perform_authorization()

    async def _refresh_token(self) -> httpx.Request:
        request = await super()._refresh_token()
        self._refresh_pending = True
        return request

    @staticmethod
    async def _refresh_rejects_credentials(response: httpx.Response) -> bool:
        """Recognize an auth rejection, not an outage or a broken token response.

        Retain HTTPX's buffered response for the SDK driver to read again.
        Only inspect a small OAuth error document; never include its body or
        credentials in diagnostics.
        """
        if response.status_code not in (400, 401, 403):
            return False
        body = await response.aread()
        if len(body) > 8192:
            return False
        try:
            payload = json.loads(body)
        except ValueError:
            return False
        error = payload.get("error") if isinstance(payload, dict) else None
        return error in ("invalid_grant", "invalid_client", "unauthorized_client", "access_denied")

    async def _handle_refresh_response(self, response: httpx.Response) -> bool:
        storage = self.context.storage
        if isinstance(storage, FileTokenStorage):
            issuer = (
                str(self.context.oauth_metadata.issuer) if self.context.oauth_metadata else None
            )
            with storage.bind_token_writes(issuer) as saved:
                refreshed = await super()._handle_refresh_response(response)
            if refreshed:
                self._credentials_entry = saved["entry"]
            elif await self._refresh_rejects_credentials(response):
                refreshed = await self._clear_tokens(reuse_winner=True)
            else:
                # The SDK clears its in-memory token on a failed response.
                # Retain the exact persisted deadline/scope/refresh token for
                # a later connection, without an automatic refresh retry here.
                self.context.clear_tokens()
        else:
            refreshed = await super()._handle_refresh_response(response)
        # Reading/parsing/saving the response is part of the owning refresh.
        # On failure/cancellation, leave the flag for that flow's cleanup.
        self._refresh_pending = False
        return refreshed

    async def _handle_token_response(self, response: httpx.Response) -> None:
        storage = self.context.storage
        if isinstance(storage, FileTokenStorage):
            issuer = (
                str(self.context.oauth_metadata.issuer) if self.context.oauth_metadata else None
            )
            with storage.bind_token_writes(issuer) as saved:
                await super()._handle_token_response(response)
            self._credentials_entry = saved["entry"]
        else:
            await super()._handle_token_response(response)

    async def _discover_for_refresh(
        self, challenge: httpx.Response | None
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        hint = extract_resource_metadata_from_www_auth(challenge) if challenge else None
        # Rediscover rather than trusting an endpoint from a previous 401.
        self.context.protected_resource_metadata = None
        self.context.oauth_metadata = None
        self.context.auth_server_url = None
        for attempt in range(2 if challenge is None else 1):
            for url in build_protected_resource_metadata_discovery_urls(
                hint, self.context.server_url
            ):
                response = yield create_oauth_metadata_request(url)
                prm = await self._parse_prm(response)
                if prm:
                    await self._validate_resource_match(prm)
                    self.context.protected_resource_metadata = prm
                    self.context.auth_server_url = str(prm.authorization_servers[0])
                    break
            if self.context.protected_resource_metadata is not None:
                break
            if attempt == 0 and challenge is None:
                # Some servers publish PRM only at the challenge's custom URL.
                # Probe without bearer tokens, client secrets or request payload.
                challenge = yield httpx.Request("GET", self.context.server_url)
                hint = extract_resource_metadata_from_www_auth(challenge)
        absent_prm = self.context.protected_resource_metadata is None
        expected = self.context.auth_server_url or self.context.get_authorization_base_url(
            self.context.server_url
        )
        # No PRM is not evidence that an alternate cached issuer moved to the
        # resource origin. Both bindings must match even for portable CIMD IDs.
        self._validate_cached_issuer(expected, refresh=True)

        for url in build_oauth_authorization_server_metadata_discovery_urls(
            self.context.auth_server_url, self.context.server_url
        ):
            response = yield create_oauth_metadata_request(url)
            metadata = await self._parse_asm(response)
            if metadata:
                self.context.oauth_metadata = metadata
                self._validate_issuer()
                break
        if self.context.oauth_metadata is None:
            raise OAuthFlowError("Cannot refresh without authorization-server metadata")
        if absent_prm:
            # Validate the legacy AS too before invalidating cached tokens. The
            # SDK will rediscover it during its normal PKCE/state/manual flow.
            raise RefreshMetadataAbsent("Protected-resource metadata is absent")

    def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if hasattr(OAuthClientProvider, "_auth_flow"):
            # SDK 1.30 wraps _auth_flow with its redirect-origin guard. Keep it.
            return super().async_auth_flow(request)
        return self._drive_sdk_flow(request, super().async_auth_flow(request))

    def _auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        return self._drive_sdk_flow(request, super()._auth_flow(request))

    async def _guard_full_discovery(
        self,
        pending: httpx.Request,
        response: httpx.Response,
        prm_urls: list[str],
        asm_urls: list[str],
    ) -> None:
        """Validate SDK rediscovery responses before resuming its generator.

        In particular the final absent PRM response must be guarded *before*
        SDK 1.30 clears an issuer-mismatched client in its origin fallback.
        SDK parsers and resource validation remain authoritative.
        """
        if prm_urls and str(pending.url) == prm_urls[0]:
            prm_urls.pop(0)
            metadata = await self._parse_prm(response)
            if metadata is not None:
                await self._validate_resource_match(metadata)
                prm_urls.clear()
                issuer = str(metadata.authorization_servers[0])
            elif prm_urls:
                return
            else:
                issuer = None
                origin = self.context.get_authorization_base_url(self.context.server_url)
                self._validate_cached_issuer(origin, refresh=True)
            asm_urls[:] = [
                str(httpx.URL(url))
                for url in build_oauth_authorization_server_metadata_discovery_urls(
                    issuer, self.context.server_url
                )
            ]
        elif asm_urls and str(pending.url) == asm_urls[0]:
            asm_urls.pop(0)
            metadata = await self._parse_asm(response)
            if metadata is not None:
                expected = self.context.auth_server_url or self.context.get_authorization_base_url(
                    self.context.server_url
                )
                if metadata.issuer != AnyHttpUrl(expected):
                    raise OAuthFlowError("Authorization metadata issuer does not match discovery")
                self._validate_cached_issuer(expected, refresh=False)
                asm_urls.clear()
            elif not asm_urls:
                if any(self._cached_issuer_bindings()):
                    raise OAuthFlowError("Cannot authorize bound credentials without AS metadata")

    async def _drive_sdk_flow(
        self, request: httpx.Request, flow: AsyncGenerator[httpx.Request, httpx.Response]
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Drive each SDK yield in its original slot; never transport guessed refreshes."""
        refresh_attempted = False
        started = False
        prm_urls: list[str] = []
        asm_urls: list[str] = []
        try:
            pending = await anext(flow)  # SDK initialization and lock acquisition.
            started = True  # Only this flow may clean up state while owning the SDK lock.
            # The SDK's first non-resource request is a proactive refresh. Its
            # generator is suspended at the refresh-response yield, not resource.
            if pending is not request:
                discovery = self._discover_for_refresh(None)
                absent = False
                try:
                    discovery_request = await anext(discovery)
                    while True:
                        discovery_response = yield discovery_request
                        discovery_request = await discovery.asend(discovery_response)
                except StopAsyncIteration:
                    pass
                except RefreshMetadataAbsent:
                    absent = True
                finally:
                    await discovery.aclose()
                refresh_attempted = True
                if absent:
                    await self._clear_tokens()
                    request.headers.pop("Authorization", None)
                    # This refresh request was NEVER sent. Resume the SDK with
                    # a failed refresh, then send its actual unauthenticated
                    # resource yield and feed back that resource's real response.
                    pending = await flow.asend(httpx.Response(400, request=pending))
                else:
                    pending = await self._refresh_token()

            while True:
                response = yield pending
                if (
                    pending is request
                    and response.status_code == 401
                    and not refresh_attempted
                    and self.context.can_refresh_token()
                ):
                    refresh_attempted = True
                    discovery = self._discover_for_refresh(response)
                    absent = False
                    try:
                        discovery_request = await anext(discovery)
                        while True:
                            discovery_response = yield discovery_request
                            discovery_request = await discovery.asend(discovery_response)
                    except StopAsyncIteration:
                        pass
                    except RefreshMetadataAbsent:
                        absent = True
                    finally:
                        await discovery.aclose()
                    if absent:
                        await self._clear_tokens()
                        request.headers.pop("Authorization", None)
                    else:
                        refresh_response = yield await self._refresh_token()
                        if await self._handle_refresh_response(refresh_response):
                            self._add_auth_header(request)
                            response = yield request
                        else:
                            request.headers.pop("Authorization", None)
                    # Here the SDK is still suspended at its RESOURCE yield:
                    # pass the original (or retried) 401, not a refresh response.
                step_up_discovery = (
                    hasattr(OAuthClientProvider, "_auth_flow")
                    and response.status_code == 403
                    and oauth_utils.extract_field_from_www_auth(response, "error")
                    == "insufficient_scope"
                    and self.context.oauth_metadata is None
                )
                if pending is request and (response.status_code == 401 or step_up_discovery):
                    # Full discovery must not reuse stale endpoint metadata.
                    self.context.protected_resource_metadata = None
                    self.context.oauth_metadata = None
                    self.context.auth_server_url = None
                    prm_urls[:] = [
                        str(httpx.URL(url))
                        for url in build_protected_resource_metadata_discovery_urls(
                            extract_resource_metadata_from_www_auth(response),
                            self.context.server_url,
                        )
                    ]
                    asm_urls.clear()
                else:
                    await self._guard_full_discovery(pending, response, prm_urls, asm_urls)
                # Failed refresh or a repeated 401 goes to the SDK's manual flow.
                pending = await flow.asend(response)
                if pending is request and not self.context.current_tokens:
                    request.headers.pop("Authorization", None)
        except StopAsyncIteration:
            return
        finally:
            if started and getattr(self, "_refresh_pending", False):
                # A canceled SDK-lock waiter never owned this refresh state.
                # Transport failure/cancellation can close the generator before
                # any refresh response arrives. Never invalidate disk on that.
                self._refresh_pending = False
                self.context.clear_tokens()
                self._initialized = False  # Reload the unchanged cache on a later call.
                request.headers.pop("Authorization", None)
            await flow.aclose()  # Release SDK lock on cancellation/error as well.


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
    return DiscoveringOAuthClientProvider(
        server_url=storage.server_url,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=signin.redirect,
        callback_handler=signin.callback,
    )


class _AuthorizedSession(ClientSession):
    """SDK session whose response timeout excludes the provider's human wait."""

    def __init__(self, read: Any, write: Any, signin: PastedSignIn, seconds: float) -> None:
        super().__init__(read, write)
        self._signin = signin
        self._response_seconds = seconds

    async def send_request(
        self,
        request: Any,
        result_type: Any,
        request_read_timeout_seconds: Any = None,
        metadata: Any = None,
        progress_callback: Any = None,
    ) -> Any:
        timeout = request_read_timeout_seconds
        seconds = (
            self._response_seconds
            if timeout is None
            else (timeout.total_seconds() if isinstance(timeout, timedelta) else float(timeout))
        )
        try:
            async with self._signin.response_budget(seconds) as budget:
                return await super().send_request(
                    request, result_type, metadata=metadata, progress_callback=progress_callback
                )
        except TimeoutError:
            if not budget.expired():
                raise
            # Preserve the SDK's response-timeout error contract.
            raise mcp_error(
                ErrorData(
                    code=MCP_TIMEOUT_CODE,
                    message=f"Timed out waiting for an MCP response after {seconds:g} seconds "
                    "(excluding the human sign-in wait).",
                )
            ) from None


class _SessionConnection:
    """One transport owner and its RPC tasks, never the external caller task.

    The owner enters/exits every transport/session/AnyIO task-group context.
    Calls run in its task group so close interrupts only session-owned work;
    caller futures receive an ordinary connection error instead of cancellation.
    """

    def __init__(self, client: AuthorizedHTTPClient) -> None:
        self.client = client
        self.ready: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self.ready.add_done_callback(self._observe_ready)
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.pending: set[asyncio.Future[Any]] = set()
        self.operations: set[asyncio.Task[Any]] = set()
        self.closing = False
        self.error: BaseException | None = None
        self.session: Any = None
        self.task = asyncio.create_task(self._run(), name="nooa-mcp-session-owner")
        self.task.add_done_callback(self._finished)

    @staticmethod
    def _observe_ready(future: asyncio.Future[Any]) -> None:
        if not future.cancelled():
            future.exception()  # Also observed by openers; avoid abandoned-future warnings.

    def _finished(self, task: asyncio.Task[Any]) -> None:
        self.closing = True
        self.client._sessions.discard(self)
        if not self.ready.done():
            self.ready.set_exception(RuntimeError("MCP session closed while opening"))

    async def _run(self) -> None:
        try:
            async with self.client._connect_transport() as session:
                self.session = session
                async with anyio.create_task_group() as group:
                    self.ready.set_result(self)
                    while True:
                        item = await self.queue.get()
                        if item is None:
                            group.cancel_scope.cancel()
                            break  # Normal exit preserves the transport's usual __aexit__ path.
                        method, args, kwargs, result = item
                        group.start_soon(self._invoke, method, args, kwargs, result)
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self.error = exc
            if not self.ready.done():
                self.ready.set_exception(
                    self.error or RuntimeError("MCP session closed while opening")
                )
        finally:
            self.closing = True
            for result in tuple(self.pending):
                if not result.done():
                    result.set_exception(self.error or RuntimeError("MCP session was closed"))

    async def _invoke(
        self, method: Any, args: Any, kwargs: Any, result: asyncio.Future[Any]
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.operations.add(task)
        try:
            if result.cancelled():
                return
            with anyio.CancelScope() as scope:

                def cancelled(future: asyncio.Future[Any]) -> None:
                    if future.cancelled():
                        scope.cancel()

                result.add_done_callback(cancelled)
                try:
                    value = await method(*args, **kwargs)
                    if not result.done():
                        result.set_result(value)
                except BaseException as exc:
                    if not result.done():
                        if not isinstance(exc, asyncio.CancelledError):
                            result.set_exception(exc)
                        elif not self.closing and anyio.current_effective_deadline() != float(
                            "-inf"
                        ):
                            # Independent RPC cancellation has no owner failure
                            # to await. Ancestor-scope/close cancellation instead
                            # leaves completion to _run after SDK groups drain,
                            # so their original HTTP/protocol error is retained.
                            result.set_exception(RuntimeError("MCP request was cancelled"))
                finally:
                    result.remove_done_callback(cancelled)
        finally:
            self.operations.discard(task)

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.session, name)
        if not inspect.iscoroutinefunction(method):
            return method

        async def call(*args: Any, **kwargs: Any) -> Any:
            if self.closing:
                raise RuntimeError("MCP session was closed")
            result = asyncio.get_running_loop().create_future()
            self.pending.add(result)
            self.queue.put_nowait((method, args, kwargs, result))
            try:
                return await result
            finally:
                self.pending.discard(result)

        return call

    def stop(self, *, cancel: bool = True) -> None:
        if not self.closing:
            self.closing = True
            if cancel:
                self.task.cancel()  # Exactly once: repeated closes must not interrupt cleanup.
            else:
                self.queue.put_nowait(None)

    async def drain(self) -> None:
        # A canceled *owner* is a normal close outcome. Cancellation of this
        # waiter is still external cancellation and must propagate, while the
        # shield lets the already-stopped owner finish its context cleanup.
        await asyncio.shield(asyncio.gather(self.task, return_exceptions=True))
        # A completed task's done callback may still be queued after an RPC
        # waiter woke up. A successful drain must remove its owner synchronously.
        self.client._sessions.discard(self)


class AuthorizedHTTPClient(MCPBaseClient):
    """A streamable-HTTP or SSE MCP client whose requests carry the provider's tokens."""

    def __init__(
        self,
        url: str,
        *,
        provider: OAuthClientProvider,
        signin: PastedSignIn,
        headers: dict[str, str] | None = None,
        transport: Literal["sse", "streamable-http"] = "streamable-http",
        tool_call_timeout: timedelta = timedelta(seconds=60),
    ) -> None:
        super().__init__(tool_call_timeout=tool_call_timeout)
        self._url = url
        self._headers = dict(headers or {})
        self._transport = transport
        self._provider = provider
        self._signin = signin
        self._closed = False
        self._sessions: set[_SessionConnection] = set()

    async def aclose(self) -> None:
        """Disable this provider and cancel/drain sessions before forgetting its cache."""
        self._closed = True
        self._signin.cancel()
        connections = list(self._sessions)
        for connection in connections:
            connection.stop()
        current = asyncio.current_task()
        await asyncio.gather(
            *(
                connection.drain()
                for connection in connections
                if current is not connection.task and current not in connection.operations
            )
        )

    @property
    def transport(self) -> Literal["sse", "stdio", "streamable-http"]:
        return self._transport

    @property
    def server_config(self) -> dict[str, Any]:
        return {"url": self._url, "headers": self._headers, "transport": self._transport}

    @asynccontextmanager
    async def connect_to_server(self) -> AsyncGenerator[ClientSession, None]:
        """Yield a session facade whose async calls run in a dedicated owner group.

        Transport/SDK contexts enter and exit on the owner task. External
        callers receive connection errors on close, never task cancellation.
        """
        if self._closed:
            raise RuntimeError(
                "MCP authentication was forgotten; connect again for a fresh sign-in"
            )
        connection = _SessionConnection(self)
        self._sessions.add(connection)
        failed = False
        try:
            session = await asyncio.shield(connection.ready)
            yield session
        except BaseException:
            failed = True
            raise
        finally:
            connection.stop(cancel=failed)
            await connection.drain()
            if not failed and connection.error is not None:
                raise connection.error

    @asynccontextmanager
    async def _connect_transport(self) -> AsyncGenerator[ClientSession, None]:
        seconds = self.tool_call_timeout.total_seconds()
        if self._transport == "sse":
            async with (
                sse_client(
                    self._url,
                    headers=self._headers or None,
                    sse_read_timeout=seconds,
                    auth=self._provider,
                ) as (read, write),
                _AuthorizedSession(read, write, self._signin, seconds) as session,
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
            _AuthorizedSession(streams[0], streams[1], self._signin, seconds) as session,
        ):
            await session.initialize()
            yield session
