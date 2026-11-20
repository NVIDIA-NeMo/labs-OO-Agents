# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict real-SDK binding and HTTP ownership for the three native families.

This module neither sends during construction nor owns retries. Borrowed SDKs
are immutable binding evidence, not a way to override the declared route.
"""

import inspect
from urllib.parse import urlsplit, urlunsplit

import anthropic
import httpx
import openai

from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.http_config import HttpConfig
from nooa.unifiedllm.route_identity import normalize_endpoint

_FAMILIES = frozenset({"openai-chat", "openai-responses", "anthropic-messages"})


def _config(message):
    return LLMFailure(FailureKind.CONFIG, message)


def _endpoint(family, value):
    value = normalize_endpoint(value)
    if value is None:
        raise _config("Native endpoint must be an explicit HTTP(S) URL")
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise _config("Native endpoint cannot contain URL auth, query or fragment")
    # The Anthropic SDK itself appends /v1/messages. Accept conventional roots
    # and version roots equivalently; never produce /v1/v1/messages.
    if family == "anthropic-messages" and parts.path.endswith("/v1"):
        value = urlunsplit(parts._replace(path=parts.path[:-3]))
    return value


def _resource(sdk, family):
    if family == "openai-chat":
        return sdk.chat.completions
    if family == "openai-responses":
        return sdk.responses
    if family == "anthropic-messages":
        return sdk.messages
    raise _config("Unsupported native API family")


def _signature_guard(sdk, family):
    required = {
        "openai-chat": {"messages", "model", "response_format", "max_completion_tokens"},
        "openai-responses": {"input", "model", "text", "reasoning", "max_output_tokens"},
        "anthropic-messages": {"messages", "model", "max_tokens", "thinking", "output_config"},
    }[family]
    if not required <= inspect.signature(_resource(sdk, family).create).parameters.keys():
        raise _config("Installed native SDK lacks required create parameters; upgrade its core SDK")


def _strict_clone(sdk, family, endpoint, key, mode, timeout):
    expected = (
        (anthropic.AsyncAnthropic if mode == "async" else anthropic.Anthropic)
        if family == "anthropic-messages"
        else (openai.AsyncOpenAI if mode == "async" else openai.OpenAI)
    )
    if type(sdk) is not expected:
        raise _config("Borrowed native SDK has an unsupported family or sync/async mode")
    # Compare the SDK's effective root without version stripping: a borrowed
    # Anthropic /v1 root really sends /v1/v1/messages and cannot be repaired by
    # mutating the caller's SDK. Declared version roots are normalized earlier.
    if normalize_endpoint(str(sdk.base_url)) != endpoint or sdk.api_key != key:
        raise _config("Borrowed native SDK route/auth differs from the declared resolved route")
    # A nonempty custom query or identity-bearing default header is significant
    # binding evidence. It must be declared, rather than silently inherited.
    _signature_guard(sdk, family)
    clone = sdk.with_options(max_retries=0, timeout=timeout)
    if clone is sdk or clone.max_retries != 0 or clone._client is not sdk._client:
        raise _config("Native SDK cannot make a retry-zero shared-transport view")
    return clone


class _NativeTransport:
    """One root with separate sync/async SDKs and exactly-once owned HTTP close.

    ``client`` can be one real SDK or an explicit {'sync': sdk, 'async': sdk}
    pair. A single borrowed mode does not manufacture/bridge the other mode.
    Caller SDKs and their HTTP clients are never closed, including on failure.
    """

    def __init__(
        self,
        family,
        endpoint,
        key,
        *,
        client=None,
        http_config=None,
        headers=None,
        query=None,
        organization=None,
        project=None,
    ):
        if family not in _FAMILIES:
            raise _config("Unsupported native API family")
        self.family = family
        self.endpoint = _endpoint(family, endpoint)
        if not isinstance(key, str) or not key.strip():
            raise _config("Native ordinary auth requires an explicit resolved API key")
        self.http_config = http_config or HttpConfig()
        if type(self.http_config) is not HttpConfig:
            raise _config("http_config must be HttpConfig")
        self._owned_sync = self._owned_async = None
        self._sync = self._async = None
        self._closed = False
        self._headers = dict(headers or {})
        self._query = dict(query or {})
        self._organization, self._project = organization, project
        timeout = self.http_config.to_httpx_timeout()
        self._timeout = timeout
        self._key = key
        if client is not None:
            if isinstance(client, dict):
                if not client or client.keys() - {"sync", "async"}:
                    raise _config("Borrowed SDK pair requires only sync/async keys")
                clients = client
            elif type(client) in {openai.OpenAI, anthropic.Anthropic}:
                clients = {"sync": client}
            elif type(client) in {openai.AsyncOpenAI, anthropic.AsyncAnthropic}:
                clients = {"async": client}
            else:
                raise _config("Borrowed client must be an ordinary real native SDK or mode pair")
            for mode, sdk in clients.items():
                clone = _strict_clone(sdk, family, self.endpoint, key, mode, timeout)
                # SDK public defaults include generated auth/content-type fields;
                # compare declared extension maps, query, org/project separately.
                if dict(sdk.default_query) != self._query:
                    raise _config("Borrowed native SDK query differs from declared route")
                for name, value in self._headers.items():
                    if sdk.default_headers.get(name) != value:
                        raise _config("Borrowed native SDK headers differ from declared route")
                custom = dict(sdk._custom_headers)
                if custom != self._headers:
                    raise _config("Borrowed native SDK custom headers must be declared exactly")
                if family != "anthropic-messages" and (
                    sdk.organization != organization or sdk.project != project
                ):
                    raise _config("Borrowed native SDK organization/project differs from route")
                if mode == "sync":
                    self._sync = clone
                else:
                    self._async = clone
        self._borrowed = client is not None

    def _initialize(self):
        """Allocate only after the existing pool owns a load lease.

        A partial failure deliberately leaves roots reachable by that pool.
        Its one retained, shielded disposal task handles all cleanup and joins.
        """
        if self._borrowed or self._sync is not None:
            return
        if self._closed:
            raise _config("Native transport is closed")
        self._owned_sync = httpx.Client(
            verify=True, limits=self.http_config.to_httpx_limits(), timeout=self._timeout
        )
        self._sync = self._construct("sync", self._owned_sync)
        self._owned_async = httpx.AsyncClient(
            verify=True, limits=self.http_config.to_httpx_limits(), timeout=self._timeout
        )
        self._async = self._construct("async", self._owned_async)

    def _construct(self, mode, http):
        cls = (
            (anthropic.AsyncAnthropic if mode == "async" else anthropic.Anthropic)
            if self.family == "anthropic-messages"
            else (openai.AsyncOpenAI if mode == "async" else openai.OpenAI)
        )
        kwargs = {
            "api_key": self._key,
            "base_url": self.endpoint,
            "max_retries": 0,
            "http_client": http,
            "timeout": self._timeout,
            "default_headers": self._headers,
            "default_query": self._query,
        }
        if self.family != "anthropic-messages":
            kwargs.update(organization=self._organization, project=self._project)
        sdk = cls(**kwargs)
        _signature_guard(sdk, self.family)
        return sdk

    def _sdk(self, mode, timeout=None):
        if self._closed:
            raise _config("Native transport is closed")
        sdk = self._sync if mode == "sync" else self._async
        if sdk is None:
            raise _config("Borrowed native SDK does not supply this execution mode")
        # SDK clones are cheap views; only the HTTP root owns close.
        return sdk if timeout is None else sdk.with_options(max_retries=0, timeout=timeout)

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._owned_sync is not None:
                self._owned_sync.close()
        finally:
            if self._owned_async is not None:
                await self._owned_async.aclose()
