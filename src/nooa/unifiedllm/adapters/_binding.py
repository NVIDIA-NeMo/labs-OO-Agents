# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private, same-route settings views used by the public loader."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._snapshots import snapshot_options

_UNSET = object()
# These select routing, transport, identity or declarations, not model settings.
_FIXED = frozenset(
    {
        "model",
        "api_base",
        "base_url",
        "api_key",
        "custom_llm_provider",
        "provider",
        "api_style",
        "adapter",
        "client",
        "http_config",
        "retry_config",
        "media_inputs",
        "media_capabilities",
        "context_window",
        "reasoning_levels",
        "reasoning_default",
        "cache_breakpoint",
        "session_affinity",
        "prompt_cache_key",
        "messages",
        "input",
        "api_version",
        "api_type",
        "deployment_id",
        "azure_endpoint",
        "azure_deployment",
        "base_model",
    }
)


# Normalize header/SDK spelling aliases only for validation, never for sending.
def _identity_key(key):
    return str(key).casefold().replace("-", "_")


_IDENTITY = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "auth",
        "host",
        "proxy",
        "proxies",
        "organization",
        "organisation",
        "org",
        "org_id",
        "organization_id",
        "project",
        "project_id",
        "api_credentials",
        "api_secret",
        "token",
        "openai_organization",
        "openai_project",
        "api_token",
        "access_token",
        "azure_ad_token",
        "azure_ad_token_provider",
        "credentials",
        "credential",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_region_name",
        "vertex_project",
        "vertex_location",
        "vertex_credentials",
        "api_host",
        "endpoint",
        "url",
        "http_client",
        "async_client",
        "default_query",
        "extra_query",
        "query",
        "params",
        "cookies",
    }
)
_HEADERS = frozenset({"headers", "extra_headers", "default_headers"})
_IDENTITY_HEADERS = (
    _IDENTITY
    | _FIXED
    | {
        "x_api_key",
        "x_auth_token",
        "x_access_token",
        "x_organization_id",
        "x_project_id",
        "x_org_id",
        "x_api_token",
        "x_goog_api_key",
        "x_goog_user_project",
        "x_amz_security_token",
        "x_amz_credential",
        "x_forwarded_host",
        "forwarded",
        "cookie",
        "set_cookie",
    }
)


def _check_settings_identity(options, *, headers=False, path="settings"):
    """Reject identity/transport carriers, including nested extra-body aliases.

    Header maps must be inspectable mappings; opaque auth/query/cookie carriers
    are unsupported for a same-route view. Ordinary provider model blocks are
    not recursively interpreted as routing (e.g. a schema property named host).
    """
    if not isinstance(options, Mapping):
        raise ValueError(f"Settings binding cannot change route/identity: unsupported {path}")
    forbidden = _IDENTITY_HEADERS
    for key, value in options.items():
        name = _identity_key(key)
        if name in forbidden:
            raise ValueError(f"Settings binding cannot change route/identity: {path}.{key}")
        if not headers and name in _HEADERS:
            _check_settings_identity(value, headers=True, path=f"{path}.{key}")
        elif not headers and name == "extra_body" and value is not None:
            _check_settings_identity(value, path=f"{path}.{key}")
        elif headers and not isinstance(value, (str, type(None))):
            raise ValueError(
                f"Settings binding cannot change route/identity: unsupported {path}.{key}"
            )


@dataclass(frozen=True)
class _List:
    items: tuple[Any, ...]


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _List(tuple(_freeze(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(value)
    return value  # SDK clients, transports and hooks are opaque, never copied.


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, _List):
        return [_thaw(item) for item in value.items]
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    return value


@dataclass(frozen=True)
class _Settings:
    model: str
    reasoning_default: str | None
    config: Mapping[str, Any]
    reasoning_levels: Any
    reasoning_level: str | None
    structured_output_mode: Any


def _bind_adapter(
    adapter,
    *,
    temperature=_UNSET,
    top_p=_UNSET,
    reasoning_level=_UNSET,
    max_output_tokens=_UNSET,
    **native,
):
    """Snapshot a built-in adapter without constructing clients or consulting routes.

    UNSET inherits; None is an explicit override, not omission: sampling/cap None
    clears the inherited value, reasoning None disables the selected level (but
    retains native defaults), and structured_output_mode=None resets to generic
    output. Native blocks replace whole blocks, using existing precedence/conflict
    checks. No arbitrary adapter/client subclass can be derived safely here.
    """
    from ._native import _BUILTINS

    if type(adapter) in _BUILTINS:
        if "send" in adapter.__dict__ or "send_sync" in adapter.__dict__:
            raise TypeError("Cannot derive native settings with overridden sends")
        from ._native_settings import _derive_native

        options = dict(native)
        for key, value in (
            ("temperature", temperature),
            ("top_p", top_p),
            ("reasoning_level", reasoning_level),
            ("max_output_tokens", max_output_tokens),
        ):
            if value is not _UNSET:
                options[key] = value
        view = copy.copy(adapter)
        view._settings = _derive_native(adapter._settings, options)
        return view

    from nooa.unifiedllm.unifiedllm import CompletionClient, ResponsesClient

    from .litellm_chat import LiteLLMChatAdapter
    from .litellm_responses import LiteLLMResponsesAdapter

    pairs = {LiteLLMChatAdapter: CompletionClient, LiteLLMResponsesAdapter: ResponsesClient}
    if type(adapter) not in pairs or type(adapter._client) is not pairs[type(adapter)]:
        raise TypeError("Settings derivation supports only built-in Chat/Responses adapters")
    if adapter._legacy_options is not None:
        raise TypeError("Cannot derive settings from a legacy call-local adapter")
    _check_settings_identity(native)
    options = {key: value for key, value in native.items() if value is not _UNSET}
    for key, value in (
        ("temperature", temperature),
        ("top_p", top_p),
        ("reasoning_level", reasoning_level),
        ("max_output_tokens", max_output_tokens),
    ):
        if value is not _UNSET:
            options[key] = value
    # Audited shallow built-in view only: preserves opaque pool and instance hooks.
    client = _settings_client(adapter)
    client.config = _thaw(_freeze(client.config))
    client._reasoning_config = client._reasoning_config.model_copy(
        update={"levels": _thaw(_freeze(client._reasoning_config.levels))}
    )
    options = snapshot_options(options)
    selection = options.get("reasoning_level", client.reasoning_level)
    if selection is not None:
        _check_settings_identity(client._reasoning_config.settings(selection))
    config = client._prepare_call_config(options)
    client._validate_request_config(
        "messages" if type(client) is CompletionClient else "input", config
    )
    mode = client.structured_output_mode
    if type(client) is CompletionClient:
        mode = "anthropic" if client._native_output_for_call(options, config) else None
    elif "structured_output_mode" in options:
        raise ValueError("Responses settings do not support structured_output_mode")
    caps = adapter.capabilities
    from nooa.unifiedllm.limits import REPLY_CAP_KEYS

    for key in REPLY_CAP_KEYS & config.keys():
        cap = config[key]
        if cap is not None and (type(cap) is not int or cap <= 0):
            raise ValueError("Reply ceiling must be a positive integer or None")
        if cap is not None and caps.max_output_tokens is not None and cap > caps.max_output_tokens:
            raise ValueError("Reply ceiling exceeds this route's known limit")
    # Validate the selected wire settings above, but freeze RAW defaults and the
    # selection separately. Baking the patch into defaults leaks disjoint fields
    # from an old level into a later rebind or transitional request override.
    from nooa.unifiedllm.reasoning import ReasoningConfig, apply_reasoning_level

    raw_options = {
        key: value
        for key, value in options.items()
        if key not in {"reasoning_level", "structured_output_mode"}
    }
    raw_config = apply_reasoning_level(
        ReasoningConfig(), client.model, client.config, raw_options, None
    )
    settings = _Settings(
        client.model,
        client._reasoning_config.default,
        _freeze(raw_config),
        _freeze(client._reasoning_config.levels),
        selection,
        mode,
    )
    client.config = _thaw(settings.config)
    client.reasoning_level = settings.reasoning_level
    client.structured_output_mode = mode
    from ._calibration import _route_calibration

    record = _route_calibration(adapter)
    view = copy.copy(adapter)
    view._route_calibration_record = record
    view._client = client
    client._adapter = view  # Completion's transport hook dispatches through this.
    view._bound_settings = settings
    view._bound_capabilities = caps
    return view


def _settings_client(adapter):
    """Resolve from the frozen record, never a mutable derived client's defaults."""
    client = copy.copy(adapter._client)
    if hasattr(adapter, "_bound_settings"):
        settings = adapter._bound_settings
        client.model = settings.model
        client.config = _thaw(settings.config)
        client._reasoning_config = client._reasoning_config.model_copy(
            update={
                "levels": _thaw(settings.reasoning_levels),
                "default": settings.reasoning_default,
            }
        )
        client.reasoning_level = settings.reasoning_level
        client.structured_output_mode = settings.structured_output_mode
    return client


def _bound_call_config(adapter, request, options):
    """Fresh wire containers per attempt; the frozen snapshot survives SDK mutation."""
    client = _settings_client(adapter)
    selection = options.get("reasoning_level", client.reasoning_level)
    if selection is not None:
        _check_settings_identity(client._reasoning_config.settings(selection))
    from nooa.unifiedllm._call_identity import current_call_identity

    # Preserve LiteLLM's configured key (including explicit None suppression)
    # and its existing case-insensitive affinity-header precedence.
    config = snapshot_options(client._prepare_call_config(options))
    identity = current_call_identity()
    if identity is not None:
        config.setdefault("prompt_cache_key", identity)
    for layer in (config, config.get("extra_body")):
        if not isinstance(layer, dict):
            continue
        layer.pop("tools", None)
        layer.pop("tool_choice", None)
        if request.output_type is not None:
            for key in ("response_format", "text_format", "output_format"):
                layer.pop(key, None)
            layer.pop("text", None)  # LiteLLM ignores text_format when text is present.
            if isinstance(layer.get("output_config"), dict):
                layer["output_config"].pop("format", None)
    config["tool_choice"] = request.tool_choice
    config["max_retries"] = config["num_retries"] = 0
    return config
