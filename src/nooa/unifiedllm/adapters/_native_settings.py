# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite, allocation-free native settings resolution (no provider discovery)."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from nooa.media_capabilities import resolve_media_capabilities
from nooa.media_inputs import validate_media_inputs
from nooa.unifiedllm.capability_settings import (
    CAPABILITY_KEYS,
    resolve_instruction_role,
    resolve_structured_output_format,
    validate_developer_role,
)
from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.http_config import HttpConfig
from nooa.unifiedllm.limits import REPLY_CAP_KEYS
from nooa.unifiedllm.reasoning import ReasoningConfig, apply_reasoning_level
from nooa.unifiedllm.route_identity import no_endpoint_override

from ._binding import _UNSET, _check_settings_identity, _freeze, _thaw
from ._native_transport import _FAMILIES, _endpoint
from ._snapshots import snapshot_options
from .base import Capabilities

_COMMON = {"temperature", "top_p", "extra_body", "extra_headers", "extra_query", "timeout"}
_KNOBS = {
    "openai-chat": _COMMON
    | {
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stop",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "n",
        "store",
        "metadata",
        "user",
        "reasoning_effort",
        "service_tier",
        "parallel_tool_calls",
        "prediction",
        "verbosity",
        "prompt_cache_key",
        "prompt_cache_retention",
        "response_format",
    },
    "openai-responses": _COMMON
    | {
        "store",
        "include",
        "metadata",
        "reasoning",
        "text",
        "service_tier",
        "parallel_tool_calls",
        "truncation",
        "user",
        "prompt_cache_key",
        "prompt_cache_retention",
        "max_tool_calls",
        "safety_identifier",
    },
    "anthropic-messages": _COMMON
    | {
        "top_k",
        "stop_sequences",
        "metadata",
        "thinking",
        "output_config",
        "service_tier",
        "cache_control",
        "container",
    },
}
_ROUTE = {
    "model",
    "model_name",
    "adapter",
    "api_base",
    "base_url",
    "api_key",
    "api_key_env",
    "client",
    "http_config",
    "retry_config",
    "context_window",
    "media_inputs",
    "reasoning_levels",
    "reasoning_default",
    "reasoning_level",
    "cache_breakpoint",
    "host_profile",
    "headers",
    "default_headers",
    "default_query",
    "organization",
    "project",
    "structured_output_mode",
    "client_type",
    "tools_supported",
    "media_capabilities",
    *CAPABILITY_KEYS,
}
_FACTS = {
    "context_window",
    "media_inputs",
    "media_capabilities",
    "reasoning_levels",
    "reasoning_default",
    "reasoning_level",
    "cache_breakpoint",
    "host_profile",
    "structured_output_mode",
    "tools_supported",
    *CAPABILITY_KEYS,
}


# The request interface (nooa.request_interface) whose media defaults a family uses.
_MEDIA_INTERFACE = {
    "openai-chat": "chat",
    "openai-responses": "responses",
    "anthropic-messages": "anthropic",
}


def _config(message):
    return LLMFailure(FailureKind.CONFIG, message)


def _positive(value, name):
    if value is not None and (type(value) is not int or value <= 0):
        raise _config(f"{name} must be a positive integer or None")
    return value


@dataclass(frozen=True)
class _NativeSettings:
    family: str
    model: str
    route: Mapping
    raw: Mapping
    selected: Mapping
    declaration: ReasoningConfig
    selection: str | None
    capabilities: Capabilities
    cache_breakpoint: str | None
    host_profile: Mapping
    replay_scope: str
    # Resolved at load from the entry's settings and detected developer role.
    output_format: str = "json_schema"
    instruction_role: str = "system"


def _compatibility_scope(family, model, route, profile, selected):
    """Route-only protected identity, not the pool's model-default/facts key.

    Reuse the existing keyed digest and weak SDK identities; no raw credential
    hashes, mutable calibration, sampling, reply caps or reasoning selections.
    Unprojectable identities isolate once at load (never a per-call registry).
    """
    from secrets import token_bytes

    from ._loader import _cache_key

    identity = {
        "base_url": route["endpoint"],
        "api_key": route["key"],
        "client": route["client"],
        "model": model,
        "headers": route["headers"],
        "default_query": route["query"],
        "organization": route["organization"],
        "project": route["project"],
        "host_profile": profile,
        "extra_headers": selected.get("extra_headers", {}),
    }
    digest = _cache_key(identity, {}, family)
    return "native-v1:" + family + ":" + (digest if digest is not None else token_bytes(32)).hex()


def _select(family, model, raw, declaration, selection, overrides=None):
    overrides = overrides or {}
    # Reject before reasoning/cap alias normalization can consume nested caps.
    recognized = (
        _ROUTE
        | REPLY_CAP_KEYS
        | set().union(*_KNOBS.values())
        | {"stream", "max_retries", "num_retries", "system", "text_format", "output_format"}
    )
    for layer in (raw, overrides):
        body = layer.get("extra_body", {})
        if isinstance(body, Mapping) and recognized & body.keys():
            raise _config("Recognized native controls must be top-level settings, not extra_body")
    try:
        params = apply_reasoning_level(declaration, model, raw, overrides, selection)
        # A valid top-level initial cache key is pinned identity, not a cheap
        # view knob. _derive_native checks new options before reaching here;
        # declaration patches are checked separately at load below.
        _check_settings_identity(
            {k: v for k, v in params.items() if k not in {"extra_query", "prompt_cache_key"}}
        )
    except (ValueError, TypeError) as exc:
        raise _config("Invalid native model settings") from exc
    # Query extensions are identity-bearing and fixed at load, never a bind knob.
    if set(params) - (_KNOBS[family] | REPLY_CAP_KEYS):
        raise _config(
            "Unsupported native model setting; use explicit extra_body for host extensions"
        )
    cap_keys = REPLY_CAP_KEYS & params.keys()
    cap = _positive(params.pop(next(iter(cap_keys))), "Reply ceiling") if cap_keys else None
    target = {
        "openai-chat": "max_completion_tokens",
        "openai-responses": "max_output_tokens",
        "anthropic-messages": "max_tokens",
    }[family]
    if cap is not None:
        params[target] = cap
    if family == "anthropic-messages" and cap is None:
        raise _config("Anthropic Messages requires an explicit positive reply cap")
    for key in ("temperature", "top_p", "timeout"):
        if key in params:
            value = params[key]
            if value is None:
                params.pop(key)
            elif (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or (key == "timeout" and value <= 0)
            ):
                raise _config(f"{key} must be a finite number")
    for key in ("frequency_penalty", "presence_penalty"):
        if key in params and (
            type(params[key]) not in (int, float) or not math.isfinite(params[key])
        ):
            raise _config(f"{key} must be a finite number")
    for key in ("seed", "top_k", "top_logprobs", "max_tool_calls"):
        if key in params and type(params[key]) is not int:
            raise _config(f"{key} must be an integer")
    for key in ("store", "logprobs", "parallel_tool_calls"):
        if key in params and type(params[key]) is not bool:
            raise _config(f"{key} must be bool")
    for key in (
        "metadata",
        "logit_bias",
        "prediction",
        "reasoning",
        "thinking",
        "text",
        "output_config",
        "cache_control",
        "extra_body",
        "extra_headers",
    ):
        if key in params and not isinstance(params[key], Mapping):
            raise _config(f"{key} must be a mapping")
    for key in (
        "reasoning_effort",
        "service_tier",
        "verbosity",
        "truncation",
        "user",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
    ):
        if key in params and not isinstance(params[key], str):
            raise _config(f"{key} must be a string")
    for key in ("include", "stop_sequences"):
        if key in params and (
            not isinstance(params[key], (list, tuple))
            or not all(isinstance(item, str) for item in params[key])
        ):
            raise _config(f"{key} must be a string sequence")
    if "stop" in params and not (
        isinstance(params["stop"], str)
        or isinstance(params["stop"], (list, tuple))
        and all(isinstance(item, str) for item in params["stop"])
    ):
        raise _config("stop must be a string or string sequence")
    if "container" in params and not isinstance(params["container"], (str, Mapping)):
        raise _config("container must be a string or mapping")
    if "n" in params and (type(params["n"]) is not int or params["n"] != 1):
        raise _config("Native Chat requires one response (n=1)")
    owned = {
        "messages",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "response_format",
        "text",
        "output_config",
        "stream",
        "n",
    }
    if owned & params.get("extra_body", {}).keys():
        raise _config("extra_body cannot override request-owned fields")
    # SDK extra_body merges at the wire top level. Recognized controls must go
    # through the ONE layered validator above, never a second bypass channel.
    if recognized & params.get("extra_body", {}).keys():
        raise _config("Recognized native controls must be top-level settings, not extra_body")
    if "response_format" in params or any(
        "format" in params.get(key, {}) for key in ("text", "output_config")
    ):
        raise _config("Structured format belongs to LLMRequest.output_type")
    _nested_controls(family, params, cap)
    # Streaming would require an internally owned accumulator, not an SDK iterator.
    return params


def _nested_controls(family, params, cap):
    """Validate native SDK-declared blocks; extensions belong in extra_body."""

    def enums(block, allowed):
        if block.keys() - allowed.keys():
            raise _config("Unsupported native nested setting")
        for key, values in allowed.items():
            if (
                key in block
                and block[key] is not None
                and (not isinstance(block[key], str) or block[key] not in values)
            ):
                raise _config("Invalid native nested setting")

    efforts = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
    if "reasoning_effort" in params and params["reasoning_effort"] not in efforts:
        raise _config("Invalid native reasoning effort")
    if "reasoning" in params:
        enums(
            params["reasoning"],
            {
                "effort": efforts,
                "summary": {"auto", "concise", "detailed"},
                "generate_summary": {"auto", "concise", "detailed"},
                "context": {"auto", "current_turn", "all_turns"},
                "mode": {"standard", "pro"},
            },
        )
    if "text" in params:
        enums(params["text"], {"verbosity": {"low", "medium", "high"}})
    if "output_config" in params:
        enums(params["output_config"], {"effort": {"low", "medium", "high", "xhigh", "max"}})
    if "thinking" in params:
        block = params["thinking"]
        kind = block.get("type")
        allowed = (
            {"type", "display", "budget_tokens"}
            if kind == "enabled"
            else {"type", "display"}
            if kind == "adaptive"
            else {"type"}
        )
        if (
            not isinstance(kind, str)
            or kind not in {"enabled", "disabled", "adaptive"}
            or block.keys() - allowed
        ):
            raise _config("Invalid native thinking block")
        if "display" in block and block["display"] not in (None, "summarized", "omitted"):
            raise _config("Invalid native thinking display")
        if kind == "enabled":
            budget = block.get("budget_tokens")
            if type(budget) is not int or budget < 1024 or cap is None or budget >= cap:
                raise _config("Native thinking budget must be >=1024 and below reply cap")
        if kind != "disabled" and (
            "top_k" in params or "top_p" in params or params.get("temperature", 1) != 1
        ):
            raise _config("Native thinking conflicts with sampling settings")
    if "prediction" in params:
        block = params["prediction"]
        content = block.get("content")
        if (
            block.keys() - {"type", "content"}
            or block.get("type") != "content"
            or not (
                isinstance(content, str)
                or isinstance(content, (list, tuple))
                and all(
                    isinstance(item, Mapping)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                    for item in content
                )
            )
        ):
            raise _config("Invalid native prediction block")
    if family == "anthropic-messages" and "temperature" in params and "top_p" in params:
        raise _config("Native Anthropic cannot combine temperature and top_p")
    for key, upper in (("temperature", 1 if family == "anthropic-messages" else 2), ("top_p", 1)):
        if key in params and not 0 <= params[key] <= upper:
            raise _config("Native sampling setting out of range")


def _canonical_aliases(family, layer):
    """Resolve synonyms within one layer; explicit layers replace inherited aliases."""
    layer = dict(layer)
    for canonical, aliases in (
        ("model", ("model", "model_name")),
        ("base_url", ("base_url", "api_base")),
        ("headers", ("headers", "default_headers")),
        ("default_query", ("default_query", "extra_query")),
    ):
        values = [layer[key] for key in aliases if key in layer]
        if len(values) > 1:
            equal = (
                _endpoint(family, values[0]) == _endpoint(family, values[1])
                if canonical == "base_url"
                else values[0] == values[1]
            )
            if not equal:
                raise _config(f"Conflicting native {canonical} aliases")
        for key in aliases:
            layer.pop(key, None)
        if values:
            layer[canonical] = values[0]
    return layer


def _resolve_native(family, name, config, overrides):
    """One detached registry snapshot; no SDK/HTTP allocation or model probing."""
    if family not in _FAMILIES:
        raise _config("Unsupported native API family")
    config, overrides = snapshot_options(config), snapshot_options(overrides)
    overrides = {
        k: v
        for k, v in overrides.items()
        if v is not _UNSET and not (k in {"api_base", "base_url"} and no_endpoint_override(v))
    }
    config = _canonical_aliases(family, config)
    overrides = _canonical_aliases(family, overrides)
    old_model = config.get("model", name)
    old_endpoint = config.get("base_url", config.get("api_base"))
    endpoint = overrides.get("base_url", overrides.get("api_base", old_endpoint))
    if endpoint is None:
        endpoint = (
            "https://api.anthropic.com"
            if family == "anthropic-messages"
            else "https://api.openai.com/v1"
        )
    endpoint = _endpoint(family, endpoint)
    model = overrides.get("model", old_model)
    if not isinstance(model, str) or not model.strip():
        raise _config("Native model must be an exact nonempty model ID")
    changed = (
        model != old_model
        or (old_endpoint is not None and _endpoint(family, old_endpoint) != endpoint)
        or (config.get("adapter", family) != family)
        or any(
            k in overrides and overrides[k] != config.get(k)
            for k in (
                "api_key",
                "api_key_env",
                "client",
                "organization",
                "project",
                "headers",
                "default_headers",
                "default_query",
            )
        )
    )
    if changed:
        for key in _FACTS - overrides.keys():
            config.pop(key, None)
    if "api_key_env" in overrides and "api_key" not in overrides:
        config.pop("api_key", None)
    params = {**config, **overrides}
    if params.get("adapter", family) != family:
        raise _config("Native adapter declaration contradicts selected family")
    # A declared legacy transport/control option must be translated or rejected,
    # including explicit False/None, not silently dropped as a guessed default.
    unknown = params.keys() - (
        _ROUTE | _KNOBS[family] | REPLY_CAP_KEYS | {"stream", "max_retries", "num_retries"}
    )
    if unknown:
        raise _config("Unsupported native load option(s): " + ", ".join(sorted(unknown)))
    if params.get("stream") is not None and params.get("stream") is not False:
        raise _config("Native streaming is not supported")
    for key in ("max_retries", "num_retries"):
        if key in params and (type(params[key]) is not int or params[key] != 0):
            raise _config("Native SDK retries must be zero; configure shared retry_config")
    if params.get("client_type") not in (None, "completion", "responses"):
        raise _config("Unsupported native authentication/API family")
    if params.get("structured_output_mode") not in (
        None,
        "anthropic" if family == "anthropic-messages" else None,
    ):
        raise _config("Unsupported native structured output mode")
    key = params.get("api_key")
    if "api_key" not in overrides and key is None:
        from nooa.unifiedllm.registry import resolve_api_key_from_config

        key = resolve_api_key_from_config(name, params)
    if not isinstance(key, str) or not key.strip():
        raise _config("Native ordinary auth requires api_key or a resolved api_key_env")
    http = params.get("http_config") or HttpConfig()
    if isinstance(http, Mapping):
        try:
            http = HttpConfig(**http)
        except (ValueError, TypeError) as exc:
            raise _config("Invalid native HttpConfig") from exc
    if type(http) is not HttpConfig:
        raise _config("Native http_config must be HttpConfig")
    try:
        declaration = ReasoningConfig(
            levels=params.get("reasoning_levels"), default=params.get("reasoning_default")
        )
        enabled = tuple(validate_media_inputs(params.get("media_inputs", [])))
        media_caps = resolve_media_capabilities(
            _MEDIA_INTERFACE[family], params.get("media_capabilities")
        )
    except (ValueError, TypeError) as exc:
        raise _config("Invalid native capability declaration") from exc
    supported = {"image", "pdf"} if family != "openai-chat" else {"image"}
    if set(enabled) - supported:
        raise _config("Declared media cannot be represented by this native family")
    for patch in (declaration.levels or {}).values():
        try:
            _check_settings_identity(patch)
        except (ValueError, TypeError) as exc:
            raise _config("Native reasoning patches cannot change route/identity") from exc
        _select(
            family,
            model,
            {k: v for k, v in params.items() if k in REPLY_CAP_KEYS},
            declaration,
            None,
            patch,
        )
    setting_keys = (_KNOBS[family] | REPLY_CAP_KEYS) - {"extra_query"}
    defaults = {k: v for k, v in config.items() if k in setting_keys}
    genuine = {k: v for k, v in overrides.items() if k in setting_keys}
    query = params.get("default_query", {})
    selection = params.get("reasoning_level")
    selected = _select(family, model, defaults, declaration, selection, genuine)
    # Keep unselected defaults, with genuine overrides replacing whole blocks/caps.
    raw = apply_reasoning_level(ReasoningConfig(), model, defaults, genuine, None)
    window = _positive(params.get("context_window"), "context_window")
    tools = params.get("tools_supported")
    if tools is not None and type(tools) is not bool:
        raise _config("tools_supported must be bool or None")
    developer = validate_developer_role(params.get("developer_role"))
    output_format = resolve_structured_output_format(params.get("structured_output_format"))
    instruction_role = resolve_instruction_role(params.get("instruction_role"), developer)
    if family == "anthropic-messages" and output_format == "json_object":
        raise _config("Anthropic Messages has no json_object structured output format")
    if family == "anthropic-messages" and instruction_role == "developer":
        raise _config("Anthropic Messages has no developer role; use instruction_role 'system'")
    caps = Capabilities(
        frozenset(enabled),
        None if declaration.levels is None else tuple(declaration.levels),
        tools,
        window,
        None,
        developer,
        None if media_caps.unrestricted else media_caps,
    )
    cache = params.get("cache_breakpoint")
    expected = "anthropic" if family == "anthropic-messages" else "openai"
    if family == "openai-chat" and cache is not None:
        raise _config("Native Chat has no explicit cache-boundary extension")
    if cache not in (None, expected):
        raise _config("Cache boundary declaration does not match native family")
    profile = params.get("host_profile") or {}
    if not isinstance(profile, Mapping) or profile.keys() - {"cost_header", "tool_id_pattern"}:
        raise _config("Native host_profile permits only finite private data knobs")
    cost_header = profile.get("cost_header")
    if cost_header is not None and (
        not isinstance(cost_header, str) or cost_header not in {"x-response-cost", "x-request-cost"}
    ):
        raise _config("Native cost header is not explicitly allowlisted")
    if profile.get("tool_id_pattern") not in (None, "alphanumeric"):
        raise _config("Unsupported native tool ID convention")
    headers = params.get("headers", params.get("default_headers", {}))
    if not isinstance(headers, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise _config("Native headers require a string map")
    try:
        _check_settings_identity(headers, headers=True)
        _check_settings_identity(query)
    except (ValueError, TypeError) as exc:
        raise _config("Native headers/query cannot override ordinary auth or route") from exc
    if not isinstance(query, Mapping) or not all(
        isinstance(k, str) and isinstance(v, (str, int, float, bool, type(None)))
        for k, v in query.items()
    ):
        raise _config("Native default_query requires a scalar string-keyed mapping")
    if family == "anthropic-messages" and any(
        params.get(k) is not None for k in ("organization", "project")
    ):
        raise _config("Anthropic ordinary auth does not support OpenAI organization/project")
    route = {
        "endpoint": endpoint,
        "key": key,
        "client": params.get("client"),
        "http_config": http,
        "headers": dict(headers),
        "query": dict(query),
        "organization": params.get("organization"),
        "project": params.get("project"),
    }
    return _NativeSettings(
        family,
        model,
        _freeze(route),
        _freeze(raw),
        _freeze(selected),
        declaration,
        selection,
        caps,
        cache,
        _freeze(profile),
        _compatibility_scope(family, model, route, dict(profile), selected),
        output_format,
        instruction_role,
    ), params.get("retry_config")


def _derive_native(settings, options):
    """Cheap immutable view; reject route carriers before resolving model blocks."""
    options = {k: v for k, v in snapshot_options(options).items() if v is not _UNSET}
    try:
        _check_settings_identity(options)
    except (ValueError, TypeError) as exc:
        raise _config("Native settings cannot change route/identity") from exc
    selection = options.pop("reasoning_level", settings.selection)
    selected = _select(
        settings.family,
        settings.model,
        _thaw(settings.raw),
        settings.declaration,
        selection,
        options,
    )
    # Freeze raw defaults separately so subsequent level changes replace whole blocks.
    raw = apply_reasoning_level(
        ReasoningConfig(), settings.model, _thaw(settings.raw), options, None
    )
    from dataclasses import replace

    scope = settings.replay_scope
    if selected.get("extra_headers", {}) != _thaw(settings.selected).get("extra_headers", {}):
        scope = _compatibility_scope(
            settings.family,
            settings.model,
            _thaw(settings.route),
            _thaw(settings.host_profile),
            selected,
        )
    return replace(
        settings,
        raw=_freeze(raw),
        selected=_freeze(selected),
        selection=selection,
        replay_scope=scope,
    )
