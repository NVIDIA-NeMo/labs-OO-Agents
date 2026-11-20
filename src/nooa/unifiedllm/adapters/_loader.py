# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Built-in public loading, credential-safe weak pool cache and view leases.

Cross-load caching requires an explicit HTTP(S) endpoint and credential. Opaque
identity/native carriers that cannot be safely projected are isolated. Registry
facts/raw native defaults are part of the protected key: reloads cannot borrow
stale facts. Sampling, selected reasoning, cap and wire timeout are view settings.
Caller-supplied SDKs are never closed; the private HTTP pool is still ours.
"""

import asyncio
import copy
import hashlib
import hmac
import inspect
import json
import secrets
import weakref
from threading import RLock
from urllib.parse import urlsplit

from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.limits import REPLY_CAP_KEYS
from nooa.unifiedllm.llm import LLM, _snapshot_capabilities
from nooa.unifiedllm.retry_config import RetryConfig
from nooa.unifiedllm.route_identity import normalize_endpoint

from ._binding import _bind_adapter, _freeze, _Settings, _thaw
from ._calibration import _sdk_identity
from ._native import _BUILTINS
from ._snapshots import snapshot_options

_KEY = secrets.token_bytes(32)
_pools = weakref.WeakValueDictionary()
_lock = RLock()
_VIEW_SETTINGS = frozenset(
    {
        "temperature",
        "top_p",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_level",
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "timeout",
    }
)


def _cache_key(params, config, selected):
    endpoints = {normalize_endpoint(params[k]) for k in ("api_base", "base_url") if params.get(k)}
    if len(endpoints) != 1 or None in endpoints:
        return None
    endpoint = next(iter(endpoints))
    if urlsplit(endpoint).scheme not in {"http", "https"}:
        return None
    credential = params.get("api_key")
    if params.get("client") is None and (not isinstance(credential, str) or not credential.strip()):
        return None
    budget = 1000

    def project(value, depth=0):
        nonlocal budget
        budget -= 1
        if budget < 0 or depth > 8:
            raise ValueError
        if type(value) in {str, int, float, bool, type(None)}:
            if type(value) is str and len(value) > 8192:
                raise ValueError
            return value
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError
            return {key: project(item, depth + 1) for key, item in sorted(value.items())}
        if type(value) in {list, tuple}:
            return [project(item, depth + 1) for item in value]
        # Only explicit supported SDK/HttpConfig identity is admitted below.
        raise ValueError

    route = snapshot_options(params)
    for alias in ("api_base", "base_url"):
        route.pop(alias, None)
    route["endpoint"] = endpoint
    sdk = route.pop("client", None)
    if type(sdk) is dict and sdk and not (sdk.keys() - {"sync", "async"}):
        sdk_identity = {mode: _sdk_identity(value) for mode, value in sdk.items()}
        if any(identity is False for identity in sdk_identity.values()):
            return None
    else:
        sdk_identity = _sdk_identity(sdk)
    if sdk_identity is False:
        return None
    route["sdk_identity"] = sdk_identity
    http = route.get("http_config")
    if http is not None:
        from nooa.unifiedllm.http_config import HttpConfig

        if type(http) is not HttpConfig:
            return None
        route["http_config"] = http.model_dump()
    # Shell retry policy is independent of transport and per loaded view.
    route.pop("retry_config", None)
    facts = {key: config[key] for key in ("context_window",) if key in config}
    try:
        data = json.dumps(project([selected, route, facts]), sort_keys=True).encode()
    except (ValueError, TypeError):
        return None
    return hmac.digest(_KEY, data, hashlib.sha256)


class _Pool:
    """Views, logical calls and dispatched provider tasks own separate leases."""

    def __init__(self, client):
        self.client = client
        self.leases = 0
        self.retired = False
        self._disposal = None

    def acquire(self):
        with _lock:
            if self.retired:
                raise LLMFailure(FailureKind.CONFIG, "LLM pool is closed")
            self.leases += 1

    def release(self):
        with _lock:
            if self.leases <= 0:
                raise RuntimeError("LLM pool lease released twice")
            self.leases -= 1
            if self.leases:
                return
            self.retired = True
            # Publish disposal atomically with retirement. No caller owns its
            # cancellation; later aclose calls can always join this same task.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.client.aclose())
            else:
                self._disposal = loop.create_task(self.client.aclose())
                self._disposal.add_done_callback(_consume_disposal)

    async def join_disposal(self):
        with _lock:
            task = self._disposal
        if task is not None:
            if task.done():
                task.result()  # Also safe after asyncio.run closed its loop.
            else:
                await asyncio.shield(task)

    async def release_async(self):
        self.release()
        await self.join_disposal()


def _consume_disposal(task):
    # The retained task preserves the failure for joiners, without an unhandled
    # background exception when synchronous close has no immediate joiner.
    try:
        task.result()
    except BaseException:
        pass


class _LegacySettingsFailure(LLMFailure, ValueError):
    """Private public-load compatibility, never a raw factory error wrapper."""

    def __init__(self, message):
        super().__init__(FailureKind.CONFIG, message)


def _legacy_declaration(levels, default=None):
    from nooa.unifiedllm.reasoning import ReasoningConfig

    try:
        return ReasoningConfig(levels=levels, default=default)
    except (TypeError, ValueError):
        raise _LegacySettingsFailure(
            "Invalid reasoning declaration; declare valid settings"
        ) from None


def _validate_legacy_settings(adapter, options, facts):
    """Pure frozen-settings validation; no transport or capability getter in catches.

    This precedes binding rather than wrapping it: unrelated hooks, properties,
    constructor errors and control exceptions retain their original identity.
    Fixed remediation messages never classify or echo exception text.
    """
    from nooa.unifiedllm.reasoning import apply_reasoning_level

    from ._binding import _UNSET, _check_settings_identity

    settings = adapter._bound_settings
    model = adapter._client.model  # Evaluate outside any validation catch.
    declaration = _legacy_declaration(_thaw(settings.reasoning_levels))
    options = {key: value for key, value in options.items() if value is not _UNSET}
    selection = options.get("reasoning_level", settings.reasoning_level)
    if selection is not None:
        try:
            patch = declaration.settings(selection)
        except (TypeError, ValueError):
            raise _LegacySettingsFailure(
                "Invalid or unknown reasoning selection; selection may be not supported"
            ) from None
        try:
            _check_settings_identity(patch)
        except (TypeError, ValueError):
            raise _LegacySettingsFailure(
                "Invalid reasoning declaration; use model settings only"
            ) from None
    try:
        effective = apply_reasoning_level(
            declaration, model, _thaw(settings.config), options, settings.reasoning_level
        )
    except (TypeError, ValueError):
        raise _LegacySettingsFailure(
            "Invalid settings; use one reply limit and avoid reasoning that conflicts with explicit settings"
        ) from None
    for key in REPLY_CAP_KEYS & effective.keys():
        cap = effective[key]
        if cap is not None and (type(cap) is not int or cap <= 0):
            raise _LegacySettingsFailure("Reply ceiling must be a positive integer or None")
        if cap is not None and (
            (facts.max_output_tokens is not None and cap > facts.max_output_tokens)
            or (facts.context_window is not None and cap >= facts.context_window)
        ):
            raise _LegacySettingsFailure("Reply ceiling exceeds this route's known limit")


def _load_public(name, params, config, selected, media_policy, supplied, require=None):
    from nooa.unifiedllm.registry import _construct_llm_client

    params = snapshot_options(params)
    settings = {key: params.pop(key) for key in tuple(params) if key in _VIEW_SETTINGS}
    extra = params.get("extra_body")
    if isinstance(extra, dict) and REPLY_CAP_KEYS & extra.keys():
        # Native cap spellings are view settings too. Keep unrelated native
        # defaults in the route key and in the frozen full extra-body layer.
        settings["extra_body"] = snapshot_options(extra)
        raw_extra = {key: value for key, value in extra.items() if key not in REPLY_CAP_KEYS}
        if raw_extra:
            params["extra_body"] = raw_extra
        else:
            params.pop("extra_body")
    # Pure declarations can fail before allocation; selected-level validation
    # stays after construction, preserving the existing failed-load lifecycle.
    _legacy_declaration(params.get("reasoning_levels"), params.get("reasoning_default"))
    retry = params.get("retry_config")
    # Do not let constructor SDK retry settings become public attempts.
    params["max_retries"] = params["num_retries"] = 0
    key = _cache_key(params, config, selected)
    with _lock:
        _legacy_behavior()  # Capture exact built-ins before any public root exists.
        pool = _pools.get(key) if key is not None else None
        created = pool is None or pool.retired
        if created:
            client = _construct_llm_client(params, snapshot_options(config), selected)
            pool = _Pool(client)
        # An in-progress load is also a lease; any post-construction failure
        # (including reading cached facts) must dispose the newly owned pool.
        pool.acquire()
        try:
            if created:
                # Resolve truthful cached route facts once, not per view.
                client._adapter._bound_capabilities = _snapshot_capabilities(
                    client._adapter.capabilities
                )
            from nooa.unifiedllm.requirements import _validate_requirements

            _validate_requirements(pool.client._adapter._bound_capabilities, require, label=name)
            # The transport root has no scalar view defaults. Seed a fresh raw
            # view with this load's inherited defaults, then resolve only truly
            # explicit settings as overrides (declared levels replace defaults).
            seed = copy.copy(pool.client._adapter)
            seed._client = copy.copy(pool.client)
            seed._client.config = snapshot_options(pool.client.config)
            seed._client.config.update(
                {key: value for key, value in settings.items() if key != "reasoning_level"}
            )
            seed._client.reasoning_level = settings.get("reasoning_level")
            seed._bound_settings = _Settings(
                _freeze(seed._client.config),
                _freeze(seed._client._reasoning_config.levels),
                seed._client.reasoning_level,
                seed._client.structured_output_mode,
            )
            from ._calibration import _route_calibration

            seed._route_calibration_record = _route_calibration(pool.client._adapter)
            explicit = {key: supplied[key] for key in settings if key in supplied}
            # Native constructor settings still count as explicit conflicts,
            # but route/identity options must never enter a settings binding.
            validation = {
                key: value
                for key, value in supplied.items()
                if key
                not in {
                    "client",
                    "model",
                    "api_base",
                    "base_url",
                    "custom_llm_provider",
                    "reasoning_levels",
                    "reasoning_default",
                }
            }
            _validate_legacy_settings(seed, validation, seed._bound_capabilities)
            bound = _bind_adapter(seed, **explicit)
            from ._view_limits import _adapter_limits

            _adapter_limits(bound, bound.capabilities)
            llm = LLM(bound, on_unsupported_media=media_policy, retry_config=retry or RetryConfig())
            llm._lease = pool
        except BaseException:
            try:
                pool.release()
            except BaseException:
                pass  # Preserve the original load/validation failure.
            raise
        if created and key is not None:
            _pools[key] = pool
        return llm


def _behavior(cls):
    """Canonical descriptors only; inherited object machinery/metadata is not reset."""
    result = {}
    for owner in reversed(cls.__mro__[:-1]):
        for name, descriptor in vars(owner).items():
            if callable(descriptor) or isinstance(
                descriptor, (property, staticmethod, classmethod)
            ):
                result[name] = descriptor
    return result


# Finite built-in identity baselines, not a route/capability registry.
# Native imports do not import LiteLLM. Legacy baselines are captured only on
# the legacy load path, before constructing the first root.
_LLM_BEHAVIOR = _behavior(LLM)
_NATIVE_BEHAVIOR = {cls: _behavior(cls) for cls in _BUILTINS}
_LEGACY_BEHAVIOR = None


def _legacy_behavior():
    global _LEGACY_BEHAVIOR
    if _LEGACY_BEHAVIOR is None:
        from nooa.unifiedllm.unifiedllm import CompletionClient, ResponsesClient

        from .litellm_chat import LiteLLMChatAdapter
        from .litellm_responses import LiteLLMResponsesAdapter

        _LEGACY_BEHAVIOR = {
            cls: _behavior(cls)
            for cls in (
                CompletionClient,
                ResponsesClient,
                LiteLLMChatAdapter,
                LiteLLMResponsesAdapter,
            )
        }
    return _LEGACY_BEHAVIOR


def _check_behavior(value, expected):
    # Static access must not execute replaced properties or noncallable hooks.
    cls = type(value)
    current = _behavior(cls)
    if (
        current.keys() != expected.keys()
        or any(
            inspect.getattr_static(cls, name) is not descriptor or name in vars(value)
            for name, descriptor in expected.items()
        )
        or any(callable(item) for item in vars(value).values())
    ):
        raise LLMFailure(FailureKind.CONFIG, "Cannot derive a view with overridden behavior")
    # Also reject foreign class replacements made before the lazy legacy
    # baseline was initialized. Same-file methods must still name their slot.
    for name, descriptor in current.items():
        method = descriptor.fget if isinstance(descriptor, property) else descriptor
        if isinstance(method, (staticmethod, classmethod)):
            method = method.__func__
        method = inspect.unwrap(method)
        owner = next(owner for owner in cls.__mro__ if name in vars(owner))
        if (
            getattr(method, "__module__", None) != owner.__module__
            or getattr(method, "__qualname__", None) != f"{owner.__qualname__}.{name}"
        ):
            raise LLMFailure(FailureKind.CONFIG, "Cannot derive a view with overridden behavior")


def _check_shell(source):
    if type(source) is not LLM:
        raise LLMFailure(
            FailureKind.CONFIG, "Settings derivation requires an exact built-in loaded LLM"
        )
    _check_behavior(source, _LLM_BEHAVIOR)


def _check_derivable(source):
    """Exact loader-owned built-ins only, under the source view lock.

    Direct adapters, arbitrary subclasses and custom dispatch hooks have no safe
    ownership/copy contract. Use a fresh alias/config route load for those.
    Native checks deliberately return before importing any legacy client.
    """
    _check_shell(source)
    if source._closed or type(source._lease) is not _Pool:
        raise LLMFailure(
            FailureKind.CONFIG, "Settings derivation requires an open loader-owned view"
        )
    from .base import Capabilities

    if type(source._capabilities) is not Capabilities:
        raise LLMFailure(FailureKind.CONFIG, "Loaded view has invalid capability facts")
    adapter = source._adapter
    from ._native import _BUILTINS

    if type(adapter) in _BUILTINS:
        from ._native_settings import _NativeSettings

        _check_behavior(adapter, _NATIVE_BEHAVIOR[type(adapter)])
        if type(adapter._settings) is not _NativeSettings:
            raise LLMFailure(FailureKind.CONFIG, "Loaded view has invalid settings")
        if adapter._transport is not source._lease.client:
            raise LLMFailure(
                FailureKind.CONFIG, "Settings derivation requires proven pool ownership"
            )
        return True
    from nooa.unifiedllm.unifiedllm import CompletionClient, ResponsesClient

    from .litellm_chat import LiteLLMChatAdapter
    from .litellm_responses import LiteLLMResponsesAdapter

    pairs = {LiteLLMChatAdapter: CompletionClient, LiteLLMResponsesAdapter: ResponsesClient}
    if type(adapter) not in pairs:
        raise LLMFailure(
            FailureKind.CONFIG, "Settings derivation requires a built-in adapter/client"
        )
    _check_behavior(adapter, _legacy_behavior()[type(adapter)])
    if type(adapter._client) is not pairs[type(adapter)]:
        raise LLMFailure(
            FailureKind.CONFIG, "Settings derivation requires a built-in adapter/client"
        )
    _check_behavior(adapter._client, _legacy_behavior()[type(adapter._client)])
    if (
        adapter._legacy_options is not None
        or type(getattr(adapter, "_bound_settings", None)) is not _Settings
    ):
        raise LLMFailure(FailureKind.CONFIG, "Settings derivation requires a loaded settings view")
    client = adapter._client
    if client._http is not source._lease.client._http:
        raise LLMFailure(FailureKind.CONFIG, "Settings derivation requires proven pool ownership")
    return False


def _derive_loaded(source, require, settings):
    from nooa.unifiedllm.requirements import _validate_requirements

    from ._binding import _UNSET, _check_settings_identity
    from ._view_limits import _adapter_limits

    settings = {key: value for key, value in settings.items() if value is not _UNSET}
    try:
        if {"client_type", "model_name", "on_unsupported_media"} & settings.keys():
            raise ValueError
        _check_settings_identity(settings)
    except (TypeError, ValueError):
        raise LLMFailure(
            FailureKind.CONFIG, "Settings cannot change route/identity; load a fresh alias/config"
        ) from None
    _check_shell(source)  # Reject changed access behavior before touching its lock.
    with source._view_lock:
        native = _check_derivable(source)
        pool = source._lease
        # Own a new view lease BEFORE source close/retirement can race. Binding
        # failures release only this lease; nothing is published or mutated.
        pool.acquire()
        try:
            _validate_requirements(source._capabilities, require)
            if not native:
                _validate_legacy_settings(source._adapter, settings, source._capabilities)
            bound = _bind_adapter(source._adapter, **settings)
            _adapter_limits(bound, source._capabilities)
            view = LLM(
                bound, on_unsupported_media=source._media_policy, retry_config=source._retry_config
            )
            view._lease = pool
            return view
        except BaseException as exc:
            try:
                pool.release()
            except BaseException:
                pass
            if native and isinstance(exc, (ValueError, TypeError)):
                raise LLMFailure(FailureKind.CONFIG, "Invalid loaded settings") from None
            raise
