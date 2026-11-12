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

from ._binding import _bind_adapter, _freeze, _Settings
from ._calibration import _sdk_identity
from .legacy import snapshot_options

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


def _load_public(name, params, config, selected, media_policy, supplied):
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
    retry = params.get("retry_config")
    # Do not let constructor SDK retry settings become public attempts.
    params["max_retries"] = params["num_retries"] = 0
    key = _cache_key(params, config, selected)
    with _lock:
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
            selection = settings.get("reasoning_level")
            if selection is not None:
                from nooa.unifiedllm.reasoning import apply_reasoning_level

                apply_reasoning_level(
                    seed._client._reasoning_config,
                    seed._client.model,
                    {},
                    {
                        key: value
                        for key, value in supplied.items()
                        if key
                        not in {"client", "model", "api_base", "base_url", "custom_llm_provider"}
                    },
                    selection,
                )
            bound = _bind_adapter(seed, **explicit)
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
