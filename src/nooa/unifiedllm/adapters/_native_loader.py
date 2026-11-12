# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native loading through the SAME protected pool and public LLM shell."""

from nooa.unifiedllm._route_calibration import _RouteCalibration
from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.llm import LLM
from nooa.unifiedllm.retry_config import RetryConfig

from ._binding import _thaw
from ._loader import _VIEW_SETTINGS, _cache_key, _lock, _Pool, _pools
from ._native import _NATIVE_TYPES
from ._native_settings import _resolve_native
from ._native_transport import _NativeTransport


def _load_native(name, config, family, media_policy, overrides, require=None):
    settings, retry = _resolve_native(family, name, config, overrides)
    from nooa.unifiedllm.requirements import _validate_requirements

    _validate_requirements(settings.capabilities, require, label=name)
    # Validate selected budgets before even allocating the transport root.
    from ._view_limits import _adapter_limits

    _adapter_limits(_NATIVE_TYPES[family](settings, None, None), settings.capabilities)
    if retry is None:
        retry = RetryConfig()
    elif retry is False:
        retry = None
    elif isinstance(retry, dict):
        retry = RetryConfig(**retry)
    elif type(retry) is not RetryConfig:
        raise LLMFailure(
            FailureKind.CONFIG, "Native retry_config must be RetryConfig, mapping or False"
        )
    route = _thaw(settings.route)
    key_params = {
        "base_url": route["endpoint"],
        "api_key": route["key"],
        "client": route["client"],
        "http_config": route["http_config"],
        "model": settings.model,
        "headers": route["headers"],
        "default_query": route["query"],
        "organization": route["organization"],
        "project": route["project"],
        "defaults": {k: v for k, v in _thaw(settings.raw).items() if k not in _VIEW_SETTINGS},
        "reasoning_levels": settings.declaration.levels,
        "reasoning_default": settings.declaration.default,
        "media_inputs": sorted(settings.capabilities.media_kinds),
        "tools_supported": settings.capabilities.tools,
        "context_window": settings.capabilities.context_window,
        "cache_breakpoint": settings.cache_breakpoint,
        "host_profile": _thaw(settings.host_profile),
    }
    key = _cache_key(key_params, {}, family)
    with _lock:
        pool = _pools.get(key) if key is not None else None
        created = pool is None or pool.retired
        if created:
            transport = _NativeTransport(family, **route)
            pool = _Pool(transport)  # Constructor above is allocation-free.
            pool._native_record = _RouteCalibration(1.0, 0.1)
        pool.acquire()  # Own initialization, including partial BaseException cleanup.
        try:
            if created:
                pool.client._initialize()
            adapter = _NATIVE_TYPES[family](settings, pool.client, pool._native_record)
            llm = LLM(adapter, on_unsupported_media=media_policy, retry_config=retry)
            llm._lease = pool
        except BaseException:
            try:
                pool.release()
            except BaseException:
                pass  # Preserve the original allocation or validation failure.
            raise
        if created and key is not None:
            _pools[key] = pool  # Atomic publication only after all validation succeeds.
        return llm
