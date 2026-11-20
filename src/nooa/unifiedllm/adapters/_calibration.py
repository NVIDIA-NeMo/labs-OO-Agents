# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private numeric route calibration bridge, not calibration accounting.

Built-in public adapters and settings views acquire a record lazily. The initial
ratio is a COPY of historical model-level calibration, not verified route data.
Public shell calls update these records from exact-wire numeric observations.
Direct legacy sends and count_tokens still use the legacy singleton exclusively.
Route-estimate consumption and shell clamping are a separate migration.

Keys are process-private keyed digests, never route strings or plain secret
hashes. No keys/credentials/endpoints are logged, traced or persisted here.
Ambiguous/implicit routes get an isolated record, not guessed sharing.
"""

import hashlib
import hmac
import json
import secrets
import weakref
from threading import RLock
from urllib.parse import urlsplit

from nooa.unifiedllm._route_calibration import _RouteCalibration
from nooa.unifiedllm.route_identity import normalize_endpoint

_KEY = secrets.token_bytes(32)
_records = weakref.WeakValueDictionary()
# Separate opaque SDK identity table; weak references never clone/own clients.
_sdk_ids: dict[int, tuple[weakref.ReferenceType, bytes]] = {}


_sdk_lock = RLock()


def _sdk_identity(sdk):
    if sdk is None:
        return None
    identity = id(sdk)

    def forget(reference):
        with _sdk_lock:
            entry = _sdk_ids.get(identity)
            if entry is not None and entry[0] is reference:
                _sdk_ids.pop(identity)

    try:
        reference = weakref.ref(sdk, forget)
    except TypeError:
        return False  # Cannot safely associate this opaque object across roots.
    with _sdk_lock:
        entry = _sdk_ids.get(identity)
        if entry is None or entry[0]() is not sdk:
            entry = (reference, secrets.token_bytes(32))
            _sdk_ids[identity] = entry
        return entry[1].hex()


def _route_key(adapter):
    """Supported explicit route facts only; unknown identity carriers isolate.

    Identity-bearing headers/extra_body and alternate auth/transport declarations
    are deliberately unsupported for cross-root sharing in this first patch.
    Binding already forbids changing those facts within a settings view.
    """
    client = adapter._client
    config = client.config
    route_fields = {"api_base", "base_url", "api_key", "custom_llm_provider", "client"}
    # Deliberately narrow cross-root sharing: arbitrary provider/native blocks,
    # headers and unknown fields can carry identity. Do not interpret them or
    # guess that they are harmless settings. Same-root derived views still share.
    scalar_settings = {
        "temperature",
        "top_p",
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "max_retries",
        "num_retries",
        "timeout",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_effort",
        "stream",
        "n",
    }
    if not isinstance(config, dict) or not isinstance(client.model, str):
        return None
    if config.keys() - route_fields - scalar_settings:
        return None
    if any(
        type(config[key]) not in {str, int, float, bool, type(None)}
        for key in config.keys() & scalar_settings
    ):
        return None
    endpoints = set()
    for key in ("api_base", "base_url"):
        value = config.get(key)
        if value is None:
            continue
        endpoint = normalize_endpoint(value)
        if endpoint is None:
            return None
        parts = urlsplit(endpoint)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None
        endpoints.add(endpoint)
    if len(endpoints) != 1:
        return None  # No endpoint => potentially environment-dependent routing.
    provider, credential = config.get("custom_llm_provider"), config.get("api_key")
    if provider is not None and not isinstance(provider, str):
        return None
    sdk = _sdk_identity(config.get("client"))
    if sdk is False or not isinstance(credential, str) or not credential.strip():
        return None  # Implicit credentials can change outside the bound view.
    facts = [type(adapter).__name__, client.model, endpoints.pop(), provider, credential, sdk]
    return hmac.digest(_KEY, json.dumps(facts).encode(), hashlib.sha256)


def _isolated_key(adapter):
    """Protect a root-local association against mutable legacy route changes.

    Only small builtin config containers/scalars are inspected; no opaque object
    is copied, serialized or repr'd. An uninspectable/cyclic/large configuration
    gets a fresh opaque key, so only an already bound view can promise reuse.
    """
    remaining = 1000

    def project(value, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 8:
            raise ValueError
        if type(value) in {str, int, float, bool, type(None)}:
            if type(value) is str and len(value) > 8192:
                raise ValueError
            return [type(value).__name__, value]
        if type(value) is dict:
            if len(value) > remaining or any(type(key) is not str for key in value):
                raise ValueError
            return ["dict", [[key, project(value[key], depth + 1)] for key in sorted(value)]]
        if type(value) in {list, tuple}:
            return [type(value).__name__, [project(item, depth + 1) for item in value]]
        identity = _sdk_identity(value)
        if identity is False:
            raise ValueError
        return ["opaque", identity]

    try:
        facts = project([adapter._client.model, adapter._client.config])
        digest = hmac.digest(_KEY, json.dumps(facts).encode(), hashlib.sha256)
    except ValueError:
        digest = secrets.token_bytes(32)
    return adapter._route_calibration_identity, digest


def _route_calibration(adapter):
    """Return the stable record associated with this built-in route/view.

    Publication uses builtin dict.setdefault (atomic in the supported CPython
    runtime), rather than a lock/wait on the event loop. Concurrent candidates
    may copy the legacy baseline, but only one is published; losing candidates
    never update/reset it. Work is local numeric/config inspection with no I/O;
    only the fallback projector enforces depth/item/value-string bounds.
    """
    from ._native import _BUILTINS

    if type(adapter) in _BUILTINS:
        return adapter._route_calibration_record

    from nooa.unifiedllm.unifiedllm import _token_calibration

    record = adapter._route_calibration_record
    if record is not None and hasattr(adapter, "_bound_settings"):
        return record  # Immutable settings views retain their original route.
    key = _route_key(adapter)
    if key is None:
        # Unsupported facts never imply equality across independent roots.
        key = _isolated_key(adapter)
    record = _records.get(key)
    if record is None:
        candidate = _RouteCalibration(
            float(_token_calibration.ratio(adapter._client.model)), float(_token_calibration._alpha)
        )
        record = _records.setdefault(key, candidate)
    adapter._route_calibration_record = record
    return record
