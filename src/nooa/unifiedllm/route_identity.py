# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative identity checks for endpoint-local context metadata.

No discovery, environment reads, credential resolution or provider-prefix guessing.
An env-var credential declaration is a scope, not equivalent to an explicit key.
"""

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _endpoint(config: Mapping[str, Any]) -> tuple[str, ...]:
    endpoints = set()
    for key in ("api_base", "base_url", "endpoint"):
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            return ("<invalid>", key)
        try:
            parts = urlsplit(value)
            # DNS/scheme case is insignificant, URL userinfo is not.
            userinfo, sep, host = parts.netloc.rpartition("@")
            netloc = userinfo + sep + host.lower() if sep else parts.netloc.lower()
            value = urlunsplit(
                (parts.scheme.lower(), netloc, parts.path.rstrip("/"), parts.query, parts.fragment)
            )
        except ValueError:
            return ("<invalid>", key)
        endpoints.add(value)
    return tuple(sorted(endpoints))


def same_context_route(
    model: str,
    config: Mapping[str, Any],
    client_type: str,
    other_model: str,
    other: Mapping[str, Any],
    other_client_type: str,
) -> bool:
    """Require the same exact routing model, endpoint, API client and credentials.

    api_base/base_url are spelling aliases only when unambiguous; conflicting
    URLs fail closed. Paths and queries remain significant. Caller-owned API
    client objects compare by identity, never by equality. Credential values
    are compared only locally and are never returned, hashed or logged.
    """
    endpoint, other_endpoint = _endpoint(config), _endpoint(other)
    return (
        model == other_model
        and client_type == other_client_type
        and len(endpoint) <= 1
        and len(other_endpoint) <= 1
        and endpoint == other_endpoint
        and config.get("client") is other.get("client")
        and config.get("custom_llm_provider") == other.get("custom_llm_provider")
        and config.get("api_key_env") == other.get("api_key_env")
        and config.get("api_key") == other.get("api_key")
    )
