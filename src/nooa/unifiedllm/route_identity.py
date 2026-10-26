# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide whether two client configurations send requests to the same route.

A ``context_window`` is only valid for the route it was declared for. These
checks compare configuration values only: they make no network calls, read no
environment variables and do not guess providers from model prefixes.
"""

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _endpoint(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the normalized endpoint URLs set in ``config``, sorted.

    ``api_base``, ``base_url`` and ``endpoint`` are collected; scheme and host are
    lowercased and a trailing slash is dropped. A non-string or unparsable value
    gives an ``("<invalid>", key)`` marker so it never matches a real endpoint.
    """
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
    """Return True if both configurations send requests to the same route.

    The model strings, client types, ``custom_llm_provider``, ``api_key_env`` and
    ``api_key`` must be equal, and the endpoints must match after normalization.
    ``api_base`` and ``base_url`` may spell the same endpoint, but a configuration
    that sets two different URLs never matches. URL paths and queries are
    significant. Client objects (``client``) must be the same object. Key values
    are only compared here; they are never returned, hashed or logged.
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
