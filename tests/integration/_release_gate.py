# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the live release-gate models from the installed registry.

The gate logic lives in this public repository; the routes it exercises do not.
Each gate case names a family such as ``"openai"`` and resolves the registry
alias ``release-gate-<family>``. The alias, with its model route, endpoint and
credential variable, is provided by a separately installed bundled-config
package (the ``nooa.bundled_configs`` entry-point group). When that package is
absent the case is skipped, which the release runner treats as a failure, so a
release can never be drafted without the gate having actually run.
"""

from __future__ import annotations

from urllib.parse import urlparse

import pytest

from nooa.unifiedllm.registry import MODELS, ensure_loaded, get_llm_client

ALIAS_PREFIX = "release-gate-"


def gate_alias(family: str) -> str:
    return f"{ALIAS_PREFIX}{family}"


def _entry(family: str) -> dict:
    ensure_loaded()
    config = MODELS.get(gate_alias(family))
    if not config:
        pytest.skip(
            f"registry alias {gate_alias(family)!r} is not installed; the private "
            "bundled-config package provides the release-gate routes"
        )
    return config


def gate_host(family: str) -> str:
    """Hostname of the family's endpoint, for request-capture hooks."""
    host = urlparse(_entry(family)["api_base"]).hostname
    assert host, f"registry alias {gate_alias(family)!r} has no usable api_base"
    return host


def gate_client(family: str, **overrides):
    """Build the family's client from its registry alias plus test-owned settings.

    Route, endpoint, credential and client type come from the alias. Request
    settings that define the test's behaviour (limits, retries, cache policy,
    reasoning options) stay here so the private entry never changes what the
    public gate asserts.
    """
    _entry(family)
    return get_llm_client(gate_alias(family), **overrides)
