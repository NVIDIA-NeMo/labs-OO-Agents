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

User, project and ``NEMO_OO_LLM_CONFIG`` registry files normally override
bundled entries. For these aliases an override would silently test a different
route, so a gate alias that differs from the bundled-config definition fails.
"""

from __future__ import annotations

from urllib.parse import urlparse

import pytest

from nooa import llm_config
from nooa.unifiedllm.registry import _load_models_from_yaml, get_llm_client, get_registry_config

ALIAS_PREFIX = "release-gate-"

# Families run by the release gate. scripts/make_release.py lists the same
# families in PROVIDER_TESTS; tests/test_make_release.py checks they match.
CACHE_RESUME_FAMILIES = ("openai", "anthropic", "gemini")
OPEN_MODEL_FAMILIES = ("deepseek", "kimi", "glm", "qwen")


class GateAliasError(Exception):
    """A gate alias exists but does not come from the bundled-config package."""


def gate_alias(family: str) -> str:
    return f"{ALIAS_PREFIX}{family}"


def _bundled_models() -> dict[str, dict]:
    """Merge only the bundled-config layer, with the registry's last-wins rules."""
    merged: dict[str, dict] = {}
    for path in llm_config.bundled_config_paths():
        for name, config in _load_models_from_yaml(path).items():
            if isinstance(config, dict):
                merged[name] = config
            else:
                merged.pop(name, None)
    return merged


def gate_config(family: str) -> dict:
    """Effective registry entry for the family's alias; empty when absent."""
    alias = gate_alias(family)
    config = get_registry_config(alias)
    if config and config != _bundled_models().get(alias):
        raise GateAliasError(
            f"registry alias {alias!r} is defined or overridden outside the "
            "bundled-config package (a user, project or NEMO_OO_LLM_CONFIG file); "
            "remove that entry so the gate uses the supplied route"
        )
    return config


def _host(config: dict) -> str | None:
    return urlparse(config.get("api_base") or "").hostname


def _entry(family: str) -> dict:
    try:
        config = gate_config(family)
    except GateAliasError as exc:
        pytest.fail(str(exc))
    if not config:
        pytest.skip(
            f"registry alias {gate_alias(family)!r} is not installed; the private "
            "bundled-config package provides the release-gate routes"
        )
    return config


def gate_host(family: str) -> str:
    """Hostname of the family's endpoint, for request-capture hooks."""
    host = _host(_entry(family))
    if not host:
        pytest.fail(
            f"registry alias {gate_alias(family)!r} has no api_base; the gate "
            "needs the endpoint host to capture its requests"
        )
    return host


def gate_client(family: str, **overrides):
    """Build the family's client from its registry alias plus test-owned settings.

    Route, endpoint and credential come from the alias. Request settings that
    define the test's behaviour (API style, limits, retries, cache policy,
    reasoning options) stay in the test so the private entry never changes what
    the public gate asserts.
    """
    _entry(family)
    return get_llm_client(gate_alias(family), **overrides)
