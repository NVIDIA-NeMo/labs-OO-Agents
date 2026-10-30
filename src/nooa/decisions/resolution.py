# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve configured decision-model clients for agents and strategies."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from nooa.decisions.client import UnifiedDecisionModel

_INSTANCE_CACHE_ATTR = "_strategy_decision_model_alias_cache"


def is_decision_model(value: Any) -> bool:
    """Return whether a value is a concrete decision-model client."""
    return not inspect.isclass(value) and callable(getattr(value, "adecide", None))


def validate_decision_model_spec(
    spec: Any,
    target_name: str,
    *,
    standalone: bool = False,
) -> None:
    """Validate a client, registry alias, or instance-bound resolver."""
    if is_decision_model(spec):
        return
    if isinstance(spec, str):
        if spec:
            return
        raise TypeError(f"decision_model for {target_name!r} cannot be an empty alias")
    if callable(spec):
        if standalone:
            raise TypeError(
                f"decision_model for standalone function {target_name!r} cannot be a callable; "
                "pass a client or registry alias"
            )
        return
    target = "standalone function" if standalone else "method"
    raise TypeError(
        f"decision_model for {target} {target_name!r} must implement adecide(), be a registry alias, "
        f"or be a callable returning a decision model; got {type(spec).__name__}"
    )


def resolve_decision_alias(
    alias: str,
    cache: dict[str, Any] | None,
    target_name: str,
    *,
    origin: str = "decision_model=",
) -> UnifiedDecisionModel:
    """Resolve and optionally cache one configured decision-model alias."""
    if not alias:
        raise TypeError(f"{origin} for {target_name!r} cannot be an empty alias")
    if cache is not None and alias in cache:
        cached = cache[alias]
        if not is_decision_model(cached):
            raise TypeError(
                f"Cached decision alias {alias!r} for {target_name!r} does not implement adecide()"
            )
        return cast("UnifiedDecisionModel", cached)

    from nooa.unifiedllm import get_decision_model

    try:
        model = get_decision_model(alias)
    except Exception as exc:
        raise RuntimeError(
            f"The {origin} alias {alias!r} for {target_name!r} could not be resolved: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if cache is not None:
        cache[alias] = model
    return model


def resolve_method_decision_model(
    spec: Any,
    agent: Any,
    method_name: str,
    *,
    origin: str = "decision_model=",
) -> UnifiedDecisionModel:
    """Resolve a method or call-site decision model against its agent instance."""
    if is_decision_model(spec):
        return cast("UnifiedDecisionModel", spec)
    if isinstance(spec, str):
        cache = getattr(agent, _INSTANCE_CACHE_ATTR, None)
        if not isinstance(cache, dict):
            cache = {}
            try:
                setattr(agent, _INSTANCE_CACHE_ATTR, cache)
            except (AttributeError, TypeError):
                cache = None
        return resolve_decision_alias(spec, cache, method_name, origin=origin)
    if not callable(spec):
        raise TypeError(
            f"{origin} for {method_name!r} must be a client, alias, or callable; "
            f"got {type(spec).__name__}"
        )
    try:
        resolved = spec(agent)
    except Exception as exc:
        raise RuntimeError(
            f"The {origin} callable for {method_name!r} raised {type(exc).__name__}: {exc}"
        ) from exc
    if not is_decision_model(resolved):
        raise TypeError(
            f"The {origin} callable for {method_name!r} returned "
            f"{type(resolved).__name__}, which does not implement adecide()"
        )
    return cast("UnifiedDecisionModel", resolved)


__all__ = [
    "is_decision_model",
    "resolve_decision_alias",
    "resolve_method_decision_model",
    "validate_decision_model_spec",
]
