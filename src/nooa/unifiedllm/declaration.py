# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config-driven provider declarations at the registry edge.

A registry alias can declare what its model id cannot prove on its own:

- ``provider`` — the canonical logical provider ("openai", "glm", ...), for
  opaque enterprise/gateway model ids whose routing string resolves to no
  provider;
- ``compat_group`` — the model compat group the id belongs to, giving an
  opaque id an opaque replay boundary derived from the declared group
  instead of none at all.

:func:`apply_alias_declaration` is called lazily by
:func:`nooa.unifiedllm.registry.get_llm_client` on first use of a declaring
alias. It translates the declaration into process-lifetime registrations
that override the built-in catalogs (compat-group membership via
:func:`register_compat_group`, reasoning capabilities via
:func:`register_reasoning_capabilities`) and returns the
:class:`ProviderIdentity` the registry attaches to the client as metadata —
never as a request parameter.

Fail-closed rules:

- No declaration → no identity; consumption-time behavior is unchanged.
- ``compat_group`` without a resolvable provider (declared or parsed) raises
  ``ValueError`` — groups are provider-scoped, so a group-only declaration
  would be a guess.
- A declared provider that contradicts what :func:`parse_model_string`
  resolves from the model string raises ``ValueError`` — replay artifacts
  must never be routed to a provider the config contradicts.
- No ``capabilities`` declaration leaves the catalog answer as-is: unknown
  providers still look up to ``None``. A ``capabilities``-only declaration
  registers an override (when a provider is resolvable) but carries no
  identity.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from typing import Any

from nooa.unifiedllm.contracts import (
    _COMPAT_GROUPS,
    _PROVIDER_ALIASES,
    ModelCompatGroup,
    ProviderIdentity,
    ReasoningCapabilities,
    derive_opaque_replay_key,
    parse_model_string,
    register_compat_group,
    register_reasoning_capabilities,
)

__all__ = ["apply_alias_declaration", "canonical_provider"]

logger = logging.getLogger(__name__)

# Serializes the registrations and the once-per-declaration bookkeeping.
# Reentrant so nested application (a declaration whose registration path
# re-enters the module) cannot self-deadlock.
_declaration_lock = threading.RLock()

#: alias -> declaration signature already applied this process. Registrations
#: are process-lifetime: re-applying an identical declaration is a no-op, and a
#: *changed* declaration (the registry was reloaded with different values)
#: re-applies and overrides what the earlier one registered.
_applied: dict[str, tuple[Any, ...]] = {}


def canonical_provider(provider: str) -> str:
    """Canonicalize a declared provider through the contracts alias map.

    ``derive_opaque_replay_key`` canonicalizes the provider with the same
    map, so going through it here is what keeps a declared group registration
    and the derived key from disagreeing about scope (a "zai" declaration
    registering under "zai" while the key derives under "glm" would never
    match).
    """
    key = provider.strip().lower()
    return _PROVIDER_ALIASES.get(key, key)


def _register_declared_group(*, provider: str, group_name: str, model: str) -> None:
    """Register the declared *model* into *group_name* under *provider*.

    A declaration may create a new group or extend a same-provider one (an
    opaque enterprise id joining a hand-verified group). Extending keeps the
    group's existing members — they were live-verified, and the declaration
    asserts one more member, not a replacement. A same-name group owned by a
    *different* provider is rejected: accepting it would move that provider's
    replay boundary, which a single alias's declaration must not do.
    """
    with _declaration_lock:
        existing = _COMPAT_GROUPS.get(group_name)
        if existing is None:
            register_compat_group(
                ModelCompatGroup(name=group_name, provider=provider, models=frozenset({model}))
            )
        elif existing.provider.lower() != provider.lower():
            raise ValueError(
                f"Compat group {group_name!r} is already declared for provider "
                f"{existing.provider!r}, but this declaration says {provider!r}; "
                "declare a different group name."
            )
        else:
            register_compat_group(
                existing.model_copy(update={"models": existing.models | frozenset({model})})
            )
        # The registration is process-wide, so a model that other same-
        # provider groups already claim now sits in several groups; lookups
        # outside this alias resolve deterministically (alphabetically first
        # group name), which may not be this one. Surface it rather than let
        # the declaration appear to take effect when it did not.
        conflicts = sorted(
            group.name
            for group in _COMPAT_GROUPS.values()
            if group.name != group_name
            and group.provider.lower() == provider.lower()
            and model.lower() in {m.lower() for m in group.models}
        )
        if conflicts:
            logger.warning(
                "Model %r is now a member of compat groups %r and %r; catalog "
                "lookups resolve to the alphabetically first group name.",
                model,
                group_name,
                conflicts,
            )


def _apply_capability_override(alias: str, provider: str, config: Mapping[str, Any]) -> None:
    """Register the alias's declared ``capabilities`` for *provider*.

    ``capabilities`` is an optional registry mapping shaped like
    :class:`ReasoningCapabilities` (``capture_kinds``, ``effort_map``,
    ``replay_field``, ...). It is validated, never trusted: an invalid
    declaration is warned about and dropped, because a wrong *capability*
    profile degrades reasoning handling without breaking the request path —
    unlike a wrong *identity*, which must fail loudly.
    """
    caps = config.get("capabilities")
    if not caps:
        return
    if not isinstance(caps, Mapping):
        logger.warning(
            "Model %r has an invalid capabilities declaration: expected a mapping, got %s; ignoring it.",
            alias,
            type(caps).__name__,
        )
        return
    try:
        register_reasoning_capabilities(provider, ReasoningCapabilities.model_validate(dict(caps)))
    except Exception as exc:
        logger.warning(
            "Model %r has an invalid capabilities declaration (%s); ignoring it.", alias, exc
        )


def apply_alias_declaration(
    alias: str,
    config: Mapping[str, Any],
    *,
    api_style: str,
    transport: str,
) -> ProviderIdentity | None:
    """Translate *alias*'s declared provider/compat_group into registrations
    plus a :class:`ProviderIdentity`, or return ``None`` when nothing is
    declared.

    The registration side effects (compat-group membership, capability
    overrides) happen once per distinct declaration for the process lifetime;
    the returned identity is a pure value and is rebuilt on every call.

    Raises:
        ValueError: the declaration is unusable — a ``compat_group`` with no
            resolvable provider, a provider that contradicts the model
            string, a group name owned by another provider, or a non-string
            declaration value.
    """
    for field, value in (
        ("provider", config.get("provider")),
        ("compat_group", config.get("compat_group")),
    ):
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"Model {alias!r} declares {field}={value!r}: expected a string or null."
            )
    declared_provider = (
        config["provider"].strip() if isinstance(config.get("provider"), str) else None
    )
    declared_group = (
        config["compat_group"].strip() if isinstance(config.get("compat_group"), str) else None
    )
    caps = config.get("capabilities")
    if not declared_provider and not declared_group and not caps:
        return None

    model_name = config.get("model_name")
    model_string = model_name if isinstance(model_name, str) and model_name else alias
    parsed = parse_model_string(model_string)

    if declared_provider:
        provider = canonical_provider(declared_provider)
    elif parsed.provider is not None:
        # No provider declared: the parser resolves one, so the declaration
        # can be honored without guessing.
        provider = parsed.provider
    elif declared_group:
        raise ValueError(
            f"Model {alias!r} declares compat_group {declared_group!r} without a provider, "
            f"and model string {model_string!r} resolves to no provider either; declare "
            "'provider' alongside 'compat_group'."
        )
    else:
        # Capabilities-only declaration for an id with no resolvable
        # provider: there is nothing to register them under.
        logger.warning(
            "Model %r declares capabilities without a resolvable provider; ignoring the "
            "declaration.",
            alias,
        )
        return None
    if parsed.provider is not None and parsed.provider != provider:
        raise ValueError(
            f"Model {alias!r} declares provider {declared_provider!r}, but model string "
            f"{model_string!r} resolves to {parsed.provider!r}; fix the declaration."
        )

    caps_key = json.dumps(caps, sort_keys=True, default=repr) if caps else None
    signature = (model_string, declared_provider, declared_group, caps_key)
    with _declaration_lock:
        if _applied.get(alias) != signature:
            if declared_group:
                _register_declared_group(
                    provider=provider, group_name=declared_group, model=parsed.model
                )
            _apply_capability_override(alias, provider, config)
            _applied[alias] = signature

    # Capabilities-only declarations register overrides but carry no
    # identity: provider/compat_group is what an identity declaration is.
    if not declared_provider and not declared_group:
        return None

    # The declared group is authoritative for THIS alias's replay boundary:
    # deriving against a mapping that contains exactly the declaration keeps
    # other groups (built-in or registered by other aliases) that also claim
    # the model from silently winning the lookup.
    declared_groups = (
        {
            declared_group: ModelCompatGroup(
                name=declared_group, provider=provider, models=frozenset({parsed.model})
            )
        }
        if declared_group
        else None
    )
    return ProviderIdentity(
        provider=provider,
        api_style=api_style,
        model=parsed.model,
        transport=transport,
        opaque_replay_key=derive_opaque_replay_key(
            provider=provider,
            api_style=api_style,
            model=parsed.model,
            compat_groups=declared_groups,
        ),
    )
