# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declared effort choices, independent of model names and provider discovery."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from nooa.unifiedllm.limits import REPLY_CAP_KEYS

_DECLARATIONS = {"reasoning_levels", "reasoning_default"}
# These select the client/request itself, not a provider's effort behavior.
_RESERVED = _DECLARATIONS | {
    "reasoning_level",
    "model",
    "api_base",
    "base_url",
    "api_key",
    "custom_llm_provider",
    "messages",
    "input",
    "extra_body",
    "client",
}


class ReasoningConfig(BaseModel):
    """Map public level names to request settings for one configured route.

    None means unknown support; an empty mapping means unsupported. The default
    documents the route's default, not a request to send it on every call.
    Declarations live in registry YAML, not a model-name table in this module.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    levels: dict[str, dict[str, Any]] | None = None
    default: str | None = None

    @model_validator(mode="after")
    def validate_declaration(self):
        """Reject malformed choices and request-control fields at construction."""
        if self.default is not None and self.default not in (self.levels or {}):
            raise ValueError("reasoning_default must name a declared reasoning level")
        for level, settings in (self.levels or {}).items():
            if not level.strip() or not settings:
                raise ValueError("reasoning_levels must have non-empty names and request settings")
            if conflict := _RESERVED & settings.keys():
                raise ValueError(
                    f"reasoning level {level!r} contains reserved fields: {sorted(conflict)}"
                )
        return self

    def settings(self, level: str) -> dict[str, Any]:
        """Validate a selection and detach its settings from the stored declaration."""
        if self.levels is None:
            raise ValueError(
                "Reasoning levels are unknown for this route; declare reasoning_levels"
            )
        if not self.levels:
            raise ValueError("Reasoning-level selection is not supported for this route")
        if not isinstance(level, str) or level not in self.levels:
            raise ValueError(
                f"Invalid reasoning level {level!r}; allowed: {', '.join(self.levels)}"
            )
        # Frozen Pydantic attributes do not freeze nested dictionaries. Reuse the
        # declaration checks so later edits cannot introduce routing controls.
        self.validate_declaration()
        # Only the small chosen configuration is copied, never conversation data.
        return deepcopy(self.levels[level])


def _replace_reply_cap(params: dict[str, Any], settings: dict[str, Any]) -> None:
    """A cap override replaces all inherited spellings, including extra_body."""
    extra = settings.get("extra_body")
    supplied = {**settings, **(extra if isinstance(extra, Mapping) else {})}
    keys = REPLY_CAP_KEYS & supplied.keys()
    if not keys:
        return
    if len(keys) > 1:
        raise ValueError("Use only one reply token limit field per configuration layer")
    for key in REPLY_CAP_KEYS:
        params.pop(key, None)
    inherited_extra = params.get("extra_body")
    if isinstance(inherited_extra, Mapping):
        params["extra_body"] = {k: v for k, v in inherited_extra.items() if k not in REPLY_CAP_KEYS}


def _promote_reply_cap(params: dict[str, Any]) -> dict[str, Any]:
    """Caps are standard request settings; some transports ignore them in extra_body."""
    extra = params.get("extra_body")
    if isinstance(extra, Mapping):
        for key in REPLY_CAP_KEYS & extra.keys():
            params[key] = extra[key]
        if REPLY_CAP_KEYS & extra.keys():
            params["extra_body"] = {k: v for k, v in extra.items() if k not in REPLY_CAP_KEYS}
    if len(REPLY_CAP_KEYS & params.keys()) > 1:
        raise ValueError("Use only one reply token limit field")
    return params


def apply_reasoning_level(
    declaration: ReasoningConfig,
    model: str,
    defaults: dict[str, Any],
    overrides: dict[str, Any],
    default_selection: str | None,
) -> dict[str, Any]:
    """Resolve effort once before either client dispatches.

    No selection leaves existing provider settings untouched. An explicit level
    replaces constructor defaults; mixing it with per-call native controls is an
    error. Route changes cannot inherit a declaration for a different endpoint.
    This affects requested effort, never stored reasoning or replay compatibility.
    """
    if _DECLARATIONS & overrides.keys():
        raise ValueError("reasoning_levels and reasoning_default belong on the client constructor")
    params = dict(defaults)
    _replace_reply_cap(params, overrides)
    params.update(overrides)
    extra = params.get("extra_body")
    if isinstance(extra, Mapping) and (set(extra) & (_DECLARATIONS | {"reasoning_level"})):
        raise ValueError("Reasoning configuration cannot be passed through extra_body")
    level = params.pop("reasoning_level", default_selection)
    if level is None:
        return _promote_reply_cap(params)
    patch = declaration.settings(level)
    if (
        any(
            key in overrides and overrides[key] != defaults.get(key)
            for key in ("api_base", "base_url", "custom_llm_provider")
        )
        or overrides.get("model", model) != model
        or overrides.get("client") is not None
    ):
        raise ValueError("Reasoning levels are route-specific; create a client for the new route")
    explicit_extra = overrides.get("extra_body")
    explicit_keys = overrides.keys() | (
        explicit_extra.keys() if isinstance(explicit_extra, Mapping) else set()
    )
    if REPLY_CAP_KEYS & patch.keys() and REPLY_CAP_KEYS & explicit_keys:
        raise ValueError(
            "reasoning_level conflicts with explicit request field(s): reply token limit"
        )
    if conflict := patch.keys() & (
        overrides.keys() | (explicit_extra.keys() if isinstance(explicit_extra, Mapping) else set())
    ):
        raise ValueError(
            f"reasoning_level conflicts with explicit request field(s): {sorted(conflict)}"
        )
    # Whole top-level values replace defaults. Authors write complete nested
    # blocks in YAML; no provider-specific merge or inheritance rules live here.
    # Remove replaced defaults from extra_body too: SDKs otherwise merge those
    # back over the selected top-level values when assembling the HTTP body.
    _replace_reply_cap(params, patch)
    extra = params.get("extra_body")
    if isinstance(extra, Mapping) and patch.keys() & extra.keys():
        params["extra_body"] = {key: value for key, value in extra.items() if key not in patch}
    params.update(patch)
    return _promote_reply_cap(params)
