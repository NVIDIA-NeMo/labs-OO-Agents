# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one rule that picks the config in force: :func:`effective`.

``effective(ConfigType)`` returns the instance of any pydantic config model in
force here. Core limits are ``TruncationConfig`` fields; a package keeps its
own limits in its own config model and reads them the same way. Core never names a package's model: it only holds the overrides,
keyed by type, at each level:

- class: ``class MyAgent(Agent, configs=[MyLimits(a=1)])``
- instance: ``MyAgent(configs=[MyLimits(a=2)])``
- method: ``@strategy(CodeActStrategy(), configs=[MyLimits(a=3)])``

``TruncationConfig`` keeps its own ``truncation=`` keyword at each level (it is
not accepted in ``configs=``). Each level names only the fields it changes
(:func:`merge_config`).
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG, TruncationConfig

__all__ = ["effective", "merge_config"]

# The resolved config of the generated-method call running now: the agent's
# config merged with the method's @strategy(truncation=...). The runtime sets
# it around each generation call (nooa.runtime.actor._execute_with_generation)
# and resets it after; None outside one.
_current_truncation_config_var: contextvars.ContextVar[TruncationConfig | None] = (
    contextvars.ContextVar("current_truncation_config", default=None)
)

# The same for every other config model: the agent's ``configs`` merged with
# the method's @strategy(configs=...), keyed by type; None outside a call.
_current_configs_var: contextvars.ContextVar[Mapping[type[BaseModel], BaseModel] | None] = (
    contextvars.ContextVar("current_configs", default=None)
)

# One defaults instance per config model (built on first use).
_defaults: dict[type[BaseModel], BaseModel] = {TruncationConfig: DEFAULT_TRUNCATION_CONFIG}


def merge_config[ConfigT: BaseModel](base: ConfigT, override: ConfigT | None) -> ConfigT:
    """``base`` with the fields ``override`` set explicitly (``model_fields_set``).

    A field holding a sub-model merges field by field the same way, so an
    override that sets one field of a sub-model keeps the base's others.
    ``None`` returns ``base``. An override built without explicit fields
    (``model_validate`` / ``model_dump`` round trips) is refused: it would
    silently change nothing.
    """
    if override is None:
        return base
    if not override.model_fields_set:
        raise ValueError(
            f"merge_with() received a config with no model_fields_set. "
            f"Was it constructed from model_dump() or model_validate()? "
            f"Config objects must be freshly constructed: "
            f"{type(override).__name__}(field=value)."
        )
    return _merge(base, override)


def _merge[ConfigT: BaseModel](base: ConfigT, override: ConfigT) -> ConfigT:
    updates: dict[str, Any] = {}
    for name in override.model_fields_set:
        value = getattr(override, name)
        base_value = getattr(base, name)
        # Merge field by field only when both hold the same model type: a
        # union-typed field may hold another model, which then replaces the
        # base's value whole (its fields need not exist on the base's model).
        if isinstance(value, BaseModel) and type(value) is type(base_value):
            value = _merge(base_value, value)
        updates[name] = value
    merged = base.model_copy(update=updates)
    # model_copy() skips validation, so check the merged values against the
    # model's own validators (cross-field rules included): an invalid merge
    # raises here. The validated copy is only a check; the result stays the
    # model_copy() one, which keeps the explicit-field metadata
    # (model_fields_set) that merge_config() relies on.
    type(base).model_validate(_validation_input(merged))
    return merged


def _validation_input(model: BaseModel) -> dict[str, Any]:
    """``model``'s values keyed the way validation reads them.

    ``model_dump()`` is not valid validation input: it keys every field by its
    *name*, while validation expects the field's alias unless the model sets
    ``populate_by_name``, so with ``extra="forbid"`` the name is an unexpected
    key; it also writes computed fields, which are output only. This walks
    ``model_fields`` instead and keeps each value as it is (a nested model
    stays a model), so re-validating runs this model's own validators -- the
    cross-field rules -- and nothing else.
    """
    data: dict[str, Any] = {}
    for name, field in type(model).model_fields.items():
        alias = field.validation_alias
        if not isinstance(alias, str):
            # An ``AliasChoices`` accepts any of its string choices; an
            # ``AliasPath`` reads a nested key, which cannot be written back as
            # one flat input. Both fall back to the field name.
            alias = next((c for c in getattr(alias, "choices", ()) if isinstance(c, str)), None)
        data[alias or name] = getattr(model, name)
    return data


def default_config[ConfigT: BaseModel](config_type: type[ConfigT]) -> ConfigT:
    """The model's own defaults (``DEFAULT_TRUNCATION_CONFIG`` for ``TruncationConfig``)."""
    config = _defaults.get(config_type)
    if config is None:
        config = _defaults[config_type] = config_type()
    return config  # type: ignore[return-value]


def config_overrides(
    configs: Iterable[BaseModel] | None, where: str
) -> dict[type[BaseModel], BaseModel]:
    """``configs=`` of ``where`` keyed by type, checked: config instances, one per type."""
    result: dict[type[BaseModel], BaseModel] = {}
    if configs is None:
        return result
    if isinstance(configs, BaseModel):
        configs = [configs]
    for config in configs:
        if not isinstance(config, BaseModel):
            raise TypeError(
                f"{where} configs= takes config model instances, got {type(config).__name__}"
            )
        if isinstance(config, TruncationConfig):
            raise TypeError(f"{where}: pass a TruncationConfig as truncation=, not in configs=")
        if type(config) in result:
            raise ValueError(f"{where} configs= names {type(config).__name__} twice")
        result[type(config)] = config
    return result


def layer_configs(
    base: Mapping[type[BaseModel], BaseModel], overrides: Mapping[type[BaseModel], BaseModel]
) -> dict[type[BaseModel], BaseModel]:
    """``base`` (resolved configs by type) with each of ``overrides`` merged over it."""
    result = dict(base)
    for config_type, override in overrides.items():
        result[config_type] = merge_config(
            result.get(config_type) or default_config(config_type), override
        )
    return result


def effective[ConfigT: BaseModel](config_type: type[ConfigT], agent: Any = None) -> ConfigT:
    """The ``config_type`` instance in force here: read every configurable limit through it.

    First match wins:

    1. The method level: inside a generated method's call (and the code,
       tools and nested calls it runs), the agent's config merged with that
       method's ``@strategy(truncation=...)`` / ``@strategy(configs=[...])``.
    2. The agent level, of ``agent`` or, when ``agent`` is ``None``, of the
       agent whose method is running: ``MyAgent(...)`` merged over ``class
       MyAgent(Agent, ...)`` merged over the defaults (``truncation=`` for
       ``TruncationConfig``, ``configs=[...]`` for any other model).
    3. The model's defaults (``DEFAULT_TRUNCATION_CONFIG``, else
       ``config_type()``), outside any agent.

    Models match by exact type. Read it when the limit is used, not when an
    object is built, so an override set later (a method call) applies. The
    lookup uses context variables: asyncio tasks inherit them, plain threads
    do not. Code that starts a thread runs its target in
    ``contextvars.copy_context().run(...)`` to keep the limits of the caller;
    ``asyncio.to_thread`` already does.
    """
    truncation = config_type is TruncationConfig
    if truncation:
        override = _current_truncation_config_var.get()
        if override is not None:
            return override  # type: ignore[return-value]
    else:
        overrides = _current_configs_var.get()
        if overrides is not None:
            found = overrides.get(config_type)
            return found if found is not None else default_config(config_type)  # type: ignore[return-value]
    if agent is None:
        from nooa.runtime.context_vars import _parent_agent_var
        from nooa.util._context import _current_agent_var

        # The agent whose method is running: any agent method call sets
        # _parent_agent_var (nooa.runtime.method_wrapper); the implemented-method
        # path of the runtime also sets _current_agent_var.
        agent = _current_agent_var.get() or _parent_agent_var.get()
    if agent is not None:
        if truncation:
            config = getattr(agent, "_truncation", None)
        else:
            config = (getattr(agent, "_configs", None) or {}).get(config_type)
        if config is not None:
            return config
    return default_config(config_type)
