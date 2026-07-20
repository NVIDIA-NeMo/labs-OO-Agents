# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed representation of an ``llm_config.yaml`` model-alias entry.

The unifiedllm registry (``MODELS``) still stores raw dicts internally so
litellm can receive arbitrary passthrough kwargs and external readers
(``nat`` plugin, ``eval_pipeline``, viewer) keep working. ``ModelConfig`` is
the *typed boundary* over one of those entries — use
:func:`nooa.config.get_model_config` /
:func:`nooa.config.resolved_config` to get validated objects
instead of poking at dicts.

Long term, ``MODELS`` itself should hold ``ModelConfig`` objects rather than
dicts (see ``unifiedllm/registry.py``); this model is the first step.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    """One model alias from ``llm_config.yaml`` (e.g. ``claude-opus-4-8``).

    Common fields are typed; ``extra="allow"`` keeps any additional keys
    (litellm passthrough such as ``num_retries``, provider-specific knobs)
    accessible as attributes. ``frozen=True`` matches the other config models
    (``CodeActConfig`` etc.) — these describe state, they aren't mutated.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    # The litellm model string actually sent to the provider. Falls back to
    # the alias name itself when omitted (handled by the registry).
    model_name: str | None = None
    api_base: str | None = None
    # Name of the env var holding the API key (NOT the key itself).
    api_key_env: str | None = None
    client_type: str | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None

    @classmethod
    def from_registry(cls, name: str, raw: dict[str, Any]) -> ModelConfig:
        """Build a :class:`ModelConfig` from a raw registry dict for *name*.

        ``model_name`` defaults to the alias *name* when the entry omits it,
        mirroring :func:`nooa.unifiedllm.get_llm_client`.
        """
        data = dict(raw)
        data.setdefault("model_name", name)
        return cls.model_validate(data)


__all__ = ["ModelConfig"]
