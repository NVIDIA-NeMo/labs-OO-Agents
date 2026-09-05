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

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


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
    # OpenAI Responses API reasoning controls, forwarded verbatim to
    # ``litellm.responses(reasoning=...)``. Typically ``{"effort": "medium",
    # "context": "all_turns"}``; ``effort`` is the thinking budget
    # (none/low/medium/high/xhigh/max, model-dependent) and ``context``
    # selects how much prior reasoning the model may attend to.
    reasoning: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "OpenAI Responses reasoning controls forwarded to the provider, "
                "e.g. {'effort': 'medium', 'context': 'all_turns'}."
            )
        ),
    ] = None
    # OpenAI Responses API server-side state policy. ``store=False`` runs the
    # session stateless (no server-retained response chain) so the client must
    # replay the full output — the ZDR-friendly mode this framework uses. Leave
    # unset to accept the provider default (``store=True`` on OpenAI).
    store: Annotated[
        bool | None,
        Field(
            description=(
                "Responses API state policy: False keeps the session stateless "
                "(client replays history); unset uses the provider default."
            )
        ),
    ] = None
    # OpenAI Responses API ``include`` list — extra output fields the provider
    # should return. For stateless reasoning continuity set
    # ``["reasoning.encrypted_content"]`` so encrypted reasoning items come back
    # for replay on the next turn.
    include: Annotated[
        list[str] | None,
        Field(
            description=(
                "Responses API output fields to include, e.g. "
                "['reasoning.encrypted_content'] for stateless reasoning replay."
            )
        ),
    ] = None

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
