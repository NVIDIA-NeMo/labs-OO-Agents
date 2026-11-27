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

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, field_validator, model_validator

from nooa.media_capabilities import (
    MediaCapabilities,
    resolve_media_capabilities,
    validate_media_capabilities,
)
from nooa.media_inputs import MediaInputs


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
    endpoint: str | None = None
    # Name of the env var holding the API key (NOT the key itself).
    api_key_env: str | None = None
    client_type: str | None = None
    api_style: str | None = None
    context_window: int | None = None
    media_inputs: MediaInputs = ()
    # Per-kind MIME types, transports and size limits; None: provider default.
    media_capabilities: dict[str, dict[str, Any]] | None = None
    decision_image_transport: Literal["json", "typesafe_multipart"] = "json"
    typesafe_danger_zone_images: StrictBool = False
    structured_output_mode: Literal["auto", "anthropic"] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    reasoning_levels: dict[str, dict[str, Any]] | None = None
    reasoning_default: str | None = None
    reasoning_level: str | None = None

    @field_validator("media_capabilities", mode="before")
    @classmethod
    def _validate_media_capabilities(cls, value: object) -> dict[str, dict[str, Any]] | None:
        return validate_media_capabilities(value)

    def resolved_media_capabilities(self) -> MediaCapabilities:
        """This model's media capabilities: its provider default plus ``media_capabilities``.

        The provider default is chosen by the request interface the entry
        sends through (``nooa.request_interface.entry_input_interface``).
        """
        from nooa.request_interface import entry_input_interface

        interface = entry_input_interface(self.model_dump(exclude_none=True))
        return resolve_media_capabilities(interface, self.media_capabilities)

    @model_validator(mode="after")
    def _validate_structured_output_route(self):
        if self.structured_output_mode == "anthropic" and (
            self.client_type not in {None, "completion"}
            or self.api_style not in {None, "chat", "anthropic"}
        ):
            raise ValueError(
                "structured_output_mode='anthropic' requires a Chat/Messages completion route"
            )
        return self

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
