# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed representation of an ``llm_config.yaml`` model-alias entry."""

from __future__ import annotations

import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nooa.unifiedllm.retry_config import RetryConfig
from nooa.unifiedllm.types import LLMRequestDefaults


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    responses: bool = False
    streaming: bool = True
    tools: bool = True
    structured_output: bool = True


class ModelConfig(BaseModel):
    """Validated, provider-neutral model alias configuration.

    ``api_base`` and ``max_tokens`` are accepted for one compatibility release.
    New configuration should use ``endpoint`` and ``request.max_output_tokens``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    model_name: str | None = None
    provider: str | None = None
    endpoint: str | None = None
    api_key_env: str | None = None
    client_type: Literal["completion", "responses"] = "completion"
    context_window: int | None = Field(default=None, gt=0)
    request: LLMRequestDefaults = Field(default_factory=LLMRequestDefaults)
    provider_options: dict[str, Any] = Field(default_factory=dict)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    retry_config: RetryConfig | Literal[False] | None = None

    # Common request defaults retained at the top level for existing files.
    temperature: float | None = None
    top_p: float | None = None
    reasoning: Any | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None

    # Descriptive/evaluation metadata used by repository model catalogs.
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    reasoning_model: bool | None = None
    max_thinking_tokens: int | None = Field(default=None, gt=0)
    max_retries: int | None = Field(default=None, ge=0)
    retry_on_empty_content: bool | None = None

    # One-release compatibility aliases. They are removed by ``registry_dict``.
    api_base: str | None = None
    max_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_and_warn_aliases(self) -> ModelConfig:
        if self.endpoint and self.api_base and self.endpoint != self.api_base:
            raise ValueError("endpoint and deprecated api_base disagree")
        if self.max_tokens is not None and self.request.max_output_tokens is not None:
            raise ValueError("request.max_output_tokens collides with deprecated max_tokens")

        request_values = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reasoning_effort": self.reasoning_effort,
        }
        for key, legacy_value in request_values.items():
            request_value = getattr(self.request, key)
            if legacy_value is not None and request_value is not None:
                raise ValueError(f"request.{key} collides with top-level {key}")

        reserved_provider_options = {
            # Connection and routing fields.
            "model",
            "provider",
            "api_key",
            "api_base",
            "base_url",
            "endpoint",
            "provider_options",
            # Provider-neutral request fields and values constructed by NOOA.
            *LLMRequestDefaults.__dataclass_fields__,
            "max_tokens",
            "messages",
            "input",
            "instructions",
            "tools",
            "response_format",
            "stream",
        }
        collisions = reserved_provider_options.intersection(self.provider_options)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"provider_options collides with reserved key(s): {names}")

        if self.api_base is not None:
            warnings.warn(
                "ModelConfig.api_base is deprecated; use endpoint (removal after one release)",
                DeprecationWarning,
                stacklevel=3,
            )
        if self.max_tokens is not None:
            warnings.warn(
                "ModelConfig.max_tokens is deprecated; use request.max_output_tokens "
                "(removal after one release)",
                DeprecationWarning,
                stacklevel=3,
            )
        return self

    @property
    def resolved_endpoint(self) -> str | None:
        return self.endpoint or self.api_base

    def registry_dict(self) -> dict[str, Any]:
        """Return the canonical read-only-compatible dictionary representation."""
        data = self.model_dump(exclude_none=True, exclude={"api_base", "max_tokens"})
        if self.resolved_endpoint is not None:
            data["endpoint"] = self.resolved_endpoint

        request = dict(data.pop("request", {}))
        # Existing top-level request fields are normalized into ``request``.
        for key in ("temperature", "top_p", "reasoning_effort"):
            value = data.pop(key, None)
            if value is not None:
                request[key] = value
        if self.max_tokens is not None:
            request["max_output_tokens"] = self.max_tokens
        data["request"] = {key: value for key, value in request.items() if value is not None}
        return data

    @classmethod
    def from_registry(cls, name: str, raw: dict[str, Any]) -> ModelConfig:
        data = dict(raw)
        data.setdefault("model_name", name)
        return cls.model_validate(data)


__all__ = ["ModelCapabilities", "ModelConfig"]
