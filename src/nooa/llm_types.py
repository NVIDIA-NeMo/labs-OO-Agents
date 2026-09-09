# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-independent values returned by :mod:`nooa.unifiedllm`."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from nooa.agentdoc import spec
from nooa.context_blocks.events import EventBase
from nooa.context_blocks.roles import Role


class ToolCall(BaseModel):
    """Provider-independent tool call exactly as emitted by the model."""

    id: str
    name: str
    arguments: str


class LLMUsage(BaseModel):
    """Normalized usage reported for one successful LLM response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    @classmethod
    def from_provider(cls, value: Any) -> LLMUsage | None:
        """Normalize common provider and LiteLLM usage shapes once."""
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if hasattr(value, "_asdict"):
            value = value._asdict()
        elif hasattr(value, "model_dump"):
            value = value.model_dump()

        def get(source: Any, key: str, default: Any = None) -> Any:
            if isinstance(source, dict):
                return source.get(key, default)
            return getattr(source, key, default)

        def first(source: Any, *keys: str) -> Any:
            for key in keys:
                result = get(source, key)
                if result is not None:
                    return result
            return None

        prompt_details = first(value, "prompt_tokens_details", "input_tokens_details")
        completion_details = first(value, "completion_tokens_details", "output_tokens_details")
        input_tokens = int(first(value, "input_tokens", "prompt_tokens") or 0)
        output_tokens = int(first(value, "output_tokens", "completion_tokens") or 0)
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=int(
                first(value, "cached_input_tokens", "cached_tokens", "cache_read_input_tokens")
                or first(prompt_details, "cached_tokens", "cache_read_input_tokens")
                or 0
            ),
            cache_write_input_tokens=int(
                first(value, "cache_write_input_tokens", "cache_creation_input_tokens") or 0
            ),
            reasoning_tokens=int(
                first(value, "reasoning_tokens")
                or first(completion_details, "reasoning_tokens")
                or 0
            ),
            total_tokens=int(first(value, "total_tokens") or input_tokens + output_tokens),
            cost_usd=float(first(value, "cost_usd", "cost") or 0.0),
        )

class LLMResponse(EventBase):
    """Canonical response produced by UnifiedLLM and persisted by NOOA.

    UnifiedLLM creates a fresh object for every call. The runtime records that
    same object; renderers project its conversational fields while telemetry
    consumers read its model and usage metadata.
    """

    _role: ClassVar[Role] = Role.ASSISTANT

    raw_response: Any = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Non-serialized provider response retained on the live in-memory object",
    )
    content: Annotated[
        str,
        spec(max_string=None),
        Field(description="Exact normalized assistant text used for replay"),
    ] = ""
    parsed: Any = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Non-serialized typed result retained on the live in-memory object",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        repr=False,
        description="Ordered public tool calls emitted on this assistant turn",
    )
    finish_reason: Literal["stop", "tool_calls", "length", "error"] = Field(
        default="stop",
        repr=False,
        description="Normalized provider finish reason",
    )
    reasoning: str | None = Field(
        default=None,
        repr=False,
        description="Provider-exposed plain reasoning returned with this assistant turn",
    )
    llm_state: dict[str, Any] | None = Field(
        default=None,
        repr=False,
        description="Opaque state returned by UnifiedLLM for exact provider replay",
    )
    usage: LLMUsage | None = Field(
        default=None,
        repr=False,
        description="Normalized token, cache, reasoning, and cost usage",
    )
    model_name: str = Field(
        default="", repr=False, description="Model identifier used for this response"
    )
    generation_id: str = Field(
        default="",
        repr=False,
        description="Generation turn that produced this response",
    )
    dynamic_context: str = Field(
        default="",
        repr=False,
        description="Trailing dynamic context rendered for this request",
    )

    @field_validator("usage", mode="before")
    @classmethod
    def _normalize_usage(cls, value: Any) -> LLMUsage | None:
        return LLMUsage.from_provider(value)

    @model_validator(mode="before")
    @classmethod
    def _separate_parsed_content(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        content = value.get("content", "")
        if content is None:
            value = dict(value)
            value["content"] = ""
            content = ""
        if isinstance(content, BaseModel):
            value = dict(value)
            value.setdefault("parsed", content)
            value["content"] = content.model_dump_json()
        elif not isinstance(content, str):
            value = dict(value)
            value["content"] = str(content)
        if not value.get("model_name"):
            raw_response = value.get("raw_response")
            raw_model = (
                raw_response.get("model")
                if isinstance(raw_response, dict)
                else getattr(raw_response, "model", None)
            )
            if isinstance(raw_model, str):
                value = dict(value)
                value["model_name"] = raw_model
        return value

    @property
    def replay_content(self) -> str:
        """Return the serializable assistant text used for event replay."""
        return self.content
