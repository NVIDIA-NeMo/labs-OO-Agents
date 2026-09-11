# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-independent values returned by :mod:`nooa.unifiedllm`."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from nooa._immutable_json import NativeJSON, freeze, json_containers
from nooa.agentdoc import spec
from nooa.context_blocks.events import EventBase
from nooa.context_blocks.roles import Role


@dataclass(frozen=True)
class CacheBoundary(Mapping[str, Any]):
    """End the stable prefix in a UnifiedLLM message list.

    Renderers and middleware pass this object through unchanged. UnifiedLLM
    consumes it after projecting assistant turns, so provider-specific expansion
    cannot move the boundary. It is never sent to a model. The read-only mapping
    is its public JSON view for integrations such as NeMo Relay, not a second
    input format: direct callers should pass CacheBoundary().
    """

    _public: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"role": "metadata", "nooa_cache_boundary": True}
    )

    def __getitem__(self, key: str) -> Any:
        return self._public[key]

    def __iter__(self):
        return iter(self._public)

    def __len__(self) -> int:
        return len(self._public)

    def public_message(self) -> dict[str, Any]:
        return dict(self._public)

    def render_message(self, content, tool_calls, *, reasoning):
        return self


class _PublicToolCall(Protocol):
    id: str
    name: str

    @property
    def arguments(self) -> str | dict[str, Any]: ...


def assistant_message(
    content: str | None,
    *,
    tool_calls: Iterable[_PublicToolCall] = (),
    reasoning: str | None = None,
) -> dict[str, Any]:
    """Build the public assistant shape used for rendering and JSON integrations.

    Only public values enter this projection; native state is never inspected.
    Captured JSON arguments stay byte-for-byte intact, while synthetic calls
    may supply dictionaries. Absent text consistently projects as an empty string.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if reasoning:
        message["reasoning_content"] = reasoning
    calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments)
                if isinstance(call.arguments, dict)
                else call.arguments,
            },
        }
        for call in tool_calls
    ]
    if calls:
        message["tool_calls"] = calls
    return message


class ToolCall(BaseModel):
    """Provider-independent tool call exactly as emitted by the model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["tool_call"] = "tool_call"
    native: NativeJSON | None = Field(default=None, repr=False)
    id: str = Field(description="Provider-assigned identifier used to match the tool result")
    name: str = Field(description="Name of the tool requested by the model")
    arguments: str = Field(description="Exact JSON argument string emitted by the model")


class AssistantText(BaseModel):
    """One assistant message, with immutable adapter-only JSON extensions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["text"] = "text"
    text: str
    native: NativeJSON | None = Field(default=None, repr=False)


class AssistantReasoning(BaseModel):
    """Readable reasoning and optional immutable adapter-only JSON extensions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["reasoning"] = "reasoning"
    text: str = ""
    native: NativeJSON | None = Field(default=None, repr=False)


AssistantPart = Annotated[
    AssistantText | ToolCall | AssistantReasoning, Field(discriminator="kind")
]


class LLMUsage(BaseModel):
    """Normalized usage reported for one successful LLM response."""

    input_tokens: int = Field(default=0, description="Total input tokens reported by the provider")
    output_tokens: int = Field(
        default=0, description="Total output tokens reported by the provider"
    )
    cached_input_tokens: int = Field(
        default=0, description="Input tokens read from the provider's prompt cache"
    )
    cache_write_input_tokens: int = Field(
        default=0, description="Input tokens written to the provider's prompt cache"
    )
    reasoning_tokens: int = Field(
        default=0, description="Output tokens attributed to reasoning by the provider"
    )
    total_tokens: int = Field(
        default=0, description="Total input and output tokens reported by the provider"
    )
    cost_usd: float = Field(
        default=0.0, description="Estimated call cost in US dollars, when available"
    )

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
                first(value, "cache_write_input_tokens", "cache_creation_input_tokens")
                or first(
                    prompt_details,
                    "cache_write_tokens",
                    "cache_write_input_tokens",
                    "cache_creation_input_tokens",
                )
                or 0
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
    """Response produced by UnifiedLLM and stored by NOOA.

    UnifiedLLM creates a fresh object for every call. The runtime records that
    same object; renderers project its conversational fields while telemetry
    consumers read its model and usage metadata.

    Parts are immutable; model_copy also strips native state on part edits
    because Pydantic's frozen fields alone do not protect copy(update=...).

    Pass this object directly back in a client's message list. Mapping access
    exposes public values only; replace the list element with dict(response)
    to edit it without native state. Nested projected containers are detached:
    editing them alone does not modify this response.
    """

    _role: ClassVar[Role] = Role.ASSISTANT

    raw_response: Any = Field(
        default=None,
        exclude=True,
        repr=False,
        description=(
            "Live provider SDK response; excluded from persistence because it is "
            "provider-specific, may not be serializable, and duplicates normalized fields"
        ),
    )
    parts: tuple[AssistantPart, ...] = Field(default=(), frozen=True, repr=False)
    replay_scope: str | None = Field(default=None, frozen=True, repr=False)
    parsed: Any = Field(
        default=None,
        exclude=True,
        repr=False,
        description=(
            "Live typed return value; excluded from persistence because arbitrary Python "
            "objects are not a durable wire format (the source JSON remains in content "
            "or provider-exposed reasoning)"
        ),
    )
    finish_reason: Literal["stop", "tool_calls", "length", "error"] = Field(
        default="stop",
        repr=False,
        description=(
            "NOOA-normalized outcome: stop, tool_calls, length, or error; provider-specific "
            "finish reasons are deliberately collapsed into these four portable values"
        ),
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
        description=(
            "Snapshot of the trailing dynamic context block included in the request that "
            "produced this response, retained for session export and debugging"
        ),
    )

    @field_validator("usage", mode="before")
    @classmethod
    def _normalize_usage(cls, value: Any) -> LLMUsage | None:
        return LLMUsage.from_provider(value)

    @model_validator(mode="before")
    @classmethod
    def _separate_parsed_content(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, dict):
            return value
        if value.get("llm_state") is not None and not (info.context or {}).get("archive"):
            raise ValueError(
                "llm_state is removed: use ordered parts and replay_scope. "
                "Legacy opaque state is discarded only by explicit archive loading."
            )
        value = dict(value)
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
        if "parts" not in value:
            # Flat archives have no trustworthy ordering. Migrate portable data
            # only; opaque state from the old representation is intentionally lost.
            parts: list[AssistantPart] = []
            if value.get("reasoning"):
                parts.append(AssistantReasoning(text=value["reasoning"]))
            parts.append(AssistantText(text=value.get("content", "")))
            parts.extend(
                ToolCall.model_validate(call).model_copy(update={"native": None})
                for call in value.get("tool_calls", [])
            )
            value["parts"] = tuple(parts)
            value["replay_scope"] = None
        for field in ("content", "tool_calls", "reasoning", "llm_state"):
            value.pop(field, None)
        return value

    @property
    def content(self) -> Annotated[str, spec(max_string=None)]:
        return "".join(part.text for part in self.parts if isinstance(part, AssistantText))

    def __instance_values__(self) -> dict[str, Any]:
        """Display readable response values, not the archived provider parts."""
        values = super().__instance_values__()
        values.update(
            (name, value)
            for name in ("content", "reasoning", "tool_calls")
            if (value := getattr(self, name))
        )
        return values

    @property
    def reasoning(self) -> str | None:
        return (
            "\n".join(
                part.text
                for part in self.parts
                if isinstance(part, AssistantReasoning) and part.text
            )
            or None
        )

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [part for part in self.parts if isinstance(part, ToolCall)]

    @property
    def replay_tool_calls(self) -> tuple[ToolCall, ...]:
        """Incomplete or malformed calls remain observable, not executable history."""
        calls = tuple(self.tool_calls)
        if self.finish_reason != "tool_calls":
            return ()
        for call in calls:
            try:
                if not isinstance(json.loads(call.arguments), dict):
                    return ()
            except json.JSONDecodeError:
                return ()
        return calls

    @property
    def is_empty(self) -> bool:
        return not (
            self.replay_content.strip()
            or self.replay_tool_calls
            or self.reasoning
            or any(part.native for part in self.parts)
        )

    def searchable_fields(self) -> dict[str, Any]:
        public = self.model_dump(exclude={"parts"})
        public.update(self.public_message())
        return public

    def render_message(self, content, tool_calls, *, reasoning):
        """Preserve the turn unless rendering changed its public parts."""
        if (
            reasoning == self.reasoning
            and content == self.content
            and tuple((call.id, call.name, call.arguments) for call in tool_calls)
            == tuple((call.id, call.name, call.arguments) for call in self.tool_calls)
        ):
            return self
        return assistant_message(content, tool_calls=tool_calls, reasoning=reasoning)

    @cached_property
    def _public_projection(self):
        """Cache immutable public values, never mutable caller-owned containers."""
        return freeze(
            assistant_message(self.content, tool_calls=self.tool_calls, reasoning=self.reasoning)
        )

    def __getitem__(self, key: str) -> Any:
        return json_containers(self._public_projection[key])

    def __iter__(self):
        return iter(self._public_projection)

    def __len__(self) -> int:
        return len(self._public_projection)

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self._public_projection else default

    def keys(self):
        return self._public_projection.keys()

    def items(self):
        return Mapping.items(self)

    def values(self):
        return Mapping.values(self)

    def __contains__(self, key: object) -> bool:
        return key in self._public_projection

    def __setitem__(self, key, value):
        raise TypeError(
            "LLMResponse is read-only. Replace the history element with "
            "a public message dict to edit it and discard native replay state."
        )

    def __delitem__(self, key):
        self.__setitem__(key, None)

    def __snapshot_data__(self) -> dict[str, Any]:
        """Durable JSON, excluding live SDK objects and honoring native serializers."""
        return self.model_dump(mode="json")

    @classmethod
    def __restore_snapshot__(cls, data: dict[str, Any]) -> LLMResponse:
        """Migrate flat archives before a generic loader filters unknown fields."""
        return cls.model_validate(data, context={"archive": True})

    def replace_parts(self, parts: tuple[AssistantPart, ...]) -> LLMResponse:
        """Edit public content without inheriting private provider state.

        Metadata stays associated with the originating event. The stored event
        is untouched; only this request's replacement loses opaque extensions.
        """
        return self.model_copy(update={"parts": parts})

    def model_copy(self, *, update=None, deep=False):
        # Pydantic's normal model_copy bypasses frozen fields and validation.
        # Metadata-only copies can share the turn; public edits cannot share
        # its native state, even when a caller uses model_copy directly.
        if (
            update
            and {"parts", "replay_scope", "content", "reasoning", "tool_calls"} & update.keys()
        ):
            if {"content", "reasoning", "tool_calls"} & update.keys():
                raise TypeError("Edit an assistant turn with replace_parts() or replace_text().")
            update = {
                **update,
                "parts": tuple(
                    part.model_copy(update={"native": None})
                    for part in update.get("parts", self.parts)
                ),
                "replay_scope": None,
                "raw_response": None,
                "parsed": None,
                "metadata": dict(update.get("metadata", self.metadata)),
            }
        result = super().model_copy(update=update, deep=deep)
        if update and "parts" in update:
            result.__dict__.pop("_public_projection", None)
        return result

    def replace_text(self, text: str) -> LLMResponse:
        """Replace visible text, retaining readable reasoning and public tool calls."""
        parts = [part for part in self.parts if not isinstance(part, AssistantText)]
        parts.insert(0, AssistantText(text=text))
        return self.replace_parts(tuple(parts))

    def public_message(self) -> dict[str, Any]:
        """On-demand, non-replay projection for relay, tracing, and token counting."""
        return json_containers(self._public_projection)

    @property
    def replay_content(self) -> str:
        """Alias of content for the formatter's event-to-message interface."""
        return self.content


# Register the public read-only protocol without replacing Pydantic's durable
# model serializer. dict(response) is portable; model_dump() is an archive.
Mapping.register(LLMResponse)
