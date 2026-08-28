# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral public contracts for LLM clients."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class LLMUsage:
    """Passive, provider-reported usage (never used for runtime control)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class LLMToolCallChunk:
    """Provider-neutral partial tool call emitted during streaming."""

    index: int = 0
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass(frozen=True)
class LLMChunk:
    """One backend-neutral streaming increment."""

    content: str = ""
    reasoning: str | None = None
    tool_calls: tuple[LLMToolCallChunk, ...] = ()
    finish_reason: str | None = None
    usage: LLMUsage | None = None


@dataclass(frozen=True)
class LLMRequestDefaults:
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: str | tuple[str, ...] | None = None
    reasoning_effort: str | None = None
    cache_key: str | None = None
    timeout: float | None = None


@dataclass(frozen=True)
class LLMCallOptions(LLMRequestDefaults):
    stream: bool = False
    tool_choice: str | dict[str, JSONValue] | None = None
    parallel_tool_calls: bool | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Structural public contract implemented by real and fake clients."""

    model: str

    @property
    def context_window(self) -> int | None: ...
    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> Any: ...
    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> Any: ...
    def stream(
        self, messages: list[dict[str, Any]], tools: list[Any] | None = None, **kwargs: Any
    ) -> Iterator[LLMChunk]: ...
    def astream(
        self, messages: list[dict[str, Any]], tools: list[Any] | None = None, **kwargs: Any
    ) -> AsyncIterator[LLMChunk]: ...
    def close(self) -> None: ...
    async def aclose(self) -> None: ...
