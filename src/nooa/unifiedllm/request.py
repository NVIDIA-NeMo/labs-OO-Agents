# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The provider-independent request sent to an adapter."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.unifiedllm.unifiedllm import Tool


class LLMRequest(BaseModel):
    """One request: the history, tools and the wanted output.

    ``messages`` and ``tools`` are kept as given (not copied or validated),
    so earlier LLMResponse objects are passed back unchanged. Model, endpoint,
    key and provider settings never go here; they belong to the client.

    While the clients still accept LiteLLM keyword arguments, ``tool_choice``,
    ``reasoning_level`` and ``max_output_tokens`` record what the call asked
    for; the wire values still come from those keyword arguments.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: SkipValidation[list[dict[str, Any] | LLMResponse | CacheBoundary]] = Field(
        default_factory=list
    )
    tools: SkipValidation[list[Tool]] = Field(default_factory=list)
    # A str other than the literals names a tool. A mapping is LiteLLM's own
    # tool_choice form, accepted until callers stop passing it.
    tool_choice: Literal["auto", "required", "none"] | str | dict[str, Any] = "auto"
    output_type: SkipValidation[type[BaseModel] | None] = None
    reasoning_level: str | None = None
    max_output_tokens: int | None = None
