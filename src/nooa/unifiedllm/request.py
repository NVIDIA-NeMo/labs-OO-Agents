# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The provider-independent request sent to an adapter."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.unifiedllm.tools import Tool


class LLMRequest(BaseModel):
    """One request: the history, tools and the wanted output.

    ``messages`` and ``tools`` are kept as given (not copied or validated),
    so earlier LLMResponse objects are passed back unchanged. Model, endpoint,
    key and provider settings never go here; they belong to the client.

    Settings are fixed at load. Unknown and removed settings are rejected.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    messages: SkipValidation[list[dict[str, Any] | LLMResponse | CacheBoundary]] = Field(
        default_factory=list
    )
    tools: SkipValidation[list[Tool]] = Field(default_factory=list)
    tool_choice: Literal["auto", "required", "none"] | str = "auto"
    output_type: SkipValidation[type[BaseModel] | None] = None
