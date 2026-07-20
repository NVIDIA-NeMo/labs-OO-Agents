# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, ConfigDict, Field

from nooa.context_blocks.formatter import (
    BlockFormatter,
    OpenAIProviderFormatter,
    ProviderFormatter,
)
from nooa.context_blocks.renderers import CachedBlockFormatter


class RenderConfig(BaseModel):
    """Controls how context blocks are formatted and how messages are assembled.

    block_formatter: How system prompt blocks are serialized (XML or Markdown).
    provider_formatter: How the message list is assembled for the LLM provider.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    block_formatter: BlockFormatter = Field(default_factory=CachedBlockFormatter)
    provider_formatter: ProviderFormatter = Field(default_factory=OpenAIProviderFormatter)
