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
    """Select serialization independently of context-view policy.

    ``block_formatter`` wraps the already-selected blocks and serializes events
    (XML, Markdown, or an application formatter). ``provider_formatter`` adapts
    neutral messages into client input, retaining replay objects and boundaries.
    Neither chooses history membership, global order, eviction, or caching.
    Python-value serialization belongs to ``context_text()`` or application code;
    final provider encoding and native replay belong to the client.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    block_formatter: BlockFormatter = Field(default_factory=CachedBlockFormatter)
    provider_formatter: ProviderFormatter = Field(default_factory=OpenAIProviderFormatter)
