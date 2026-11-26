# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nooa.context_blocks.formatter import (
    BlockFormatter,
    OpenAIProviderFormatter,
    ProviderFormatter,
)
from nooa.context_blocks.live_context import coerce_live_context
from nooa.context_blocks.renderers import CachedBlockFormatter


class RenderConfig(BaseModel):
    """Controls how context blocks are formatted and how messages are assembled.

    block_formatter: How system prompt blocks are serialized (XML or Markdown).
    provider_formatter: How the message list is assembled for the LLM provider.

    ``RenderConfig(live_context=...)`` builds the default
    :class:`CachedBlockFormatter` with that live-context policy (a policy
    object, its dict form or a bare kind name). The policy is the formatter's;
    read it back through :attr:`live_context`. Giving both ``live_context`` and
    ``block_formatter`` is an error.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    block_formatter: BlockFormatter = Field(default_factory=CachedBlockFormatter)
    provider_formatter: ProviderFormatter = Field(default_factory=OpenAIProviderFormatter)

    @model_validator(mode="before")
    @classmethod
    def _formatter_from_live_context(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or "live_context" not in data:
            return data
        data = dict(data)
        policy = data.pop("live_context")
        if policy is None:
            return data
        if "block_formatter" in data:
            raise ValueError("RenderConfig takes live_context or block_formatter, not both")
        data["block_formatter"] = CachedBlockFormatter(live_context=coerce_live_context(policy))
        return data

    @property
    def live_context(self) -> Any:
        """The formatter's live-context policy, or ``None`` for a formatter without one."""
        return getattr(self.block_formatter, "live_context", None)
