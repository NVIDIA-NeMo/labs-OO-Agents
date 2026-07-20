# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarizer configuration for TokenBudgetSummarizer and MethodSummarizer."""

from pydantic import BaseModel, ConfigDict


class TokenBudgetConfig(BaseModel):
    """Config for TokenBudgetSummarizer.

    Set via: TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig(...))
    """

    model_config = ConfigDict(frozen=True)

    max_tokens: int = 100_000
    preserve_recent: int = 10
    target_chars: int = 1000

    def merge_with(self, other: "TokenBudgetConfig") -> "TokenBudgetConfig":
        if not other.model_fields_set:
            raise ValueError(
                "merge_with() received a config with no model_fields_set. "
                "Was it constructed from model_dump() or model_validate()? "
                "Config objects must be freshly constructed: TokenBudgetConfig(field=value)."
            )
        return self.model_copy(update={k: getattr(other, k) for k in other.model_fields_set})


class MethodSummarizerConfig(BaseModel):
    """Config for MethodSummarizer.

    Set via: MethodSummarizer.install(agent, config=MethodSummarizerConfig(...))
    """

    model_config = ConfigDict(frozen=True)

    min_events: int = 3
    exclude_root: bool = True
    target_chars: int = 1000

    def merge_with(self, other: "MethodSummarizerConfig") -> "MethodSummarizerConfig":
        if not other.model_fields_set:
            raise ValueError(
                "merge_with() received a config with no model_fields_set. "
                "Was it constructed from model_dump() or model_validate()? "
                "Config objects must be freshly constructed: MethodSummarizerConfig(field=value)."
            )
        return self.model_copy(update={k: getattr(other, k) for k in other.model_fields_set})
