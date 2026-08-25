# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light configuration models for interactive hosts."""

from typing import Literal

from pydantic import BaseModel


class SummarizationConfig(BaseModel):
    """Configuration for history summarization.

    ``max_tokens`` defaults to ``None`` meaning "80% of the LLM's context
    window, resolved at install time." The old 100K absolute was fine when
    models had ~200K context but fired at ~10% usage on 1M-context models
    like Opus 4.8, making summarization feel constant. Set an explicit
    integer to pin a specific threshold.
    """

    policy: Literal["token_budget", "none"] = "token_budget"
    max_tokens: int | None = None
    preserve_recent: int = 10
    target_chars: int = 4000


__all__ = ["SummarizationConfig"]
