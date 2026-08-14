# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM capability requirements used by strategy-aware client selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class LLMRequirements:
    """Capabilities a strategy needs from the LLM request path."""

    function_tools: bool = False
    structured_result: bool = False
    multi_turn_tools: bool = False
    reasoning: Literal["preserve_model_default", "none"] | None = None
