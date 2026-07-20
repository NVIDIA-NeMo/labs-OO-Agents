# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 02: Structured output — Pydantic return types with auto-retry.

uv run python examples/quickstart/02_structured_outputs.py
"""

from typing import Literal

from nooa.util.quickstart import *


class FeedbackAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    topics: list[str]
    urgency: Literal["low", "medium", "high"]
    summary: str
    confidence: float = Field(ge=0, le=1)  # Pydantic constraints enforced!


class FeedbackAgent(Agent, llm=llm):
    """Agent for analyzing customer feedback with structured output."""

    async def analyze_feedback(self, text: str) -> FeedbackAnalysis:
        """Analyze customer feedback comprehensively."""
        ...


@autorun
async def main():
    agent = FeedbackAgent()
    result = await agent.analyze_feedback("Broken feature, needs immediate fix!")
    print(result)  # Guaranteed valid FeedbackAnalysis instance
