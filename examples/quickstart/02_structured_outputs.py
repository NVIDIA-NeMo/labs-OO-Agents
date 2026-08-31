# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 02: Structured output — Pydantic return types with auto-retry.

uv run python examples/quickstart/02_structured_outputs.py
"""

from typing import Literal

from nooa.util.quickstart import *


class FeedbackAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall tone of the feedback."
    )
    topics: list[str] = Field(description="Specific subjects mentioned by the customer.")
    urgency: Literal["low", "medium", "high"] = Field(
        description="How quickly the feedback needs a response."
    )
    summary: str = Field(description="One-sentence faithful summary.")
    confidence: float = Field(ge=0, le=1, description="Confidence in the analysis from 0 to 1.")


class FeedbackAgent(Agent, llm=llm):
    """Agent for analyzing customer feedback with structured output."""

    @strategy(PredictStrategy())
    async def analyze_feedback(self, text: str) -> FeedbackAnalysis:
        """Analyze customer feedback comprehensively."""
        ...


@autorun
async def main():
    agent = FeedbackAgent()
    result = await agent.analyze_feedback("Broken feature, needs immediate fix!")
    print(result)  # Guaranteed valid FeedbackAnalysis instance
