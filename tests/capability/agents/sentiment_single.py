# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single sentiment classification agent."""

from typing import Annotated, Literal

from nooa import Agent


class SentimentSingleAgent(Agent):
    """You are specialist for sentiment classification."""

    async def classify(
        self, text: Annotated[str, "The text to classify"]
    ) -> Literal["positive", "negative", "neutral"]:
        """Classify the sentiment of the text."""
        ...
