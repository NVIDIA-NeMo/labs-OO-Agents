# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capability test agents for eval_pipeline.

These agents are used to test model capabilities via the evaluation pipeline.
"""

from nooa import Agent


class SentimentAgent(Agent):
    """Agent that classifies sentiment of text."""

    async def classify_single(self, text: str) -> str:
        """Classify the sentiment of a single text.

        Args:
            text: The text to classify

        Returns:
            One of: "positive", "negative", "neutral"
        """
        ...

    async def classify_batch(self, texts: list[str]) -> list[str]:
        """Classify the sentiment of multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of sentiment labels ("positive", "negative", "neutral")
        """
        ...


class CalculateAgent(Agent):
    """Agent that performs calculations."""

    async def calculate_single(self, a: int, b: int) -> int:
        """Multiply two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            The product a * b
        """
        ...

    async def calculate_batch(self, pairs: list[tuple[int, int]]) -> list[int]:
        """Multiply multiple pairs of numbers.

        Args:
            pairs: List of (a, b) tuples

        Returns:
            List of products [a1*b1, a2*b2, ...]
        """
        ...
