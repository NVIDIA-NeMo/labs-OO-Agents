# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large data handling agent - tests truncation system."""

from nooa import Agent


class LargeDataAgent(Agent):
    """You are an agent that processes very large text data efficiently.

    You will receive text that may be extremely large (hundreds of KB or more).
    You must handle it smartly without trying to print or display the entire content.
    """

    async def find_marker(self, data: str, marker: str) -> dict[str, int | bool]:
        """Find a marker string in potentially very large data.

        Args:
            data: A potentially very large text (may be 100KB-1MB+)
            marker: A marker string to find in the data

        Returns:
            dict with:
                - found: True if marker was found, False otherwise
                - position: Index where marker was found (-1 if not found)
                - data_length: Length of the data received
        """
        ...

    async def extract_around_marker(self, data: str, marker: str, context_chars: int = 50) -> str:
        """Extract text around a marker from large data.

        Args:
            data: A potentially very large text
            marker: A marker string to find
            context_chars: Number of characters to extract before and after marker

        Returns:
            Extracted text centered around the marker, or empty string if not found.
            Should return approximately context_chars before + marker + context_chars after.
        """
        ...

    async def count_pattern(self, data: str, pattern: str) -> dict[str, int]:
        """Count occurrences of a pattern in large data.

        Args:
            data: A potentially very large text
            pattern: Pattern string to count

        Returns:
            dict with:
                - count: Number of times pattern appears
                - data_length: Length of the data
        """
        ...
