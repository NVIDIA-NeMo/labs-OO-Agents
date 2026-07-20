# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Needle-in-haystack test agent."""

from nooa import Agent


class NeedleTestWrapper(Agent):
    """You are an agent that can find the needle in the haystack."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data1: list[str] = []
        self.data2: list[str] = []
        self.data3: list[str] = []

    async def _call_agent(self, data1: list[str], data2: list[str], data3: list[str]) -> str:
        """Set the data for the needle test and find the negative sentiment.

        Args:
            data1: List of sentences
            data2: List of sentences
            data3: List of sentences

        Returns:
            The sentence with negative sentiment.
        """
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        return await self.find_negative_sentiment()

    async def find_negative_sentiment(self) -> str:
        """Find and return the single sentence with negative sentiment in the data (
        self.data1, self.data2, self.data3). Return the sentence exactly as written."""
        ...
