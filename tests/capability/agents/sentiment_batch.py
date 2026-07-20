# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch sentiment classification agent."""

from typing import Annotated, Literal

from nooa import Agent
from nooa.tools.method_writing_lib import MethodWriting


class SentimentBatchAgent(Agent):
    """You are an agent that classifies sentiment of multiple texts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method_writing = MethodWriting()

    async def classify(
        self, texts: Annotated[list[str], "The texts to classify"]
    ) -> list[Literal["positive", "negative", "neutral"]]:
        """Classify the sentiment of multiple texts."""
        ...
