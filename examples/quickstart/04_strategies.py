# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405,F401
"""Quickstart 04: Strategies — PredictStrategy vs CodeActStrategy.

uv run python examples/quickstart/04_strategies.py
"""

import os  # for LLM exec_globals (visible by default)
from typing import Annotated

from nooa.config import CodeActConfig
from nooa.tools import ShellTools
from nooa.util.quickstart import *


class AnalysisAgent(Agent, llm=llm):
    """Agent demonstrating different strategy options."""

    shell = ShellTools()  # External tool - LLM can call this too

    @strategy(PredictStrategy())
    async def classify_sentiment(self, text: str) -> str:
        """Classify as positive, negative, or neutral."""
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=10)))
    async def perform_task(
        self,
        request: str,
    ) -> Annotated[str, "Your answer"]:
        """Perform the task requested by the user and provide a friendly response."""
        ...


@autorun
async def main():
    agent = AnalysisAgent()

    # PredictStrategy (fast, single-shot)
    sentiment = await agent.classify_sentiment("I love this product! Best purchase ever!")
    print("Sentence: I love this product! Best purchase ever!")
    print(f"Sentiment: {sentiment}\n")

    # CodeActStrategy (can execute Python code, call methods, iterate)
    requests = [
        "What is the current working directory?",
        "List the files in the current directory.",
    ]
    for request in requests:
        response = await agent.perform_task(request=request)
        print(f"Request: {request}")
        print(f"Response: {response}\n")
