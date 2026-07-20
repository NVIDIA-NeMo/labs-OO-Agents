# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send agent traces to Langfuse via OTLP.

Langfuse is an open-source LLM observability platform.  The ``exporters.langfuse()``
factory reads connection details from environment variables and sends spans over
OTLP/HTTP using Basic auth.

Prerequisites:
    - Langfuse server running (self-hosted or cloud)
    - Environment variables set:
        LANGFUSE_HOST          — e.g. https://cloud.langfuse.com
        LANGFUSE_PUBLIC_KEY    — project public key
        LANGFUSE_SECRET_KEY    — project secret key

Usage:
    uv run python examples/advanced/tracing_langfuse.py
"""

import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel

from examples.util.example_llm import qwen
from nooa import Agent

load_dotenv(override=True)


class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str


class PersonAnalysis(BaseModel):
    category: str
    insights: list[str]
    score: float


class SentimentAgent(Agent, llm=qwen):
    async def sentiment_analysis(self, text: str) -> str:
        """Analyze the sentiment of the text.

        Return one of: 'positive', 'negative', or 'neutral'
        """
        ...

    async def compute(self, instructions: str, numbers: list[int]) -> float:
        """Perform a computation based on the instructions and a list of numbers."""
        ...

    async def analyze_person(self, person: PersonInfo) -> PersonAnalysis:
        """Analyze a person's profile and provide structured insights.

        Given person.name, person.age, and person.occupation:
        - Categorize them into a demographic/professional category
        - Provide 2-3 insightful observations about their career stage
        - Assign a career potential score from 0.0 to 10.0

        Return a PersonAnalysis object with category, insights list, and score.
        """
        ...


async def main() -> None:
    from nooa.tracing import enable_tracing, exporters

    print("=" * 80)
    print("Langfuse Tracing Example")
    print("=" * 80)

    enable_tracing(exporters=[exporters.langfuse()])
    ui_url = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    agent = SentimentAgent()

    for sentence in [
        "I love this product!",
        "I hate this product!",
        "I'm neutral about this product.",
    ]:
        result = await agent.sentiment_analysis(sentence)
        print(f"  '{sentence}' -> {result}")

    result = await agent.compute("Add the numbers", [1, 2, 3, 4, 5])
    print(f"  'Add the numbers' -> {result}")

    person = PersonInfo(name="Alice Smith", age=28, occupation="Software Engineer")
    analysis = await agent.analyze_person(person)
    print(f"\n  Person Analysis: {analysis.category} (score={analysis.score})")

    print(f"\nView traces: {ui_url}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
