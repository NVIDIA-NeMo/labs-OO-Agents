# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send agent traces to Arize Phoenix via OTLP.

Phoenix is an open-source LLM observability platform from Arize AI.  This example
uses the generic ``exporters.otlp()`` factory since Phoenix accepts standard
OTLP/HTTP traces.

Prerequisites:
    - Phoenix server running (self-hosted or cloud)
    - Environment variables (optional):
        PHOENIX_HOST     — default: http://localhost:6006
        PHOENIX_API_KEY  — for authenticated endpoints

Usage:
    uv run python examples/advanced/tracing_phoenix.py
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

    host = os.getenv("PHOENIX_HOST", "http://localhost:6006")
    api_key = os.getenv("PHOENIX_API_KEY")
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}

    print("=" * 80)
    print("Phoenix Tracing Example")
    print("=" * 80)

    enable_tracing(
        exporters=[
            exporters.otlp(
                endpoint=f"{host.rstrip('/')}/v1/traces",
                headers=headers,
            ),
        ]
    )

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

    print(f"\nView traces: {host}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
