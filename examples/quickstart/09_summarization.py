# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 09: Summarization — automatic history compression with TokenBudgetSummarizer.

uv run python examples/quickstart/09_summarization.py
"""

from nooa.agentdoc import spec
from nooa.agents import TokenBudgetSummarizer
from nooa.config import TokenBudgetConfig
from nooa.util.quickstart import *


class InterviewAgent(Agent, llm=llm):
    """A technical interviewer conducting a multi-turn conversation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        spec(self, "events", hidden=False)  # Expose event querying

    async def ask(self, candidate_answer: str) -> str:
        """Continue the technical interview based on the candidate's latest answer.

        Ask a relevant follow-up or move to a new topic as appropriate.
        Keep track of what has already been covered.
        """
        ...

    async def evaluate(self) -> str:
        """Based on the full interview so far, provide a brief candidate evaluation."""
        ...


ANSWERS = [
    "I have 3 years of Python experience, mostly building REST APIs.",
    "I use FastAPI for most projects — I like its async support and automatic OpenAPI docs.",
    "For databases, I usually go with PostgreSQL and SQLAlchemy as the ORM.",
    "I handle errors with try/except blocks and return proper HTTP status codes.",
    "For testing I write unit tests with pytest and mock out external dependencies.",
    "I've used Redis for caching and Celery for background task queues.",
    "I deploy on AWS — mostly ECS for long-running services and Lambda for event-driven jobs.",
]


@autorun
async def main():
    agent = InterviewAgent()
    TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig(max_tokens=1000))

    for i, answer in enumerate(ANSWERS, 1):
        await agent.ask(answer)
        n = len(agent.events.query())
        print(f"  {i:<6}  {n:>14}")

    evaluation = await agent.evaluate()
    print(f"\nEvaluation: {evaluation}")
