# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405

from nooa.util.quickstart import *


class ConversationAgent(Agent, llm=llm):
    """Agent demonstrating that event history persists across method calls."""

    async def greet(self, name: str) -> str:
        """Greet the user by name."""
        ...

    async def recall(self) -> str:
        """What was the name of the person you just greeted?"""
        ...


@autorun
async def main():
    agent = ConversationAgent()

    # First call — introduces the name into the event history
    greeting = await agent.greet("Alice")
    print(f"Greeting: {greeting}")

    # Second call — no name passed, but the agent remembers from event history
    memory = await agent.recall()
    print(f"Recall:   {memory}")
