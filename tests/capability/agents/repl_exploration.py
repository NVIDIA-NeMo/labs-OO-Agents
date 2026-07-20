# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REPL Exploration Test Agent. Uses REPL exploration to find a secret message."""

from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CodeActConfig


class ReplExplorationTestAgent(Agent):
    """You are an agent that must retrieve the secret message."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=20)))
    async def retrieve_secret_message(self) -> str:
        """Retrieve the secret message."""
        ...

    async def first_riddle(self) -> str:
        """You are given the first riddle for free."""
        return "What is the capital of France?"

    async def second_riddle(self, first_riddle_answer: str) -> str:
        """You are given the second riddle only if the given first riddle answer is correct."""
        if "paris" not in first_riddle_answer.lower():
            raise Exception("The first riddle answer is incorrect.")
        return "What is the name of our galaxy?"

    async def third_riddle(self, second_riddle_answer: str) -> str:
        """You are given the third riddle only if the given second riddle answer is correct."""
        if "milky way" not in second_riddle_answer.lower():
            raise Exception("The second riddle answer is incorrect.")
        return "What is the name of the planet you are on?"

    async def fourth_riddle(self, third_riddle_answer: str) -> str:
        """You are given the fourth riddle only if the given third riddle answer is correct."""
        if "earth" not in third_riddle_answer.lower():
            raise Exception("The third riddle answer is incorrect.")
        return "What is the name of the language you are speaking?"

    async def secret_message(self, fourth_riddle_answer: str) -> str:
        """You are given the secret message only if the given fourth riddle answer is correct."""
        if "english" not in fourth_riddle_answer.lower():
            raise Exception("The fourth riddle answer is incorrect.")
        return "You are a genius!"
