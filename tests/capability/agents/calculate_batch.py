# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch calculation agent - interprets natural language math for multiple inputs."""

from nooa import Agent
from nooa.tools.method_writing_lib import MethodWriting


class CalculateBatchAgent(Agent):
    """You are an agent that performs multiple calculations from natural language."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method_writing = MethodWriting()

    async def calculate(self, items: list[dict]) -> list[int | float]:
        """Perform multiple calculations described in natural language.

        Args:
            items: List of calculation items, each with:
                   - a: First number (int or float)
                   - b: Second number (int or float)
                   - calculation: Natural language description

        Returns:
            List of computed results.
        """
        ...
