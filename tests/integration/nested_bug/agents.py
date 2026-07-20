# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent for reproducing nested history ordering bug.

This agent demonstrates the bug where nested agent calls cause tool_call_id
ordering issues in the message history.
"""

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies.codeact import CodeActStrategy


class NestedBugAgent(Agent):
    """Agent demonstrating the nested history bug.

    The bug occurs when outer_method's execute_python code calls inner_method.
    The tool_call_id ordering in history becomes invalid.
    """

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def outer_method(self, value: int) -> str:
        """Process a value by DELEGATING to inner_method.

        CRITICAL: You MUST call self.inner_method(value) to get the result.
        Do NOT compute the result yourself - you MUST delegate to inner_method.
        Required steps:

        1. Use execute_python to call: `result = await self.inner_method(value)` to get the processed value
        2. Then use execute_python again to return it wrapped by calling `return_result(self.wrap(result))`
        """
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3)))
    async def inner_method(self, x: int) -> str:
        """Process a single value and return a formatted string.

        use execute_python call `return_result(self.wrap(x))`
        """
        ...

    def wrap(self, x):
        """Wrap the variable and make it pretty."""
        return f"hello {x}!"
