# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Child Agent inheriting from a parent in a DIFFERENT module (issue 259)."""

from __future__ import annotations

from nooa import strategy
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient
from tests.helpers.cross_module_parent import ParentAgent

# Child-module symbols.
CHILD_CONSTANT = 99
COLLIDING_NAME = "from-child"  # collides with the parent module; leaf must win


_CHILD_LLM = FakeLLMClient()


class ChildAgent(ParentAgent, llm=_CHILD_LLM):
    """Child agent in its own module; inherits ParentAgent from another module."""

    @strategy(CodeActStrategy())
    async def child_task(self) -> int:
        """Run code that relies on parent-module module-level symbols."""
        ...
