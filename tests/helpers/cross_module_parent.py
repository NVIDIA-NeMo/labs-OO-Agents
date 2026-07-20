# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parent Agent + module-level symbols for cross-module inheritance tests (issue 259).

The child agent lives in ``cross_module_child.py`` (a DIFFERENT module). These
module-level names must reach the child's generated-code globals.
"""

from __future__ import annotations

from typing import Annotated

from nooa import Agent, hidden, strategy
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient

# Visible module-level symbols — must appear in a child agent's exec_globals.
SHARED_CONSTANT = 7
COLLIDING_NAME = "from-parent"  # child redefines this; leaf must win


def shared_util(x: int) -> int:
    """Double x — a parent-module helper the child should be able to call."""
    return x * 2


class ParentModel:
    """A parent-module type that should surface as an available type."""

    def __init__(self, value: int):
        self.value = value


# Hidden module-level symbols — must NOT leak into exec_globals.
API_KEY: Annotated[str, hidden] = "secret-key"


@hidden
def hidden_parent_util(x: int) -> int:
    return x


with hidden:
    HIDDEN_PARENT_SECRET = "should-not-leak"  # noqa: S105 - intentional test fixture


_PARENT_LLM = FakeLLMClient()


class ParentAgent(Agent, llm=_PARENT_LLM):
    """Parent agent defined in its own module."""

    @strategy(CodeActStrategy())
    async def parent_task(self) -> int:
        """Use shared_util and SHARED_CONSTANT from this module."""
        ...
