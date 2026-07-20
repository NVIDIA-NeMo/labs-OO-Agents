# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test agent with asyncio alias."""

import asyncio as aio

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


class AgentAsyncioAlias(Agent, llm=FakeLLMClient()):
    """Agent with asyncio alias."""

    async def run(self):
        await aio.sleep(0.1)
        return "done"
