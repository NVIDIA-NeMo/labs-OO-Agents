# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test agent with explicit import asyncio."""

import asyncio

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


class AgentWithImportAsyncio(Agent, llm=FakeLLMClient()):
    """Agent with explicit import asyncio."""

    async def run(self):
        await asyncio.sleep(0.1)
        return "done"
