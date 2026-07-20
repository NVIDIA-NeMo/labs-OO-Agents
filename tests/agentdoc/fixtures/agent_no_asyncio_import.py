# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test agent without any asyncio imports."""

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


class AgentNoAsyncioImport(Agent, llm=FakeLLMClient()):
    """Agent without any asyncio imports."""

    def run(self):
        return "done"
