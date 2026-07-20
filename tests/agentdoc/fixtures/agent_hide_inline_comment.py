# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test agent with inline hide comment that shouldn't affect next import."""

import math  # noqa: F401
import sys  # agentdoc: hide  # noqa: F401

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


class AgentHideInlineComment(Agent, llm=FakeLLMClient()):
    """Agent with inline hide comment."""

    def run(self):
        return math.sqrt(16)
