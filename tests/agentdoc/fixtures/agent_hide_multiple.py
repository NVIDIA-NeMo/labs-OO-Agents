# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test agent with multiple hide patterns."""

import json  # noqa: F401
import math  # noqa: F401

# agentdoc: hide
import os  # noqa: F401
import sys  # agentdoc: hide  # noqa: F401

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


class AgentHideMultiple(Agent, llm=FakeLLMClient()):
    """Agent with multiple hide patterns."""

    def run(self):
        return math.sqrt(16) + len(json.dumps({}))
