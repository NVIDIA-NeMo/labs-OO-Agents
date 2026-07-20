# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test agent with hide on from import."""

import math  # noqa: F401
from collections import Counter  # noqa: F401
from datetime import datetime  # agentdoc: hide  # noqa: F401

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


class AgentHideFromImport(Agent, llm=FakeLLMClient()):
    """Agent with hide on from import."""

    def run(self):
        return math.sqrt(16) + len(Counter([1, 2, 3]))
