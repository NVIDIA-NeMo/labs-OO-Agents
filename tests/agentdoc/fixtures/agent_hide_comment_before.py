# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test agent with hide comment on line before import."""

import json  # noqa: F401

# agentdoc: hide
import os  # noqa: F401

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


class AgentHideCommentBefore(Agent, llm=FakeLLMClient()):
    """Agent with comment before import."""

    def run(self):
        return json.dumps({"test": "data"})
