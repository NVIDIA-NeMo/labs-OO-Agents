# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test agent with multiple imports."""

import json  # noqa: F401
from asyncio import sleep  # noqa: F401
from datetime import datetime  # noqa: F401

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


class AgentMultipleImports(Agent, llm=FakeLLMClient()):
    """Agent with multiple imports."""

    pass
