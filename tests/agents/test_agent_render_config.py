# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Agent.__init__ render_config= parameter (Task 17)."""

from nooa.context_blocks.formatter import MarkdownBlockFormatter
from nooa.context_blocks.render_config import RenderConfig
from nooa.context_blocks.renderers import CachedBlockFormatter
from nooa.unifiedllm import FakeLLMClient


def make_llm():
    return FakeLLMClient.with_tool_call("return_result", {"result": "done"})


def test_agent_accepts_render_config():
    from nooa import Agent

    rc = RenderConfig(block_formatter=MarkdownBlockFormatter())

    class MyAgent(Agent, llm=make_llm()):
        pass

    agent = MyAgent(render_config=rc)
    assert agent.render_config is rc
    assert isinstance(agent.render_config.block_formatter, MarkdownBlockFormatter)


def test_agent_default_render_config():
    from nooa import Agent

    class MyAgent(Agent, llm=make_llm()):
        pass

    agent = MyAgent()
    assert isinstance(agent.render_config, RenderConfig)
    assert isinstance(agent.render_config.block_formatter, CachedBlockFormatter)
