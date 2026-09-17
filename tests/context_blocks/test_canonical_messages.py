# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context rendering has one public message shape; transports project it later."""

import inspect

import nooa.context_blocks as context_blocks
from nooa.context_blocks import formatter
from nooa.context_blocks.models import RenderedMessage, Role
from nooa.context_blocks.render_config import RenderConfig
from nooa.context_blocks.renderer import render_context
from nooa.llm_types import CacheBoundary, LLMResponse


def test_canonical_conversion_preserves_replay_and_cache_objects():
    turn = LLMResponse(content="answer")
    boundary = CacheBoundary()
    messages = [
        RenderedMessage(role=Role.USER, content="question"),
        RenderedMessage(role=Role.ASSISTANT, content="answer", replay_message=turn),
        RenderedMessage(role=Role.METADATA, replay_message=boundary),
    ]
    result = formatter.to_messages(messages)
    assert result[0] == {"role": "user", "content": "question"}
    assert result[1] is turn
    assert result[2] is boundary


def test_no_provider_selection_in_context_rendering():
    assert "provider_formatter" not in inspect.signature(render_context).parameters
    assert "provider_formatter" not in RenderConfig.model_fields
    for name in (
        "ProviderFormatter",
        "OpenAIProviderFormatter",
        "AnthropicProviderFormatter",
        "ResponsesProviderFormatter",
    ):
        assert not hasattr(formatter, name)
        assert not hasattr(context_blocks, name)
