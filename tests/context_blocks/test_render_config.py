# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import BaseModel, ValidationError

from nooa.context_blocks.formatter import (
    MarkdownBlockFormatter,
    OpenAIProviderFormatter,
)
from nooa.context_blocks.render_config import RenderConfig
from nooa.context_blocks.renderers import CachedBlockFormatter


def test_render_config_is_pydantic_model():
    assert issubclass(RenderConfig, BaseModel)


def test_render_config_defaults():
    c = RenderConfig()
    assert isinstance(c.block_formatter, CachedBlockFormatter)
    assert isinstance(c.provider_formatter, OpenAIProviderFormatter)


def test_render_config_frozen():
    c = RenderConfig()
    with pytest.raises(ValidationError):
        c.block_formatter = MarkdownBlockFormatter()


def test_render_config_custom_formatters():
    c = RenderConfig(block_formatter=MarkdownBlockFormatter())
    assert isinstance(c.block_formatter, MarkdownBlockFormatter)
    assert isinstance(c.provider_formatter, OpenAIProviderFormatter)
