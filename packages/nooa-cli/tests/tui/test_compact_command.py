# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /compact command — event queue must survive summarization failure.

Regression tests for GitLab issue #146: /compact was clearing the event queue
when summarization failed, causing data loss.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa_cli.tui.commands import CommandHandler, CommandRegistry
from nooa_cli.tui.output import TextOutput


@pytest.fixture
def mock_frontend():
    frontend = AsyncMock()
    frontend.render = AsyncMock()
    frontend.get_input = AsyncMock(return_value="")
    frontend.start_thinking = AsyncMock()
    frontend.stop_thinking = AsyncMock()
    frontend.is_connected = True
    return frontend


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.default_model = "test-model"
    return config


@pytest.fixture
def mock_agent(mock_config):
    agent = MagicMock()
    agent._llm = MagicMock()
    agent.event_manager = MagicMock()
    agent.event_manager.clear = MagicMock()
    agent.event_manager.keys = MagicMock(return_value=["tag1", "tag2", "tag3"])
    agent.event_manager.collapse = MagicMock()
    agent.context_stats = MagicMock()
    agent.context_stats.total_tokens = 5000
    return agent


@pytest.fixture
def registry(mock_frontend, mock_config, mock_agent):
    return CommandRegistry(
        frontend=mock_frontend,
        config=mock_config,
        agent=mock_agent,
        skills_dirs=[Path(".cursor/skills")],
        mcp_file=Path(".mcp.json"),
    )


@pytest.fixture
def handler(registry, mock_frontend):
    return CommandHandler(registry=registry, frontend=mock_frontend)


# ============================================================================
# /compact — summarization failure must NOT clear the event queue (#146)
# ============================================================================


@pytest.mark.asyncio
async def test_compact_summarization_failure_preserves_events(handler, mock_agent):
    """When summarization raises, /compact must NOT call event_manager.clear().

    This is the core regression test for #146.
    """
    # Set up a summarizer that fails
    summarizer = MagicMock()
    summarizer._render_range_to_markdown = MagicMock(return_value="# History")
    summarizer.summarize = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    mock_agent._summarizers = [summarizer]

    result = await handler.handle("/compact")

    # Events must NOT be cleared
    mock_agent.event_manager.clear.assert_not_called()
    # The command should still succeed (it's a warning, not an error)
    assert result.success is True
    text_outputs = [o for o in result.outputs if isinstance(o, TextOutput)]
    assert any(
        "failed" in o.content.lower() or "Summarization failed" in o.content for o in text_outputs
    )
    # Should mention the error
    assert any("LLM unavailable" in o.content for o in text_outputs)


@pytest.mark.asyncio
async def test_compact_summarization_failure_does_not_set_compact_done(handler, mock_agent):
    """Failed summarization should not signal compact_done."""
    summarizer = MagicMock()
    summarizer._render_range_to_markdown = MagicMock(return_value="# History")
    summarizer.summarize = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    mock_agent._summarizers = [summarizer]

    result = await handler.handle("/compact")

    # compact_done should NOT be set on failure — renaming should not be triggered
    assert result.compact_done is False


@pytest.mark.asyncio
async def test_compact_summarization_success(handler, mock_agent):
    """Happy path: successful summarization collapses events."""
    summarizer = MagicMock()
    summarizer._render_range_to_markdown = MagicMock(return_value="# History")
    summarizer.config = MagicMock()
    summarizer.config.target_chars = 2000
    summarizer.summarize = AsyncMock(return_value="Summary of conversation")
    mock_agent._summarizers = [summarizer]
    # After collapse, fewer keys
    mock_agent.event_manager.keys = MagicMock(
        side_effect=[["tag1", "tag2", "tag3"], ["summary_tag"]]
    )

    result = await handler.handle("/compact")

    # Collapse should be called with the correct range
    mock_agent.event_manager.collapse.assert_called_once_with(
        "tag1", "tag3", "Summary of conversation"
    )
    mock_agent.event_manager.clear.assert_not_called()
    assert result.success is True
    assert result.compact_done is True
    text_outputs = [o for o in result.outputs if isinstance(o, TextOutput)]
    assert any("Compacted" in o.content for o in text_outputs)


@pytest.mark.asyncio
async def test_compact_no_summarizer_clears_events(handler, mock_agent):
    """Without a summarizer, /compact falls back to clearing events (expected behavior)."""
    mock_agent._summarizers = []

    result = await handler.handle("/compact")

    mock_agent.event_manager.clear.assert_called_once()
    assert result.success is True
    assert result.compact_done is True


@pytest.mark.asyncio
async def test_compact_empty_history(handler, mock_agent):
    """With no events, /compact shows info message."""
    mock_agent.event_manager.keys = MagicMock(return_value=[])

    result = await handler.handle("/compact")

    assert result.success is True
    mock_agent.event_manager.clear.assert_not_called()
    text_outputs = [o for o in result.outputs if isinstance(o, TextOutput)]
    assert any("Nothing to compact" in o.content for o in text_outputs)


@pytest.mark.asyncio
async def test_compact_no_rich_live_spinner(handler, mock_agent, mock_frontend):
    """No rich.live spinner during summarization — it flickers layered on the pt app.

    Regression: /compact used frontend.start_thinking() (a rich.live.Live at ~10fps)
    on top of prompt_toolkit's full-screen app, repainting through emit_block /
    run_in_terminal and fighting pt's own status spinner = visible flicker. The
    progress indicator is now a single static status block.
    """
    summarizer = MagicMock()
    summarizer._render_range_to_markdown = MagicMock(return_value="# History")
    summarizer.summarize = AsyncMock(side_effect=RuntimeError("boom"))
    mock_agent._summarizers = [summarizer]

    await handler.handle("/compact")

    mock_frontend.start_thinking.assert_not_called()
    mock_frontend.stop_thinking.assert_not_called()
    # A single static "Summarizing history…" status line is rendered instead.
    rendered = [c.args[0] for c in mock_frontend.render.call_args_list if c.args]
    assert any(isinstance(o, TextOutput) and "Summarizing history" in o.content for o in rendered)
