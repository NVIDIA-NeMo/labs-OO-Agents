# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for context-window usage shown in the TUI.

Two surfaces:

1. The status-bar rule between turns includes a compact ``"ctx N%"``
   label whenever the agent has a bounded context and the runtime has
   rendered at least one generation.
2. The TUI agent registers a dynamic ``context_usage`` block in its
   ``context_manager`` so the LLM sees the same stats every turn.
"""

from types import SimpleNamespace

# ── _context_usage_label -----------------------------------------------------


def _make_session_for_label(context_stats):
    """Build a minimal Session whose ``_context_usage_label`` we can call."""
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session.agent = SimpleNamespace(context_stats=context_stats)
    return session


def _stats(*, total, model_context_window=None, reserved_output_tokens=None):
    """Real ContextWindowStats for the label helper.

    ``total`` is the provider-reported prompt-token count (None before the
    first response). The percentage is against the usable window (model
    window minus the output-token reserve).
    """
    from nooa.context_blocks.models import ContextWindowStats

    return ContextWindowStats(
        context_blocks_count=0,
        events_count=0,
        prompt_tokens=total,
        model_context_window=model_context_window,
        reserved_output_tokens=reserved_output_tokens,
    )


def test_context_usage_label_placeholder_when_no_stats():
    # No generation context yet → placeholder, not blank.
    session = _make_session_for_label(None)
    assert session._context_usage_label() == "ctx —"


def test_context_usage_label_placeholder_before_first_response():
    # Stats exist (render ran) but the provider has not reported usage yet.
    session = _make_session_for_label(_stats(total=None, model_context_window=200_000))
    assert session._context_usage_label() == "ctx —"


def test_context_usage_label_placeholder_when_no_window():
    # Provider tokens known but the model window is unknown → can't compute %.
    session = _make_session_for_label(_stats(total=1234, model_context_window=None))
    assert session._context_usage_label() == "ctx —"


def test_context_usage_label_shows_integer_percent():
    # 16 000 / 80 000 = 20.0%
    session = _make_session_for_label(_stats(total=16_000, model_context_window=80_000))
    assert session._context_usage_label() == "ctx 20%"


def test_context_usage_label_rounds():
    # 13 000 / 80 000 = 16.25% → "ctx 16%"
    session = _make_session_for_label(_stats(total=13_000, model_context_window=80_000))
    assert session._context_usage_label() == "ctx 16%"


def test_context_usage_label_accounts_for_output_reserve():
    # 30 000 / (100 000 − 40 000 usable) = 50%, not 30% of the raw window.
    session = _make_session_for_label(
        _stats(total=30_000, model_context_window=100_000, reserved_output_tokens=40_000)
    )
    assert session._context_usage_label() == "ctx 50%"


def test_context_usage_label_uses_model_context_window():
    # 20 000 / 200 000 = 10%
    session = _make_session_for_label(_stats(total=20_000, model_context_window=200_000))
    assert session._context_usage_label() == "ctx 10%"


def test_context_usage_label_keeps_last_exact_value_while_next_call_is_pending():
    session = _make_session_for_label(_stats(total=9_000, model_context_window=100_000))
    assert session._context_usage_label() == "ctx 9%"

    session.agent.context_stats = _stats(total=None, model_context_window=100_000)

    assert session._context_usage_label() == "ctx 9%"


def test_context_usage_label_does_not_reuse_value_for_different_window():
    session = _make_session_for_label(_stats(total=9_000, model_context_window=100_000))
    assert session._context_usage_label() == "ctx 9%"

    session.agent.context_stats = _stats(total=None, model_context_window=200_000)

    assert session._context_usage_label() == "ctx —"


# ── TUI agent registers the context_usage dynamic block -----------------------


def test_tui_agent_installs_context_usage_dynamic_block():
    """BaseTUIAgent.__init__ registers a dynamic context_usage block.

    The expression depends only on ``self.context_stats`` so it stays
    None-safe on the very first turn (when no generation has run).
    """
    import os

    # Don't actually hit any LLM — pin to a FakeLLMClient and a
    # minimum-viable config.
    from pathlib import Path

    from nooa_cli.tui.agent import TUIAgent
    from nooa_cli.tui.config import AgentConfig, SummarizationConfig

    from nooa.unifiedllm import FakeLLMClient

    agent_cfg = AgentConfig(
        working_dir=Path(os.getcwd()),
        summarization=SummarizationConfig(policy="none"),
    )
    agent = TUIAgent(llm=FakeLLMClient(), config=agent_cfg)

    # Dynamic context keys are stored on the context manager
    cm = agent.context_manager
    keys = list(cm.keys())
    assert "context_usage" in keys, f"context_usage missing; have: {keys}"
