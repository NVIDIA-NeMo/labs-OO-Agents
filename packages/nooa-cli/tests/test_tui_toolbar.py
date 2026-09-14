# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured TUI toolbar providers."""

from pathlib import Path

from nooa_cli.tui.toolbar import ToolbarContext, ToolbarRegistry


def test_toolbar_renders_configured_items_in_order():
    registry = ToolbarRegistry(load_plugins=False)
    context = ToolbarContext(
        model="provider/claude-example",
        working_directory=Path("/work/repo"),
        context_usage="ctx 20%",
        session_id="12345678-abcd",
        session_title="migration",
    )

    assert registry.render(["model", "cwd", "context", "session"], context) == (
        "example · repo · ctx 20% · migration [12345678]"
    )


def test_toolbar_accepts_registered_extension():
    registry = ToolbarRegistry(load_plugins=False)
    registry.register("branch", lambda context: "dev/tui")

    assert registry.render(["branch"], ToolbarContext("model", Path("."), "")) == "dev/tui"


def test_token_usage_uses_total_input_and_cache_reads_only():
    from nooa_cli.tui.toolbar import format_token_usage

    from nooa.unifiedllm import LLMUsage

    usage = LLMUsage(
        input_tokens=12_345,
        output_tokens=456,
        cached_input_tokens=9_876,
        cache_write_input_tokens=2_000,
    )
    context = ToolbarContext("model", Path("."), "", token_usage=format_token_usage(usage))
    assert ToolbarRegistry(load_plugins=False).render(["tokens"], context) == (
        "↑ 12.3k ↓ 456 cache 80%"
    )


def test_token_usage_distinguishes_unknown_usage_from_zero_cache_hits():
    from nooa_cli.tui.toolbar import format_token_usage

    from nooa.unifiedllm import LLMUsage

    assert format_token_usage(None) == "↑ — ↓ — cache —"
    assert format_token_usage(LLMUsage()) == "↑ 0 ↓ 0 cache —"
    assert format_token_usage(LLMUsage(input_tokens=1_200_000, output_tokens=1_500)) == (
        "↑ 1.2m ↓ 1.5k cache 0%"
    )
