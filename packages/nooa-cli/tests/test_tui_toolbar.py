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
