# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The deliberately retained built-in TUI slash-command surface."""

from nooa_cli.tui.commands import CommandRegistry


def test_builtin_command_surface_is_explicit_and_pruned():
    assert set(CommandRegistry.get_all_command_classes()) == {
        "activity",
        "clear",
        "compact",
        "context",
        "edit",
        "events",
        "exit",
        "help",
        "jobs",
        "keep-going",
        "memory",
        "memories",
        "mcp",
        "model",
        "models",
        "quit",
        "reasoning",
        "reflection",
        "session",
        "show-diffs",
        "show-python",
        "skills",
        "theme",
        "todos",
        "toolbar",
        "trace-url",
    }
