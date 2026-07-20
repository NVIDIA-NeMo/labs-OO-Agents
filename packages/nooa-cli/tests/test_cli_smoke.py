# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: verify the nooa CLI package is importable and the main entry point exists."""


def test_cli_importable():
    from nooa_cli import main

    assert callable(main)


def test_commands_discoverable():
    from nooa_cli.commands import discover_commands

    commands = list(discover_commands())
    assert len(commands) > 0
    names = [name for name, _ in commands]
    assert "start-dev" in names
    assert "eval" in names
    assert "config" in names
