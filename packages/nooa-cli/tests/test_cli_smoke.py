# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: verify the nooa CLI package is importable and the main entry point exists."""

import subprocess
import sys


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
    assert "tui" in names


def test_cli_discovery_does_not_import_tracing_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import nooa_cli; assert 'nooa.tracing' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
