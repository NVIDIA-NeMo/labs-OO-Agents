# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that ShellTools.grep accepts and forwards a timeout parameter."""

import pytest

from nooa.tools.shell_tools_legacy import ShellToolsLegacy as ShellTools


@pytest.fixture
async def shell(tmp_path):
    """Create a ShellTools instance with a temp directory as cwd."""
    s = ShellTools(cwd=tmp_path)
    # Create a test file to grep
    (tmp_path / "hello.py").write_text("# hello world\nfoo = 1\nbar = 2\n")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_grep_works_without_explicit_timeout(shell, tmp_path):
    """grep() works without passing timeout (uses default 30s)."""
    result = await shell.grep("foo", "hello.py")
    assert result.total_matches == 1


@pytest.mark.asyncio
async def test_grep_accepts_timeout_parameter(shell, tmp_path):
    """grep() should accept a timeout parameter without raising TypeError."""
    result = await shell.grep("foo", "hello.py", timeout=60.0)
    assert result.total_matches == 1


@pytest.mark.asyncio
async def test_grep_custom_timeout_succeeds(shell, tmp_path):
    """A custom timeout value is accepted and works for a tiny file."""
    # timeout=5 should be plenty for a tiny file
    result = await shell.grep("bar", "hello.py", timeout=5.0)
    assert result.total_matches == 1
