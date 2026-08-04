# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that TUI bang (!) commands work with ShellTools-based agents."""

import tempfile
from unittest.mock import AsyncMock, MagicMock

from nooa_cli.tui.session import Session


class TestBangCommand:
    """Verify that ! commands route through the TUI's own ShellTools."""

    async def test_bang_command_works_with_shell_agent(self):
        """An agent with self.shell must handle ! commands via TUI-owned shell."""
        # Create a mock agent with shell but no bash
        agent = MagicMock()
        del agent.bash  # ensure no bash attribute
        agent.shell = MagicMock()
        agent.shell.cwd = tempfile.gettempdir()

        # Create a minimal session with a pre-set _bang_shell mock
        session = object.__new__(Session)
        session.agent = agent
        bang_shell = MagicMock()
        bang_shell.cwd = tempfile.gettempdir()  # match agent.shell.cwd so no sync cd happens
        bang_shell.run = AsyncMock(
            return_value=MagicMock(
                stdout="hello",
                stderr="",
                returncode=0,
            )
        )
        session._bang_shell = bang_shell

        # The session should detect shell support
        assert hasattr(session.agent, "shell")
        assert not hasattr(session.agent, "bash")

        # Call _handle_bang — this should NOT show the warning
        frontend = AsyncMock()
        session.frontend = frontend

        await session._handle_bang("!echo hello")

        # Should have called the TUI's own bang shell, not the agent's
        bang_shell.run.assert_called_once_with("echo hello")
        agent.shell.run.assert_not_called()

    @staticmethod
    def _session_with_bang_shell() -> Session:
        agent = MagicMock()
        del agent.bash
        agent.shell = MagicMock()
        agent.shell.cwd = tempfile.gettempdir()

        session = object.__new__(Session)
        session.agent = agent
        bang_shell = MagicMock()
        bang_shell.cwd = tempfile.gettempdir()
        bang_shell.run = AsyncMock(return_value=MagicMock(stdout="", stderr="", returncode=0))
        session._bang_shell = bang_shell
        session.frontend = AsyncMock()
        return session

    async def test_bang_strips_whitespace(self):
        """Leading/trailing whitespace after ! is stripped before running."""
        session = self._session_with_bang_shell()

        await session._handle_bang("!  git status  ")

        session._bang_shell.run.assert_called_once_with("git status")

    async def test_bang_empty_command_does_not_run(self):
        """! with no command returns early without calling the shell."""
        session = self._session_with_bang_shell()

        assert await session._handle_bang("!") is None
        session._bang_shell.run.assert_not_called()

    async def test_bang_whitespace_only_does_not_run(self):
        """! followed by only whitespace returns early without calling the shell."""
        session = self._session_with_bang_shell()

        assert await session._handle_bang("!   ") is None
        session._bang_shell.run.assert_not_called()
