# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Session._restore_terminal — three code paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_session():
    """Build a minimal Session-like object with just the restore method."""
    from nooa_cli.tui.session import Session

    # Session.__init__ requires several args; we bypass it and attach
    # the method + required state directly.
    obj = object.__new__(Session)
    obj._saved_termios = None
    return obj


class TestRestoreTerminalTermiosSuccess:
    """Path 1: termios available, saved attrs present, stdin is a tty."""

    def test_restores_saved_attrs(self):
        session = _make_session()
        fake_attrs = [1, 2, 3, 4, 5, 6, []]
        session._saved_termios = fake_attrs
        mock_termios = MagicMock()

        with patch("sys.stdin") as mock_stdin, patch.dict("sys.modules", {"termios": mock_termios}):
            mock_stdin.isatty.return_value = True
            mock_stdin.fileno.return_value = 0
            session._restore_terminal()

        mock_termios.tcsetattr.assert_called_once_with(0, mock_termios.TCSANOW, fake_attrs)

    def test_noop_when_no_saved_attrs(self):
        session = _make_session()
        session._saved_termios = None

        mock_termios = MagicMock()
        with patch("sys.stdin") as mock_stdin, patch.dict("sys.modules", {"termios": mock_termios}):
            mock_stdin.isatty.return_value = True
            session._restore_terminal()

        mock_termios.tcsetattr.assert_not_called()

    def test_noop_when_not_a_tty(self):
        session = _make_session()
        session._saved_termios = [1, 2, 3]

        mock_termios = MagicMock()
        with patch("sys.stdin") as mock_stdin, patch.dict("sys.modules", {"termios": mock_termios}):
            mock_stdin.isatty.return_value = False
            session._restore_terminal()

        mock_termios.tcsetattr.assert_not_called()


class TestRestoreTerminalTermiosOSError:
    """Path 2: termios raises OSError → falls back to stty."""

    def test_falls_back_to_stty_on_oserror(self):
        session = _make_session()
        session._saved_termios = [1, 2, 3]

        mock_termios = MagicMock()
        mock_termios.tcsetattr.side_effect = OSError("device not configured")

        with (
            patch("sys.stdin") as mock_stdin,
            patch.dict("sys.modules", {"termios": mock_termios}),
            patch("subprocess.run") as mock_run,
        ):
            mock_stdin.isatty.return_value = True
            mock_stdin.fileno.return_value = 0
            session._restore_terminal()

        mock_run.assert_called_once_with(["stty", "sane"], stdin=mock_stdin, check=False)


class TestRestoreTerminalNoTermios:
    """Path 3: termios not available (ImportError) → falls back to stty."""

    def test_falls_back_to_stty_when_no_termios(self):
        session = _make_session()
        session._saved_termios = None  # Would be None if termios wasn't available at save time

        def _raise_import(name, *args, **kwargs):
            if name == "termios":
                raise ImportError("No module named 'termios'")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__

        with (
            patch("sys.stdin") as mock_stdin,
            patch("subprocess.run") as mock_run,
            patch("builtins.__import__", side_effect=_raise_import),
        ):
            mock_stdin.isatty.return_value = True
            session._restore_terminal()

        mock_run.assert_called_once_with(["stty", "sane"], stdin=mock_stdin, check=False)
