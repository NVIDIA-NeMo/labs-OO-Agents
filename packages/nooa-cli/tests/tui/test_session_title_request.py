# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session titles are requested through the normal agent turn."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from nooa_cli.tui.commands import SessionCommand
from nooa_cli.tui.session import Session


def _session(*, name: str | None = None, user_named: bool = False) -> Session:
    session = Session.__new__(Session)
    session._session_title_requested = False
    session._session_manager = SimpleNamespace(name=name, user_named=user_named)
    session.agent = SimpleNamespace(request_session_title=MagicMock())
    return session


def test_unnamed_session_requests_title_once() -> None:
    session = _session()

    assert session._request_session_title("Fix the TUI") is True
    assert session._request_session_title("A later message") is False
    session.agent.request_session_title.assert_called_once_with("Fix the TUI")


def test_existing_title_skips_automatic_request() -> None:
    session = _session(name="Existing title")

    assert session._request_session_title("Fix the TUI") is False
    session.agent.request_session_title.assert_not_called()


def test_user_selected_title_skips_automatic_request() -> None:
    session = _session(user_named=True)

    assert session._request_session_title("Fix the TUI") is False
    session.agent.request_session_title.assert_not_called()


def test_manual_session_rename_requires_a_title() -> None:
    command = SessionCommand(MagicMock(), MagicMock(), MagicMock())

    valid, error = command.validate_args(["rename"])
    assert valid is False
    assert error == "Usage: /session rename <name>"
