# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session titles are requested through the normal agent turn."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from nooa_cli.interactive.session_title import SessionTitleRequest
from nooa_cli.tui.commands import SessionCommand


def _agent(*, name: str | None = None, user_named: bool = False):
    return SimpleNamespace(
        _session_manager=SimpleNamespace(
            info=SimpleNamespace(title=name, title_is_user_set=user_named)
        ),
        request_session_title=MagicMock(),
    )


def test_unnamed_session_requests_title_once() -> None:
    agent = _agent()
    request = SessionTitleRequest()

    assert request.request(agent, "Fix the TUI") is True
    assert request.request(agent, "A later message") is False
    agent.request_session_title.assert_called_once_with("Fix the TUI")


def test_existing_title_skips_automatic_request() -> None:
    agent = _agent(name="Existing title")

    assert SessionTitleRequest().request(agent, "Fix the TUI") is False
    agent.request_session_title.assert_not_called()


def test_user_selected_title_skips_automatic_request() -> None:
    agent = _agent(user_named=True)

    assert SessionTitleRequest().request(agent, "Fix the TUI") is False
    agent.request_session_title.assert_not_called()


def test_swapping_sessions_requests_a_title_for_the_new_session():
    agent = _agent(name="Previous session")
    request = SessionTitleRequest()
    assert request.request(agent, "Existing conversation") is False
    agent._session_manager = SimpleNamespace(
        info=SimpleNamespace(title=None, title_is_user_set=False)
    )
    assert request.request(agent, "New conversation") is True
    agent.request_session_title.assert_called_once_with("New conversation")


def test_manual_session_rename_requires_a_title() -> None:
    command = SessionCommand(MagicMock(), MagicMock(), MagicMock())

    valid, error = command.validate_args(["rename"])
    assert valid is False
    assert error == "Usage: /session rename <name>"
