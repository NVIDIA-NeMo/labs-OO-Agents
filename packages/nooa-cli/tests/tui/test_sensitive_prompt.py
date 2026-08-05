# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-app masked prompt used by manual MCP OAuth."""

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

from nooa_cli.tui.subapp import ChoicePromptView, SensitiveTextPromptView, TextPromptView
from nooa_cli.tui.tui_application import TUIApplication
from prompt_toolkit.formatted_text import ANSI


def test_sensitive_prompt_masks_value_and_submits():
    view = SensitiveTextPromptView("OAuth", "Paste the authorization code")
    for character in "code-123":
        view.handle_key("text", character)

    rendered = view.render(80, 10)
    assert "code-123" not in rendered
    assert "•" * len("code-123") in rendered
    assert view.handle_key("enter") == "close"
    assert view.value == "code-123"


def test_sensitive_prompt_accepts_keys_reserved_by_explorer_views():
    view = SensitiveTextPromptView("OAuth", "Paste")
    for action in ("quit", "resume", "j", "k", "slash"):
        assert view.handle_key(action) == "handled"
    assert view.handle_key("backspace") == "handled"
    assert view.handle_key("enter") == "close"
    assert view.value == "qrjk"


def test_sensitive_prompt_escape_cancels_without_value():
    view = SensitiveTextPromptView("OAuth", "Paste")
    view.handle_key("text", "secret")
    assert view.handle_key("escape") == "close"
    assert view.value is None


def test_sensitive_prompt_strips_terminal_controls_from_server_url():
    view = SensitiveTextPromptView("OAuth", "https://example.test/\x1b[2J\u202eauthorize")
    rendered = view.render(80, 10)
    assert "\x1b" not in rendered
    assert "\u202e" not in rendered


def test_sensitive_prompt_scrolls_long_authorization_url():
    message = "Authorize: https://example.test/" + "a" * 500 + "\nTAIL_VISIBLE"
    view = SensitiveTextPromptView("OAuth", message)

    first = view.render(40, 8)
    assert "TAIL_VISIBLE" not in first
    view.handle_key("end")
    last = view.render(40, 8)

    assert "TAIL_VISIBLE" in last


def test_sensitive_prompt_renders_short_clickable_authorization_link():
    url = "https://login.example.test/authorize?" + "state=" + "a" * 500
    view = SensitiveTextPromptView("OAuth", "Authorize in your browser.", link_url=url)

    rendered = view.render(60, 10)
    fragments = ANSI(rendered).__pt_formatted_text__()
    visible = "".join(text for style, text in fragments if style != "[ZeroWidthEscape]")
    escapes = "".join(text for style, text in fragments if style == "[ZeroWidthEscape]")

    assert "Open authorization URL" in visible
    assert url not in visible
    assert f"\x1b]8;;{url}\x07" in escapes
    assert "Ctrl+Y copy URL" in visible


def test_sensitive_prompt_copies_complete_authorization_url():
    copied = []
    url = "https://login.example.test/authorize?state=" + "a" * 500
    view = SensitiveTextPromptView(
        "OAuth",
        "Authorize in your browser.",
        link_url=url,
        copy_handler=lambda value: copied.append(value) or True,
    )

    assert view.handle_key("copy") == "handled"
    assert copied == [url]
    assert "URL copied" in view.render(80, 10)


def test_sensitive_prompt_rejects_non_http_link_targets():
    view = SensitiveTextPromptView(
        "OAuth",
        "Authorize in your browser.",
        link_url="javascript:alert(1)",
    )

    assert "Open authorization URL" not in view.render(80, 10)
    assert view.handle_key("copy") == "handled"


def test_clipboard_falls_back_to_complete_osc52_payload(monkeypatch):
    url = "https://login.example.test/authorize?state=" + "a" * 500
    output = MagicMock()
    app = TUIApplication.__new__(TUIApplication)
    app._app = SimpleNamespace(output=output)
    monkeypatch.setattr("nooa_cli.tui.tui_application.shutil.which", lambda _name: None)

    assert app._copy_to_clipboard(url) is True
    sequence = output.write_raw.call_args.args[0]
    encoded = sequence.removeprefix("\x1b]52;c;").removesuffix("\x07")
    assert base64.b64decode(encoded).decode("utf-8") == url
    output.flush.assert_called_once_with()


def test_text_prompt_shows_default_and_returns_edited_value():
    view = TextPromptView("Alias", "Choose an alias", default="nemotron")
    assert "nemotron" in view.render(80, 8)
    view.handle_key("text", "-fast")
    assert view.handle_key("enter") == "close"
    assert view.value == "nemotron-fast"


def test_choice_prompt_filters_and_selects():
    view = ChoicePromptView(
        "Model", "Choose a model", ["nvidia/nemotron", "openai/gpt", "meta/llama"]
    )
    for character in "gpt":
        view.handle_key("text", character)

    rendered = view.render(80, 10)
    assert "openai/gpt" in rendered
    assert "nemotron" not in rendered
    assert view.handle_key("enter") == "close"
    assert view.value == "openai/gpt"


def test_choice_prompt_supports_arrow_selection_and_escape():
    view = ChoicePromptView("Model", "Choose", ["one", "two"])
    view.handle_key("down")
    assert view.handle_key("enter") == "close"
    assert view.value == "two"

    cancelled = ChoicePromptView("Model", "Choose", ["one"])
    assert cancelled.handle_key("escape") == "close"
    assert cancelled.value is None
