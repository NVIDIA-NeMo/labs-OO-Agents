# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-app masked prompt used by manual MCP OAuth."""

from nooa_cli.tui.subapp import ChoicePromptView, SensitiveTextPromptView, TextPromptView


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
