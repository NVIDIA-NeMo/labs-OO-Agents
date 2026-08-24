# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-app masked prompt used by manual MCP OAuth."""

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

from nooa_cli.tui.commands import _mcp_oauth_markdown_link
from nooa_cli.tui.subapp import ChoicePromptView, SensitiveTextPromptView, TextPromptView
from nooa_cli.tui.tui_application import TUIApplication, _is_raw_mouse_report


def test_mcp_oauth_markdown_link_shows_complete_clickable_url():
    url = "https://login.example.test/authorize?state=" + "a" * 500 + "&scope=read%20write"

    assert _mcp_oauth_markdown_link(url) == f"[{url}](<{url}>)"


def test_mcp_oauth_markdown_link_escapes_label_metacharacters():
    url = "https://example.test/a_[b]?scope=*read*"

    assert _mcp_oauth_markdown_link(url) == (
        r"[https://example.test/a\_\[b\]?scope=\*read\*]"
        f"(<{url}>)"
    )


def test_mcp_oauth_markdown_link_rejects_unsafe_target():
    assert _mcp_oauth_markdown_link("javascript:alert(1)") is None
    assert _mcp_oauth_markdown_link("https://example.test/a b") is None


def test_raw_numeric_mouse_report_is_filtered():
    assert _is_raw_mouse_report("\x1b[0;12;34M") is True


def test_non_mouse_numeric_csi_sequence_is_not_filtered():
    assert _is_raw_mouse_report("\x1b[31m") is False


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


def test_sensitive_prompt_stays_in_bounded_dynamic_area():
    view = SensitiveTextPromptView("OAuth", "Authorize in your browser.")

    rendered = view.render(120, 40)

    assert len(rendered.splitlines()) == view.max_height
    assert view.max_height < 40


def test_sensitive_prompt_scrolls_long_authorization_url():
    message = "Authorize: https://example.test/" + "a" * 500 + "\nTAIL_VISIBLE"
    view = SensitiveTextPromptView("OAuth", message)

    first = view.render(40, 8)
    assert "TAIL_VISIBLE" not in first
    view.handle_key("end")
    last = view.render(40, 8)

    assert "TAIL_VISIBLE" in last


def test_sensitive_prompt_renders_safe_authorization_copy_hint():
    url = "https://login.example.test/authorize?" + "state=" + "a" * 500
    view = SensitiveTextPromptView("OAuth", "Authorize in your browser.", link_url=url)

    rendered = view.render(60, 10)

    assert "Authorization URL ready" in rendered
    assert url not in rendered
    assert "\x1b" not in rendered
    assert "Ctrl+Y copy URL" in rendered


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


def test_remote_clipboard_prefers_osc52_over_host_pbcopy(monkeypatch):
    output = MagicMock()
    app = TUIApplication.__new__(TUIApplication)
    app._app = SimpleNamespace(output=output)
    monkeypatch.setenv("SSH_CONNECTION", "client server")
    pbcopy = MagicMock()
    monkeypatch.setattr("nooa_cli.tui.tui_application.shutil.which", pbcopy)

    result = app._copy_to_clipboard_result("remote text")

    assert result.success is True
    assert result.transport == "osc52"
    pbcopy.assert_not_called()
    assert "\x1b]52;c;" in output.write_raw.call_args.args[0]


def test_clipboard_reports_size_and_transport_failures(monkeypatch):
    output = MagicMock()
    output.write_raw.side_effect = OSError("terminal rejected OSC 52")
    app = TUIApplication.__new__(TUIApplication)
    app._app = SimpleNamespace(output=output)
    monkeypatch.setenv("SSH_TTY", "/dev/pts/1")

    oversized = app._copy_to_clipboard_result("x" * 100_001)
    failed = app._copy_to_clipboard_result("copy me")

    assert oversized.success is False
    assert "100 KB" in oversized.reason
    assert failed.success is False
    assert "terminal rejected" in failed.reason


def test_local_clipboard_prefers_platform_command_over_osc52(monkeypatch):
    output = MagicMock()
    app = TUIApplication.__new__(TUIApplication)
    app._app = SimpleNamespace(output=output)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("SBX_NO_DISPLAY", raising=False)
    monkeypatch.setattr(
        "nooa_cli.tui.tui_application.shutil.which",
        lambda name: "/usr/bin/wl-copy" if name == "wl-copy" else None,
    )
    run = MagicMock()
    monkeypatch.setattr("nooa_cli.tui.tui_application.subprocess.run", run)

    result = app._copy_to_clipboard_result("local text")

    assert result.success is True
    assert result.transport == "local"
    run.assert_called_once()
    assert run.call_args.args[0] == ["/usr/bin/wl-copy"]
    assert run.call_args.kwargs["input"] == b"local text"
    output.write_raw.assert_not_called()


def test_displayless_sandbox_ignores_xclip_shim_and_uses_osc52(monkeypatch):
    output = MagicMock()
    app = TUIApplication.__new__(TUIApplication)
    app._app = SimpleNamespace(output=output)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("SBX_NO_DISPLAY", "1")
    monkeypatch.setattr(
        "nooa_cli.tui.tui_application.shutil.which",
        lambda name: f"/usr/local/bin/{name}" if name in {"wl-copy", "xclip"} else None,
    )
    run = MagicMock()
    monkeypatch.setattr("nooa_cli.tui.tui_application.subprocess.run", run)

    result = app._copy_to_clipboard_result("sandbox text")

    assert result.success is True
    assert result.transport == "osc52"
    run.assert_not_called()
    assert "\x1b]52;c;" in output.write_raw.call_args.args[0]


def test_xclip_is_available_with_x_display(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("SBX_NO_DISPLAY", raising=False)
    monkeypatch.setattr(
        "nooa_cli.tui.tui_application.shutil.which",
        lambda name: "/usr/bin/xclip" if name == "xclip" else None,
    )

    assert TUIApplication._local_clipboard_command() == (
        "/usr/bin/xclip",
        "-selection",
        "clipboard",
    )
