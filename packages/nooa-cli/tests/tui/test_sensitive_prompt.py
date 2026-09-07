# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the container-hosted overlay prompts used by manual MCP OAuth."""

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

from nooa_cli.tui.prompt_overlay import ChoiceOverlay, PromptOverlay
from nooa_cli.tui.tui_application import TUIApplication, _is_raw_mouse_report


def test_raw_numeric_mouse_report_is_filtered():
    assert _is_raw_mouse_report("\x1b[0;12;34M") is True


def test_non_mouse_numeric_csi_sequence_is_not_filtered():
    assert _is_raw_mouse_report("\x1b[31m") is False


def _overlay(**kwargs):
    return PromptOverlay(SimpleNamespace(render_counter=0), "OAuth", "Authorize.", **kwargs)


def _join(fragments):
    return "".join(fragment[1] for fragment in fragments)


def test_prompt_overlay_masks_sensitive_buffer():
    view = _overlay(masked=True)
    view.handle_key("text", "secret")
    assert view.buffer.text == "secret"
    processors = list(view.input_control.input_processors)
    assert any(type(processor).__name__ == "PasswordProcessor" for processor in processors)
    assert view.handle_key("enter") == "close"
    assert view.value == "secret"


def test_prompt_overlay_accepts_keys_reserved_by_explorer_views():
    view = _overlay()
    for action in ("quit", "resume", "j", "k", "slash"):
        assert view.handle_key(action) == "handled"
    assert view.handle_key("enter") == "close"
    assert view.buffer.text == "qrjk/"
    assert view.value == "qrjk/"


def test_prompt_overlay_escape_cancels_without_value():
    view = _overlay()
    view.handle_key("text", "secret")
    assert view.handle_key("escape") == "close"
    assert view.value is None


def test_prompt_overlay_preserves_spaces_and_collapses_only_linebreaks():
    """Ordinary spaces must survive; only line breaks collapse."""
    view = _overlay()
    view.handle_key("space")
    assert view.buffer.text == " "
    view.handle_key("text", "word ")
    assert view.buffer.text == " word "

    pasted = _overlay()
    pasted.handle_key("text", "line one\nline two\n")
    # Interior line breaks become single spaces; a trailing break is dropped
    # entirely so pasting a callback URL plus its newline adds no padding.
    assert pasted.buffer.text == "line one line two"


def test_link_fragments_require_strict_web_targets():
    """file:// URLs and format characters must never become clickable links."""
    from nooa_cli.tui.prompt_overlay import _link_fragments

    def visible(fragments):
        return "".join(f[1] for f in fragments if "[ZeroWidthEscape]" not in f[0])

    assert "URL unavailable" in visible(_link_fragments("file:///etc/passwd"))
    assert "URL unavailable" in visible(_link_fragments("https://example.test/\u202eevil"))
    # A bare port has netloc=":443" but no hostname — must never link.
    assert "URL unavailable" in visible(_link_fragments("https://:443"))
    ok = _link_fragments("https://login.example.test/authorize?state=abc")
    assert "https://login.example.test/authorize?state=abc" in visible(ok)


def test_prompt_overlay_strips_terminal_controls_from_message():
    view = PromptOverlay(
        SimpleNamespace(render_counter=0),
        "OAuth",
        "https://example.test/\x1b[2J\u202eauthorize",
    )
    assert "\x1b" not in view.message
    assert "\u202e" not in view.message


def test_prompt_overlay_shows_clickable_authorization_url():
    url = "https://login.example.test/authorize?state=" + "a" * 300
    view = _overlay(link_url=url)
    body = view._body_text()
    assert url in _join(body)
    escapes = "".join(fragment[1] for fragment in body if "[ZeroWidthEscape]" in fragment[0])
    assert f"\x1b]8;;{url}\x1b\\" in escapes


def test_prompt_overlay_rejects_non_http_link_targets():
    view = _overlay(link_url="javascript:alert(1)")
    body = view._body_text()
    visible = _join(fragment for fragment in body if "[ZeroWidthEscape]" not in fragment[0])
    assert "javascript:" not in visible
    assert "URL unavailable" in visible


def test_prompt_overlay_copies_complete_authorization_url():
    url = "https://login.example.test/authorize?state=" + "a" * 500
    copied = []
    view = _overlay(
        link_url=url,
        copy_handler=lambda value: copied.append(value) or True,
    )
    assert view.handle_key("copy") == "handled"
    assert copied == [url]
    assert "URL copied" in _join(view._footer_text())


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


async def test_text_prompt_returns_edited_value():
    from .tui_app_harness import TUIHarness

    async with TUIHarness() as h:
        prompt = asyncio.create_task(h.app.prompt_text("Alias", "Choose an alias", "nemotron"))
        await h.wait_for(lambda: h.app.active_subview is not None)
        view = h.app.active_subview
        assert isinstance(view, PromptOverlay)
        assert view.buffer.text == "nemotron"
        await h.type_keys("-fast")
        await h.press("enter")
        assert await asyncio.wait_for(prompt, timeout=1) == "nemotron-fast"


async def test_choice_prompt_filters_and_selects():
    from .tui_app_harness import TUIHarness

    async with TUIHarness() as h:
        prompt = asyncio.create_task(
            h.app.prompt_choice(
                "Model", "Choose a model", ["nvidia/nemotron", "openai/gpt", "meta/llama"]
            )
        )
        await h.wait_for(lambda: h.app.active_subview is not None)
        view = h.app.active_subview
        assert isinstance(view, ChoiceOverlay)
        await h.type_keys("gpt")
        await h.press("enter")
        assert await asyncio.wait_for(prompt, timeout=1) == "openai/gpt"


def test_choice_overlay_clamps_cursor_when_filter_shrinks():
    """Enter after a filter shrink must select in-range, never IndexError."""
    from nooa_cli.tui.prompt_overlay import ChoiceOverlay

    app = SimpleNamespace(output=MagicMock())
    app.output.get_size.return_value = SimpleNamespace(rows=40, columns=80)
    view = ChoiceOverlay(app, "Model", "Choose", [f"model-{i}" for i in range(10)])

    for _ in range(6):
        view.handle_key("down")
    assert view.cursor == 6

    # Filter shrinks the match set to one entry before any repaint.
    view.handle_key("text", "9")
    matches = view._clamped_matches()
    assert len(matches) == 1
    assert view.cursor == 0

    assert view.handle_key("enter") == "close"
    assert view.value == "model-9"


def test_choice_overlay_enter_after_empty_filter_selects_valid_option():
    """Keys queue before repaint; enter must not crash on a stale cursor."""
    from nooa_cli.tui.prompt_overlay import ChoiceOverlay

    app = SimpleNamespace(output=MagicMock())
    app.output.get_size.return_value = SimpleNamespace(rows=40, columns=80)
    view = ChoiceOverlay(app, "Model", "Choose", ["alpha", "beta", "gamma", "delta"])

    view.handle_key("end")
    assert view.handle_key("enter") == "close"
    assert view.value == "delta"


async def test_choice_prompt_supports_arrow_selection_and_escape():
    from .tui_app_harness import TUIHarness

    async with TUIHarness() as h:
        prompt = asyncio.create_task(h.app.prompt_choice("Model", "Choose", ["one", "two"]))
        await h.wait_for(lambda: h.app.active_subview is not None)
        await h.press("down")
        await h.press("enter")
        assert await asyncio.wait_for(prompt, timeout=1) == "two"

        cancelled = asyncio.create_task(h.app.prompt_choice("Model", "Choose", ["one"]))
        await h.wait_for(lambda: h.app.active_subview is not None)
        await h.press("escape")
        assert await asyncio.wait_for(cancelled, timeout=1) == ""


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
