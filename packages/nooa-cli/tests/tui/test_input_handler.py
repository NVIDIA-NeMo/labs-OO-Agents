# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TUIInputHandler — prompt_continuation, key bindings."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nooa_cli.tui.input_handler import TUIInputHandler, create_key_bindings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    registry.get_completions.return_value = {"/help": "Show help", "/exit": "Exit"}
    return registry


@pytest.fixture
def handler(mock_registry):
    return TUIInputHandler(mock_registry)


# ---------------------------------------------------------------------------
# Indentation: prompt_continuation must be "" so wrapped lines don't indent
# ---------------------------------------------------------------------------


class TestPromptContinuation:
    @pytest.mark.asyncio
    async def test_get_input_passes_empty_prompt_continuation(self, handler):
        """get_input() must pass prompt_continuation='' to suppress wrap indent.

        Without this, prompt_toolkit indents continuation lines to match the
        prompt width (e.g. '/Volumes/dev/dev/tui006 (sonnet-4-5) ❯ ' = 44 chars),
        making wrapped text appear indented on the second line.
        """
        with patch.object(handler.session, "prompt_async", new_callable=AsyncMock) as mock_prompt:
            mock_prompt.return_value = "hello"
            await handler.get_input("❯ ")

        _, kwargs = mock_prompt.call_args
        assert "prompt_continuation" in kwargs, (
            "prompt_async was not called with prompt_continuation kwarg"
        )
        assert kwargs["prompt_continuation"] == "", (
            f"Expected prompt_continuation='', got {kwargs['prompt_continuation']!r}"
        )

    @pytest.mark.asyncio
    async def test_get_input_returns_stripped_result(self, handler):
        """get_input() strips leading/trailing whitespace from the result."""
        with patch.object(handler.session, "prompt_async", new_callable=AsyncMock) as mock_prompt:
            mock_prompt.return_value = "  hello world  "
            result = await handler.get_input("❯ ")

        assert result == "hello world"


# ---------------------------------------------------------------------------
# Key bindings: Enter submits, c-j (Shift+Enter on iTerm2) inserts newline
# ---------------------------------------------------------------------------


class TestKeyBindings:
    def test_enter_binding_exists(self):
        """The 'enter' key (ControlM) must be bound (to submit)."""
        from prompt_toolkit.keys import Keys

        bindings = create_key_bindings()
        bound_keys = [b.keys for b in bindings.bindings]
        assert any(keys == (Keys.ControlM,) for keys in bound_keys), (
            "enter/ControlM is not bound — submitting won't work"
        )

    def test_escape_enter_binding_exists(self):
        """Alt/Option+Enter must be bound as newline fallback."""
        from prompt_toolkit.keys import Keys

        bindings = create_key_bindings()
        bound_key_tuples = [b.keys for b in bindings.bindings]
        assert any(keys == (Keys.Escape, Keys.ControlM) for keys in bound_key_tuples), (
            "Alt/Option+Enter (Escape+ControlM) is not bound"
        )

    def test_c_j_binding_exists(self):
        """ControlJ (\\n) must be bound to insert newline.

        iTerm2 sends \\n (0x0a, ControlJ) for Shift+Enter.
        """
        from prompt_toolkit.keys import Keys

        bindings = create_key_bindings()
        bound_keys = [b.keys for b in bindings.bindings]
        assert any(keys == (Keys.ControlJ,) for keys in bound_keys), (
            "c-j (ControlJ) is not bound — Shift+Enter on iTerm2 will submit"
        )


# ---------------------------------------------------------------------------
# End-to-end: Shift+Enter inserts newline
# ---------------------------------------------------------------------------


class TestShiftEnter:
    @pytest.mark.asyncio
    async def test_iterm2_shift_enter_inserts_newline(self, mock_registry):
        """Full pipeline: iTerm2 Shift+Enter (\\n / ControlJ) → newline, not submit."""
        import io

        from nooa_cli.tui.input_handler import create_key_bindings
        from prompt_toolkit import PromptSession
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import create_output

        with create_pipe_input() as inp:
            out = create_output(stdout=io.StringIO())
            session = PromptSession(
                input=inp,
                output=out,
                key_bindings=create_key_bindings(),
                multiline=True,
            )
            # "hi" + ControlJ (iTerm2 Shift+Enter) + "there" + regular Enter
            inp.send_text("hi\nthere\r")
            result = await session.prompt_async("> ", prompt_continuation="")

        assert result == "hi\nthere", (
            f"Expected 'hi\\nthere', got {result!r} — "
            "iTerm2 Shift+Enter (ControlJ) submitted instead of inserting newline"
        )


# ---------------------------------------------------------------------------
# End-to-end: @ inline mention triggers completion and substitutes on submit
# ---------------------------------------------------------------------------


class TestMentionKeyBindings:
    def test_at_binding_exists(self):
        """The '@' key must be bound so typing it can trigger completion."""
        bindings = create_key_bindings()
        bound = [b.keys for b in bindings.bindings]
        assert any(keys == ("@",) for keys in bound), "'@' is not bound"

    @pytest.mark.asyncio
    async def test_at_completion_fires_on_mention(self, tmp_path):
        """Typing '@<dir>/' populates the completion menu from the filesystem.

        Drives the real prompt_toolkit Buffer through the same
        ``_set_completions_sync`` path the '@' keybinding uses, so this proves
        the keybinding's logic wires into the shared Completer — without
        racing prompt_toolkit's async render loop (which clears completion
        state on submit).
        """
        from nooa_cli.tui.input_handler import (
            _MENTION_RE,
            SlashCommandCompleter,
            _set_completions_sync,
        )
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document

        (tmp_path / "alpha.md").touch()
        (tmp_path / "beta.md").touch()

        registry = MagicMock()
        registry.get_active_help.return_value = {}

        text = f"see @{tmp_path}/"
        # The '@' keybinding gates on this regex before triggering completion.
        assert _MENTION_RE.search(text) is not None

        buf = Buffer(completer=SlashCommandCompleter(registry))
        buf.set_document(Document(text, len(text)), bypass_readonly=True)
        _set_completions_sync(buf)

        assert buf.complete_state is not None, "@ mention did not open a completion menu"
        displays = [c.display_text for c in buf.complete_state.completions]
        assert any("alpha.md" in d for d in displays), displays
        assert any("beta.md" in d for d in displays), displays
