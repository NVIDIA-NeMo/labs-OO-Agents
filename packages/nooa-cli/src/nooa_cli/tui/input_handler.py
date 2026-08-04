# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input handling with history, completion, and multi-line support.

Uses prompt_toolkit for readline-like input experience.
The actual completion logic lives in ``completer.py`` (shared with the web
frontend); this module is just the prompt_toolkit adapter.
"""

import re
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer as PtCompleter
from prompt_toolkit.completion import Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from .completer import _MENTION_PATTERN, Completer
from .theme import COLORS

if TYPE_CHECKING:
    from .commands import CommandRegistry

# The keybindings trigger completion when an @ mention is being typed at the
# cursor — same pattern the engine uses (anchored at end-of-buffer here).
_MENTION_RE = re.compile(_MENTION_PATTERN + r"?\Z")


class SlashCommandCompleter(PtCompleter):
    """prompt_toolkit adapter over the shared ``Completer`` engine."""

    def __init__(self, registry: "CommandRegistry"):
        self._completer = Completer(registry=registry)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if (
            not text.startswith("/")
            and not text.startswith("!")
            and _MENTION_RE.search(text) is None
        ):
            return

        for item in self._completer.complete(text):
            # Calculate how much text to replace: the item.text is the full
            # replacement, so we insert the suffix after what's already typed.
            suffix = item.text[len(text) :] if item.text.startswith(text) else item.text
            if not suffix and item.text == text:
                # Exact match — still yield it so prompt_toolkit's
                # completion menu never receives an empty list (which
                # crashes _get_menu_width with max() on empty iterable).
                yield Completion(
                    "",
                    start_position=0,
                    display=item.display,
                    display_meta=item.description,
                )
                continue
            yield Completion(
                suffix if item.text.startswith(text) else item.text,
                start_position=0 if item.text.startswith(text) else -len(text),
                display=item.display,
                display_meta=item.description,
            )


def _set_completions_sync(buffer) -> None:
    """Generate completions synchronously and set them on the buffer.

    Avoids the prompt_toolkit race in ``Buffer.start_completion()`` which
    creates an empty ``CompletionState`` before completions are loaded,
    causing ``_get_menu_width`` to call ``max()`` on an empty sequence.

    Uses ``Buffer._set_completions`` (private API) with a fallback to the
    public ``start_completion`` for forward-compatibility.
    """
    from prompt_toolkit.completion import CompleteEvent

    if not buffer.completer:
        return

    completions = list(
        buffer.completer.get_completions(buffer.document, CompleteEvent(completion_requested=True))
    )
    if completions:
        try:
            buffer._set_completions(completions=completions)
            # Ensure no completion is pre-selected (matches start_completion(select_first=False)).
            if buffer.complete_state:
                buffer.complete_state.go_to_index(None)
        except (AttributeError, TypeError):
            # Fallback if private API changes in a future prompt_toolkit release.
            buffer.start_completion(select_first=False)
    else:
        # No matches — dismiss any visible menu.
        if buffer.complete_state:
            buffer.cancel_completion()


def create_key_bindings(vi_mode: bool = False) -> KeyBindings:
    """Create key bindings for multi-line input and slash command completion.

    - Enter: Submit
    - Alt+Enter or Option+Enter: Insert newline for multi-line input
    - Shift+Enter: Insert newline (requires iTerm2/kitty/WezTerm with CSI u mode)
    - Any character after /: Keep completion menu open (disabled in vi mode to
      avoid clashing with vi normal-mode movement keys)

    Note: Most terminals send Alt+Enter as Escape followed by Enter.
    Shift+Enter works in terminals that send \\x1b[13;2u (Kitty keyboard protocol).

    Args:
        vi_mode: When True, skip letter-key completion bindings that would
                 conflict with vi normal-mode commands.
    """
    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event):
        """Enter submits the current input."""
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def _(event):
        """Shift+Enter on iTerm2 sends \\n (ControlJ) — insert a newline."""
        event.current_buffer.insert_text("\n")

    @bindings.add("escape", "enter")
    def _(event):
        """Alt+Enter (Option+Enter without CSI u mode) inserts a newline."""
        event.current_buffer.insert_text("\n")

    # Trigger and maintain completion for slash commands
    def _handle_char_for_completion(event, char: str):
        """Insert character and trigger completion if in a slash command."""
        buffer = event.current_buffer
        buffer.insert_text(char)

        # Check if we're typing a slash or bang command, or an inline @ mention.
        text = buffer.text[: buffer.cursor_position]
        if (
            text.startswith("/")
            or text.startswith("!")
            or (("\n/" in text) and text.rsplit("\n", 1)[-1].startswith("/"))
            or _MENTION_RE.search(text) is not None
        ):
            # Generate completions synchronously and set them directly.
            # buffer.start_completion() creates an empty CompletionState before
            # completions load, which races with the renderer: prompt_toolkit's
            # _get_menu_width calls max() on the empty list → ValueError.
            try:
                _set_completions_sync(buffer)
            except (IndexError, ValueError):
                # prompt_toolkit race: completion state accessed with empty list.
                # Safe to ignore — the next keystroke will retry.
                pass

    if not vi_mode:
        # Bind alphanumeric + path-relevant punctuation to maintain completion.
        # Skipped in vi mode: letters are vi normal-mode movement/action keys.
        _completion_chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "._-~"  # common in file paths: dotfiles, underscores, hyphens, home ~
        )
        for char in _completion_chars:

            @bindings.add(char)
            def _(event, c=char):
                _handle_char_for_completion(event, c)

        @bindings.add(" ")
        def _(event):
            _handle_char_for_completion(event, " ")

        @bindings.add("c-h")  # Backspace
        @bindings.add("backspace")
        def _(event):
            """Handle backspace and re-trigger completion if still in / or ! command."""
            buffer = event.current_buffer
            buffer.delete_before_cursor(1)

            text = buffer.text[: buffer.cursor_position]
            if text.startswith("/") or text.startswith("!") or _MENTION_RE.search(text) is not None:
                try:
                    _set_completions_sync(buffer)
                except (IndexError, ValueError):
                    pass

    # Always bind "/" — in vi mode only trigger completion when the buffer
    # already starts with "/" (i.e. we're already in a slash command), so that
    # vi normal-mode "/" search still fires on an empty buffer.
    @bindings.add("/")
    def _(event):
        buffer = event.current_buffer
        text_before = buffer.text[: buffer.cursor_position]
        if vi_mode and not text_before.startswith("/"):
            # Let vi handle "/" search on an empty/non-slash buffer
            buffer.insert_text("/")
        else:
            _handle_char_for_completion(event, "/")

    # "!" on an empty buffer starts a bang command; trigger completion immediately.
    @bindings.add("!")
    def _(event):
        buffer = event.current_buffer
        buffer.insert_text("!")
        text = buffer.text[: buffer.cursor_position]
        if text == "!" or text.startswith("!"):
            _set_completions_sync(buffer)

    # "@" starts an inline file/dir mention; trigger completion immediately
    # whenever it lands at start-of-line or after whitespace.
    @bindings.add("@")
    def _(event):
        buffer = event.current_buffer
        buffer.insert_text("@")
        text = buffer.text[: buffer.cursor_position]
        if _MENTION_RE.search(text) is not None:
            try:
                _set_completions_sync(buffer)
            except (IndexError, ValueError):
                pass

    return bindings


def create_prompt_style() -> Style:
    """Create prompt_toolkit style using Catppuccin colors."""
    return Style.from_dict(
        {
            # Prompt
            "prompt": COLORS["green"],
            "prompt.continuation": COLORS["overlay1"],
            # Completion menu
            "completion-menu": f"bg:{COLORS['surface0']} {COLORS['text']}",
            "completion-menu.completion": f"bg:{COLORS['surface0']} {COLORS['text']}",
            "completion-menu.completion.current": f"bg:{COLORS['surface2']} {COLORS['mauve']}",
            "completion-menu.meta": COLORS["overlay1"],
            "completion-menu.meta.current": COLORS["lavender"],
            # Scrollbar
            "scrollbar.background": COLORS["surface0"],
            "scrollbar.button": COLORS["surface2"],
        }
    )


class TUIInputHandler:
    """Handle user input with history, completion, and multi-line support."""

    def __init__(self, registry: "CommandRegistry", vi_mode: bool = False):
        """Initialize the input handler.

        Args:
            registry: CommandRegistry instance to get completion options from
            vi_mode: Enable vi keybindings (normal/insert mode). When True,
                     letter-key completion bindings are skipped to avoid
                     conflicts with vi movement commands.
        """
        from prompt_toolkit.shortcuts import CompleteStyle

        self.vi_mode = vi_mode
        self.history = InMemoryHistory()
        self.completer = SlashCommandCompleter(registry)
        self.key_bindings = create_key_bindings(vi_mode=vi_mode)
        self.style = create_prompt_style()

        self.session = PromptSession(
            history=self.history,
            completer=self.completer,
            key_bindings=self.key_bindings,
            style=self.style,
            multiline=True,  # Allow Shift+Enter newlines; Enter binding submits
            complete_while_typing=False,  # We handle it manually via key bindings
            complete_style=CompleteStyle.COLUMN,  # Single column with descriptions
            enable_history_search=True,  # Ctrl+R for reverse search
            vi_mode=vi_mode,
        )

    def refresh_style(self) -> None:
        """Rebuild prompt_toolkit style from current theme colors."""
        self.style = create_prompt_style()
        self.session.style = self.style

    async def get_input(self, prompt: str = "You: ", default: str = "", bottom_toolbar=None) -> str:
        """Get input from user with history and completion.

        Args:
            prompt: The prompt to display
            default: Pre-fill the input buffer with this text
            bottom_toolbar: Optional callable returning toolbar text (for spinner)

        Returns:
            User input string (may be multi-line)

        Raises:
            EOFError: If Ctrl+D is pressed
            KeyboardInterrupt: If Ctrl+C is pressed
        """
        kwargs: dict = {
            "prompt_continuation": "",
            "default": default,
        }
        if bottom_toolbar is not None:
            kwargs["bottom_toolbar"] = bottom_toolbar
        result = await self.session.prompt_async(
            [("class:prompt", prompt)],
            **kwargs,
        )
        return result.strip()

    def invalidate(self) -> None:
        """Force a redraw of the prompt (used by spinner to update toolbar)."""
        if self.session.app:
            self.session.app.invalidate()
