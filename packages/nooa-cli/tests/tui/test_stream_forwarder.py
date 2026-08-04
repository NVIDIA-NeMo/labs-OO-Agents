# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``_StrayStreamForwarder`` and ``install_stray_stream_capture``.

The forwarder has to do three things reliably:

1. Emit each complete line through ``emit_block``, styled so stdout and
   stderr are visually distinct in the transcript.
2. Buffer partial lines and hold them until the next ``\\n`` or an
   explicit ``flush()`` so progress-bar / streaming-log writes don't
   splatter mid-line into the queue.
3. Install and uninstall symmetrically so nothing leaks into subsequent
   tests or into the terminal after the TUI exits.
"""

from __future__ import annotations

import io
import sys

from nooa_cli.tui.stream_forwarder import (
    _StrayStreamForwarder,
    install_stray_stream_capture,
)


def _mk(prefix: str = "· ", color: str = "2") -> tuple[_StrayStreamForwarder, list[str]]:
    emitted: list[str] = []
    original = io.StringIO()
    fw = _StrayStreamForwarder(
        original,
        emitted.append,
        prefix=prefix,
        ansi_color=color,
    )
    return fw, emitted


class TestLineBuffering:
    def test_complete_line_is_emitted(self):
        fw, emitted = _mk()
        fw.write("hello\n")
        assert emitted == ["\x1b[2m· hello\x1b[0m\n"]

    def test_partial_line_is_buffered_until_newline(self):
        fw, emitted = _mk()
        fw.write("par")
        fw.write("tial ")
        assert emitted == []  # nothing flushed yet
        fw.write("line\n")
        assert emitted == ["\x1b[2m· partial line\x1b[0m\n"]

    def test_multiple_newlines_in_one_write_emit_multiple_lines(self):
        fw, emitted = _mk()
        fw.write("one\ntwo\nthree\n")
        assert emitted == [
            "\x1b[2m· one\x1b[0m\n",
            "\x1b[2m· two\x1b[0m\n",
            "\x1b[2m· three\x1b[0m\n",
        ]

    def test_mixed_terminated_and_unterminated_in_one_write(self):
        fw, emitted = _mk()
        fw.write("first\nsecond")  # second has no trailing \n
        assert emitted == ["\x1b[2m· first\x1b[0m\n"]
        fw.flush()
        assert emitted == [
            "\x1b[2m· first\x1b[0m\n",
            "\x1b[2m· second\x1b[0m\n",
        ]

    def test_flush_without_pending_is_noop(self):
        fw, emitted = _mk()
        fw.flush()
        assert emitted == []

    def test_empty_write_does_nothing(self):
        fw, emitted = _mk()
        fw.write("")
        assert emitted == []

    def test_write_returns_character_count(self):
        fw, _ = _mk()
        assert fw.write("hello\n") == 6
        assert fw.write("") == 0


class TestStyling:
    def test_stdout_prefix_and_dim_color(self):
        """The stdout marker defaults to ``· `` with ANSI dim (2)."""
        fw, emitted = _mk(prefix="· ", color="2")
        fw.write("plain\n")
        assert emitted == ["\x1b[2m· plain\x1b[0m\n"]

    def test_stderr_prefix_and_red_color(self):
        """Stderr uses ``! `` with ANSI red (31)."""
        fw, emitted = _mk(prefix="! ", color="31")
        fw.write("boom\n")
        assert emitted == ["\x1b[31m! boom\x1b[0m\n"]


class TestDuckTyping:
    def test_isatty_reports_false(self):
        """Libraries that check isatty() must see False so they don't
        emit raw ANSI sequences that we'd then capture as literal
        characters inside our styled line."""
        fw, _ = _mk()
        assert fw.isatty() is False

    def test_forwards_attribute_access_to_original(self):
        """Non-write attributes (``encoding`` etc.) fall through."""
        original = io.StringIO()
        original.mode = "w"  # type: ignore[attr-defined]
        fw = _StrayStreamForwarder(original, lambda _: None, prefix="· ", ansi_color="2")
        assert fw.mode == "w"

    def test_writelines_emits_each_entry(self):
        fw, emitted = _mk()
        fw.writelines(["a\n", "b\n"])
        assert emitted == [
            "\x1b[2m· a\x1b[0m\n",
            "\x1b[2m· b\x1b[0m\n",
        ]


class TestEmitBlockFailureSafety:
    def test_emit_block_exception_falls_through_to_original_stream(self):
        """If the TUI's emit_block is broken for any reason, we must
        still see the write somewhere — otherwise crashes are silent."""
        original = io.StringIO()

        def broken_emit(_: str) -> None:
            raise RuntimeError("block queue exploded")

        fw = _StrayStreamForwarder(original, broken_emit, prefix="! ", ansi_color="31")
        fw.write("critical\n")
        assert "critical" in original.getvalue()


class TestInstallUninstall:
    def test_install_replaces_both_streams_uninstall_restores(self):
        original_out = sys.stdout
        original_err = sys.stderr
        try:
            uninstall = install_stray_stream_capture(lambda _: None)
            assert isinstance(sys.stdout, _StrayStreamForwarder)
            assert isinstance(sys.stderr, _StrayStreamForwarder)
            assert sys.stdout is not original_out
            assert sys.stderr is not original_err
            uninstall()
            assert sys.stdout is original_out
            assert sys.stderr is original_err
        finally:
            # Defensive: if something above raised, make sure we don't
            # leak the forwarder into subsequent tests.
            if isinstance(sys.stdout, _StrayStreamForwarder):
                sys.stdout = original_out
            if isinstance(sys.stderr, _StrayStreamForwarder):
                sys.stderr = original_err

    def test_installed_stdout_stderr_emit_through_callback(self):
        emitted: list[str] = []
        original_out = sys.stdout
        original_err = sys.stderr
        try:
            uninstall = install_stray_stream_capture(emitted.append)
            # Write via the installed wrappers — simulating any library.
            sys.stdout.write("hello\n")
            sys.stderr.write("boom\n")
            assert emitted == [
                "\x1b[2m· hello\x1b[0m\n",
                "\x1b[31m! boom\x1b[0m\n",
            ]
        finally:
            uninstall()
            if sys.stdout is not original_out:
                sys.stdout = original_out
            if sys.stderr is not original_err:
                sys.stderr = original_err

    def test_uninstall_flushes_partial_lines(self):
        """A partial line sitting in the buffer at shutdown must not be
        lost — flush on uninstall so it lands in the transcript."""
        emitted: list[str] = []
        original_out = sys.stdout
        try:
            uninstall = install_stray_stream_capture(emitted.append)
            sys.stdout.write("half-line-no-newline")
            assert emitted == []  # buffered
            uninstall()
            assert any("half-line-no-newline" in chunk for chunk in emitted)
        finally:
            if sys.stdout is not original_out:
                sys.stdout = original_out


class TestDeduplication:
    """Tests for consecutive line deduplication."""

    def test_repeated_lines_collapsed(self):
        """Repeated identical lines produce one emit + a dedup summary."""
        emitted: list[str] = []
        from nooa_cli.tui.stream_forwarder import _StrayStreamForwarder

        fwd = _StrayStreamForwarder(sys.stdout, emitted.append, prefix="· ", ansi_color="2")
        fwd.write("rate limit\n")
        fwd.write("rate limit\n")
        fwd.write("rate limit\n")
        fwd.write("different line\n")

        # First "rate limit" emitted immediately, then summary "(×2 more)",
        # then "different line" emitted.
        assert len(emitted) == 3
        assert "rate limit" in emitted[0]
        assert "×2 more" in emitted[1]
        assert "different line" in emitted[2]

    def test_single_line_no_dedup_suffix(self):
        """A single line (no repeats) emits normally without dedup suffix."""
        emitted: list[str] = []
        from nooa_cli.tui.stream_forwarder import _StrayStreamForwarder

        fwd = _StrayStreamForwarder(sys.stdout, emitted.append, prefix="· ", ansi_color="2")
        fwd.write("hello\n")
        fwd.write("world\n")
        assert len(emitted) == 2
        assert "×" not in emitted[0]
        assert "×" not in emitted[1]

    def test_flush_emits_dedup_summary(self):
        """Calling flush() emits pending dedup summary."""
        emitted: list[str] = []
        from nooa_cli.tui.stream_forwarder import _StrayStreamForwarder

        fwd = _StrayStreamForwarder(sys.stdout, emitted.append, prefix="· ", ansi_color="2")
        fwd.write("retry\n")
        fwd.write("retry\n")
        fwd.write("retry\n")
        fwd.flush()
        # First emit + summary
        assert len(emitted) == 2
        assert "×2 more" in emitted[1]


class TestBlankLineSuppression:
    """Tests for blank/whitespace-only line suppression."""

    def test_blank_lines_suppressed(self):
        """Empty lines are not emitted."""
        emitted: list[str] = []
        from nooa_cli.tui.stream_forwarder import _StrayStreamForwarder

        fwd = _StrayStreamForwarder(sys.stdout, emitted.append, prefix="· ", ansi_color="2")
        fwd.write("\n")
        fwd.write("   \n")
        fwd.write("\n")
        assert emitted == []

    def test_whitespace_only_suppressed(self):
        """Whitespace-only lines are suppressed."""
        emitted: list[str] = []
        from nooa_cli.tui.stream_forwarder import _StrayStreamForwarder

        fwd = _StrayStreamForwarder(sys.stdout, emitted.append, prefix="· ", ansi_color="2")
        fwd.write("  \t  \n")
        assert emitted == []


class TestOnStrayCallback:
    """Tests for the on_stray callback mechanism."""

    def test_on_stray_called_for_emitted_lines(self):
        """on_stray receives (content, 'emitted') for lines that are displayed."""
        emitted: list[str] = []
        stray_calls: list[tuple[str, str]] = []
        from nooa_cli.tui.stream_forwarder import _StrayStreamForwarder

        fwd = _StrayStreamForwarder(
            sys.stdout,
            emitted.append,
            prefix="· ",
            ansi_color="2",
            on_stray=lambda c, d: stray_calls.append((c, d)),
        )
        fwd.write("visible line\n")
        assert len(stray_calls) == 1
        assert stray_calls[0] == ("visible line", "emitted")

    def test_on_stray_called_for_repeated_lines(self):
        """on_stray receives (content, 'repeated') for deduplicated lines."""
        stray_calls: list[tuple[str, str]] = []
        from nooa_cli.tui.stream_forwarder import _StrayStreamForwarder

        fwd = _StrayStreamForwarder(
            sys.stdout,
            lambda _: None,
            prefix="· ",
            ansi_color="2",
            on_stray=lambda c, d: stray_calls.append((c, d)),
        )
        fwd.write("same\n")
        fwd.write("same\n")
        assert ("same", "emitted") in stray_calls
        assert ("same", "repeated") in stray_calls

    def test_on_stray_called_for_suppressed_noise(self):
        """on_stray receives (content, 'suppressed') for noise-pattern lines."""
        stray_calls: list[tuple[str, str]] = []
        from nooa_cli.tui.stream_forwarder import _StrayStreamForwarder

        fwd = _StrayStreamForwarder(
            sys.stdout,
            lambda _: None,
            prefix="· ",
            ansi_color="2",
            on_stray=lambda c, d: stray_calls.append((c, d)),
        )
        fwd.write("Give Feedback / Get Help: something\n")
        assert len(stray_calls) == 1
        assert stray_calls[0][1] == "suppressed"
