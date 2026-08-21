# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nooa_cli.tui.session import _build_user_bar
from nooa_cli.tui.terminal_safety import (
    hyperlink_at_plain_offset,
    normalize_transcript_block,
    sanitize_transcript_ansi,
    strip_safe_ansi,
)
from rich.cells import cell_len


def test_transcript_normalizer_allows_style_but_exposes_terminal_commands() -> None:
    raw = "\x1b[31mred\x1b[0m\x1b[2J\x1b]52;c;Y2xpcGJvYXJk\x07\r\x07\x08"

    normalized = normalize_transcript_block(raw)

    assert "\x1b[31m" in normalized
    assert "\x1b[2J" not in normalized
    assert "\x1b]52;" not in normalized
    assert "\x07" not in normalized
    assert "\x08" not in normalized
    visible = strip_safe_ansi(normalized)
    assert r"\x1b[2J" in visible
    assert r"\x1b]52;c;Y2xpcGJvYXJk\x07" in visible
    assert r"\r\x07\x08" in visible
    assert visible.endswith("\n")


def test_transcript_normalizer_preserves_well_formed_hyperlinks() -> None:
    hyperlink = "\x1b]8;;https://example.com\x1b\\example\x1b]8;;\x1b\\\n"
    normalized = normalize_transcript_block(hyperlink)
    assert hyperlink in normalized
    assert normalized.endswith("\x1b]8;;\x1b\\")


def test_hyperlink_cannot_smuggle_a_c1_string_terminator() -> None:
    smuggled = "\x1b]8;;https://example.com\x9c\x1b[2Jowned\x07"
    normalized = normalize_transcript_block(smuggled)

    assert "\x1b]8;" not in normalized
    assert "\x9c" not in normalized
    assert "\x1b[2J" not in normalized
    assert r"\x9c\x1b[2J" in normalized


def test_private_csi_ending_in_m_is_not_mistaken_for_graphic_rendition() -> None:
    # XTMODKEYS uses final byte ``m`` but changes terminal input behavior; only
    # the ordinary numeric/semicolon/colon SGR grammar is presentation-safe.
    normalized = normalize_transcript_block("before\x1b[>4;2mafter")

    assert "\x1b[>4;2m" not in normalized
    assert r"\x1b[>4;2m" in normalized


def test_transcript_normalizer_wraps_ansi_and_graphemes_inside_safe_width() -> None:
    normalized = normalize_transcript_block(
        "\x1b[31mab界cafe\N{COMBINING ACUTE ACCENT}👨‍👩‍👧‍👦z\x1b[0m",
        columns=5,
    )

    visible_lines = strip_safe_ansi(normalized).splitlines()
    assert "".join(visible_lines) == "ab界café👨‍👩‍👧‍👦z"
    assert all(cell_len(line) <= 5 for line in visible_lines)


def test_grapheme_wider_than_pathological_safe_width_is_replaced() -> None:
    normalized = normalize_transcript_block("界", columns=1)

    assert strip_safe_ansi(normalized) == "�\n"


def test_user_bar_is_control_safe_cell_aware_and_reserves_final_column() -> None:
    class _App:
        @staticmethod
        def transcript_columns() -> int:
            return 9  # ten physical columns, with one deliberately reserved

    colors = {"text": "#cdd6f4", "surface2": "#585b70"}
    bar = _build_user_bar("wide 界\x1b[2J\r", _App(), colors)  # type: ignore[arg-type]

    assert "\x1b[2J" not in bar
    visible_lines = strip_safe_ansi(sanitize_transcript_ansi(bar)).splitlines()
    assert visible_lines
    assert all(cell_len(line) == 9 for line in visible_lines)


def test_hyperlink_hit_testing_accepts_only_http_targets() -> None:
    linked = "before \x1b]8;id=7;https://example.test/path\x1b\\label\x1b]8;;\x1b\\ after"

    assert hyperlink_at_plain_offset(linked, 7) == "https://example.test/path"
    assert hyperlink_at_plain_offset(linked, 11) == "https://example.test/path"
    assert hyperlink_at_plain_offset(linked, 6) is None
    assert hyperlink_at_plain_offset(linked, 12) is None

    unsafe = "\x1b]8;;file:///tmp/secret\x1b\\local\x1b]8;;\x1b\\"
    assert hyperlink_at_plain_offset(unsafe, 0) is None
