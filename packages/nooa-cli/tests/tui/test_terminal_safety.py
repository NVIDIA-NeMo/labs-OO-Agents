# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nooa_cli.tui.session import _build_user_bar
from nooa_cli.tui.terminal_safety import (
    hyperlink_at_plain_offset,
    normalize_transcript_block,
    safe_http_url,
    safe_hyperlink_spans,
    safe_hyperlink_target,
    sanitize_transcript_ansi,
    strip_safe_ansi,
)
from rich.cells import cell_len


def test_visible_code_line_uses_terminal_cells_for_tab_stops() -> None:
    from nooa_cli.tui.copyable_markdown import visible_code_line

    rendered, source_map = visible_code_line("界\t")

    assert rendered == "界  "
    assert source_map == ((0, 1), (1, 2), (1, 2))


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
    assert bar.startswith("\x1b[38;5;59;49;7m" + "▔" * 9)
    assert "\x1b[38;5;189;48;5;59m" in bar

    assert "\x1b[2J" not in bar
    visible_lines = strip_safe_ansi(sanitize_transcript_ansi(bar)).splitlines()
    assert len(visible_lines) >= 3
    assert visible_lines[0] == "▔" * 9
    assert visible_lines[-1] == "▁" * 9
    assert all(cell_len(line) == 9 for line in visible_lines)

    styled_lines = bar.splitlines()
    edge_style = "\x1b[38;5;59;49;7m"
    assert styled_lines[0] == edge_style + "▔" * 9 + "\x1b[0m"
    assert styled_lines[-1] == edge_style + "▁" * 9 + "\x1b[0m"

    from prompt_toolkit.formatted_text import ANSI, to_formatted_text

    parsed_edge_style = to_formatted_text(ANSI(styled_lines[0]))[0][0]
    assert parsed_edge_style == "#5f5f5f bg:ansidefault reverse"


def test_hyperlink_target_length_is_bounded() -> None:
    prefix = "https://example.test/"
    exact = prefix + "a" * (2_048 - len(prefix))

    assert safe_http_url(exact) == exact
    assert safe_http_url(exact + "a") is None


def test_safe_http_url_rejects_terminal_controls() -> None:
    for codepoint in (*range(0x20), 0x7F, *range(0x80, 0xA0)):
        url = f"https://example.test/a{chr(codepoint)}b"
        assert safe_http_url(url) is None, f"accepted U+{codepoint:04X}"


def test_safe_hyperlink_target_accepts_absolute_file_urls_only() -> None:
    assert safe_hyperlink_target("file:///tmp/example.py") == "file:///tmp/example.py"
    assert safe_hyperlink_target("file://path/to/file") == "file:///path/to/file"
    assert safe_hyperlink_target("file://localhost/tmp/example.py") == (
        "file://localhost/tmp/example.py"
    )
    assert safe_hyperlink_target("file:relative/path") is None
    assert safe_hyperlink_target("file:////server/share") is None
    assert safe_hyperlink_target("file://localhost//server/share") is None
    assert safe_hyperlink_target("file:///%2Fserver/share") is None
    assert safe_hyperlink_target("file://%2Fserver/share") is None
    assert safe_hyperlink_target("file://%5C%5Cserver/share") is None
    assert safe_hyperlink_target(r"file:///\\server\share") is None
    assert safe_hyperlink_target("file:///tmp/%00unsafe") is None
    assert safe_hyperlink_target("javascript:alert(1)") is None


def test_safe_hyperlink_spans_reports_safe_label_bounds() -> None:
    linked = "before \x1b]8;id=7;https://example.test/path\x1b\\label\x1b]8;;\x1b\\ after"
    local = "\x1b]8;;file:///tmp/example.py\x1b\\local\x1b]8;;\x1b\\"
    unsafe = "\x1b]8;;javascript:alert(1)\x1b\\script\x1b]8;;\x1b\\"

    assert safe_hyperlink_spans(linked) == ((7, 12, "https://example.test/path"),)
    assert safe_hyperlink_spans(local) == ((0, 5, "file:///tmp/example.py"),)
    assert safe_hyperlink_spans(unsafe) == ()


def test_hyperlink_hit_testing_accepts_safe_browser_and_file_targets() -> None:
    linked = "before \x1b]8;id=7;https://example.test/path\x1b\\label\x1b]8;;\x1b\\ after"

    assert hyperlink_at_plain_offset(linked, 7) == "https://example.test/path"
    assert hyperlink_at_plain_offset(linked, 11) == "https://example.test/path"
    assert hyperlink_at_plain_offset(linked, 6) is None
    assert hyperlink_at_plain_offset(linked, 12) is None

    local = "\x1b]8;;file:///tmp/example.py\x1b\\local\x1b]8;;\x1b\\"
    assert hyperlink_at_plain_offset(local, 0) == "file:///tmp/example.py"

    unsafe = "\x1b]8;;javascript:alert(1)\x1b\\script\x1b]8;;\x1b\\"
    assert hyperlink_at_plain_offset(unsafe, 0) is None
