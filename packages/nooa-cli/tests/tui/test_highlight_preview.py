# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone inline-code palette preview."""

from nooa_cli.tui import theme
from nooa_cli.tui.highlight_preview import CANDIDATES, _read_key, contrast_ratio, main, render


def test_preview_offers_three_accessible_candidates_per_theme() -> None:
    assert CANDIDATES.keys() == theme.THEMES.keys()
    for theme_name, candidates in CANDIDATES.items():
        assert len(candidates) == 3
        assert candidates[0].name == "Balanced"
        for candidate in candidates:
            assert contrast_ratio(candidate.foreground, candidate.background) >= 4.5
            assert contrast_ratio(candidate.background, theme.THEMES[theme_name]["base"]) >= 3


def test_balanced_candidate_matches_live_inline_code_palette() -> None:
    for name, candidates in CANDIDATES.items():
        balanced = candidates[0]
        palette = theme.THEMES[name]
        assert balanced.foreground == palette["inline_code_fg"]
        assert balanced.background == palette["inline_code_bg"]


def test_plain_preview_shows_inline_code_choices_without_ansi() -> None:
    output = render("latte", selected=2, color=False)

    assert "theme: latte" in output
    assert "1. Balanced" in output
    assert "2. Cool" in output
    assert "❯ 3. Warm" in output
    assert "inline code" in output
    assert "uv run pytest" in output
    assert "\x1b[" not in output


def test_plain_cli_prints_every_theme(capsys) -> None:
    assert main(["--plain"]) == 0
    output = capsys.readouterr().out
    for theme_name in CANDIDATES:
        assert f"theme: {theme_name}" in output


def test_windows_arrow_key_sequence_is_normalized(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    keys = iter(["\xe0", "H"])
    monkeypatch.setattr("nooa_cli.tui.highlight_preview.os.name", "nt")
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(getwch=lambda: next(keys)))

    assert _read_key() == "\x1b[A"
