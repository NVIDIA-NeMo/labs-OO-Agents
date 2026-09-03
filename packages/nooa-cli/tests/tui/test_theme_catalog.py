# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Theme catalog validation and discovery tests."""

from __future__ import annotations

import textwrap

import pytest
from nooa_cli.tui.theme_catalog import (
    ThemeRecord,
    contrast_ratio,
    load_user_themes,
    normalize_color,
    parse_theme,
    validate_palette,
)


def _base16(**extra):
    data = {
        "base00": "181818",
        "base01": "282828",
        "base02": "383838",
        "base03": "585858",
        "base04": "b8b8b8",
        "base05": "d8d8d8",
        "base06": "e8e8e8",
        "base07": "f8f8f8",
        "base08": "ab4642",
        "base09": "dc9656",
        "base0A": "f7ca88",
        "base0B": "a1b56c",
        "base0C": "86c1b9",
        "base0D": "7cafc2",
        "base0E": "ba8baf",
        "base0F": "a16946",
    }
    return {"scheme": "Test Scheme", "slug": "test-scheme", **data, **extra}


def test_normalize_color_accepts_hashless_rgb_and_rejects_style_strings() -> None:
    assert normalize_color("AABBCC") == "#aabbcc"
    with pytest.raises(ValueError, match="six-digit RGB"):
        normalize_color("bold red")


def test_base16_theme_maps_standard_colors_to_nooa_roles() -> None:
    record = parse_theme(_base16(variant="dark"), fallback_id="unused", source="test")

    assert record.id == "test-scheme"
    assert record.name == "Test Scheme"
    assert record.palette["base"] == "#181818"
    assert record.palette["text"] == "#d8d8d8"
    assert record.palette["red"] == "#ab4642"
    assert record.palette["green"] == "#a1b56c"
    assert record.palette["blue"] == "#7cafc2"
    assert record.palette["selection_bg"] == "#383838"


def test_standard_solarized_theme_gets_accessible_derived_highlights() -> None:
    data = {
        "scheme": "Solarized Dark",
        "slug": "solarized-dark",
        "variant": "dark",
        "base00": "002b36",
        "base01": "073642",
        "base02": "586e75",
        "base03": "657b83",
        "base04": "839496",
        "base05": "93a1a1",
        "base06": "eee8d5",
        "base07": "fdf6e3",
        "base08": "dc322f",
        "base09": "cb4b16",
        "base0A": "b58900",
        "base0B": "859900",
        "base0C": "2aa198",
        "base0D": "268bd2",
        "base0E": "6c71c4",
        "base0F": "d33682",
    }

    record = parse_theme(data, fallback_id="solarized-dark", source="test")

    for role in ("search_current", "inline_code"):
        assert contrast_ratio(record.palette[f"{role}_fg"], record.palette[f"{role}_bg"]) >= 4.5
    assert contrast_ratio(record.palette["inline_code_bg"], record.palette["base"]) >= 3


def test_base16_theme_accepts_semantic_overrides() -> None:
    record = parse_theme(
        _base16(
            id="custom",
            variant="dark",
            inline_code_fg="#11111b",
            inline_code_bg="#89b4fa",
            search_match_fg="#11111b",
            search_current_fg="#11111b",
            diff_added="#a6e3a1",
            diff_removed="#f38ba8",
        ),
        fallback_id="unused",
        source="test",
    )

    assert record.id == "custom"
    assert record.palette["inline_code_bg"] == "#89b4fa"
    assert record.palette["diff_added"] == "#a6e3a1"
    assert record.source == "test"


def test_layered_theme_discovery_prefers_project_and_isolates_bad_files(
    tmp_path, monkeypatch
) -> None:
    user = tmp_path / "user"
    project = tmp_path / "project"
    (user / "themes").mkdir(parents=True)
    (project / "themes").mkdir(parents=True)
    (user / "themes" / "shared.yaml").write_text(
        textwrap.dedent(
            """
            scheme: User copy
            slug: shared
            variant: dark
            base00: '000000'
            base01: '010101'
            base02: '020202'
            base03: '030303'
            base04: '808080'
            base05: 'eeeeee'
            base06: 'f0f0f0'
            base07: 'ffffff'
            base08: 'ff0000'
            base09: 'ff8800'
            base0A: 'ffff00'
            base0B: '00ff00'
            base0C: '00ffff'
            base0D: '0088ff'
            base0E: 'ff00ff'
            base0F: '884400'
            syntax_theme: monokai
            """
        )
    )
    (project / "themes" / "shared.yaml").write_text(
        (user / "themes" / "shared.yaml").read_text().replace("User copy", "Prøject copy")
    )
    (project / "themes" / "bad.yaml").write_text("palette: [not, a, mapping]\n")
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project))
    builtin = ThemeRecord("builtin", "Builtin", {}, "monokai", "dark", "builtin")

    records, diagnostics = load_user_themes({"builtin": builtin})

    assert records["shared"].name == "Prøject copy"
    assert records["shared"].source.startswith("project:")
    assert len(diagnostics) == 1
    assert "bad.yaml" in diagnostics[0]


def test_partial_base24_extension_is_rejected() -> None:
    data = _base16(base10="ffffff")
    with pytest.raises(ValueError, match="base10 through base17"):
        parse_theme(data, fallback_id="partial", source="test")


def test_unreadable_base16_theme_is_rejected() -> None:
    data = _base16(**{f"base{index:02X}": "777777" for index in range(16)})
    with pytest.raises(ValueError, match="insufficient contrast"):
        parse_theme(data, fallback_id="unreadable", source="test")


def test_complete_base24_extension_is_accepted() -> None:
    data = _base16(
        variant="dark",
        **{f"base{index:02X}": "ffffff" for index in range(16, 24)},
    )
    record = parse_theme(data, fallback_id="base24", source="test")
    assert record.id == "test-scheme"
    assert record.palette["feedback_success"] == "#ffffff"


def test_feedback_text_requires_wcag_aa_contrast() -> None:
    from nooa_cli.tui import theme

    palette = dict(theme.THEMES["mocha"])
    palette["feedback_info"] = "#777777"
    palette["base"] = "#222222"
    with pytest.raises(ValueError, match=r"info \(3\.[0-9]+:1\)"):
        validate_palette(palette)


def test_builtins_define_valid_semantic_roles() -> None:
    from nooa_cli.tui.theme import THEME_RECORDS

    for record in THEME_RECORDS.values():
        validate_palette(record.palette)


def test_builtins_expose_every_semantic_role() -> None:
    from nooa_cli.tui.theme import THEME_RECORDS
    from nooa_cli.tui.theme_catalog import SEMANTIC_KEYS

    for record in THEME_RECORDS.values():
        assert set(SEMANTIC_KEYS) <= record.palette.keys()
        validate_palette(record.palette)
