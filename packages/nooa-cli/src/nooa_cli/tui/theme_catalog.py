# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discovery and validation for built-in, native, and Base16 TUI themes."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import yaml
from pygments.styles import get_style_by_name

logger = logging.getLogger(__name__)
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_MAX_THEME_BYTES = 64 * 1024
_MAX_THEME_FILES = 64

SEMANTIC_KEYS = (
    "text_primary",
    "text_muted",
    "text_subtle",
    "surface_raised",
    "border_default",
    "feedback_success",
    "feedback_error",
    "feedback_warning",
    "feedback_info",
    "selection_fg",
    "selection_bg",
    "search_match_fg",
    "search_match_bg",
    "search_current_fg",
    "search_current_bg",
    "focus_accent",
    "user_message_fg",
    "user_message_bg",
    "inline_code_fg",
    "inline_code_bg",
    "code_path",
    "code_number",
    "diff_added",
    "diff_removed",
)


@dataclass(frozen=True, slots=True)
class ThemeRecord:
    id: str
    name: str
    palette: dict[str, str]
    syntax_theme: str
    variant: str
    source: str
    description: str = ""
    author: str = ""


def normalize_color(value: object) -> str:
    """Normalize a six-digit RGB value and reject style-string injection."""
    text = str(value).strip()
    if not text.startswith("#"):
        text = "#" + text
    if not _HEX.fullmatch(text):
        raise ValueError(f"Expected six-digit RGB color, got {value!r}")
    return text.lower()


def semanticize(palette: dict[str, str]) -> dict[str, str]:
    """Fill documented semantic UI roles from one normalized base palette."""
    result = dict(palette)
    defaults = {
        "text_primary": result["text"],
        "text_muted": result["subtext1"],
        "text_subtle": result["overlay1"],
        "surface_raised": result["surface0"],
        "border_default": result["surface2"],
        "feedback_success": result["green"],
        "feedback_error": result["red"],
        "feedback_warning": result["yellow"],
        "feedback_info": result["blue"],
        "selection_fg": result["text"],
        "selection_bg": result["surface2"],
        "search_match_fg": result["crust"],
        "search_match_bg": result["yellow"],
        "search_current_fg": result["crust"],
        "search_current_bg": result["sky"],
        "focus_accent": result["lavender"],
        "user_message_fg": result["text"],
        "user_message_bg": result["surface2"],
        "inline_code_fg": result["text"],
        "inline_code_bg": result["surface0"],
        "code_path": result["teal"],
        "code_number": result["peach"],
    }
    for key, value in defaults.items():
        result.setdefault(key, value)
    result.setdefault("diff_added", result["feedback_success"])
    result.setdefault("diff_removed", result["feedback_error"])
    return result


def _base16_palette(data: dict[str, Any]) -> dict[str, str]:
    folded = {str(key).lower(): value for key, value in data.items()}
    bases = {f"base{index:02X}": normalize_color(folded[f"base{index:02x}"]) for index in range(16)}
    palette = semanticize(
        {
            "base": bases["base00"],
            "mantle": bases["base01"],
            "crust": bases["base00"],
            "surface0": bases["base01"],
            "surface1": bases["base02"],
            "surface2": bases["base03"],
            "overlay0": bases["base03"],
            "overlay1": bases["base04"],
            "overlay2": bases["base04"],
            "text": bases["base05"],
            "subtext1": bases["base04"],
            "subtext0": bases["base03"],
            "red": bases["base08"],
            "maroon": bases["base08"],
            "peach": bases["base09"],
            "yellow": bases["base0A"],
            "green": bases["base0B"],
            "teal": bases["base0C"],
            "sky": bases["base0C"],
            "sapphire": bases["base0D"],
            "blue": bases["base0D"],
            "lavender": bases["base0D"],
            "mauve": bases["base0E"],
            "pink": bases["base0E"],
            "rosewater": bases["base0F"],
            "flamingo": bases["base0F"],
        }
    )
    dark, light = bases["base00"], bases["base07"]
    for role, background in (
        ("selection", bases["base02"]),
        ("search_match", bases["base0A"]),
        ("search_current", bases["base0D"]),
        ("inline_code", bases["base0D"]),
    ):
        foreground = _contrast_foreground(background, dark, light)
        if contrast_ratio(foreground, background) < 4.5:
            foreground, background = palette["text_primary"], palette["surface_raised"]
        if role == "inline_code" and contrast_ratio(background, palette["base"]) < 3:
            candidates = (
                bases["base0A"],
                bases["base0B"],
                bases["base0C"],
                bases["base04"],
                bases["base05"],
                bases["base06"],
                bases["base07"],
            )
            for candidate in candidates:
                candidate_foreground = _contrast_foreground(candidate, dark, light)
                if (
                    contrast_ratio(candidate_foreground, candidate) >= 4.5
                    and contrast_ratio(candidate, palette["base"]) >= 3
                ):
                    foreground, background = candidate_foreground, candidate
                    break
        palette[f"{role}_fg"] = foreground
        palette[f"{role}_bg"] = background
    palette["user_message_fg"] = palette["selection_fg"]
    palette["user_message_bg"] = palette["selection_bg"]
    base24_keys = {f"base{index:02x}" for index in range(16, 24)}
    focus_candidates = [bases["base0D"], bases["base0B"], bases["base0A"]]
    if base24_keys <= folded.keys():
        palette.update(
            feedback_error=normalize_color(folded["base11"]),
            feedback_success=normalize_color(folded["base12"]),
            feedback_warning=normalize_color(folded["base13"]),
            feedback_info=normalize_color(folded["base14"]),
        )
        focus_candidates.append(normalize_color(folded["base14"]))
    for role in ("success", "error", "warning", "info"):
        key = f"feedback_{role}"
        if contrast_ratio(palette[key], palette["base"]) < 4.5:
            palette[key] = palette["text_primary"]
    palette["diff_added"] = palette["feedback_success"]
    palette["diff_removed"] = palette["feedback_error"]
    palette["focus_accent"] = max(
        focus_candidates,
        key=lambda color: contrast_ratio(color, palette["base"]),
    )
    return palette


def parse_theme(data: dict[str, Any], *, fallback_id: str, source: str) -> ThemeRecord:
    """Parse a Base16/Base24 YAML mapping with optional semantic overrides."""
    theme_id = str(data.get("id") or data.get("slug") or fallback_id).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", theme_id):
        raise ValueError(f"Invalid theme id {theme_id!r}")
    variant_value = data.get("variant")
    variant = str(variant_value) if variant_value is not None else ""
    if variant and variant not in {"dark", "light"}:
        raise ValueError("variant must be dark or light")
    folded_keys = {str(key).lower() for key in data}
    base24 = {f"base{index:02x}" for index in range(16, 24)}
    present_base24 = base24 & folded_keys
    if present_base24 and present_base24 != base24:
        raise ValueError("Base24 extensions must include every key from base10 through base17")

    missing = [f"base{index:02X}" for index in range(16) if f"base{index:02x}" not in folded_keys]
    if missing:
        raise ValueError("theme must contain all Base16 base00..base0F colors")
    palette = _base16_palette(data)
    for key in SEMANTIC_KEYS:
        if key in data:
            palette[key] = normalize_color(data[key])
    if not variant:
        variant = "dark" if _luminance(palette["base"]) < 0.5 else "light"
    syntax = str(data.get("syntax_theme") or ("github-dark" if variant == "dark" else "vs"))
    get_style_by_name(syntax)
    validate_palette(palette)
    return ThemeRecord(
        theme_id,
        str(data.get("name") or data.get("scheme") or theme_id),
        palette,
        syntax,
        variant,
        source,
        str(data.get("description") or ""),
        str(data.get("author") or ""),
    )


def _luminance(color: str) -> float:
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio for two normalized RGB colors."""
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _contrast_foreground(background: str, dark: str, light: str) -> str:
    """Choose whichever supplied foreground is more readable on *background*."""
    return max((dark, light), key=lambda color: contrast_ratio(color, background))


def validate_palette(palette: dict[str, str]) -> None:
    """Reject themes whose required text/highlight pairs are not accessible."""
    text_pairs = {
        "primary text": (palette["text_primary"], palette["base"]),
        "selection": (palette["selection_fg"], palette["selection_bg"]),
        "user message": (palette["user_message_fg"], palette["user_message_bg"]),
        "search match": (palette["search_match_fg"], palette["search_match_bg"]),
        "current search match": (palette["search_current_fg"], palette["search_current_bg"]),
        "inline code": (palette["inline_code_fg"], palette["inline_code_bg"]),
        "success": (palette["feedback_success"], palette["base"]),
        "error": (palette["feedback_error"], palette["base"]),
        "warning": (palette["feedback_warning"], palette["base"]),
        "info": (palette["feedback_info"], palette["base"]),
        "diff added": (palette["diff_added"], palette["base"]),
        "diff removed": (palette["diff_removed"], palette["base"]),
    }
    failures = [
        f"{role} ({contrast_ratio(foreground, background):.2f}:1)"
        for role, (foreground, background) in text_pairs.items()
        if contrast_ratio(foreground, background) < 4.5
    ]
    accents = {
        "focus accent": palette["focus_accent"],
        "inline-code surface": palette["inline_code_bg"],
    }
    failures.extend(
        f"{role} ({contrast_ratio(color, palette['base']):.2f}:1)"
        for role, color in accents.items()
        if contrast_ratio(color, palette["base"]) < 3
    )
    if failures:
        raise ValueError("insufficient contrast: " + ", ".join(failures))


def load_user_themes(builtins: dict[str, ThemeRecord]) -> tuple[dict[str, ThemeRecord], list[str]]:
    """Overlay user then project theme files; malformed files become diagnostics."""
    from nooa.paths import get_project_dir, get_user_dir

    records = dict(builtins)
    diagnostics: list[str] = []
    for label, directory in (
        ("user", get_user_dir("themes")),
        ("project", get_project_dir("themes")),
    ):
        if not directory.is_dir():
            continue
        paths = sorted((*directory.glob("*.yaml"), *directory.glob("*.yml")))[:_MAX_THEME_FILES]
        for path in paths:
            try:
                if not path.is_file() or path.stat().st_size > _MAX_THEME_BYTES:
                    raise ValueError("theme file is not regular or exceeds 64 KiB")
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("theme document must be a mapping")
                record = parse_theme(data, fallback_id=path.stem, source=f"{label}:{path}")
                records[record.id] = record
            except Exception as exc:
                message = f"Skipped theme {path}: {exc}"
                diagnostics.append(message)
                logger.warning(message)
    return records, diagnostics
