# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive true-color preview for Markdown inline-code palettes."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import TextIO

RESET = "\x1b[0m"
CLEAR = "\x1b[2J\x1b[H"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"


@dataclass(frozen=True)
class InlineCodePalette:
    """One inline-code foreground/background candidate."""

    name: str
    description: str
    foreground: str
    background: str


CANDIDATES: dict[str, tuple[InlineCodePalette, ...]] = {
    "mocha": (
        InlineCodePalette("Balanced", "blue chip", "#11111b", "#89b4fa"),
        InlineCodePalette("Cool", "teal chip", "#11111b", "#94e2d5"),
        InlineCodePalette("Warm", "peach chip", "#11111b", "#fab387"),
    ),
    "latte": (
        InlineCodePalette("Balanced", "blue chip", "#ffffff", "#1e66f5"),
        InlineCodePalette("Cool", "teal chip", "#ffffff", "#007b83"),
        InlineCodePalette("Warm", "rose chip", "#ffffff", "#a83f55"),
    ),
    "vsdark": (
        InlineCodePalette("Balanced", "blue chip", "#1e1e1e", "#9cdcfe"),
        InlineCodePalette("Cool", "teal chip", "#1e1e1e", "#4ec9b0"),
        InlineCodePalette("Warm", "peach chip", "#1e1e1e", "#ce9178"),
    ),
    "vslight": (
        InlineCodePalette("Balanced", "blue chip", "#ffffff", "#005fb8"),
        InlineCodePalette("Cool", "green chip", "#ffffff", "#107c10"),
        InlineCodePalette("Warm", "red chip", "#ffffff", "#a31515"),
    ),
}


def _luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio for two ``#rrggbb`` colors."""
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _paint(text: str, candidate: InlineCodePalette, *, enabled: bool) -> str:
    if not enabled:
        return text
    fg = tuple(int(candidate.foreground[index : index + 2], 16) for index in (1, 3, 5))
    bg = tuple(int(candidate.background[index : index + 2], 16) for index in (1, 3, 5))
    return f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]};48;2;{bg[0]};{bg[1]};{bg[2]}m{text}{RESET}"


def render(theme: str, selected: int = 0, *, color: bool = True) -> str:
    """Render three inline-code choices for one theme."""
    lines = [
        f"NOOA INLINE-CODE LAB  ·  theme: {theme}",
        "The actual `inline code` chip is previewed below for every option.",
        "",
    ]
    for index, candidate in enumerate(CANDIDATES[theme]):
        marker = "❯" if index == selected else " "
        sample = _paint(" inline code ", candidate, enabled=color)
        command = _paint("uv run pytest", candidate, enabled=color)
        path = _paint("theme.py", candidate, enabled=color)
        lines.extend(
            [
                f"{marker} {index + 1}. {candidate.name} — {candidate.description}",
                f"   Use {sample} inside prose; run {command} after editing {path}.",
                f"   {candidate.foreground} on {candidate.background}  "
                f"contrast {contrast_ratio(candidate.foreground, candidate.background):.2f}:1",
                "",
            ]
        )
    lines.append(
        "←/→ or h/l: theme   ↑/↓ or j/k: choice   1–3: choice   Enter: print choice   q: quit"
    )
    return "\n".join(lines)


def _read_key() -> str:
    if os.name == "nt":  # pragma: no cover - Windows-only fallback
        import msvcrt

        key = msvcrt.getwch()
        if key in {"\x00", "\xe0"}:
            scan_code = msvcrt.getwch()
            return {
                "H": "\x1b[A",
                "P": "\x1b[B",
                "K": "\x1b[D",
                "M": "\x1b[C",
            }.get(scan_code, key + scan_code)
        return key
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    old = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        first = sys.stdin.read(1)
        return first + sys.stdin.read(2) if first == "\x1b" else first
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, old)


def run_interactive(theme: str, *, output: TextIO = sys.stdout) -> int:
    """Run the keyboard-driven preview and print the selected candidate."""
    names = list(CANDIDATES)
    theme_index = names.index(theme)
    selected = 0
    output.write(HIDE_CURSOR)
    try:
        while True:
            current_theme = names[theme_index]
            output.write(CLEAR + render(current_theme, selected) + "\n")
            output.flush()
            key = _read_key()
            if key in {"q", "Q", "\x03"}:
                return 0
            if key in {"\x1b[D", "h", "H"}:
                theme_index = (theme_index - 1) % len(names)
            elif key in {"\x1b[C", "l", "L"}:
                theme_index = (theme_index + 1) % len(names)
            elif key in {"\x1b[A", "k", "K"}:
                selected = (selected - 1) % 3
            elif key in {"\x1b[B", "j", "J"}:
                selected = (selected + 1) % 3
            elif key in {"1", "2", "3"}:
                selected = int(key) - 1
            elif key in {"\r", "\n"}:
                choice = CANDIDATES[current_theme][selected]
                output.write(
                    CLEAR
                    + f"Selected: {current_theme} / {choice.name}\n"
                    + f"inline code: {choice.foreground} on {choice.background}\n"
                )
                return 0
    finally:
        output.write(SHOW_CURSOR)
        output.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare NOOA inline-code highlight palettes")
    parser.add_argument("--theme", choices=CANDIDATES, default="mocha")
    parser.add_argument("--plain", action="store_true", help="print without color or controls")
    args = parser.parse_args(argv)
    if args.plain:
        print("\n\n".join(render(name, color=False) for name in CANDIDATES))
        return 0
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("interactive preview requires a terminal; use --plain for text output")
    return run_interactive(args.theme)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
