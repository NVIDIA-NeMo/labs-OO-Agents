# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Theme system for the NOOA TUI.

Four built-in palettes — all use the same Catppuccin key names so every
consumer (frontend.py, console.py, splash.py) works without changes:

  mocha   — Catppuccin Mocha (dark)         [default]
  latte   — Catppuccin Latte (light)
  vsdark  — Visual Studio Code Dark+
  vslight — Visual Studio Code Light

Usage::

    from .theme import set_theme, COLORS, create_theme
    set_theme("latte")          # switches COLORS in-place
    console._theme = create_theme()
"""

from rich.console import Console, ConsoleOptions, RenderResult
from rich.syntax import Syntax
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Palettes — all share the same key names (Catppuccin semantic roles)
# ---------------------------------------------------------------------------

_MOCHA: dict[str, str] = {
    "rosewater": "#f5e0dc",
    "flamingo": "#f2cdcd",
    "pink": "#f5c2e7",
    "mauve": "#cba6f7",
    "red": "#f38ba8",
    "maroon": "#eba0ac",
    "peach": "#fab387",
    "yellow": "#f9e2af",
    "green": "#a6e3a1",
    "teal": "#94e2d5",
    "sky": "#89dceb",
    "sapphire": "#74c7ec",
    "blue": "#89b4fa",
    "lavender": "#b4befe",
    "text": "#cdd6f4",
    "subtext1": "#bac2de",
    "subtext0": "#a6adc8",
    "overlay2": "#9399b2",
    "overlay1": "#7f849c",
    "overlay0": "#6c7086",
    "surface2": "#585b70",
    "surface1": "#45475a",
    "surface0": "#313244",
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
}

_LATTE: dict[str, str] = {
    "rosewater": "#dc8a78",
    "flamingo": "#dd7878",
    "pink": "#ea76cb",
    "mauve": "#8839ef",
    "red": "#d20f39",
    "maroon": "#e64553",
    "peach": "#fe640b",
    "yellow": "#df8e1d",
    "green": "#40a02b",
    "teal": "#179299",
    "sky": "#04a5e5",
    "sapphire": "#209fb5",
    "blue": "#1e66f5",
    "lavender": "#7287fd",
    "text": "#4c4f69",
    "subtext1": "#5c5f77",
    "subtext0": "#6c6f85",
    "overlay2": "#7c7f93",
    "overlay1": "#8c8fa1",
    "overlay0": "#9ca0b0",
    "surface2": "#acb0be",
    "surface1": "#bcc0cc",
    "surface0": "#ccd0da",
    "base": "#eff1f5",
    "mantle": "#e6e9ef",
    "crust": "#dce0e8",
}

# VS Code Dark+ — keys mapped to Catppuccin semantic roles
_VS_DARK: dict[str, str] = {
    "rosewater": "#d4d4d4",
    "flamingo": "#d4d4d4",
    "pink": "#c586c0",
    "mauve": "#c586c0",
    "red": "#f48771",
    "maroon": "#f48771",
    "peach": "#ce9178",
    "yellow": "#dcdcaa",
    "green": "#4ec9b0",
    "teal": "#4ec9b0",
    "sky": "#9cdcfe",
    "sapphire": "#9cdcfe",
    "blue": "#569cd6",
    "lavender": "#9cdcfe",
    "text": "#d4d4d4",
    "subtext1": "#9d9d9d",
    "subtext0": "#808080",
    "overlay2": "#6a6a6a",
    "overlay1": "#5a5a5a",
    "overlay0": "#4a4a4a",
    "surface2": "#3e3e42",
    "surface1": "#333337",
    "surface0": "#252526",
    "base": "#1e1e1e",
    "mantle": "#181818",
    "crust": "#111111",
}

# VS Code Light — keys mapped to Catppuccin semantic roles
_VS_LIGHT: dict[str, str] = {
    "rosewater": "#000000",
    "flamingo": "#000000",
    "pink": "#af00db",
    "mauve": "#6f42c1",
    "red": "#a31515",
    "maroon": "#cd3131",
    "peach": "#a31515",
    "yellow": "#795e26",
    "green": "#008000",
    "teal": "#267f99",
    "sky": "#0451a5",
    "sapphire": "#0451a5",
    "blue": "#0000ff",
    "lavender": "#0451a5",
    "text": "#000000",
    "subtext1": "#444444",
    "subtext0": "#6a6a6a",
    "overlay2": "#737373",
    "overlay1": "#919191",
    "overlay0": "#b4b4b4",
    "surface2": "#cccccc",
    "surface1": "#e0e0e0",
    "surface0": "#f0f0f0",
    "base": "#ffffff",
    "mantle": "#f8f8f8",
    "crust": "#ececec",
}

THEMES: dict[str, dict[str, str]] = {
    "mocha": _MOCHA,
    "latte": _LATTE,
    "vsdark": _VS_DARK,
    "vslight": _VS_LIGHT,
}

# Pygments styles selected to match each UI palette. Syntax renderers keep
# their existing transparent/default background policy; this mapping only
# changes token colours.
SYNTAX_THEMES: dict[str, str] = {
    "mocha": "monokai",
    "latte": "gruvbox-light",
    "vsdark": "github-dark",
    "vslight": "vs",
}

# ---------------------------------------------------------------------------
# Active palette — a mutable dict updated in-place so that callers who did
# `from .theme import COLORS` keep a live reference after set_theme().
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = dict(_MOCHA)
_active_name: str = "mocha"


def get_theme() -> str:
    """Return the name of the currently active theme."""
    return _active_name


def get_syntax_theme(name: str | None = None) -> str:
    """Return the Pygments style paired with a built-in UI theme."""
    selected = _active_name if name is None else name
    try:
        return SYNTAX_THEMES[selected]
    except KeyError as exc:
        raise ValueError(f"Unknown theme {selected!r}. Choose from: {', '.join(THEMES)}") from exc


class ThemeSyntax(Syntax):
    """Syntax highlighting that follows the active UI theme when replayed."""

    def __init__(self, code: str, lexer: object, **kwargs: object) -> None:
        self._uses_active_theme = "theme" not in kwargs
        if self._uses_active_theme:
            kwargs["theme"] = get_syntax_theme()
        super().__init__(code, lexer, **kwargs)  # type: ignore[arg-type]

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self._uses_active_theme:
            self._theme = self.get_theme(get_syntax_theme())
        yield from super().__rich_console__(console, options)


def set_theme(name: str) -> None:
    """Switch the active theme.  Updates COLORS in-place.

    Args:
        name: One of ``mocha``, ``latte``, ``vsdark``, ``vslight``.

    Raises:
        ValueError: If *name* is not a known theme.
    """
    global _active_name
    if name not in THEMES:
        raise ValueError(f"Unknown theme {name!r}. Choose from: {', '.join(THEMES)}")
    _active_name = name
    COLORS.clear()
    COLORS.update(THEMES[name])


def create_theme() -> Theme:
    """Build a Rich Theme from the currently active palette."""
    c = COLORS
    return Theme(
        {
            # NOOA branding
            "nooa": f"bold {c['mauve']}",
            "tagline": f"italic {c['pink']}",
            # User interface
            "user": f"bold {c['green']}",
            "user.prompt": f"bold {c['green']}",
            "agent": f"bold {c['mauve']}",
            "agent.response": c["text"],
            # Status messages
            "status": c["overlay1"],
            "success": f"bold {c['green']}",
            "error": f"bold {c['red']}",
            "warning": f"bold {c['yellow']}",
            "info": f"bold {c['blue']}",
            # Panels and borders
            "panel.border": c["mauve"],
            "panel.title": f"bold {c['mauve']}",
            # Tables
            "table.header": f"bold {c['lavender']}",
            "table.border": c["surface2"],
            # Commands
            "command": f"bold {c['sapphire']}",
            "command.arg": c["sky"],
            # Spinners
            "spinner": c["mauve"],
            "spinner.text": c["subtext1"],
            # Code/technical
            "code": c["peach"],
            "path": c["teal"],
            "number": c["peach"],
            # History tags
            "tag": c["blue"],
            "tag.summary": c["yellow"],
            # MCP/Skills
            "mcp": c["sapphire"],
            "skill": c["pink"],
            "skill.active": f"bold {c['green']}",
        }
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

CATPPUCCIN_THEME = create_theme()
