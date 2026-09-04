# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Theme system for the NOOA TUI.

Four built-in palettes provide a shared base palette plus documented semantic
UI roles. Compatible native and Base16 YAML themes are discovered at startup:

  mocha   — Catppuccin Mocha (dark)         [default]
  latte   — Catppuccin Latte (light)
  vsdark  — Visual Studio Code Dark+
  vslight — Visual Studio Code Light

Usage::

    from .theme import set_theme, COLORS, create_theme
    set_theme("latte")          # switches COLORS in-place
    console._theme = create_theme()
"""

from pygments.token import Generic
from rich.console import Console, ConsoleOptions, RenderResult
from rich.style import Style
from rich.syntax import PygmentsSyntaxTheme, Syntax, SyntaxTheme
from rich.theme import Theme

from .theme_catalog import SEMANTIC_KEYS, ThemeRecord, load_user_themes, parse_theme

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
    "inline_code_fg": "#89b4fa",
    "inline_code_bg": "#1e1e2e",
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
    "inline_code_fg": "#1858c7",
    "inline_code_bg": "#eff1f5",
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
    "inline_code_fg": "#569cd6",
    "inline_code_bg": "#1e1e1e",
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
    "inline_code_fg": "#005fb8",
    "inline_code_bg": "#ffffff",
}

_BUILTIN_SEMANTICS = {
    "mocha": {
        "selection_fg": "#cdd6f4",
        "selection_bg": "#585b70",
        "user_message_fg": "#cdd6f4",
        "user_message_bg": "#585b70",
        "search_match_fg": "#11111b",
        "search_match_bg": "#f9e2af",
        "search_current_fg": "#11111b",
        "search_current_bg": "#89dceb",
        "focus_accent": "#b4befe",
    },
    "latte": {
        "selection_fg": "#4c4f69",
        "selection_bg": "#c8d8f4",
        "user_message_fg": "#4c4f69",
        "user_message_bg": "#c8d8f4",
        "search_match_fg": "#4c4f69",
        "search_match_bg": "#f1dfb8",
        "search_current_fg": "#20233a",
        "search_current_bg": "#7aa2e8",
        "focus_accent": "#1e66f5",
        "feedback_success": "#287a1f",
        "feedback_warning": "#765000",
        "feedback_info": "#1858c7",
    },
    "vsdark": {
        "selection_fg": "#d4d4d4",
        "selection_bg": "#264f78",
        "user_message_fg": "#d4d4d4",
        "user_message_bg": "#264f78",
        "search_match_fg": "#f5e6b3",
        "search_match_bg": "#4b4632",
        "search_current_fg": "#ffffff",
        "search_current_bg": "#365f86",
        "focus_accent": "#569cd6",
    },
    "vslight": {
        "selection_fg": "#000000",
        "selection_bg": "#add6ff",
        "user_message_fg": "#000000",
        "user_message_bg": "#add6ff",
        "search_match_fg": "#3b2f1d",
        "search_match_bg": "#f4ddb5",
        "search_current_fg": "#101820",
        "search_current_bg": "#75b8f5",
        "focus_accent": "#005fb8",
    },
}
for _theme_id, _overrides in _BUILTIN_SEMANTICS.items():
    _BUILTIN_PALETTE = {"mocha": _MOCHA, "latte": _LATTE, "vsdark": _VS_DARK, "vslight": _VS_LIGHT}[
        _theme_id
    ]
    _BUILTIN_PALETTE.update(_overrides)


_BUILTIN_THEME_METADATA = {
    "mocha": ("Catppuccin Mocha", "dark", "monokai", "Catppuccin's dark palette"),
    "latte": ("Catppuccin Latte", "light", "gruvbox-light", "Catppuccin's light palette"),
    "vsdark": ("Visual Studio Dark+", "dark", "github-dark", "Visual Studio Code Dark+"),
    "vslight": ("Visual Studio Light", "light", "vs", "Visual Studio Code Light"),
}
_BUILTIN_PALETTES = {"mocha": _MOCHA, "latte": _LATTE, "vsdark": _VS_DARK, "vslight": _VS_LIGHT}


def _base24_scheme(theme_id: str) -> dict[str, str]:
    """Express a built-in palette through the public Base24 theme schema."""
    palette = _BUILTIN_PALETTES[theme_id]
    metadata = _BUILTIN_THEME_METADATA[theme_id]
    scheme = {
        "id": theme_id,
        "name": metadata[0],
        "variant": metadata[1],
        "syntax_theme": metadata[2],
        "description": metadata[3],
        "base00": palette["base"],
        "base01": palette["surface0"],
        "base02": palette["surface1"],
        "base03": palette["surface2"],
        "base04": palette["subtext1"],
        "base05": palette["text"],
        "base06": palette["rosewater"],
        "base07": palette["text"],
        "base08": palette["red"],
        "base09": palette["peach"],
        "base0A": palette["yellow"],
        "base0B": palette["green"],
        "base0C": palette["teal"],
        "base0D": palette["blue"],
        "base0E": palette["mauve"],
        "base0F": palette["flamingo"],
        "base10": palette["crust"],
        "base11": palette["red"],
        "base12": palette["green"],
        "base13": palette["yellow"],
        "base14": palette["blue"],
        "base15": palette["pink"],
        "base16": palette["sky"],
        "base17": palette["lavender"],
    }
    scheme.update({key: palette[key] for key in SEMANTIC_KEYS if key in palette})
    return scheme


_BUILTIN_RECORDS = {
    theme_id: parse_theme(_base24_scheme(theme_id), fallback_id=theme_id, source="builtin")
    for theme_id in _BUILTIN_THEME_METADATA
}
THEME_RECORDS, THEME_DIAGNOSTICS = load_user_themes(_BUILTIN_RECORDS)
THEMES: dict[str, dict[str, str]] = {
    theme_id: record.palette for theme_id, record in THEME_RECORDS.items()
}
SYNTAX_THEMES: dict[str, str] = {
    theme_id: record.syntax_theme for theme_id, record in THEME_RECORDS.items()
}


def reload_themes() -> tuple[str, ...]:
    """Reload user/project theme files while preserving shared mapping identity."""
    global THEME_DIAGNOSTICS, _active_name
    records, diagnostics = load_user_themes(_BUILTIN_RECORDS)
    THEME_RECORDS.clear()
    THEME_RECORDS.update(records)
    THEMES.clear()
    THEMES.update((theme_id, record.palette) for theme_id, record in records.items())
    SYNTAX_THEMES.clear()
    SYNTAX_THEMES.update((theme_id, record.syntax_theme) for theme_id, record in records.items())
    THEME_DIAGNOSTICS = diagnostics
    if _active_name not in THEMES:
        _active_name = "mocha"
    COLORS.clear()
    COLORS.update(THEMES[_active_name])
    return tuple(THEME_RECORDS)


def theme_names() -> tuple[str, ...]:
    """Return all discovered theme IDs in deterministic display order."""
    return tuple(THEME_RECORDS)


def get_theme_record(name: str | None = None) -> ThemeRecord:
    """Return metadata and semantic colors for one discovered theme."""
    selected = _active_name if name is None else name
    try:
        return THEME_RECORDS[selected]
    except KeyError as exc:
        raise ValueError(
            f"Unknown theme {selected!r}. Choose from: {', '.join(THEME_RECORDS)}"
        ) from exc


# ---------------------------------------------------------------------------
# Active palette — a mutable dict updated in-place so that callers who did
# `from .theme import COLORS` keep a live reference after set_theme().
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = dict(THEMES["mocha"])
_active_name: str = "mocha"


def get_theme() -> str:
    """Return the name of the currently active theme."""
    return _active_name


def get_syntax_theme(name: str | None = None) -> str:
    """Return the Pygments style paired with a discovered UI theme."""
    selected = _active_name if name is None else name
    try:
        return SYNTAX_THEMES[selected]
    except KeyError as exc:
        raise ValueError(f"Unknown theme {selected!r}. Choose from: {', '.join(THEMES)}") from exc


class SemanticSyntaxTheme(SyntaxTheme):
    """Pygments syntax colors with semantic added/removed diff colors."""

    def __init__(self, syntax_theme: str, palette: dict[str, str]) -> None:
        self._base = PygmentsSyntaxTheme(syntax_theme)
        self._palette = palette

    def get_style_for_token(self, token_type: object) -> Style:
        if token_type in Generic.Inserted:
            return Style(color=self._palette["diff_added"])
        if token_type in Generic.Deleted:
            return Style(color=self._palette["diff_removed"])
        return self._base.get_style_for_token(token_type)  # type: ignore[arg-type]

    def get_background_style(self) -> Style:
        return self._base.get_background_style()


def create_syntax_theme(name: str | None = None) -> SemanticSyntaxTheme:
    """Create syntax colors for a discovered theme, including semantic diffs."""
    selected = get_theme() if name is None else name
    return SemanticSyntaxTheme(get_syntax_theme(selected), THEMES[selected])


class ThemeSyntax(Syntax):
    """Syntax highlighting that follows the active UI theme when replayed."""

    def __init__(self, code: str, lexer: object, **kwargs: object) -> None:
        self._uses_active_theme = "theme" not in kwargs
        if self._uses_active_theme:
            kwargs["theme"] = create_syntax_theme()
        super().__init__(code, lexer, **kwargs)  # type: ignore[arg-type]

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self._uses_active_theme:
            self._theme = create_syntax_theme()
        yield from super().__rich_console__(console, options)


def set_theme(name: str) -> None:
    """Switch the active theme.  Updates COLORS in-place.

    Args:
        name: A built-in or discovered user theme ID.

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
            "status": c["text_subtle"],
            "success": f"bold {c['feedback_success']}",
            "error": f"bold {c['feedback_error']}",
            "warning": f"bold {c['feedback_warning']}",
            "info": f"bold {c['feedback_info']}",
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
            # Inline code stays subtle: bold semantic text, no filled chip.
            "markdown.code": f"bold {c['inline_code_fg']}",
            "path": c["code_path"],
            "number": c["code_number"],
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
