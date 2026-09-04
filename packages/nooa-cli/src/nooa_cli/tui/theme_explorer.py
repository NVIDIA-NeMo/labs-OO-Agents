# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-screen browser for discovered TUI themes."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console
from rich.syntax import Syntax

from . import theme
from .explorer_base import ExplorerConfig, ExplorerModel, ExplorerView
from .theme import SemanticSyntaxTheme, get_theme, set_theme
from .theme_catalog import ThemeRecord

if TYPE_CHECKING:
    from .theme_gallery import GalleryTheme


def _rgb(color: str) -> str:
    value = color.lstrip("#")
    return ";".join(str(int(value[index : index + 2], 16)) for index in (0, 2, 4))


def _style(text: str, foreground: str, background: str | None = None, *, bold: bool = False) -> str:
    code = f"{'1;' if bold else ''}38;2;{_rgb(foreground)}"
    if background is not None:
        code += f";48;2;{_rgb(background)}"
    return f"\x1b[{code}m{text}\x1b[0m"


def _syntax_lines(code: str, lexer: str, *, record: ThemeRecord, width: int) -> list[str]:
    """Render a compact syntax sample with the same Pygments theme as the TUI."""
    output = io.StringIO()
    console = Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
        width=max(20, width),
    )
    console.print(
        Syntax(
            code,
            lexer,
            theme=SemanticSyntaxTheme(record.syntax_theme, record.palette),
            background_color="default",
            word_wrap=True,
            padding=0,
        )
    )
    return output.getvalue().rstrip("\n").splitlines()


@dataclass(frozen=True, slots=True)
class ThemeExplorerRow:
    """One discovered theme and its source metadata."""

    id: str
    title: str
    variant: str
    source: str
    description: str
    record: ThemeRecord = field(compare=False, hash=False)
    search_text: str = ""


def build_theme_rows(records: dict[str, ThemeRecord] | None = None) -> list[ThemeExplorerRow]:
    """Build searchable rows from the active theme catalog."""
    catalog = theme.THEME_RECORDS if records is None else records
    return [
        ThemeExplorerRow(
            record.id,
            record.name,
            record.variant,
            record.source,
            record.description,
            record,
            "\n".join(
                (
                    record.id,
                    record.name,
                    record.variant,
                    record.source,
                    record.description,
                    record.author,
                )
            ),
        )
        for record in catalog.values()
    ]


class ThemeExplorerView(ExplorerView):
    """Preview discovered themes; Enter commits and q restores the opening theme."""

    item_name = "theme"
    list_heading = "  id               variant  source"
    list_height = 4
    quit_from_list = True

    def __init__(
        self,
        rows: list[ThemeExplorerRow] | None = None,
        *,
        refresh: Callable[[], None] | None = None,
        persist: Callable[[str], None] | None = None,
    ) -> None:
        if rows is None:
            theme.reload_themes()
            values = build_theme_rows()
        else:
            values = rows
        model = ExplorerModel(values)
        if values:
            active = next((i for i, row in enumerate(values) if row.id == get_theme()), 0)
            model.cursor = active
        skipped = len(theme.THEME_DIAGNOSTICS) if rows is None else 0
        title = "Theme Browser" + (f" · {skipped} invalid skipped" if skipped else "")
        super().__init__(model, ExplorerConfig(title=title, actions={"enter": "Apply"}))
        self._opening_theme = get_theme()
        self._previewed_theme = self._opening_theme
        self._committed = False
        self._refresh = refresh or (lambda: None)
        self._persist = persist or (lambda _name: None)

    def format_row(self, row: ThemeExplorerRow, width: int) -> str:
        state = "active" if row.id == get_theme() else ""
        return f"{row.id:<16} {row.variant:<7}  {row.source:<12} {state}"[:width]

    def detail_lines(self, row: ThemeExplorerRow, width: int) -> list[str]:
        p = row.record.palette
        swatches = "  ".join(
            _style(f" {label} ", fg, bg, bold=bold)
            for label, fg, bg, bold in (
                ("inline code", p["inline_code_fg"], None, True),
                ("selected", p["selection_fg"], p["selection_bg"], False),
                ("match", p["search_match_fg"], p["search_match_bg"], False),
                ("current", p["search_current_fg"], p["search_current_bg"], False),
            )
        )
        feedback = "  ".join(
            _style(label, p[key])
            for label, key in (
                ("success", "feedback_success"),
                ("warning", "feedback_warning"),
                ("error", "feedback_error"),
                ("info", "feedback_info"),
            )
        )
        return [
            f"{row.title} ({row.id})",
            f"Variant: {row.variant}    Source: {row.source}",
            f"Syntax: {row.record.syntax_theme}",
            row.description,
            "",
            "Semantic highlights",
            swatches,
            "",
            "Feedback and accents",
            feedback,
            "",
            _style(" Primary text ", p["text_primary"], p["base"]),
            _style(" Muted text ", p["text_muted"], p["surface_raised"]),
            "",
            "Syntax-highlighted code",
            *_syntax_lines(
                'def greet(name: str) -> str:\n    return f"Hello, {name}!"',
                "python",
                record=row.record,
                width=width,
            ),
            "",
            "Unified diff",
            *_syntax_lines(
                "@@ -1,2 +1,2 @@\n-old_value = 1\n+new_value = 2",
                "diff",
                record=row.record,
                width=width,
            ),
            "",
            "Enter applies and saves this theme. Esc or q closes and restores the opening theme.",
        ]

    def _preview_current(self) -> None:
        row = self.model.current
        if row is None or row.id == self._previewed_theme:
            return
        set_theme(row.id)
        self._previewed_theme = row.id
        self._refresh()

    def on_selection_changed(self) -> None:
        self._preview_current()

    def handle_action(self, action: str, row: ThemeExplorerRow | None) -> str:
        if action != "enter" or row is None:
            return "ignored"
        set_theme(row.id)
        self._previewed_theme = row.id
        self._persist(row.id)
        self._committed = True
        self._refresh()
        return "close"

    def on_open(self) -> None:
        self._preview_current()

    def on_close(self) -> None:
        if not self._committed and get_theme() != self._opening_theme:
            set_theme(self._opening_theme)
            self._refresh()


class ThemeGalleryView(ThemeExplorerView):
    """Browse remote schemes without mutating the active theme until installation."""

    item_name = "scheme"

    def __init__(
        self,
        entries: dict[str, GalleryTheme],
        *,
        install: Callable[[GalleryTheme], None],
    ) -> None:
        self._entries = entries
        self._install = install
        self._install_error = ""
        super().__init__(build_theme_rows({key: value.record for key, value in entries.items()}))
        self.title = "Theme Gallery"
        self.config.title = self.title

    def _preview_current(self) -> None:
        """The detail pane previews remote colors without registering them globally."""

    def on_selection_changed(self) -> None:
        self._install_error = ""

    def detail_lines(self, row: ThemeExplorerRow, width: int) -> list[str]:
        lines = super().detail_lines(row, width)
        lines[-1] = "Enter installs, applies, and saves this theme. Esc or q closes."
        if self._install_error:
            lines.extend(
                (
                    "",
                    _style(
                        f"Install failed: {self._install_error}",
                        row.record.palette["feedback_error"],
                    ),
                )
            )
        return lines

    def handle_action(self, action: str, row: ThemeExplorerRow | None) -> str:
        if action != "enter" or row is None:
            return "ignored"
        try:
            self._install(self._entries[row.id])
        except Exception as exc:
            self._install_error = str(exc)
            return "handled"
        self._committed = True
        return "close"
