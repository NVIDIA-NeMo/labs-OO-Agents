# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ARC-AGI-3 Solver: Grid class for frame representation and analysis.

Inspired by arc_agi/problem.py:Board but tailored for ARC-AGI-3's 64x64 grids
with values 0-15 and episode metadata. Includes a YAML representer so that
Grid objects serialize as compact text when rendered in LLM prompts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import ClassVar

import yaml


# ARC-AGI-3 official color palette (16 colors, indices 0-15)
# Source: official ARC-AGI-3 SDK color map
ARC3_COLORS = [
    "white",       # 0  #FFFFFF
    "light_gray",  # 1  #CCCCCC
    "gray",        # 2  #999999
    "dark_gray",   # 3  #666666
    "near_black",  # 4  #333333
    "black",       # 5  #000000
    "pink",        # 6  #E53AA3
    "light_pink",  # 7  #FF7BCC
    "red",         # 8  #F93C31
    "blue",        # 9  #1E93FF
    "light_blue",  # 10 #88D8F1
    "yellow",      # 11 #FFDC00
    "orange",      # 12 #FF851B
    "maroon",      # 13 #921231
    "green",       # 14 #4FCC30
    "purple",      # 15 #A356D6
]


def _col_header_lines(
    width: int,
    sep: str,
    fmt: str,
    row_prefix_width: int = 0,
    col_offset: int = 0,
) -> list[str]:
    """Build ruler-style column header lines (tens + units).

    Tens line: dot-filled with the tens digit at every 10th position.
    Units line: repeating 0-9 cycle for each column.

    Returns 2 lines. ``row_prefix_width`` pads them so they align with
    row-enumerated data lines (accounts for the ``{idx}|`` prefix).
    """
    cell_width = 2 if (fmt == "dec" and sep) else 1
    tens: list[str] = []
    for c in range(col_offset, col_offset + width):
        if c % 10 == 0:
            tens.append(str(c // 10).rjust(cell_width))
        else:
            tens.append(".".rjust(cell_width))
    units = [
        str(c % 10).rjust(cell_width)
        for c in range(col_offset, col_offset + width)
    ]
    prefix = " " * row_prefix_width
    return [prefix + sep.join(tens), prefix + sep.join(units)]


KEYBOARD_FOOTER = """
                                ↑ UP
                        ← LEFT  ↓ DOWN  RIGHT →"""


def _coerce_add_grid_orientation(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower().replace("-", "_")
    if text in {"always", "true", "1", "yes", "on"}:
        return True
    if text in {
        "",
        "never",
        "false",
        "0",
        "no",
        "off",
        "when_keyboard_active",
    }:
        return False
    raise ValueError(
        "add_grid_orientation must be never, always, or "
        f"when_keyboard_active; got {value!r}"
    )


@dataclass
class Grid:
    """A single grid frame in ARC-AGI-3 with metadata.

    Represents a 64x64 grid snapshot captured during gameplay, along with
    the episode step and level at which it was captured.

    Rendering defaults (sep, fmt) are controlled via class-level attributes
    set once at startup by Grid.configure(). Individual instances inherit
    from the class defaults automatically.
    """

    # Class-level rendering defaults — set once via Grid.configure()
    _default_sep: ClassVar[str] = ""
    _default_fmt: ClassVar[str] = "hex"  # 'hex' (0-9,a-f) or 'dec' (0-15)
    _default_enumerate_rows: ClassVar[bool] = False
    _default_enumerate_cols: ClassVar[bool] = False
    _default_mark_click: ClassVar[bool] = False
    _default_add_grid_orientation: ClassVar[bool] = False

    data: list[list[int]]  # 64x64, values 0-15
    step: int | None = None   # Episode step when captured (None = unknown)
    level: int | None = None  # levels_completed when captured (None = unknown)

    @classmethod
    def configure(
        cls,
        sep: str | None = None,
        fmt: str | None = None,
        enumerate_rows: bool | None = None,
        enumerate_cols: bool | None = None,
        mark_click: bool | None = None,
        add_grid_orientation: bool | str | None = None,
    ) -> None:
        """Set class-level rendering defaults from solver config.

        Call once at startup. All Grid/DiffGrid instances will use these
        defaults in render_text() and YAML serialization.
        """
        if sep is not None:
            cls._default_sep = sep
        if fmt is not None:
            cls._default_fmt = fmt
        if enumerate_rows is not None:
            cls._default_enumerate_rows = enumerate_rows
        if enumerate_cols is not None:
            cls._default_enumerate_cols = enumerate_cols
        if mark_click is not None:
            cls._default_mark_click = mark_click
        if add_grid_orientation is not None:
            cls._default_add_grid_orientation = _coerce_add_grid_orientation(
                add_grid_orientation
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Return (height, width) of the grid."""
        if not self.data:
            return (0, 0)
        return (len(self.data), len(self.data[0]))

    def diff_count(self, other: Grid) -> int:
        """Return the number of pixels that differ between this grid and another."""
        count = 0
        for y, (row_a, row_b) in enumerate(zip(self.data, other.data)):
            for x, (a, b) in enumerate(zip(row_a, row_b)):
                if a != b:
                    count += 1
        return count

    def unique_colors(self) -> list[int]:
        """Return sorted list of unique color values in the grid."""
        colors: set[int] = set()
        for row in self.data:
            colors.update(row)
        return sorted(colors)

    def color_histogram(self) -> dict[int, int]:
        """Return a mapping of color value to pixel count."""
        counts: Counter[int] = Counter()
        for row in self.data:
            counts.update(row)
        return dict(sorted(counts.items()))

    @staticmethod
    def copy_data(data: list[list[int]]) -> list[list[int]]:
        """Return a shallow copy of 2D grid data (isolates row references)."""
        return [row[:] for row in data]

    def to_list(self) -> list[list[int]]:
        """Return the grid data as a plain list."""
        return self.data

    def render_text(
        self,
        compact: bool = True,
        click_rc: tuple[int, int] | None = None,
        add_header: bool = True,
        row_offset: int = 0,
        col_offset: int = 0,
    ) -> str:
        """Render the grid as a text string for display.

        Uses class-level defaults for sep and fmt (set via Grid.configure()).
        When enumerate_rows / enumerate_cols are enabled, row indices and/or
        ruler-style column headers are added around the data.

        Args:
            compact: Use compact hex/dec rendering (True) or color-name rendering.
            click_rc: Optional (row, col) of the last CLICK target. When
                ``_default_mark_click`` is enabled, that pixel is rendered as
                ``*`` and a legend line ``* = <value> (<color>/<int>)`` is
                appended after the grid body.
            add_header: When True, prepend the ``Grid step=... level=...``
                metadata header. Defaults to True.
            row_offset: Display offset for enumerated row labels.
            col_offset: Display offset for enumerated column labels.
        """
        sep = self._default_sep
        fmt = self._default_fmt
        enum_rows = self._default_enumerate_rows
        enum_cols = self._default_enumerate_cols
        mark_click = self._default_mark_click
        # Column enumeration not supported for dec without separator
        if enum_cols and fmt == "dec" and not sep:
            enum_cols = False

        # Resolve effective click position
        click_row: int | None = None
        click_col: int | None = None
        click_legend: str | None = None
        if mark_click and click_rc is not None:
            cr, cc = click_rc
            height_check, width_check = self.shape
            if 0 <= cr < height_check and 0 <= cc < width_check:
                click_row, click_col = cr, cc
                val = self.data[cr][cc]
                color_name = ARC3_COLORS[val] if 0 <= val < len(ARC3_COLORS) else "?"
                if fmt == "hex":
                    val_str = hex(val)[2:] if 0 <= val < 16 else "?"
                else:
                    val_str = str(val)
                click_legend = f"* = {val_str} ({color_name}/{val})"

        height, width = self.shape
        lines: list[str] = []
        parts = ["Grid"]
        if self.step is not None:
            parts.append(f"step={self.step}")
        if self.level is not None:
            parts.append(f"level={self.level}")
        parts.append(f"shape={self.shape}")
        header = " ".join(parts)
        if add_header:
            lines.append(header)

        if not compact:
            for row in self.data:
                names = [ARC3_COLORS[v] if 0 <= v < len(ARC3_COLORS) else "?" for v in row]
                lines.append(" ".join(names))
            return "\n".join(lines)

        # --- compact path ---
        row_prefix_width = 0
        if enum_rows and height > 0:
            row_prefix_width = len(str(row_offset + height - 1)) + 1  # digits + '|'

        if enum_cols and width > 0:
            col_headers = _col_header_lines(
                width, sep, fmt, row_prefix_width, col_offset,
            )
            lines.extend(col_headers)
            lines.append("-" * len(col_headers[-1]))

        # Determine whether dec values need fixed-width padding
        dec_pad = enum_cols and fmt == "dec" and sep
        cell_width = 2 if dec_pad else 0  # 0 means no padding

        if fmt == "dec":
            for i, row in enumerate(self.data):
                cells: list[str] = []
                for j, v in enumerate(row):
                    if i == click_row and j == click_col:
                        cells.append("*".rjust(cell_width) if cell_width else "*")
                    elif cell_width:
                        cells.append(str(v).rjust(cell_width))
                    else:
                        cells.append(str(v))
                row_str = sep.join(cells)
                if enum_rows:
                    row_str = f"{i + row_offset:>{row_prefix_width - 1}}|{row_str}"
                lines.append(row_str)
        else:
            char_map = {i: hex(i)[2:] for i in range(16)}
            for i, row in enumerate(self.data):
                cells = []
                for j, v in enumerate(row):
                    if i == click_row and j == click_col:
                        cells.append("*")
                    else:
                        cells.append(char_map.get(v, "?"))
                row_str = sep.join(cells)
                if enum_rows:
                    row_str = f"{i + row_offset:>{row_prefix_width - 1}}|{row_str}"
                lines.append(row_str)

        if click_legend:
            lines.append(click_legend)

        return "\n".join(lines)

    @classmethod
    def from_numpy(cls, arr, step: int | None = None, level: int | None = None) -> Grid:
        """Create a Grid from a numpy array."""
        return cls(data=arr.tolist(), step=step, level=level)

    @classmethod
    def empty(cls, height: int = 64, width: int = 64, fill: int = 0) -> Grid:
        """Create an empty grid filled with a single value."""
        return cls(data=[[fill] * width for _ in range(height)])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return self.data == other.data

    def __repr__(self) -> str:
        h, w = self.shape
        colors = self.unique_colors()
        parts = [f"Grid({h}x{w}"]
        if self.step is not None:
            parts.append(f"step={self.step}")
        if self.level is not None:
            parts.append(f"level={self.level}")
        parts.append(f"colors={colors})")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# DiffGrid: spatial diff between two consecutive grids
# ---------------------------------------------------------------------------

# Sentinel value for unchanged pixels in DiffGrid.data
UNCHANGED = -1


@dataclass
class DiffGrid(Grid):
    """Spatial diff between two consecutive grids.

    Values per pixel:
        -1 (UNCHANGED) = pixel did not change
        0-15           = new color at that pixel after the action

    Rendered as: '.' for unchanged, color value for the new color.
    This gives the LLM a visual map of exactly where and what changed.
    """

    num_changed: int = 0

    @classmethod
    def from_grids(cls, prev: Grid, curr: Grid) -> DiffGrid:
        """Compute the spatial diff from prev to curr."""
        data: list[list[int]] = []
        count = 0
        for row_p, row_c in zip(prev.data, curr.data):
            diff_row: list[int] = []
            for p, c in zip(row_p, row_c):
                if p != c:
                    diff_row.append(c)
                    count += 1
                else:
                    diff_row.append(UNCHANGED)
            data.append(diff_row)
        return cls(
            data=data, step=curr.step, level=curr.level, num_changed=count,
        )

    @classmethod
    def no_change(cls, grid: Grid) -> DiffGrid:
        """Create an empty diff (nothing changed)."""
        h, w = grid.shape
        return cls(
            data=[[UNCHANGED] * w for _ in range(h)],
            step=grid.step, level=grid.level, num_changed=0,
        )

    def render_text(
        self,
        compact: bool = True,
        click_rc: tuple[int, int] | None = None,
        add_header: bool = True,
        row_offset: int = 0,
        col_offset: int = 0,
    ) -> str:
        """Render the diff grid: '.' for unchanged, color value for new color.

        Args:
            compact: Use compact hex/dec rendering (True) or color-name rendering.
            click_rc: Optional (row, col) of the last CLICK target. When
                ``_default_mark_click`` is enabled, that pixel is rendered as
                ``*`` and a legend line is appended.
            add_header: When True, prepend the ``DiffGrid ...`` metadata
                header. Defaults to True.
            row_offset: Display offset for enumerated row labels.
            col_offset: Display offset for enumerated column labels.
        """
        sep = self._default_sep
        fmt = self._default_fmt
        enum_rows = self._default_enumerate_rows
        enum_cols = self._default_enumerate_cols
        mark_click = self._default_mark_click
        if enum_cols and fmt == "dec" and not sep:
            enum_cols = False

        # Resolve effective click position (use the *underlying* grid value
        # for the legend, not the diff sentinel)
        click_row: int | None = None
        click_col: int | None = None
        click_legend: str | None = None
        if mark_click and click_rc is not None:
            cr, cc = click_rc
            height_check, width_check = self.shape
            if 0 <= cr < height_check and 0 <= cc < width_check:
                click_row, click_col = cr, cc
                raw = self.data[cr][cc]
                val = raw if raw != UNCHANGED else None
                if val is not None:
                    color_name = ARC3_COLORS[val] if 0 <= val < len(ARC3_COLORS) else "?"
                    val_str = hex(val)[2:] if (fmt == "hex" and 0 <= val < 16) else str(val)
                    click_legend = f"* = {val_str} ({color_name}/{val}) [changed]"
                else:
                    click_legend = "* = . (unchanged)"

        height, width = self.shape
        hdr_parts = ["DiffGrid"]
        if self.step is not None:
            hdr_parts.append(f"step={self.step}")
        if self.level is not None:
            hdr_parts.append(f"level={self.level}")
        hdr_parts.append(f"changed={self.num_changed}")
        hdr_parts.append(f"shape={self.shape}")
        header = " ".join(hdr_parts)
        lines: list[str] = [header] if add_header else []

        if not compact:
            for row in self.data:
                cells = []
                for v in row:
                    if v == UNCHANGED:
                        cells.append(".")
                    elif 0 <= v < len(ARC3_COLORS):
                        cells.append(ARC3_COLORS[v])
                    else:
                        cells.append("?")
                lines.append(" ".join(cells))
            return "\n".join(lines)

        # --- compact path ---
        row_prefix_width = 0
        if enum_rows and height > 0:
            row_prefix_width = len(str(row_offset + height - 1)) + 1

        if enum_cols and width > 0:
            col_headers = _col_header_lines(
                width, sep, fmt, row_prefix_width, col_offset,
            )
            lines.extend(col_headers)
            lines.append("-" * len(col_headers[-1]))

        dec_pad = enum_cols and fmt == "dec" and sep
        cell_width = 2 if dec_pad else 0

        if fmt == "dec":
            for i, row in enumerate(self.data):
                cells: list[str] = []
                for j, v in enumerate(row):
                    if i == click_row and j == click_col:
                        cells.append("*".rjust(cell_width) if cell_width else "*")
                    elif v == UNCHANGED:
                        cells.append(".".rjust(cell_width) if cell_width else ".")
                    elif cell_width:
                        cells.append(str(v).rjust(cell_width))
                    else:
                        cells.append(str(v))
                row_str = sep.join(cells)
                if enum_rows:
                    row_str = f"{i + row_offset:>{row_prefix_width - 1}}|{row_str}"
                lines.append(row_str)
        else:
            char_map = {i: hex(i)[2:] for i in range(16)}
            for i, row in enumerate(self.data):
                cells = []
                for j, v in enumerate(row):
                    if i == click_row and j == click_col:
                        cells.append("*")
                    elif v == UNCHANGED:
                        cells.append(".")
                    else:
                        cells.append(char_map.get(v, "?"))
                row_str = sep.join(cells)
                if enum_rows:
                    row_str = f"{i + row_offset:>{row_prefix_width - 1}}|{row_str}"
                lines.append(row_str)

        if click_legend:
            lines.append(click_legend)

        return "\n".join(lines)

    def changed_cells(self) -> list[tuple[int, int, int]]:
        """Return list of (row, col, new_color) for every changed pixel."""
        cells = []
        for y, row in enumerate(self.data):
            for x, v in enumerate(row):
                if v != UNCHANGED:
                    cells.append((y, x, v))
        return cells

    def __repr__(self) -> str:
        h, w = self.shape
        parts = [f"DiffGrid({h}x{w}"]
        if self.step is not None:
            parts.append(f"step={self.step}")
        if self.level is not None:
            parts.append(f"level={self.level}")
        parts.append(f"changed={self.num_changed})")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# YAML representers: Grid / DiffGrid -> compact text block in YAML output
# ---------------------------------------------------------------------------

def _grid_yaml_representer(dumper: yaml.Dumper, grid: Grid) -> yaml.Node:
    """Represent a Grid as a literal block scalar using render_text()."""
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", grid.render_text(compact=True), style="|"
    )


yaml.add_representer(Grid, _grid_yaml_representer)
yaml.add_representer(DiffGrid, _grid_yaml_representer)
