# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: the session rule must never occupy the terminal's final column.

A full-bleed rule wraps the cursor to the next row; when run_in_terminal
(emit_block) repaints this non-full-screen app after a SIGWINCH resize,
prompt_toolkit's erase is sized to the pre-resize frame and leaves a stale
rule (of the old width) plus a blank gap line in the scrollback. Repeated
resizes stack these into the "resize clutter" the status bar showed.

See ``format_session_rule`` in ``tui_application``.
"""

from __future__ import annotations

import pytest
from nooa_cli.tui.tui_application import format_session_rule


def _rendered_width(fragments: list[tuple[str, str]]) -> int:
    return sum(len(text) for _style, text in fragments)


@pytest.mark.parametrize("cols", [20, 21, 40, 80, 81, 120, 200])
def test_rule_never_occupies_final_column_no_label(cols: int) -> None:
    """Without a label the rule must be strictly narrower than the terminal."""
    width = _rendered_width(format_session_rule(cols, ""))
    assert width == cols - 1, f"cols={cols}: rule width {width} != {cols - 1}"
    assert width < cols, f"cols={cols}: rule must not span the final column"


@pytest.mark.parametrize("cols", [20, 40, 80, 120, 200])
@pytest.mark.parametrize("label", ["", "12:00 · opus · ctx 31% · Session [abc123]", "x"])
def test_rule_never_occupies_final_column_with_label(cols: int, label: str) -> None:
    """With any label the rendered fragments must still fit in cols-1."""
    width = _rendered_width(format_session_rule(cols, label))
    assert width <= cols - 1, (
        f"cols={cols} label={label!r}: rendered width {width} exceeds cols-1={cols - 1}"
    )


def test_rule_degenerate_narrow_terminal() -> None:
    """A pathologically narrow terminal never produces a negative/zero width."""
    for cols in (1, 2, 3):
        frags = format_session_rule(cols, "")
        assert _rendered_width(frags) >= 1
        # And with a label longer than the terminal, it still renders something.
        frags = format_session_rule(cols, "a-very-long-label")
        assert _rendered_width(frags) >= 1


@pytest.mark.parametrize("cols", range(20, 130))
def test_rule_never_full_bleed_for_any_label_length(cols: int) -> None:
    """Brute-force the never-full-bleed invariant across every label length.

    Regression for the off-by-one where ``len(label) == cols - 2`` made the
    dash fill round back up to 1 and the line re-occupied the final column.
    """
    for length in range(0, cols + 3):
        width = _rendered_width(format_session_rule(cols, "x" * length))
        assert width < cols, (
            f"cols={cols} label_len={length}: width {width} occupies the final column"
        )


def test_rule_label_is_preserved() -> None:
    """The label text is rendered verbatim (only the dash fill is clamped)."""
    label = "ctx 50% · Sess [deadbeef]"
    frags = format_session_rule(120, label)
    rendered = "".join(text for _style, text in frags)
    assert label in rendered
