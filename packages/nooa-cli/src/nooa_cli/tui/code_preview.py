# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact code-preview formatter used by the TUI activity line."""

from __future__ import annotations


def _code_preview(code: str, max_cols: int = 100) -> str:
    """Return a compact preview of *code* for the activity line.

    Shape:
      * First line is comment (``#…``): keep the comment (white-styled
        downstream) AND the next non-blank code line (grey). Max 2.
      * First line is code: show just that line (grey). Max 1.

    Filters out lines matching ``return_result(...)`` — CodeAct
    boilerplate the user doesn't care about in a preview. Long lines
    truncate to ``max_cols`` with an ellipsis.
    """
    raw = [ln for ln in code.splitlines() if ln.strip()]
    # Drop the CodeAct-internal return_result(...) scaffolding.
    lines = [ln for ln in raw if not ln.lstrip().startswith("return_result(")]
    if not lines:
        return ""

    def _clip(ln: str) -> str:
        return ln if len(ln) <= max_cols else ln[: max_cols - 1] + "…"

    first = lines[0]
    if first.lstrip().startswith("#"):
        result = [_clip(first)]
        if len(lines) > 1:
            result.append(_clip(lines[1]))
        if len(lines) > 2:
            result[-1] += "…"
        return "\n".join(result)

    # No comment — one line only, suffix with … if there was more.
    clipped = _clip(first)
    if len(lines) > 1:
        clipped += "…"
    return clipped
