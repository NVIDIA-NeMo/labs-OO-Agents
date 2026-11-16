# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One head/tail split of a character budget, for the places that keep both ends of a text (private).

``cut_text(text, limit)`` is a whole string within ``limit`` chars:
``render_output(cut(...))``, so itself when it fits, else ``HEAD``,
``[... N chars omitted; full output not kept]``, ``TAIL``.
``text_stream(limit)`` is the ``TextStream`` of the same split, for output
that arrives over time. ``split(limit)`` is the split itself: ``tail``
chars from the end (``None``: half, rounded down), the rest from the start.
"""

from __future__ import annotations

from nooa.snapshot import Text, TextStream, cut, render_output

__all__ = ["cut_halves", "cut_text", "split", "text_stream"]


def split(limit: int, tail: int | None = None) -> tuple[int, int]:
    """``(head, tail)`` of a ``limit``-char budget: ``tail`` from the end (``None``: ``limit // 2``)."""
    tail_chars = limit // 2 if tail is None else tail
    return limit - tail_chars, tail_chars


def cut_halves(text: str, limit: int, tail: int | None = None) -> Text:
    """``text`` as a ``Text`` keeping ``split(limit, tail)`` of it (complete when it fits)."""
    head_chars, tail_chars = split(limit, tail)
    return cut(text, head=head_chars, tail=tail_chars)


def cut_text(text: str, limit: int, tail: int | None = None) -> str:
    """``text`` within ``limit`` chars, in ``render_output``'s notation (itself when it fits)."""
    return render_output(cut_halves(text, limit, tail))


def text_stream(limit: int, tail: int | None = None) -> TextStream:
    """A chars ``TextStream`` keeping ``split(limit, tail)``: the first and last chars of the output."""
    head_chars, tail_chars = split(limit, tail)
    return TextStream(unit="chars", head=head_chars, tail=tail_chars)
