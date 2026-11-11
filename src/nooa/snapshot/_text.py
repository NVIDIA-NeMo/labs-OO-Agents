# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure functions on ``Text``: ``cut`` a whole string or bytes, ``merge`` two ``Text``s.

They read no stream state and no live objects; they share only the nodes
(``_model``) and the UTF-8 cut points (``_codec``).
"""

from __future__ import annotations

from typing import Literal

from nooa.snapshot._codec import complete_prefix, encode_text, safe_start
from nooa.snapshot._model import Text

__all__ = ["cut"]


def cut(
    data: str | bytes,
    *,
    head: int,
    tail: int,
    unit: Literal["chars", "bytes"] | None = None,
) -> Text:
    """``data``, already whole in memory, as a ``Text`` keeping its first ``head`` and last ``tail`` units.

    - ``data``: a ``str`` or ``bytes`` (``bytearray`` and ``memoryview``
      are read as bytes).
    - ``head``, ``tail``: the units kept from the start and from the end.
    - ``unit`` (default: ``"chars"`` for a ``str``, ``"bytes"`` for bytes):
      ``"bytes"`` on a ``str`` counts and keeps its UTF-8 encoding (as a
      bytes ``TextStream`` would); ``"chars"`` on bytes raises ``TypeError``.

    The same ``Text`` a ``TextStream(unit=unit, head=head, tail=tail)``
    returns after one ``write(data)`` and ``close()``, without the stream:
    complete when ``data`` fits in ``head + tail``; else cut with reason
    ``"max_length"``, a bytes cut on whole code points. Nothing is spilled
    (``full`` is ``None``). Render it with ``render_output`` (or
    ``render_python``). Raises ``ValueError`` for a negative ``head`` or
    ``tail`` or an unknown ``unit``.

    >>> from nooa.snapshot import cut, render_output
    >>> print(render_output(cut("0123456789", head=3, tail=2)))
    012
    [... 5 chars omitted; full output not kept]
    89
    """
    if head < 0 or tail < 0:
        raise ValueError("head and tail must be non-negative")
    if isinstance(data, str):
        if unit is None or unit == "chars":
            return _cut(data, head, tail, "chars")
        if unit == "bytes":
            return _cut(encode_text(data), head, tail, "bytes")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        if unit is None or unit == "bytes":
            return _cut(bytes(data), head, tail, "bytes")
        if unit == "chars":
            raise TypeError("cut(unit='chars') takes str; cut bytes with unit='bytes'")
    else:
        raise TypeError(f"cut takes str or bytes, not {type(data).__name__}")
    raise ValueError(f"unit must be 'chars' or 'bytes', got {unit!r}")


def _cut(data: str | bytes, head: int, tail: int, unit: Literal["chars", "bytes"]) -> Text:
    if len(data) <= head + tail:
        return Text(head=data, length=len(data), unit=unit)
    kept_head, kept_tail = data[:head], data[len(data) - tail :] if tail else data[:0]
    if isinstance(kept_head, bytes) and isinstance(kept_tail, bytes):  # whole code points only
        kept_head = kept_head[: complete_prefix(kept_head)]
        kept_tail = kept_tail[safe_start(kept_tail, 0) :]
    return Text(head=kept_head, tail=kept_tail, length=len(data), unit=unit, reason="max_length")
