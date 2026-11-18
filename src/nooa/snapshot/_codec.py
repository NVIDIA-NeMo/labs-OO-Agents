# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoding and accounting shared by capture, rendering and streams (a leaf module).

What lives here: the canonical JSON encoding (``dumps``) and its size
accounting (``Container``), the typed smallest fallbacks and the opaque
label, string prefixes, JSON Pointers, the mapping-key conversion, and the
UTF-8 helpers (cut points, decoding with replacement). Capture and
``render_json`` both use them, so capture's sizes and ``render_json``'s
output agree exactly, and every cut follows the same code point rules.

The helpers take built-in values only and run no user code. Each caller
keeps its own policy where it differs: ``json_key(live=True)`` is capture's
(keys of a live value), ``live=False`` the renderer's (keys of a detached
snapshot).
"""

from __future__ import annotations

import json
import math
from typing import Any

# The canonical JSON options: ASCII-only, no NaN/Infinity, ", " and ": "
# separators, keys in insertion order. Every size in this package is measured
# in this encoding.
_JSON_KWARGS: dict[str, Any] = {
    "ensure_ascii": True,
    "allow_nan": False,
    "separators": (", ", ": "),
    "sort_keys": False,
}
_OPAQUE_PREFIX = "<opaque: "  # the opaque label is "<opaque: module.QualName>"
_OPAQUE_SUFFIX = ">"


# --- canonical JSON -----------------------------------------------------------------


def dumps(value: Any) -> str:
    """``value`` as canonical JSON text, the encoding every size here is measured in.

    Raises whatever ``json.dumps`` raises: ``ValueError`` for a non-finite
    float, ``TypeError`` for a value JSON cannot hold.

    >>> dumps({"a": [1, None]})
    '{"a": [1, null]}'
    """
    return json.dumps(value, **_JSON_KWARGS)


def pointer_child(path: str, component: str) -> str:
    """``path`` with one more JSON Pointer component, escaped as RFC 6901 says.

    >>> pointer_child("/a", "b/c~d")
    '/a/b~1c~0d'
    """
    escaped = component.replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def json_key(key: Any, *, live: bool) -> str | None:
    """The JSON object key that mapping key ``key`` becomes, or ``None`` when it has none.

    Only exact ``str``, ``int``, ``float``, ``bool`` and ``None`` keys
    convert, as ``json.dumps`` converts them (numbers by ``repr``); no user
    code runs.

    Args:
        key: the mapping key.
        live: ``True`` for a key of a live value (capture): an ``int`` over
            128 bits (unbounded decimal conversion) and a non-finite
            ``float`` are refused. ``False`` for a key of a detached
            snapshot (the renderers): converted as they are.

    >>> json_key(1, live=True), json_key(None, live=True), json_key((1,), live=True)
    ('1', 'null', None)
    """
    key_type = type(key)
    if key_type is str:
        return key
    if key is None:
        return "null"
    if key_type is bool:
        return "true" if key else "false"
    if key_type is int:
        if live and int.bit_length(key) > 128:
            return None
        return int.__repr__(key)
    if key_type is float:
        if live and not math.isfinite(key):
            return None
        return float.__repr__(key)
    return None


class Container:
    """The accounting of one JSON array or object being built: its size, entries and stop reason.

    ``size`` is the encoded size so far (``[]`` / ``{}``: 2); each entry after
    the first costs a 2-character separator, an object entry also its
    ``"key": `` (``overhead``). ``count`` is the entries committed and
    ``reason`` the first reason the container stopped (``None`` while it
    goes on). Capture and ``render_json`` both size with it, so their sizes
    agree exactly.

    Args:
        max_size: the most characters the encoded container may take.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.size = 2
        self.count = 0
        self.reason: Any = None  # the first reason to stop, a ``Reason``

    def stop(self, reason: str) -> None:
        """Record ``reason`` (a ``Reason``) as why the container stops, unless it already stopped."""
        if self.reason is None:
            self.reason = reason

    def room(self, overhead: int = 0) -> int:
        """The characters left for the next entry's value, after its separator and ``overhead``."""
        separator = 2 if self.count else 0
        return self.max_size - self.size - separator - overhead

    def slot(self, key: str | None = None) -> tuple[int, int] | None:
        """Make room for the next entry: ``(overhead, available)``, or ``None``.

        Args:
            key: the entry's JSON key, for an object; ``None`` for an array.

        Returns:
            ``overhead``, the characters of ``"key": `` (0 for an array), and
            ``available``, the characters left for the value. ``None`` (and
            the container stops with ``"max_length"``) when the key, or a
            value of at least one character, does not fit. JSON escaping
            never makes a key shorter, so an obviously oversized key is
            refused before it is encoded.
        """
        overhead = 0
        if key is not None:
            if len(key) + 2 > self.room():
                self.stop("max_length")
                return None
            overhead = len(dumps(key)) + 2
        available = self.room(overhead)
        if available < 1:
            self.stop("max_length")
            return None
        return overhead, available

    def commit(self, overhead: int, size: int) -> None:
        """Count an entry whose value took ``size`` characters, plus its ``overhead`` and separator."""
        self.size += (2 if self.count else 0) + overhead + size
        self.count += 1

    def pop(self, cost: int) -> None:
        """Remove the last entry, which cost ``cost`` characters (its separator included)."""
        self.size -= cost
        self.count -= 1

    def resume(self) -> None:
        """Forget the stop reason, after removed entries made room again."""
        self.reason = None


def fallback_candidates(type_name: str) -> tuple[Any, ...]:
    """The smallest JSON values that still say a value's type, by built-in type name, best first.

    A value too large for its slot keeps its type in JSON when the room
    allows (``""`` for a ``str``, ``[]`` for a list or tuple, ``{}`` for a
    dict, ``0`` for an int...); ``""`` or ``0`` for anything else. A
    ``type_name`` that is not an exact ``str`` is never compared (no
    ``__eq__`` runs).

    >>> fallback_candidates("list"), fallback_candidates("float")
    (([],), (0.0, 0))
    """
    if type(type_name) is not str:
        return ("", 0)  # never compare a foreign object (its ``__eq__`` could run)
    if type_name == "str":
        return ("",)
    if type_name in ("list", "tuple"):
        return ([],)
    if type_name == "dict":
        return ({},)
    if type_name == "bool":
        return (False, 0)
    if type_name == "int":
        return (0,)
    if type_name == "float":
        return (0.0, 0)
    if type_name == "NoneType":
        return (None, 0)
    return ("", 0)


def fallback_json(type_name: str, max_size: int) -> tuple[Any, int]:
    """The first of ``fallback_candidates(type_name)`` that fits ``max_size``.

    Returns:
        ``(value, size)``: the JSON value and its encoded size; ``(0, 1)``
        when none fits (callers give at least one character).
    """
    for candidate in fallback_candidates(type_name):
        encoded = dumps(candidate)
        if len(encoded) <= max_size:
            return candidate, len(encoded)
    # Callers only descend when at least one JSON character is available.
    return 0, 1


def fit_string(value: str, max_size: int) -> tuple[Any, int]:
    """The longest prefix of ``value`` whose JSON encoding fits ``max_size``.

    Returns:
        ``(prefix, size)``, the prefix and its encoded size; ``(0, 1)``
        when not even ``""`` fits (``max_size`` below 2).

    >>> fit_string("héllo", 8)
    ('h', 3)
    """
    if max_size < 2:
        return 0, 1
    raw_limit = max_size - 2
    candidate = value if len(value) <= raw_limit else value[:raw_limit]
    encoded = dumps(candidate)
    if len(encoded) <= max_size:
        return candidate, len(encoded)
    low, high = 0, len(candidate)
    while low < high:
        midpoint = (low + high + 1) // 2
        if len(dumps(candidate[:midpoint])) <= max_size:
            low = midpoint
        else:
            high = midpoint - 1
    prefix = candidate[:low]
    return prefix, len(dumps(prefix))


def opaque_name_budget(max_size: int) -> int:
    """How many characters of a type name fit in an opaque label of ``max_size`` JSON characters (at least 0)."""
    return max(0, max_size - len(_OPAQUE_PREFIX) - len(_OPAQUE_SUFFIX) - 2)


def opaque_json(type_name: str, fallback_type: str, max_size: int) -> tuple[Any, int]:
    """The ``"<opaque: Type>"`` label of ``type_name`` fitted to ``max_size``.

    Args:
        type_name: the qualified type name to show (cut to fit).
        fallback_type: the built-in type name whose smallest fallback is
            used when the label does not fit (``""``: the generic one).
        max_size: the room, in JSON characters.

    Returns:
        ``(value, size)``: the label, or a prefix of it, and its size; the
        smallest fallback when not even that fits.
    """
    fallback = fallback_json(fallback_type, max_size)
    if max_size < 2:
        return fallback
    label = f"{_OPAQUE_PREFIX}{type_name[: opaque_name_budget(max_size)]}{_OPAQUE_SUFFIX}"
    built = fit_string(label, max_size)
    return built if built[1] <= max_size else fallback


# --- UTF-8 ----------------------------------------------------------------------------
# Three different questions: where to cut a head (``safe_end``), where a tail
# may start (``safe_start``) and how much of a growing stream is complete
# (``complete_prefix``). They are not interchangeable. None of them validates
# UTF-8: invalid bytes are kept as they are and shown with replacement.


def safe_end(data: bytes | bytearray, end: int) -> int:
    """``end`` moved back to a UTF-8 code point boundary of ``data``, for a head ``data[:end]``.

    A code point that does not fit whole is left out, never kept beyond
    ``end`` (``max_string`` and the budget are upper bounds). Returns 0 when
    even the first code point does not fit, and at most ``len(data)``.

    >>> safe_end("aé".encode(), 2)
    1
    """
    cut = min(end, len(data))
    while 0 < cut < len(data) and data[cut] & 0xC0 == 0x80:
        cut -= 1
    return cut


def safe_start(data: bytes | bytearray, start: int) -> int:
    """``start`` (at least 0) moved forward past UTF-8 continuation bytes, for a tail ``data[start:]``.

    >>> safe_start("éa".encode(), 1)
    2
    """
    start = max(0, start)
    while start < len(data) and data[start] & 0xC0 == 0x80:
        start += 1
    return start


def complete_prefix(data: bytes | bytearray) -> int:
    """The length of ``data`` without a trailing UTF-8 sequence that more bytes could complete.

    Bytes that cannot start or continue a valid sequence count as complete.

    >>> complete_prefix("a€".encode()[:3]), complete_prefix(bytes([97, 255]))
    (1, 2)
    """
    size = len(data)
    index = size - 1
    continuation = 0
    while index >= 0 and continuation < 4 and data[index] & 0xC0 == 0x80:
        index -= 1
        continuation += 1
    if index < 0 or continuation >= 4:
        return size  # not a sequence that more bytes could complete
    lead = data[index]
    if lead < 0x80:
        need = 1
    elif lead >> 5 == 0b110:
        need = 2
    elif lead >> 4 == 0b1110:
        need = 3
    elif lead >> 3 == 0b11110:
        need = 4
    else:
        return size
    return index if size - index < need else size


def decode(data: bytes | bytearray) -> str:
    """``data`` decoded as UTF-8, invalid bytes as U+FFFD (how every renderer shows bytes)."""
    return bytes(data).decode("utf-8", errors="replace")


def encode_text(text: str) -> bytes:
    """``text`` as UTF-8 bytes; never raises.

    A lone surrogate from ``surrogateescape`` decoding becomes its original
    byte; any other lone surrogate is written as the text ``\\udXXX``. This
    is how a bytes ``TextStream`` and ``cut(unit="bytes")`` encode a
    ``str``, and how a chars stream writes its overflow file.
    """
    try:
        return text.encode("utf-8", errors="surrogateescape")
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="backslashreplace")
