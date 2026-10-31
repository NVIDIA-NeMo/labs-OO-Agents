# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared codecs and accounting of ``nooa.snapshot`` (private leaf module).

Canonical JSON encoding and size accounting (#342's fallbacks, the opaque
label, string prefixes, pointers, mapping keys) and UTF-8 helpers (cut
points, decoding with replacement counting). ``_capture``, ``_render`` and
``_stream`` use them; each keeps its own safety policy: capture reads live
values and never runs user code, rendering only reads detached nodes. These
helpers take built-in values only and run no user code themselves.
"""

from __future__ import annotations

import json
import math
from typing import Any

_JSON_KWARGS: dict[str, Any] = {
    "ensure_ascii": True,
    "allow_nan": False,
    "separators": (", ", ": "),
    "sort_keys": False,
}
OPAQUE_PREFIX = "<opaque: "
OPAQUE_SUFFIX = ">"
REPLACEMENT = "\ufffd"
REPLACEMENT_BYTES = REPLACEMENT.encode("utf-8")


# --- canonical JSON -----------------------------------------------------------------


def dumps(value: Any) -> str:
    """Encode with the canonical options used by accounting and by ``render_json``."""
    return json.dumps(value, **_JSON_KWARGS)


def pointer_child(path: str, component: str) -> str:
    """Append one RFC 6901-escaped component to a JSON Pointer path."""
    escaped = component.replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def json_key(key: Any, *, live: bool) -> str | None:
    """The JSON object key a mapping key becomes, else None (not convertible).

    Only exact ``str``, ``int``, ``float``, ``bool`` and ``None`` keys
    convert, as ``json.dumps`` converts them (numbers by ``repr``). The two
    policies stay explicit: ``live=True`` is capture's, for keys of a live
    value: an ``int`` over 128 bits (unbounded decimal conversion) and a
    non-finite ``float`` are refused. ``live=False`` is the renderer's, for
    the keys of a detached snapshot: converted as they are.
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


def fallback_candidates(type_name: str) -> tuple[Any, ...]:
    """#342's type-oriented smallest JSON fallbacks, by built-in type name."""
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
    """The first fallback for ``type_name`` that fits ``max_size`` (else ``0``)."""
    for candidate in fallback_candidates(type_name):
        encoded = dumps(candidate)
        if len(encoded) <= max_size:
            return candidate, len(encoded)
    # Callers only descend when at least one JSON character is available.
    return 0, 1


def fit_string(value: str, max_size: int) -> tuple[Any, int]:
    """The longest prefix of ``value`` whose JSON encoding fits ``max_size`` (#342)."""
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


def opaque_json(type_name: str, fallback_type: str, max_size: int) -> tuple[Any, int]:
    """#342's ``"<opaque: Type>"`` label fitted to ``max_size``, else a smallest fallback."""
    fallback = fallback_json(fallback_type, max_size)
    if max_size < 2:
        return fallback
    name_budget = max(0, max_size - len(OPAQUE_PREFIX) - len(OPAQUE_SUFFIX) - 2)
    label = f"{OPAQUE_PREFIX}{type_name[:name_budget]}{OPAQUE_SUFFIX}"
    built = fit_string(label, max_size)
    return built if built[1] <= max_size else fallback


# --- UTF-8 ----------------------------------------------------------------------------
# Three different questions: where to cut a head (``safe_end``), where a tail
# may start (``safe_start``) and how much of a growing stream is complete
# (``complete_prefix``). They are not interchangeable.


def safe_end(data: bytes | bytearray, end: int) -> int:
    """``end`` moved back to a UTF-8 code point boundary (0 when the first one is cut).

    A code point that does not fit whole is left out, never kept beyond
    ``end`` (``max_string`` and the budget are upper bounds).
    """
    cut = min(end, len(data))
    while 0 < cut < len(data) and data[cut] & 0xC0 == 0x80:
        cut -= 1
    return cut


def safe_start(data: bytes | bytearray, start: int) -> int:
    """``start`` moved forward past UTF-8 continuation bytes."""
    start = max(0, start)
    while start < len(data) and data[start] & 0xC0 == 0x80:
        start += 1
    return start


def complete_prefix(data: bytes | bytearray) -> int:
    """Length of ``data`` without a trailing, incomplete UTF-8 sequence."""
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


def decode_counted(data: bytes | bytearray) -> tuple[str, int]:
    """``data`` decoded as UTF-8 with replacement, and the replacements introduced."""
    text = bytes(data).decode("utf-8", errors="replace")
    introduced = text.count(REPLACEMENT) - bytes(data).count(REPLACEMENT_BYTES)
    return text, max(0, introduced)
