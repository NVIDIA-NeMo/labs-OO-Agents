# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docstring placeholders (``{expr}``) rendered through ``nooa.snapshot``, bounded.

One rule for every docstring that expands placeholders: method docstrings and
Task prompts (``ActorRuntime.expand_variables``) and the class-docstring system
prompt (``Agent._resolve_system_prompt``).

- ``{expr}`` with no conversion and no format spec: a ``str`` value is raw
  text, cut to the agent's ``TruncationConfig.placeholder_chars``
  (``nooa.snapshot.cut``, head and tail around ``render_output``'s cut line;
  ``None``: whole); any other value is
  ``render_python(capture(value, limits))`` under the agent's prefill limits
  (its ``TruncationConfig.prefill_format``), so a value is read without running
  its code, as a parameter in the prefill is.
- ``{expr!r}`` / ``{expr!s}`` / ``{expr!a}`` or ``{expr:spec}``: applied as
  written, then the resulting string is cut the same way.

A small ``str``, number, ``None``, bool, list, tuple or dict renders as
``str(value)`` did. To insert a value's ``str()`` text instead of its
snapshot notation (a ``Path`` as ``/workspace/repo`` rather than
``PosixPath('/workspace/repo')``), write ``{str(expr)}`` or ``{expr!s}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nooa.config.truncation_config import FormatConfig

__all__ = ["bound_placeholder_text", "render_placeholder"]


def bound_placeholder_text(text: str, max_chars: int | None) -> str:
    """``text`` whole up to ``max_chars`` (``None``: always), else its first and last halves around the cut line."""
    if max_chars is None:
        return text
    from nooa._bounded_text import cut_text

    return cut_text(text, max_chars)


def _value_text(value: Any, prefill: FormatConfig | None, max_chars: int | None) -> str:
    """A non-string value as ``render_python(capture(value))`` under the prefill limits."""
    from nooa.agentdoc._pformat import _value_limits
    from nooa.snapshot import capture, render_python

    limits = prefill
    if limits is None:
        from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

        limits = DEFAULT_TRUNCATION_CONFIG.prefill_format
    snapshot = capture(value, _value_limits(limits.max_length, limits.max_string, limits.max_depth))
    return bound_placeholder_text(render_python(snapshot), max_chars)


def render_placeholder(
    value: Any,
    conversion: str | None = None,
    format_spec: str = "",
    *,
    max_chars: int | None,
    prefill: FormatConfig | None = None,
) -> str:
    """The bounded text a placeholder's ``value`` inserts (see the module).

    ``conversion`` is ``"r"``, ``"s"``, ``"a"`` or ``None``; ``format_spec``
    is the text after ``:`` (``""``: none). ``max_chars``: the agent's
    effective ``TruncationConfig.placeholder_chars``, the most characters the
    result has (``None``: whole). ``prefill``: the agent's effective
    ``TruncationConfig.prefill_format``, the limits of a non-string value
    (``None``: the defaults). A format spec the value rejects raises, as
    ``format()`` does; the caller decides what to insert then.
    """
    if conversion is None and not format_spec:
        if isinstance(value, str):
            return bound_placeholder_text(str(value), max_chars)
        return _value_text(value, prefill, max_chars)
    if conversion == "r":
        value = repr(value)
    elif conversion == "s":
        value = str(value)
    elif conversion == "a":
        value = ascii(value)
    text = format(value, format_spec) if format_spec else str(value)
    return bound_placeholder_text(text, max_chars)
