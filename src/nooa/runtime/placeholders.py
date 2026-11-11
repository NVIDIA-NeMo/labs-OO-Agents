# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docstring placeholders (``{expr}``) rendered through ``nooa.snapshot``, bounded.

One rule for every docstring that expands placeholders: method docstrings and
Task prompts (``ActorRuntime.expand_variables``) and the class-docstring system
prompt (``Agent._resolve_system_prompt``).

- ``{expr}`` with no conversion and no format spec: a ``str`` value is raw
  text, cut to ``PLACEHOLDER_TEXT_CHARS`` (``nooa.snapshot.cut``, head and
  tail around ``render_output``'s cut line); any other value is
  ``render_python(capture(value, limits))`` under the prefill limits
  (``TruncationConfig.prefill_format``), so a value is read without running
  its code, as a parameter in the prefill is.
- ``{expr!r}`` / ``{expr!s}`` / ``{expr!a}`` or ``{expr:spec}``: applied as
  written, then the resulting string is cut the same way.

A small ``str``, number, ``None``, bool, list, tuple or dict renders as
``str(value)`` did.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PLACEHOLDER_TEXT_CHARS", "bound_placeholder_text", "render_placeholder"]

PLACEHOLDER_TEXT_CHARS = 50_000
"""Characters one placeholder may insert: the preview bound (``Limits().max_chars``)."""


def bound_placeholder_text(text: str) -> str:
    """``text`` whole up to ``PLACEHOLDER_TEXT_CHARS``, else its first and last halves around the cut line."""
    if len(text) <= PLACEHOLDER_TEXT_CHARS:
        return text
    from nooa.snapshot import cut, render_output

    tail = PLACEHOLDER_TEXT_CHARS // 2
    return render_output(cut(text, head=PLACEHOLDER_TEXT_CHARS - tail, tail=tail))


def _value_text(value: Any) -> str:
    """A non-string value as ``render_python(capture(value))`` under the prefill limits."""
    from nooa.agentdoc._pformat import _value_limits
    from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
    from nooa.snapshot import capture, render_python

    limits = DEFAULT_TRUNCATION_CONFIG.prefill_format
    snapshot = capture(value, _value_limits(limits.max_length, limits.max_string, limits.max_depth))
    return bound_placeholder_text(render_python(snapshot))


def render_placeholder(value: Any, conversion: str | None = None, format_spec: str = "") -> str:
    """The bounded text a placeholder's ``value`` inserts (see the module).

    ``conversion`` is ``"r"``, ``"s"``, ``"a"`` or ``None``; ``format_spec``
    is the text after ``:`` (``""``: none). A format spec the value rejects
    raises, as ``format()`` does; the caller decides what to insert then.
    """
    if conversion is None and not format_spec:
        if isinstance(value, str):
            return bound_placeholder_text(str(value))
        return _value_text(value)
    if conversion == "r":
        value = repr(value)
    elif conversion == "s":
        value = str(value)
    elif conversion == "a":
        value = ascii(value)
    text = format(value, format_spec) if format_spec else str(value)
    return bound_placeholder_text(text)
