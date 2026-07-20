# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared cleanup functions for model output.

These functions handle common "model-helping" transforms: stripping markdown
fences, removing XML wrappers, etc.  They are called from strategy-level and
runtime-level intercept points where model output is cleaned up before further
processing.

Each function is idempotent and safe to call on already-clean input.
"""

from __future__ import annotations

import re

# ── Code fence stripping ───────────────────────────────────────────


def strip_code_fences(code: str) -> tuple[str, str | None]:
    """Strip markdown code fences from LLM-generated code.

    Handles ````` ```python ```, ````` ```py ```, ````` ``` ```, and any other
    language tag (````` ```bash ```, etc.).  Requires balanced opening and
    closing fences.

    Returns:
        Tuple of (cleaned_code, fence_token) where fence_token is the
        opening fence string (e.g. "```python") if fences were stripped,
        or None if no fences were found.
    """
    stripped = code.strip()
    # Multiline form: ```lang\nCODE\n```
    # The opening line after the lang tag must contain only whitespace before
    # the newline. This preserves code when the LLM puts code on the same
    # line as the fence (e.g. "```python print('x')\n```").
    fence_pattern = r"^(```\w*)[ \t]*\n(.*?)\n?```$"
    match = re.match(fence_pattern, stripped, re.DOTALL)
    if match:
        return match.group(2).strip(), match.group(1)
    # Inline form: ```lang CODE``` (no newline) — handles same-line cases
    # like "```python print('x')```".
    inline_pattern = r"^(```\w*)\s*(.*?)\s*```$"
    match = re.match(inline_pattern, stripped, re.DOTALL)
    if match:
        return match.group(2).strip(), match.group(1)
    return code, None


# ── XML wrapper stripping ──────────────────────────────────────────


def strip_xml_wrapper(content: str) -> tuple[str, str | None]:
    """Strip a single outermost XML wrapper tag from content.

    Matches ``<tagname ...>CONTENT</tagname>`` with or without attributes.
    Only strips if the entire string is a single wrapper (anchored ``^...$``).

    Does NOT check for nested XML or raise on malformed input — callers
    that need strict behavior should check the result themselves.

    Returns:
        Tuple of (inner_content, tag_name) where tag_name is the stripped
        tag (e.g. "assistant_message") or None if no wrapper was found.
    """
    content = content.strip()
    if not content.startswith("<"):
        return content, None

    # Single pattern handles both with-attributes and without-attributes cases.
    # Use .* (not .+) so empty-content tags <tag></tag> still match.
    pattern = r"^<([a-zA-Z_][\w-]*)(?:\s[^>]*)?>(.*)</\1>$"
    match = re.match(pattern, content, re.DOTALL)
    if match:
        return match.group(2).strip(), match.group(1)

    return content, None
