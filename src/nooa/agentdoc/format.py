# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal formatting utilities for agentdoc."""

from typing import Any

from nooa.agentdoc._pformat import _pformat_to_str as _pformat
from nooa.agentdoc.doc_config import DocConfig


def format_value_summary(value: Any, config: DocConfig) -> str:
    """Format a value with truncation for summary view.

    Uses _pformat for readable formatting with smart truncation (Rich-style).
    Shows truncation counts: '+N' for strings, '... +N' for containers.

    Args:
        value: Value to format
        config: Configuration controlling formatting

    Returns:
        Formatted string representation
    """
    if value is None:
        return "None"

    # Simple types don't need _pformat
    if isinstance(value, (int, float, bool)):
        return str(value)

    # Use _pformat for everything (Rich-style formatting with truncation and preview)
    try:
        return _pformat(
            value,
            max_length=config.max_list_items,
            max_string=config.max_value_chars,
            max_depth=2,
        )
    except Exception:  # _pformat may raise ImportError, AttributeError, RecursionError, etc.
        return f"{type(value).__name__}(...)"


def truncate_docstring(docstring: str | None, max_lines: int = 1) -> str:
    """Truncate a docstring to a specified number of lines.

    Args:
        docstring: Docstring to truncate (or None)
        max_lines: Maximum number of lines to return

    Returns:
        Truncated docstring
    """
    if not docstring:
        return ""

    lines = docstring.strip().split("\n")
    if len(lines) <= max_lines:
        return docstring.strip()

    return "\n".join(lines[:max_lines]).strip()
