# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rich-compatible pretty printing for agent code execution.

Provides pprint() with Rich-compatible API but no Rich dependency.
Used by LLM-generated code for smart truncation of large data structures.

Implementation is provided by agentdoc._pformat - this module re-exports
the public API for use in agent-generated code.
"""

from typing import Any

# Import pprint from agentdoc (the source of truth)
from nooa.agentdoc import pprint as _agentdoc_pprint
from nooa.agentdoc._pformat import _pformat_to_str as _pformat


def pprint(
    _object: Any,
    *,
    indent_guides: bool = True,
    max_length: int | None = None,
    max_string: int | None = None,
    max_depth: int | None = None,
    expand_all: bool = False,
) -> None:
    """Pretty print with optional truncation (Rich-compatible API).

    Formats data structures with intelligent truncation for large containers
    and strings. API matches rich.pretty.pprint() for LLM familiarity.

    Args:
        _object: Object to print.
        indent_guides: Show indent guide lines (aesthetic only in our impl).
        max_length: Max elements per container (None=unlimited, Rich default).
        max_string: Max string characters (None=unlimited, Rich default).
        max_depth: Max nesting depth (None=unlimited, Rich default).
        expand_all: Always expand containers (default: compact for small ones).

    Examples:
        pprint(data)                      # No truncation (Rich default)
        pprint(data, max_length=50)       # Limit to 50 elements
        pprint(text, max_string=500)      # Limit string length
        pprint(nested, max_depth=3)       # Limit nesting

    Note:
        Variables persist after truncation. If you need more data,
        slice it: print(data[100:200])
    """
    _agentdoc_pprint(
        _object,
        max_length=max_length,
        max_string=max_string,
        max_depth=max_depth,
        expand_all=expand_all,
    )


# Re-export _pformat for internal use (e.g., CodeActStrategy return value formatting)
__all__ = ["_pformat", "pprint"]
