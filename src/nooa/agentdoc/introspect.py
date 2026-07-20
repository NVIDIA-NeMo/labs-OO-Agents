# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc.introspect — fine-grained introspection views.

Lower-level alternatives to ``doc()`` when you need just one slice of a type
rather than the full rendered contract:

    from nooa.agentdoc.introspect import methods, variables

    methods(MyAgent)                   # callable names + signatures only
    methods(MyAgent, detail="full")    # with docstrings
    variables(my_agent)                # field names + current values only

These are composable building blocks. ``doc()`` builds its output from the same
underlying data — use these when you need a narrower view, or when you want to
mix the output into your own prompt template.

``methods()`` respects both ``@hidden`` annotations and the ``_`` name convention.
``variables()`` respects ``@hidden``, ``Annotated[T, hidden]``, and the ``_``
name prefix convention (same filtering as ``doc()``). For the full rendered API
contract (fields + methods + docstrings), use ``doc()`` instead.
"""

from __future__ import annotations

from nooa.agentdoc.core import methods, variables

__all__ = [
    "methods",
    "variables",
]
