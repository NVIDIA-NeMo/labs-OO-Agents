# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc.visibility — inspect visibility decisions for framework authors.

When building a framework on top of agentdoc, use these helpers to mirror the
same filtering that ``spec()`` and ``hidden`` produce — so your exec_globals,
prompt context, or schema builder includes exactly the same names that
documentation shows.

    from nooa.agentdoc.visibility import filter_module_globals, is_hidden_field

    # Populate exec_globals from a module, respecting hidden annotations
    agent_globals = filter_module_globals(my_module)

    # Guard a field before including it in a prompt payload
    if not is_hidden_field(MyAgent, "api_key"):
        payload["api_key"] = agent.api_key

Not part of the two-step user workflow (spec → doc).
Import explicitly when building a framework on top of agentdoc.
"""

from __future__ import annotations

from nooa.agentdoc._visibility import (
    filter_module_globals,
    filter_mro_module_globals,
    is_hidden_field,
    is_hidden_method,
    is_hidden_module_variable,
    iter_agent_mro_modules,
)

__all__ = [
    "filter_module_globals",
    "filter_mro_module_globals",
    "is_hidden_field",
    "is_hidden_method",
    "is_hidden_module_variable",
    "iter_agent_mro_modules",
]
