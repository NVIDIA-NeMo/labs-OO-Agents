# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test helper for deriving parameter names from a signature string."""

import ast


def param_names_from_signature(signature: str) -> list[str]:
    """Ordered parameter names (excluding ``self``) from a signature string.

    Uses AST so it mirrors ``inspect.signature().parameters`` exactly — robust to
    commas inside annotations/defaults, ``*args``/``**kwargs``, and the ``*``/``/``
    separators — rather than hand-splitting on commas. Lets test helpers populate a
    CurrentCall's ``param_names`` the way real calls (built via ``from_method``) do.
    """
    a = ast.parse(f"def _f{signature}: ...").body[0].args
    names: list[str] = []
    for group in (a.posonlyargs, a.args):
        names.extend(p.arg for p in group)
    if a.vararg is not None:
        names.append(a.vararg.arg)
    names.extend(p.arg for p in a.kwonlyargs)
    if a.kwarg is not None:
        names.append(a.kwarg.arg)
    return [n for n in names if n != "self"]
