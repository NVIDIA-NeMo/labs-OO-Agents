# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve an attribute chain such as ``self.shell.read`` to a live callable.

Used by ``nooa.errors.formatting`` to name the un-awaited async call in an
error hint after a cell has failed. This module has no import-time
dependencies on the rest of nooa.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any


def resolve_static_chain(node: ast.expr, namespace: dict[str, Any]) -> object | None:
    """Resolve ``a.b.c`` against live objects without invoking descriptors.

    Each step uses ``inspect.getattr_static``; a missing attribute or a
    ``property`` along the way ends resolution (returns None).
    """
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in namespace:
        return None
    value: Any = namespace[node.id]
    for name in reversed(names):
        try:
            value = inspect.getattr_static(value, name)
        except AttributeError:
            return None
        if isinstance(value, property):
            return None
    return value


def is_coroutine_function(value: object) -> bool:
    """``inspect.iscoroutinefunction`` that sees through static/classmethod and wrappers."""
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    try:
        if inspect.iscoroutinefunction(value):
            return True
        return inspect.iscoroutinefunction(inspect.unwrap(value))  # type: ignore[arg-type]
    except Exception:
        return False
