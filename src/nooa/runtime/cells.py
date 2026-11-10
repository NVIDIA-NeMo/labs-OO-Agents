# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The module that classes and functions defined in owned cells belong to.

Every :class:`~nooa.runtime.execution_namespace.ExecutionNamespace` seeds
``__name__`` with this module's name, so a class a cell defines reports
``__module__ == "nooa.runtime.cells"`` (not ``builtins``) and its instances
serialize under the stable name ``nooa.runtime.cells.<qualname>``. Looking an
attribute up here finds such a class in a live namespace (the most recently
created one first), which is how a snapshot restore in this process recreates
the instance. A class only a previous process defined no longer exists; its
instances are dropped on restore (with a warning), the other values survive.
"""

from __future__ import annotations

import weakref
from typing import Any

_namespaces: list[weakref.ref] = []


def register(namespace: Any) -> None:
    """Make the top-level classes and functions of ``namespace.globals`` resolvable here."""
    _namespaces[:] = [ref for ref in _namespaces if ref() is not None]
    _namespaces.append(weakref.ref(namespace))


def __getattr__(name: str) -> Any:
    for ref in reversed(list(_namespaces)):
        namespace = ref()
        if namespace is None:
            continue
        value = namespace.globals.get(name)
        if getattr(value, "__module__", None) == __name__ and isinstance(value, type):
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
