# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loud-fail on cross-boundary mutation of non-``self`` module-level state.

A sandbox cell runs in a forked worker with its *own* copy of the agent's
module-level globals. Only ``self.*`` brokers to the live parent — module-level
objects do not, so a cell that mutates one (``SOME_DICT[k] = v``,
``some_list.append(...)``, ``some_obj.attr = x``) would succeed locally while the
parent never sees it: a **silent divergence** from the in-process backend where
that state is shared.

This module makes that mutation **loud**. ``freeze_module_state`` wraps mutable
module-level values in a read-only view whose reads/calls/iteration forward
normally but whose mutation entry points raise :class:`SandboxStateError`.

Best-effort, matching the rest of this in-process guard layer: it catches the
common vectors (dict/list/set mutators + item/attr assignment) but is not a
security boundary — the OS layer (Landlock/seccomp/rlimits) is. Its job is to
turn a correctness footgun into an actionable error, not to contain a hostile
cell.
"""

from __future__ import annotations

import types
from typing import Any

__all__ = ["SandboxStateError", "ReadOnlyView", "freeze_module_state"]


class SandboxStateError(RuntimeError):
    """A sandbox cell tried to mutate non-``self`` module-level state."""


_MESSAGE = (
    "cannot mutate module-level state {name!r} inside a sandbox cell: cells get a "
    "private copy of module globals, so the change would not reach the parent. Put "
    "shared mutable state on `self` (brokered) or return it via return_result()."
)

# Well-known in-place mutators of the builtin containers.
_MUTATOR_METHODS = frozenset(
    {
        # list
        "append",
        "extend",
        "insert",
        "remove",
        "pop",
        "clear",
        "sort",
        "reverse",
        # dict
        "update",
        "setdefault",
        "popitem",
        # set
        "add",
        "discard",
        "difference_update",
        "intersection_update",
        "symmetric_difference_update",
    }
)


class ReadOnlyView:
    """A read-only proxy: reads/calls/iteration forward; mutation raises."""

    __slots__ = ("_ro_target", "_ro_name")

    def __init__(self, target: Any, name: str) -> None:
        object.__setattr__(self, "_ro_target", target)
        object.__setattr__(self, "_ro_name", name)

    def _raise(self) -> None:
        raise SandboxStateError(_MESSAGE.format(name=object.__getattribute__(self, "_ro_name")))

    # -- attribute protocol -------------------------------------------------
    def __getattr__(self, attr: str) -> Any:
        if attr in _MUTATOR_METHODS:
            self._raise()
        return getattr(object.__getattribute__(self, "_ro_target"), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        self._raise()

    def __delattr__(self, attr: str) -> None:
        self._raise()

    # -- item protocol ------------------------------------------------------
    def __getitem__(self, key: Any) -> Any:
        return object.__getattribute__(self, "_ro_target")[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._raise()

    def __delitem__(self, key: Any) -> None:
        self._raise()

    # -- read-only protocols forwarded to the target ------------------------
    def __iter__(self):
        return iter(object.__getattribute__(self, "_ro_target"))

    def __reversed__(self):
        return reversed(object.__getattribute__(self, "_ro_target"))

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_ro_target"))

    def __contains__(self, item: Any) -> bool:
        return item in object.__getattribute__(self, "_ro_target")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return object.__getattribute__(self, "_ro_target")(*args, **kwargs)

    def __eq__(self, other: Any) -> bool:
        return object.__getattribute__(self, "_ro_target") == other

    def __hash__(self) -> int:
        return hash(object.__getattribute__(self, "_ro_target"))

    def __bool__(self) -> bool:
        return bool(object.__getattribute__(self, "_ro_target"))

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_ro_target"))

    def __str__(self) -> str:
        return str(object.__getattribute__(self, "_ro_target"))


# Values that must NOT be wrapped: they are read-only-by-nature or are the
# callables/types/namespaces a cell uses normally.
_IMMUTABLE = (int, float, complex, bool, str, bytes, frozenset, type(None), tuple, range)
_NEVER_WRAP = (
    types.ModuleType,
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
)


def _is_mutable_state(value: Any) -> bool:
    """True if ``value`` is module-level mutable *data* worth guarding."""
    if (
        isinstance(value, ReadOnlyView)
        or isinstance(value, _IMMUTABLE)
        or isinstance(value, _NEVER_WRAP)
    ):
        return False
    if isinstance(value, (list, dict, set)):
        return True
    # A plain instance with a writable __dict__ (excludes slotted/atomic objects).
    return hasattr(value, "__dict__") and isinstance(getattr(value, "__dict__", None), dict)


def freeze_module_state(ns: dict[str, Any]) -> dict[str, Any]:
    """Wrap mutable module-level values in ``ns`` with a read-only view, in place.

    Only non-underscore data values are wrapped (dunders and framework builtins
    are left alone). Returns ``ns`` for convenience.
    """
    for name, value in list(ns.items()):
        if name.startswith("__"):
            continue
        if _is_mutable_state(value):
            ns[name] = ReadOnlyView(value, name)
    return ns
