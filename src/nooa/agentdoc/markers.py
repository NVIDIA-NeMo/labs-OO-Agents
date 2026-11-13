# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The visibility markers and where ``spec()`` stores its keywords: the contract agentdoc and its readers share.

- ``hidden``: the marker of ``Annotated[T, hidden]``, ``@hidden`` and
  ``with hidden:``.
- ``SpecAnnotation``: what ``spec(**kwargs)`` returns without a target, as
  ``Annotated`` metadata (``Annotated[T, spec(hidden=True)]``) or as a
  decorator. Its keywords are its ``kwargs`` dict.
- ``CLASS_METADATA_ATTR``: the attribute holding a class's or callable's
  ``@spec(...)`` keywords (a ``dict``).
- ``FIELD_METADATA_ATTR``: the attribute holding the ``spec(obj, field, ...)``
  overrides declared on a class or an instance (``{field: {keyword: value}}``,
  in the object's own ``__dict__``).

agentdoc (``doc``, ``pformat``, ``spec``) and ``nooa.snapshot`` (which
fields a capture may show) both read the markers from here, so a field one
of them hides the other hides too. Standard library only: importing it never
imports a renderer or the snapshot, so neither package needs the other to
know what is hidden.
"""

from __future__ import annotations

import contextlib
import functools
import sys
import threading
import types
from typing import Any, Literal, cast

__all__ = ["CLASS_METADATA_ATTR", "FIELD_METADATA_ATTR", "SpecAnnotation", "hidden"]

CLASS_METADATA_ATTR = "_agentdoc_docs"
"""The attribute of a class or callable that holds its ``@spec(...)`` keywords."""
FIELD_METADATA_ATTR = "_agentdoc_fields_docs"
"""The attribute of a class or instance that holds its ``spec(obj, field, ...)`` overrides."""


def _object_namespace(obj: Any) -> Any:
    """Return ``obj.__dict__`` without invoking an instance ``__getattr__``."""
    try:
        return object.__getattribute__(obj, "__dict__")
    except (AttributeError, TypeError):
        return {}


def get_docs_metadata(obj: Any) -> dict[str, Any]:
    """Return the spec() metadata dict for obj (never None)."""
    return getattr(obj, CLASS_METADATA_ATTR, None) or {}


def set_docs_metadata(obj: Any, **kwargs: Any) -> None:
    """Merge kwargs into the spec() metadata dict on obj."""
    with contextlib.suppress(AttributeError, TypeError):
        existing = dict(getattr(obj, CLASS_METADATA_ATTR, None) or {})
        existing.update(kwargs)
        setattr(obj, CLASS_METADATA_ATTR, existing)


def get_field_metadata(obj: Any, field: str) -> dict[str, Any]:
    """Return per-field spec() metadata for obj's own declarations only (never None)."""
    fields_meta = _object_namespace(obj).get(FIELD_METADATA_ATTR) or {}
    return fields_meta.get(field) or {}


def set_field_metadata(obj: Any, field: str, **kwargs: Any) -> None:
    """Merge kwargs into the per-field spec() metadata on obj."""
    with contextlib.suppress(AttributeError, TypeError):
        fields_meta = _object_namespace(obj).get(FIELD_METADATA_ATTR)
        if fields_meta is None:
            fields_meta = {}
            setattr(obj, FIELD_METADATA_ATTR, fields_meta)
        existing = fields_meta.get(field) or {}
        existing.update(kwargs)
        fields_meta[field] = existing


class _Hidden:
    """Hide a field, callable, or module-level name from documentation.

    Use as ``@hidden``, ``Annotated[T, hidden]``, or ``with hidden:``.
    """

    def __init__(self):
        self._local = threading.local()

    @property
    def _stack(self) -> list[tuple[types.ModuleType, set[str]]]:
        """Thread-local stack of (module, snapshot) for nested ``with hidden:``."""
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return cast(list[tuple[types.ModuleType, set[str]]], self._local.stack)

    def __call__(self, func: Any) -> Any:
        """Use as @hidden decorator on methods or module-level functions."""
        try:
            func._agentdoc_hidden = True
        except AttributeError:
            # property/cached_property have no __dict__; store on the inner
            # function so is_hidden_method() can find it.
            if isinstance(func, property) and func.fget is not None:
                func.fget._agentdoc_hidden = True  # type: ignore[attr-defined]
            elif isinstance(func, functools.cached_property):
                func.func._agentdoc_hidden = True  # type: ignore[attr-defined]
        return func

    def __enter__(self):
        frame = sys._getframe(1)
        if frame.f_locals is not frame.f_globals:
            raise RuntimeError(
                "'with hidden:' must be used at module level, not inside a function or class body"
            )
        module_name = frame.f_globals.get("__name__")
        module = sys.modules.get(module_name) if module_name else None
        if module is None:
            raise RuntimeError(
                "hidden must be used at module level, could not determine calling module."
            )
        return self._enter_for_module(module)

    def _enter_for_module(self, module: types.ModuleType):
        """Enter the block for a specific module (used by tests)."""
        self._stack.append((module, set(module.__dict__.keys())))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        if not self._stack:
            raise RuntimeError("_Hidden.__exit__ called without matching __enter__")
        module, snapshot = self._stack.pop()
        new_names = set(module.__dict__.keys()) - snapshot
        existing: set[str] = getattr(module, "_agentdoc_hidden_names", set())
        module._agentdoc_hidden_names = existing | new_names  # type: ignore[attr-defined]
        return False

    def __repr__(self):
        return "hidden"


hidden = _Hidden()
"""The ``hidden`` marker: ``Annotated[T, hidden]``, ``@hidden``, ``with hidden:``."""


def _sync_hidden(target: Any, hidden_value: bool | None) -> None:
    """Sync the ``_agentdoc_hidden`` attribute on *target* based on *hidden_value*.

    Shared by ``SpecAnnotation.__call__`` (decorator form) and
    ``Spec.__call__`` (imperative form).
    """
    if hidden_value is True:
        with contextlib.suppress(AttributeError, TypeError):
            target._agentdoc_hidden = True
    elif hidden_value is False:
        try:
            target._agentdoc_hidden = False
        except (AttributeError, TypeError):
            # C-extension method: fall back to class-level metadata
            owner = getattr(target, "__objclass__", None)
            mname = getattr(target, "__name__", None)
            if owner is not None and mname is not None:
                set_field_metadata(owner, mname, hidden=False)


class SpecAnnotation:
    """A specification returned by ``spec(**kwargs)`` when no positional target is given.

    Works as:
    - Annotated metadata: ``Annotated[str, spec(hidden=True)]``
    - Decorator: ``@spec(hidden=True)``
    """

    __slots__ = ("kwargs",)

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__[T](self, target: T) -> T:
        """Apply as decorator: ``@spec(...)``."""
        # Sync _agentdoc_hidden attribute so is_hidden_method() and the _ prefix
        # override check in _extract_methods() both see the correct state.
        _sync_hidden(target, self.kwargs.get("hidden"))
        set_docs_metadata(target, **self.kwargs)
        return target

    def __repr__(self) -> str:
        kw = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"spec({kw})"
