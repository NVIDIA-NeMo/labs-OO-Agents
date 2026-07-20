# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visibility control for agent namespace (exec_globals).

Self-contained within agentdoc.

Simplified model: everything (module-level and agent-level) is visible by default.
Explicitly hide with:
- @hidden on functions (module-level or methods)
- Annotated[T, hidden] on variables (module-level or fields)
- ``with hidden:`` for imports and unannotated names at module level.

filter_module_globals() returns all module names except those hidden by the above.
Blocked modules are still stripped by the runtime after filtering.
"""

from __future__ import annotations

import functools
import inspect
import sys
import threading
import types
import typing
from typing import Annotated, Any, cast


def _is_dunder(name: str) -> bool:
    """True if name is a module/class dunder (e.g. __name__, __file__)."""
    return name.startswith("__") and name.endswith("__")


class _Hidden:
    """Exclude fields, methods, or imports from documentation.

    Works in three roles:
    1. Decorator: @hidden on functions (module-level or methods)
    2. Annotation marker: Annotated[T, hidden] on variables (module-level or fields)
    3. Context manager: ``with hidden:`` — names defined in the block are hidden at module level
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
    ) -> bool:
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


def is_hidden_method(func: Any) -> bool:
    """Check if a method is marked with @hidden.

    Also checks the underlying function for descriptors where @hidden is
    applied as the inner decorator — e.g. ``@property @hidden def foo``
    stores the marker on ``property.fget``, not on the property wrapper.
    """
    if getattr(func, "_agentdoc_hidden", False) is True:
        return True
    # @property @hidden: hidden marker lives on fget, not the property object
    if isinstance(func, property) and func.fget is not None:
        return getattr(func.fget, "_agentdoc_hidden", False) is True
    # @cached_property @hidden: hidden marker lives on func
    if isinstance(func, functools.cached_property):
        return getattr(func.func, "_agentdoc_hidden", False) is True
    return False


def is_hidden_field(cls: Any, name: str) -> bool:
    """Check if a field is hidden via Annotated[T, hidden] or spec() imperative annotation.

    Accepts both types and instances. Priority:
    1. Instance-level spec(self, "field", hidden=True/False) — highest priority.
       Allows per-instance opt-in: ``spec(self, "context", hidden=False)`` in __init__.
    2. Class-level spec(MyClass, "field", hidden=True/False) — overrides MRO scan.
    3. MRO annotation scan — stops at the first class that declares the annotation.

    Uses typing.get_type_hints() to resolve string annotations from
    ``from __future__ import annotations`` and forward references.
    When full class resolution fails (e.g. parent annotations reference
    types not importable in the subclass module), falls back to resolving
    just the target field's annotation in the declaring class's module.
    """
    from nooa.agentdoc._metadata import get_field_metadata

    # Instance-level override: spec(self, "field", hidden=False/True) in __init__
    if not isinstance(cls, type):
        instance_hidden = get_field_metadata(cls, name).get("hidden")
        if instance_hidden is True:
            return True
        if instance_hidden is False:
            return False
        cls = type(cls)

    # Class-level imperative override — spec(MyClass, "field", hidden=True/False).
    # Walk the MRO (leaf → base) so an override declared on a parent class is
    # honored on the child; the most-derived declaration wins.  get_field_metadata
    # is intentionally own-dict-only, so we iterate the MRO here.
    for klass in cls.__mro__:
        imperative_hidden = get_field_metadata(klass, name).get("hidden")
        if imperative_hidden is True:
            return True
        if imperative_hidden is False:
            return False

    for klass in cls.__mro__:
        if name not in inspect.get_annotations(klass):
            continue
        # get_type_hints resolves string annotations and forward refs
        try:
            resolved = typing.get_type_hints(klass, include_extras=True)
        except (NameError, AttributeError, TypeError):
            resolved = None

        hint = (
            resolved.get(name) if resolved is not None else _resolve_single_annotation(klass, name)
        )

        if hint is None:
            return False
        origin = typing.get_origin(hint)
        if origin is Annotated:
            args = typing.get_args(hint)
            from nooa.agentdoc._docs import SpecAnnotation

            return any(
                arg is hidden
                or (isinstance(arg, SpecAnnotation) and arg.kwargs.get("hidden") is True)
                for arg in args[1:]
            )
        return False
    return False


def _resolve_single_annotation(klass: type, name: str) -> Any:
    """Resolve a single annotation when full get_type_hints fails.

    This handles the case where ``from __future__ import annotations`` stores
    annotations as strings and ``get_type_hints(klass)`` fails because OTHER
    annotations in the MRO reference types not available in the current module.
    """
    raw = inspect.get_annotations(klass).get(name)
    if raw is None or not isinstance(raw, str):
        return raw

    mod = sys.modules.get(klass.__module__)
    ns = vars(mod) if mod else {}
    try:
        return eval(raw, ns)  # noqa: S307
    except (NameError, AttributeError, TypeError, SyntaxError, ValueError):
        return None


def is_hidden_module_variable(module: types.ModuleType, name: str) -> bool:
    """Check if a module-level variable is annotated with Annotated[T, hidden].

    Uses the module's annotations (via inspect.get_annotations) and resolves
    string annotations (e.g. from ``from __future__ import annotations``) via the module's namespace.
    """
    ann = inspect.get_annotations(module).get(name)
    if ann is None:
        return False
    if isinstance(ann, str):
        ns = vars(module).copy()
        try:
            ann = eval(ann, ns)  # noqa: S307
        except (NameError, AttributeError, TypeError, SyntaxError, ValueError):
            return False
    origin = typing.get_origin(ann)
    if origin is not Annotated:
        return False
    args = typing.get_args(ann)
    return any(arg is hidden for arg in args[1:])


def filter_module_globals(module: types.ModuleType) -> dict[str, Any]:
    """Return the module's globals filtered to names visible in documentation.

    All module-level names are visible by default; hide explicitly with
    ``@hidden``, ``Annotated[T, hidden]``, or ``with hidden:``.

    Excludes names added to ``_agentdoc_hidden_names`` (via ``with hidden:``),
    callables marked ``@hidden``, variables annotated ``Annotated[T, hidden]``,
    and internal agentdoc/dunder names.
    """
    hidden_names: set[str] = getattr(module, "_agentdoc_hidden_names", set())
    result: dict[str, Any] = {}
    # Snapshot items to avoid "dictionary changed size during iteration" (e.g. lazy module attrs)
    items = list(module.__dict__.items())
    for k, v in items:
        if k in hidden_names:
            continue
        if _is_dunder(k) or k.startswith("_agentdoc_"):
            continue
        if callable(v) and getattr(v, "_agentdoc_hidden", False) is True:
            continue
        if is_hidden_module_variable(module, k):
            continue
        result[k] = v
    return result


def iter_agent_mro_modules(agent_class: type) -> list[types.ModuleType]:
    """Ordered (base -> leaf) list of distinct user-defined modules across the MRO.

    These are the modules whose module-level names contribute to an agent's
    generated-code globals. When an ``Agent`` subclass inherits from a parent
    ``Agent`` (or plain mixin) defined in another module, the parent module's
    module-level functions, constants, and types must also be exposed —
    otherwise the child's ``execute_python()`` REPL hits ``NameError`` on names
    the parent's generation methods rely on.

    Framework base modules (``nooa`` and submodules) and ``builtins``
    are skipped so only user / mixin modules appear — EXCEPT the leaf class's
    own module, which is always included (this preserves prior behavior for
    agents defined inside the framework, e.g. tests and synthesized standalone
    agents). Duplicate modules are de-duplicated, retaining the first
    (base-most) position; ordering is base -> leaf so that callers merging in
    iteration order let the most-derived (leaf) module win on name collisions.
    """
    leaf_module = inspect.getmodule(agent_class)
    modules: list[types.ModuleType] = []
    seen: set[int] = set()
    # reversed(__mro__) => object first, leaf last (base -> leaf order).
    for cls in reversed(agent_class.__mro__):
        module = inspect.getmodule(cls)
        if module is None or id(module) in seen:
            continue
        name = getattr(module, "__name__", "")
        is_framework = name == "builtins" or name == "nooa" or name.startswith("nooa.")
        if module is not leaf_module and is_framework:
            continue
        seen.add(id(module))
        modules.append(module)
    return modules


def filter_mro_module_globals(agent_class: type) -> dict[str, Any]:
    """Merge :func:`filter_module_globals` across the agent class MRO.

    Modules from :func:`iter_agent_mro_modules` are merged base -> leaf so the
    most-derived (leaf) module wins on name collisions, matching Python's own
    name-resolution intuition. This is the multi-module generalization of
    ``filter_module_globals`` used to build generated-code globals for agents
    that inherit across modules.
    """
    result: dict[str, Any] = {}
    for module in iter_agent_mro_modules(agent_class):
        result.update(filter_module_globals(module))
    return result
