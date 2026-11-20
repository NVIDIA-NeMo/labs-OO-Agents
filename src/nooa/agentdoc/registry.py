# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Type-based extractor registry for agentdoc.

Allows libraries to register custom TypeInfo extractors for types you don't control,
providing TypeInfo (and optionally instance values) for third-party types.

This is the recommended approach for third-party library integration.
"""

import inspect
import re
import sys
import threading
from collections.abc import Callable
from typing import Any, NamedTuple

from nooa.agentdoc._info import CallableInfo, ModuleInfo, TypeInfo

_registry_lock = threading.RLock()

# Type for extractor functions
# Returns TypeInfo for types, or (TypeInfo, dict) for instances
TypeInfoExtractorFunc = Callable[[Any], TypeInfo | tuple[TypeInfo, dict[str, Any]]]

# Module extractor: receives the module object, returns ModuleInfo
ModuleInfoExtractorFunc = Callable[[Any], ModuleInfo]

# Global registry: type -> extractor function
_extractor_registry: dict[type, TypeInfoExtractorFunc] = {}

# Global registry: module qualified name -> extractor function
_module_extractor_registry: dict[str, ModuleInfoExtractorFunc] = {}

# Callable extractor: receives the function object, returns CallableInfo
CallableInfoExtractorFunc = Callable[[Any], CallableInfo]

# Global registry: id(callable) -> (callable, extractor). Keyed by identity because
# many callables are not hashable by value; holding the callable keeps its id unique.
_callable_extractor_registry: dict[int, tuple[Any, CallableInfoExtractorFunc]] = {}


def register_type_info_extractor(
    target_type: type,
) -> Callable[[TypeInfoExtractorFunc], TypeInfoExtractorFunc]:
    """Decorator to register a TypeInfo extractor for a type.

    The extractor function receives the object (type or instance) and returns:
    - TypeInfo for types
    - (TypeInfo, values_dict) tuple for instances

    Args:
        target_type: The type to register the extractor for

    Returns:
        Decorator function

    Example:
        @register_type_info_extractor(ThirdPartyClass)
        def extract_third_party(obj) -> TypeInfo | tuple[TypeInfo, dict]:
            if isinstance(obj, type):
                return TypeInfo(
                    name=obj.__name__,
                    base="ThirdParty",
                    fields=[...],
                    methods=[...],
                    docstring=obj.__doc__,
                )
            else:
                type_info = extract_third_party(type(obj))
                values = {"field": obj.field}
                return (type_info, values)
    """

    def decorator(func: TypeInfoExtractorFunc) -> TypeInfoExtractorFunc:
        with _registry_lock:
            _extractor_registry[target_type] = func
        return func

    return decorator


def get_type_info_extractor(obj: Any) -> TypeInfoExtractorFunc | None:
    """Get the registered TypeInfo extractor for an object's type.

    Checks the object's type and all base classes (MRO order) for a
    registered extractor.

    Args:
        obj: Object to get extractor for (type or instance)

    Returns:
        Extractor function if registered, None otherwise
    """
    # Get the type to check
    target_type = obj if isinstance(obj, type) else type(obj)

    # Check type and all base classes in MRO order
    with _registry_lock:
        for base_type in target_type.__mro__:
            if base_type in _extractor_registry:
                return _extractor_registry[base_type]

    return None


def unregister_type_info_extractor(target_type: type) -> None:
    """Remove a type from the extractor registry.

    Mainly useful for testing.

    Args:
        target_type: Type to unregister
    """
    with _registry_lock:
        _extractor_registry.pop(target_type, None)


def register_module_info_extractor(
    target_module: Any,
) -> Callable[[ModuleInfoExtractorFunc], ModuleInfoExtractorFunc]:
    """Decorator to register a ModuleInfo extractor for a module.

    The extractor receives the module object and returns a ModuleInfo.
    Use this to provide a curated view of third-party modules for doc().

    Args:
        target_module: The module object to register the extractor for

    Returns:
        Decorator function

    Example:
        import numpy
        from nooa.agentdoc import spec
        from nooa.agentdoc.ext import ModuleInfo, CallableInfo

        @spec.define_doc(numpy)
        def _(mod) -> ModuleInfo:
            return ModuleInfo(
                name="numpy",
                docstring="NumPy — N-dimensional array computing.",
                functions=[CallableInfo(name="array", ...)],
                submodules=[("numpy.linalg", "Linear algebra.")],
            )
    """
    import inspect

    if not inspect.ismodule(target_module):
        raise TypeError(
            f"register_module_info_extractor expects a module, got {type(target_module)}"
        )
    module_name = target_module.__name__

    def decorator(func: ModuleInfoExtractorFunc) -> ModuleInfoExtractorFunc:
        with _registry_lock:
            _module_extractor_registry[module_name] = func
        return func

    return decorator


def get_module_info_extractor(module: Any) -> ModuleInfoExtractorFunc | None:
    """Get the registered ModuleInfo extractor for a module.

    Args:
        module: A module object

    Returns:
        Extractor function if registered, None otherwise
    """
    module_name = getattr(module, "__name__", None)
    if module_name is None:
        return None
    with _registry_lock:
        return _module_extractor_registry.get(module_name)


def unregister_module_info_extractor(target_module: Any) -> None:
    """Remove a module from the extractor registry.

    Mainly useful for testing.

    Args:
        target_module: Module object to unregister
    """
    import inspect

    if not inspect.ismodule(target_module):
        return
    with _registry_lock:
        _module_extractor_registry.pop(target_module.__name__, None)


def register_callable_info_extractor(
    target_callable: Any,
) -> Callable[[CallableInfoExtractorFunc], CallableInfoExtractorFunc]:
    """Decorator to register a CallableInfo extractor for one function object.

    Use this for third-party functions whose own docstring is too long or whose
    signature cannot be introspected (e.g. ``numpy.zeros``). The extractor receives
    the function object and returns a CallableInfo; it must not call the function.

    Args:
        target_callable: The function (or other non-type callable) to document

    Returns:
        Decorator function
    """
    if isinstance(target_callable, type) or not callable(target_callable):
        raise TypeError(
            f"register_callable_info_extractor expects a non-type callable, got {target_callable!r}"
        )

    def decorator(func: CallableInfoExtractorFunc) -> CallableInfoExtractorFunc:
        with _registry_lock:
            _callable_extractor_registry[id(target_callable)] = (target_callable, func)
        return func

    return decorator


def get_callable_info_extractor(obj: Any) -> CallableInfoExtractorFunc | None:
    """Get the CallableInfo extractor registered for this exact callable, if any."""
    with _registry_lock:
        entry = _callable_extractor_registry.get(id(obj))
    if entry is None or entry[0] is not obj:
        return None
    return entry[1]


def unregister_callable_info_extractor(target_callable: Any) -> None:
    """Remove a callable from the extractor registry (mainly for tests)."""
    with _registry_lock:
        entry = _callable_extractor_registry.get(id(target_callable))
        if entry is not None and entry[0] is target_callable:
            del _callable_extractor_registry[id(target_callable)]


# Missing-attribute hints: owner key -> {attribute name: hint}. The owner key is a
# module's __name__ or a class's "module.qualname"; see register_attribute_hint.
_attribute_hint_registry: dict[str, dict[str, str]] = {}

_MODULE_ATTRIBUTE_ERROR = re.compile(r"^module '([^']+)' has no attribute ")

# Missing-attribute hint providers: owner-key prefix -> provider(obj, name, owner_key),
# consulted when no static hint matches (generic "did you mean" for a library).
AttributeHintProvider = Callable[[Any, str, str], str | None]
_attribute_hint_providers: dict[str, AttributeHintProvider] = {}


def _owner_key(owner: Any) -> str:
    if inspect.ismodule(owner):
        return owner.__name__
    return f"{owner.__module__}.{owner.__qualname__}"


def register_attribute_hint(owner: Any, name: str, hint: str) -> None:
    """Register what to use instead of ``owner.<name>``, an attribute that does not exist.

    ``owner`` is a class (its subclasses and instances inherit the hint) or a module.
    Agents often guess plausible names (``UsdGeom.Subset.CreateSubset``); the hint
    is appended to the AttributeError that model-facing error formatting shows.
    """
    if not (inspect.ismodule(owner) or isinstance(owner, type)):
        raise TypeError(f"register_attribute_hint expects a class or module, got {owner!r}")
    with _registry_lock:
        _attribute_hint_registry.setdefault(_owner_key(owner), {})[name] = hint


def get_attribute_hint(error: BaseException) -> str | None:
    """The registered hint for an ``AttributeError`` (``error.obj``/``error.name``), or None."""
    if not isinstance(error, AttributeError):
        return None
    name = getattr(error, "name", None)
    obj = getattr(error, "obj", None)
    if not isinstance(name, str) or obj is None:
        return None
    keys: list[str] = []
    if inspect.ismodule(obj):
        # A package that forwards to an extension module (pxr.UsdGeom -> _usdGeom)
        # raises with the extension as obj; the message names the package.
        match = _MODULE_ATTRIBUTE_ERROR.match(str(error))
        keys = [match.group(1)] if match else []
        keys.append(obj.__name__)
    else:
        cls = obj if isinstance(obj, type) else type(obj)
        keys = [_owner_key(base) for base in cls.__mro__]
    with _registry_lock:
        for key in keys:
            hint = _attribute_hint_registry.get(key, {}).get(name)
            if hint is not None:
                return hint
        providers = list(_attribute_hint_providers.items())
    if not keys or name.startswith("__"):
        return None
    owner_key = keys[0]
    for prefix, provider in providers:
        if owner_key != prefix and not owner_key.startswith(prefix + "."):
            continue
        try:
            hint = provider(obj, name, owner_key)
        except Exception:  # noqa: BLE001 - a hint must never break error formatting
            hint = None
        if hint:
            return hint
    return None


def register_attribute_hint_provider(prefix: str, provider: AttributeHintProvider) -> None:
    """Compute hints for missing attributes of owners under ``prefix`` (e.g. ``"pxr"``).

    ``provider(obj, name, owner_key)`` gets the AttributeError's ``obj``, the missing
    ``name`` and the owner's key (module name or ``"module.qualname"``); it returns a
    hint or None. Static hints (:func:`register_attribute_hint`) win.
    """
    with _registry_lock:
        _attribute_hint_providers[prefix] = provider


def unregister_attribute_hint_provider(prefix: str) -> None:
    """Remove a hint provider (mainly for tests)."""
    with _registry_lock:
        _attribute_hint_providers.pop(prefix, None)


def unregister_attribute_hint(owner: Any, name: str) -> None:
    """Remove a missing-attribute hint (mainly for tests)."""
    with _registry_lock:
        _attribute_hint_registry.get(_owner_key(owner), {}).pop(name, None)


class CuratedMember(NamedTuple):
    """Where :func:`find_curated_member` found a callable's curated entry.

    ``owner`` is the registered class or module view (``TypeInfo``/``ModuleInfo``)
    whose entry ``member`` documents the callable, and ``owner_name`` its display
    name. When an unbound method is shared by several curated classes and the class
    that defines it is not curated, ``owner`` is ``None``, ``owner_name`` names the
    defining class, and ``shared`` holds each curated class's ``(name, entry)``.
    """

    owner_name: str
    owner: TypeInfo | ModuleInfo | None
    member: CallableInfo
    shared: tuple[tuple[str, CallableInfo], ...] = ()


def _is_member_of(owner: Any, name: str, obj: Any) -> bool:
    """True when ``owner.<name>`` is ``obj`` (or the same method, re-bound on access)."""
    try:
        attr = getattr(owner, name, None)
    except Exception:  # noqa: BLE001 - odd metaclass __getattr__
        return False
    if attr is None:
        return False
    if attr is obj:
        return True
    bound_self = getattr(obj, "__self__", None)
    if bound_self is not None and not isinstance(bound_self, type):
        # Method bound to an instance (img.resize): its class resolves the same
        # function or descriptor as ``owner`` unless a subclass overrides it.
        if (
            not isinstance(owner, type)
            or inspect.ismodule(bound_self)
            or not isinstance(bound_self, owner)
        ):
            return False
        # The bound function must be this attribute (stage.GetUsedLayers is not
        # Usd.Stage.Open); a bound builtin has no __func__, so match its name.
        func = getattr(obj, "__func__", None)
        if func is not None:
            if func is not attr and getattr(attr, "__func__", None) is not func:
                return False
        elif getattr(obj, "__name__", name) != name:
            return False
        try:
            return getattr(type(bound_self), name, None) is attr
        except Exception:  # noqa: BLE001 - odd metaclass __getattr__
            return False
    try:
        # Classmethods and bound builtins compare equal but are new objects per access.
        return bool(attr == obj)
    except Exception:  # noqa: BLE001 - equality must never break doc()
        return False


def _entry(info: TypeInfo | ModuleInfo, name: str) -> CallableInfo | None:
    """The curated entry called ``name`` in a class or module view."""
    entries = info.methods if isinstance(info, TypeInfo) else info.functions or []
    for entry in entries:
        if entry.name.rsplit(".", 1)[-1] == name:
            return entry
    return None


def _view(extractor: Any, target: Any) -> TypeInfo | ModuleInfo | None:
    try:
        result = extractor(target)
    except Exception:  # noqa: BLE001 - an adapter must never break doc()
        return None
    return result[0] if isinstance(result, tuple) else result


def _defining_class(cls: type, name: str) -> type | None:
    """The class in ``cls``'s MRO whose own namespace holds ``name``."""
    for base in getattr(cls, "__mro__", ()):
        if name in getattr(base, "__dict__", {}):
            return base
    return None


def _display_name(cls: type, views: list[tuple[type, TypeInfo]]) -> str:
    """Name a class the way its module's curated views name their classes."""
    for other, info in views:
        if other.__module__ == cls.__module__ and info.name.endswith("." + other.__qualname__):
            return info.name[: -len(other.__qualname__)] + cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def _class_views() -> list[tuple[type, TypeInfo]]:
    """Every registered class with its curated view (to name an uncurated class alike)."""
    with _registry_lock:
        entries = list(_extractor_registry.items())
    views = []
    for cls, extractor in entries:
        info = _view(extractor, cls)
        if isinstance(info, TypeInfo):
            views.append((cls, info))
    return views


def _class_members(
    obj: Any, name: str, entries: list[tuple[type, Any]]
) -> list[tuple[type, TypeInfo, CallableInfo]]:
    """Registered classes whose attribute ``name`` is ``obj`` and whose view lists it."""
    found = []
    for cls, extractor in entries:
        if not _is_member_of(cls, name, obj):
            continue
        info = _view(extractor, cls)
        entry = _entry(info, name) if isinstance(info, TypeInfo) else None
        if entry is not None:
            found.append((cls, info, entry))
    return found


def _choose(obj: Any, name: str, found: list[tuple[type, TypeInfo, CallableInfo]]) -> CuratedMember:
    """Pick the entry for ``obj`` among the curated classes that expose it as ``name``."""
    bound_self = getattr(obj, "__self__", None)
    if bound_self is not None and not isinstance(bound_self, type):
        # Bound to an instance: its class is known, so its most derived curated class wins.
        mro = list(type(bound_self).__mro__)
        cls, info, entry = min(found, key=lambda f: mro.index(f[0]) if f[0] in mro else len(mro))
        return CuratedMember(info.name, info, entry)
    if len(found) == 1:
        cls, info, entry = found[0]
        return CuratedMember(info.name, info, entry)
    # Unbound and shared (Boost.Python exposes one function object on every schema
    # class): the access does not say which class was named. The defining class's
    # entry fits all of them; without one, list each curated class's entry.
    definer = _defining_class(found[0][0], name)
    for cls, info, entry in found:
        if cls is definer:
            return CuratedMember(info.name, info, entry)
    found = sorted(found, key=lambda f: f[1].name)
    shared = tuple((info.name, entry) for _cls, info, entry in found)
    owner_name = _display_name(definer, _class_views()) if definer is not None else found[0][1].name
    return CuratedMember(owner_name, None, found[0][2], shared)


def find_curated_member(obj: Any) -> CuratedMember | None:
    """Find the curated class or module view that documents callable ``obj``.

    Matches registered classes (:func:`register_type_info_extractor`) and modules
    (:func:`register_module_info_extractor`) whose attribute, named in their curated
    view, is ``obj``: by ``obj.__name__`` first, then any curated name, so aliases
    (``Vt.IntArray.FromNumpy is Vt.IntArray.FromBuffer``) resolve too. Returns a
    :class:`CuratedMember` or ``None``. Only attributes are read; ``obj`` is never
    called.
    """
    if isinstance(obj, type) or not callable(obj):
        return None
    name = getattr(obj, "__name__", None)
    with _registry_lock:
        entries = list(_extractor_registry.items())
        module_entries = list(_module_extractor_registry.items())
    if isinstance(name, str) and name:
        found = _class_members(obj, name, entries)
        if found:
            return _choose(obj, name, found)
    # Aliases: a curated name other than __name__ that resolves to obj.
    for cls, extractor in entries:
        info = _view(extractor, cls)
        if not isinstance(info, TypeInfo):
            continue
        for entry in info.methods:
            alias = entry.name.rsplit(".", 1)[-1]
            if alias != name and _is_member_of(cls, alias, obj):
                found = _class_members(obj, alias, entries)
                if found:
                    return _choose(obj, alias, found)
    # Module-level functions of a curated module (Gf.Dot, UsdGeom.GetStageUpAxis).
    for module_name, extractor in module_entries:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        info = _view(extractor, module)
        if not isinstance(info, ModuleInfo):
            continue
        for entry in info.functions or []:
            short = entry.name.rsplit(".", 1)[-1]
            if _is_member_of(module, short, obj):
                return CuratedMember(info.name, info, entry)
    return None


def clear_registry() -> None:
    """Clear all registered extractors.

    Mainly useful for testing.
    """
    with _registry_lock:
        _extractor_registry.clear()
        _module_extractor_registry.clear()
        _attribute_hint_registry.clear()
        _attribute_hint_providers.clear()
        _callable_extractor_registry.clear()
