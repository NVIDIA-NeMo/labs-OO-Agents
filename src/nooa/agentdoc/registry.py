# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Type-based extractor registry for agentdoc.

Allows libraries to register custom TypeInfo extractors for types you don't control,
providing TypeInfo (and optionally instance values) for third-party types.

This is the recommended approach for third-party library integration.
"""

import threading
from collections.abc import Callable
from typing import Any

from nooa.agentdoc._info import ModuleInfo, TypeInfo

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


def clear_registry() -> None:
    """Clear all registered extractors.

    Mainly useful for testing.
    """
    with _registry_lock:
        _extractor_registry.clear()
        _module_extractor_registry.clear()
