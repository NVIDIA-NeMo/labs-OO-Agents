# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed orchestration for complete agentdoc documents.

Formatting lives in :mod:`_pformat`; this module owns object normalization and the
identity-based reference graph shared by ``doc()`` and direct ``pformat()``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from nooa.agentdoc._discover import (
    _extract_types_from_hint,
    _IdentityTypeSet,
    _is_custom_type,
    discover_referenced_types,
)
from nooa.agentdoc._info import CallableInfo, ModuleInfo, TypeInfo
from nooa.agentdoc._introspection import (
    _extract_instance_values,
    _is_structured_instance,
)
from nooa.agentdoc._metadata import is_expand_false
from nooa.agentdoc._structured import extract_callable_info, extract_module_info, extract_type_info
from nooa.agentdoc._visibility import is_hidden_field
from nooa.agentdoc.protocols import SupportsInstanceValues
from nooa.agentdoc.registry import get_module_info_extractor, get_type_info_extractor

StructuredInfo = TypeInfo | CallableInfo | ModuleInfo


@dataclass
class DocumentSubject:
    """A primary object normalized once for formatting and graph discovery."""

    obj: Any
    info: StructuredInfo | None = None
    values: dict[str, Any] | None = None
    represented_type: type | None = None


def _instance_dict(obj: Any) -> dict[str, Any] | None:
    try:
        value = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_instance(obj: Any) -> tuple[TypeInfo, dict[str, Any]] | None:
    if not _is_structured_instance(obj, respect_custom_repr=False):
        return None

    obj_type = type(obj)
    instance_meta = (_instance_dict(obj) or {}).get("_agentdoc_fields_docs") or {}
    has_overrides = any(meta.get("hidden") is False for meta in instance_meta.values())
    extractor = get_type_info_extractor(obj_type)
    if extractor:
        result = extractor(obj)
        if isinstance(result, tuple):
            info, values = result
        else:
            info, values = result, _extract_instance_values(obj, result)
    elif isinstance(obj, SupportsInstanceValues):
        info = extract_type_info(obj_type)
        if has_overrides:
            raw = extract_type_info(obj_type, _skip_protocol=True, _include_hidden=True)
            info = TypeInfo(
                info.name,
                info.base,
                [f for f in raw.fields if not is_hidden_field(obj, f.name)],
                info.methods,
                info.docstring,
            )
        values = obj.__instance_values__()
    else:
        info = extract_type_info(obj_type)
        if has_overrides:
            raw = extract_type_info(obj_type, _skip_protocol=True, _include_hidden=True)
            info = TypeInfo(
                info.name,
                info.base,
                [f for f in raw.fields if not is_hidden_field(obj, f.name)],
                info.methods,
                info.docstring,
            )
        values = _extract_instance_values(obj, info)

    info = TypeInfo(
        info.name,
        info.base,
        [f for f in info.fields if not is_hidden_field(obj, f.name)],
        info.methods,
        info.docstring,
    )
    return info, values


def normalize_subject(obj: Any) -> DocumentSubject:
    """Extract a primary exactly once, preserving non-document values."""
    if isinstance(obj, (TypeInfo, CallableInfo, ModuleInfo)):
        return DocumentSubject(obj, obj)
    if isinstance(obj, type):
        return DocumentSubject(obj, extract_type_info(obj), represented_type=obj)
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return DocumentSubject(obj, extract_callable_info(obj))
    if inspect.ismodule(obj):
        extractor = get_module_info_extractor(obj)
        info = extractor(obj) if extractor else extract_module_info(obj)
        return DocumentSubject(obj, info)
    extracted = _extract_instance(obj)
    if extracted is None:
        return DocumentSubject(obj)
    info, values = extracted
    return DocumentSubject(obj, info, values, type(obj))


def reference_seeds(subject: DocumentSubject) -> list[type]:
    """Return visible declared and runtime reference types for a subject."""
    obj, info = subject.obj, subject.info
    if isinstance(info, TypeInfo) and subject.represented_type is not None:
        visible = {field.name for field in info.fields}
        seeds = list(
            discover_referenced_types(
                subject.represented_type,
                field_names=visible if subject.values is not None else None,
                type_info=info,
            )
        )
        if subject.values:
            runtime = _IdentityTypeSet()
            for name, value in subject.values.items():
                if name.startswith("_") or is_hidden_field(obj, name):
                    continue
                if callable(value) and not isinstance(value, type):
                    continue
                _extract_types_from_hint(value if isinstance(value, type) else type(value), runtime)
            seeds.extend(item for item in runtime if _is_custom_type(item))
        return seeds
    if isinstance(info, CallableInfo) and (inspect.isfunction(obj) or inspect.ismethod(obj)):
        return list(discover_referenced_types(obj))
    return []


def collect_references(
    seed_types: list[type], *, exclude: list[type], max_depth: int
) -> list[DocumentSubject]:
    """Traverse a bounded graph, deduplicating and excluding strictly by identity."""
    seen = {id(item) for item in exclude}
    found: list[DocumentSubject] = []
    frontier = [item for item in seed_types if id(item) not in seen and not is_expand_false(item)]
    for _ in range(max_depth):
        if not frontier:
            break
        following: list[type] = []
        for ref_type in frontier:
            if id(ref_type) in seen:
                continue
            seen.add(id(ref_type))
            subject = normalize_subject(ref_type)
            found.append(subject)
            following.extend(reference_seeds(subject))
        frontier = [
            item for item in following if id(item) not in seen and not is_expand_false(item)
        ]
    return sorted(
        found,
        key=lambda subject: (
            subject.obj.__name__,
            subject.obj.__module__,
            subject.obj.__qualname__,
        ),
    )


def normalize_subjects(objs: list[Any]) -> list[DocumentSubject]:
    """Normalize primaries once, deduplicating by identity in caller order."""
    unique: list[Any] = []
    seen: set[int] = set()
    for obj in objs:
        if id(obj) not in seen:
            seen.add(id(obj))
            unique.append(obj)
    return [normalize_subject(obj) for obj in unique]


def collect_subject_references(
    subjects: list[DocumentSubject], *, max_depth: int
) -> list[DocumentSubject]:
    """Collect references for subjects, excluding all represented primary types."""
    if max_depth <= 0:
        return []
    seeds = [seed for subject in subjects for seed in reference_seeds(subject)]
    primaries = [
        subject.represented_type for subject in subjects if subject.represented_type is not None
    ]
    return collect_references(seeds, exclude=primaries, max_depth=max_depth)