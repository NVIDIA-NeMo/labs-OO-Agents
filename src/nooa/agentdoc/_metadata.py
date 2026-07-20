# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata storage for spec() annotations.

spec() metadata is stored in two places:
- On callables/classes: attached as ``_agentdoc_docs`` attribute
- Field-level overrides on types/instances: attached as ``_agentdoc_fields_docs``

Both are plain dicts keyed by keyword name (hidden, description, expand, ...).
"""

from __future__ import annotations

import contextlib
from typing import Any

_DOCS_ATTR = "_agentdoc_docs"
_FIELDS_ATTR = "_agentdoc_fields_docs"


def get_docs_metadata(obj: Any) -> dict[str, Any]:
    """Return the spec() metadata dict for obj (never None)."""
    return getattr(obj, _DOCS_ATTR, None) or {}


def set_docs_metadata(obj: Any, **kwargs: Any) -> None:
    """Merge kwargs into the spec() metadata dict on obj."""
    with contextlib.suppress(AttributeError, TypeError):
        existing = dict(getattr(obj, _DOCS_ATTR, None) or {})
        existing.update(kwargs)
        setattr(obj, _DOCS_ATTR, existing)


def get_field_metadata(obj: Any, field: str) -> dict[str, Any]:
    """Return per-field spec() metadata for obj's own declarations only (never None)."""
    try:
        fields_meta = vars(obj).get(_FIELDS_ATTR) or {}
    except TypeError:
        fields_meta = {}
    return fields_meta.get(field) or {}


def set_field_metadata(obj: Any, field: str, **kwargs: Any) -> None:
    """Merge kwargs into the per-field spec() metadata on obj."""
    with contextlib.suppress(AttributeError, TypeError):
        try:
            fields_meta = vars(obj).get(_FIELDS_ATTR)
        except TypeError:
            fields_meta = None
        if fields_meta is None:
            fields_meta = {}
            setattr(obj, _FIELDS_ATTR, fields_meta)
        existing = fields_meta.get(field) or {}
        existing.update(kwargs)
        fields_meta[field] = existing


def is_expand_false(type_hint: Any) -> bool:
    """True if *type_hint* is a class annotated with @docs(expand=False)."""
    if not isinstance(type_hint, type):
        return False
    meta = get_docs_metadata(type_hint)
    return meta.get("expand") is False
