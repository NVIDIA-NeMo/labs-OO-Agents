# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata storage for spec() annotations.

spec() metadata is stored in two places:
- On callables/classes: attached as ``_agentdoc_docs`` attribute
- Field-level overrides on types/instances: attached as ``_agentdoc_fields_docs``

Both are plain dicts keyed by keyword name (hidden, description, expand, ...).
The attribute names and the storage helpers live in ``nooa.agentdoc.markers``
(the contract shared with ``nooa.snapshot``); this module re-exports them.
"""

from __future__ import annotations

from typing import Any

from nooa.agentdoc.markers import CLASS_METADATA_ATTR as _DOCS_ATTR
from nooa.agentdoc.markers import FIELD_METADATA_ATTR as _FIELDS_ATTR
from nooa.agentdoc.markers import (
    _object_namespace,
    get_docs_metadata,
    get_field_metadata,
    set_docs_metadata,
    set_field_metadata,
)

__all__ = [
    "_DOCS_ATTR",
    "_FIELDS_ATTR",
    "_object_namespace",
    "get_docs_metadata",
    "get_field_metadata",
    "is_expand_false",
    "set_docs_metadata",
    "set_field_metadata",
]


def is_expand_false(type_hint: Any) -> bool:
    """True if *type_hint* is a class annotated with @docs(expand=False)."""
    if not isinstance(type_hint, type):
        return False
    meta = get_docs_metadata(type_hint)
    return meta.get("expand") is False
