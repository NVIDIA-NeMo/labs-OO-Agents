# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc.ext — Extension layer for custom extractors and library authors.

Use `spec.define_doc()` to register custom TypeInfo extractors for third-party
types. `register_type_info_extractor` is the lower-level alternative for
advanced use cases.

```python
from nooa.agentdoc import spec
from nooa.agentdoc.ext import TypeInfo, FieldInfo

@spec.define_doc(MyThirdPartyClass)
def _(cls_or_instance) -> TypeInfo | tuple[TypeInfo, dict]:
    type_info = TypeInfo(
        name="MyThirdPartyClass",
        base=None,
        fields=[FieldInfo(name="url", type="str")],
        methods=[],
        docstring="My third-party class.",
    )
    if isinstance(cls_or_instance, type):
        return type_info
    # For instances, also return current field values
    return type_info, {"url": cls_or_instance.url}
```
"""

from typing import Any

from nooa.agentdoc._info import (
    REQUIRED,
    CallableInfo,
    FieldInfo,
    ModuleInfo,
    TypeInfo,
)
from nooa.agentdoc._structured import (
    extract_callable_info,
    extract_module_info,
    extract_type_info,
    format_type,
)
from nooa.agentdoc.doc_config import DocConfig
from nooa.agentdoc.protocols import (
    SupportsCallableInfo,
    SupportsInstanceValues,
    SupportsTypeInfo,
)
from nooa.agentdoc.registry import (
    clear_registry,
    get_type_info_extractor,
    register_type_info_extractor,
    unregister_type_info_extractor,
)


def has_type_info(obj: Any) -> bool:
    """Return True if obj implements the __type_info__ protocol."""
    return hasattr(obj, "__type_info__") and callable(getattr(obj, "__type_info__", None))


__all__ = [
    # Info types
    "TypeInfo",
    "CallableInfo",
    "ModuleInfo",
    "FieldInfo",
    "REQUIRED",
    # Extraction functions
    "extract_type_info",
    "extract_callable_info",
    "extract_module_info",
    "format_type",
    # Configuration
    "DocConfig",
    # Protocols
    "SupportsTypeInfo",
    "SupportsCallableInfo",
    "SupportsInstanceValues",
    # Registry
    "register_type_info_extractor",
    "get_type_info_extractor",
    "unregister_type_info_extractor",
    "clear_registry",
    # Helpers
    "has_type_info",
]
