# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for curated adapters: signatures read from the installed library.

``lib_callable(func, "one-line doc")`` pairs a curated one-liner with the real
signature of ``func`` (``inspect.signature``), so a curated view cannot drift from
the installed version. String annotations (``from __future__ import annotations``)
are shown unquoted. The function is never called.
"""

from __future__ import annotations

import enum
import inspect
from typing import Any

from nooa.agentdoc.ext import CallableInfo


def _annotation(annotation: Any) -> str:
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


def _default(value: Any) -> str:
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    return repr(value)


def signature_parts(func: Any, *, annotate: bool = True) -> tuple[str, str]:
    """``(params, return_type)`` of ``func`` as display strings."""
    sig = inspect.signature(func)
    params = []
    previous_kind = None
    for param in sig.parameters.values():
        kind = param.kind
        if kind is inspect.Parameter.KEYWORD_ONLY and previous_kind not in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            params.append("*")
        if previous_kind is inspect.Parameter.POSITIONAL_ONLY and kind is not (
            inspect.Parameter.POSITIONAL_ONLY
        ):
            params.append("/")
        prefix = {inspect.Parameter.VAR_POSITIONAL: "*", inspect.Parameter.VAR_KEYWORD: "**"}
        text = prefix.get(kind, "") + param.name
        has_annotation = annotate and param.annotation is not inspect.Parameter.empty
        if has_annotation:
            text += f": {_annotation(param.annotation)}"
        if param.default is not inspect.Parameter.empty:
            text += (" = " if has_annotation else "=") + _default(param.default)
        params.append(text)
        previous_kind = kind
    if previous_kind is inspect.Parameter.POSITIONAL_ONLY:
        params.append("/")
    returns = ""
    if annotate and sig.return_annotation is not inspect.Signature.empty:
        returns = _annotation(sig.return_annotation)
    return f"({', '.join(params)})", returns


def lib_callable(
    func: Any,
    docstring: str,
    *,
    name: str | None = None,
    annotate: bool = True,
    return_type: str | None = None,
    is_classmethod: bool = False,
) -> CallableInfo:
    """A CallableInfo with ``func``'s real signature and a curated docstring."""
    params, returns = signature_parts(func, annotate=annotate)
    return CallableInfo(
        name=name or func.__name__,
        signature=params,
        return_type=returns if return_type is None else return_type,
        docstring=docstring,
        is_classmethod=is_classmethod,
    )
