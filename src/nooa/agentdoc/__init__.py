# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc — runtime Python documentation for Python objects.

Two-step mental model:

  1. spec() — specify how the type renders: descriptions, visibility, hints.
  2. doc()  — get the documentation: the API contract, ready for a prompt.

Quick start:

    from nooa.agentdoc import spec, hidden, doc, pformat, truncating_pformat

    class MyAgent:
        api_key: Annotated[str, hidden] = ""          # excluded from documentation
        label: Annotated[str, spec(description="Display name")] = "agent"

    agent = MyAgent()
    doc(MyAgent)   # → API contract (type view)
    doc(agent)     # → API contract with current values substituted
    pformat(agent) # → compact repr: MyAgent(label='agent')
"""

import io
import sys
from typing import Annotated, Any

from nooa._version import __version__ as __version__
from nooa.agentdoc._docs import spec
from nooa.agentdoc._pformat import _pformat
from nooa.agentdoc._truncating_stream import (
    FileBackedTruncatingStringIO,
    TruncatingStringIO,
)
from nooa.agentdoc._visibility import hidden
from nooa.agentdoc.core import doc
from nooa.agentdoc.doc_config import DocConfig

__submodules__ = ["ext", "introspect", "visibility", "adapters"]


def truncating_pformat(
    obj: Annotated[Any, "Object to format"],
    *,
    max_chars: Annotated[
        int | None, "Hard char cap on non-string rendering via TruncatingStringIO"
    ] = None,
    **kwargs: Any,
) -> str:
    """Format *obj* as a string. Strings pass through verbatim; non-strings go
    through :func:`pformat` with the supplied structural kwargs.

    Per-value bounds (``max_string``, ``max_length``, ``max_depth``) come from
    ``kwargs``.  ``max_chars`` is an independent hard cap on the total rendered
    output for non-string objects via :class:`TruncatingStringIO`.

    Raises:
        ValueError: if ``max_chars`` is set and <= 0.
    """
    if max_chars is not None and max_chars <= 0:
        raise ValueError(f"truncating_pformat max_chars must be > 0 or None, got {max_chars}")

    if isinstance(obj, str):
        return obj

    if max_chars is None:
        from io import StringIO

        stream = StringIO()
    else:
        stream = TruncatingStringIO(limit=max_chars)
    _pformat(obj, stream, **kwargs)
    return stream.getvalue()


def pformat(
    obj: Annotated[Any, "Object to format"],
    *,
    console: Any = None,
    indent_guides: bool = True,
    max_length: Annotated[int | None, "Max elements per container; None = unlimited"] = None,
    max_string: Annotated[int | None, "Max string chars; None = unlimited"] = None,
    max_depth: Annotated[int | None, "Max nesting depth; None = unlimited"] = None,
    expand_all: Annotated[bool, "Always expand containers to multiple lines"] = False,
    concise: Annotated[bool, "Show first-line docstrings only"] = False,
    instance_mode: Annotated[
        str, "Instance format: 'repr' for repr-style, 'type' for type structure"
    ] = "repr",
    unquote_strings: Annotated[bool, "Untruncated strings rendered verbatim"] = False,
) -> str:
    """Format an object as a string with smart truncation.

    Drop-in replacement for ``rich.pretty.pformat()``.
    For user-defined instances, ``hidden`` fields are automatically excluded.
    Respects ``@spec(expand=False)`` on field types — shown as ``ClassName()`` rather than expanded.

    ``console`` and ``indent_guides`` are accepted for Rich API compatibility but have no effect.

    To cap total output size (preventing OOM on huge objects), use :func:`truncating_pformat`
    which applies a hard ``max_chars`` limit via :class:`TruncatingStringIO`.
    """
    # console and indent_guides are intentionally ignored for Rich compatibility
    del console, indent_guides

    stream: io.StringIO = io.StringIO()

    _pformat(
        obj,
        stream,
        max_length=max_length,
        max_string=max_string,
        max_depth=max_depth,
        expand_all=expand_all,
        concise=concise,
        instance_mode=instance_mode,
    )

    result = stream.getvalue()

    # unquote_strings: for top-level strings, _pformat renders repr ('hello').
    # Strip surrounding quotes so the string renders verbatim in context blocks.
    # Truncated strings (str(len=N,...) marker) pass through unchanged.
    if unquote_strings and isinstance(obj, str):
        if max_string is None or len(obj) <= max_string:
            # Untruncated: strip outer quotes from repr output.
            for q in ("'''", '"""'):
                if result.startswith(q) and result.endswith(q):
                    return result[len(q) : -len(q)]
            if len(result) >= 2 and result[0] == result[-1] and result[0] in ("'", '"'):
                return result[1:-1]

    return result


def pprint(
    obj: Annotated[Any, "Object to print"],
    *,
    console: Any = None,
    indent_guides: bool = True,
    max_length: Annotated[int | None, "Max elements per container; None = unlimited"] = None,
    max_string: Annotated[int | None, "Max string chars; None = unlimited"] = None,
    max_depth: Annotated[int | None, "Max nesting depth; None = unlimited"] = None,
    expand_all: Annotated[bool, "Always expand containers to multiple lines"] = False,
    concise: Annotated[bool, "Show first-line docstrings only"] = False,
    instance_mode: Annotated[
        str, "Instance format: 'repr' for repr-style, 'type' for type structure"
    ] = "repr",
) -> None:
    """Pretty-print an object with smart truncation. Prints to stdout.

    Drop-in replacement for ``rich.pretty.pprint()``.
    Writes directly to ``sys.stdout`` via stream-based formatting so that
    stdout capture (via ``ContextVarStream``) bounds output during formatting.

    ``console`` and ``indent_guides`` are accepted for Rich API compatibility but have no effect.
    """
    # console and indent_guides are intentionally ignored for Rich compatibility
    del console, indent_guides

    _pformat(
        obj,
        sys.stdout,
        max_length=max_length,
        max_string=max_string,
        max_depth=max_depth,
        expand_all=expand_all,
        concise=concise,
        instance_mode=instance_mode,
    )
    sys.stdout.write("\n")


__all__ = [
    "spec",
    "hidden",
    "doc",
    "DocConfig",
    "pformat",
    "pprint",
    "truncating_pformat",
    "FileBackedTruncatingStringIO",
    "TruncatingStringIO",
]
