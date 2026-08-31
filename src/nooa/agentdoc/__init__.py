# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn live Python objects into prompt-ready API documentation.

Show an agent what an object can do—not just what it contains::

    from typing import Annotated

    from nooa.agentdoc import doc, hidden, pformat, spec

    class Assistant:
        api_key: Annotated[str, hidden] = ""
        name: Annotated[str, spec(description="Display name")] = "Ada"

        def greet(self, person: str) -> str:
            \"""Greet a person by name.\"""

    assistant = Assistant()
    assistant.locale = "en-US"

    doc(Assistant)      # declared API and defaults
    doc(assistant)      # the same API, enriched with live state
    pformat(assistant)  # compact value view

Use ``spec()`` and ``hidden`` to shape the contract, ``doc()`` to reveal it,
and ``pformat()`` when values alone are enough.
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
# Keep doc(agentdoc) as a capability map; individual callables provide details on demand.
__agentdoc_concise_members__ = True


def truncating_pformat(
    obj: Annotated[Any, "Object to format"],
    *,
    max_chars: Annotated[
        int | None, "Hard char cap on non-string rendering via TruncatingStringIO"
    ] = None,
    **kwargs: Any,
) -> str:
    """Format a value, with a hard total-size cap for non-string objects.

    Strings pass through unchanged. Other options are forwarded to ``pformat()``.
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
    """Render a compact, value-focused representation.

    Hidden fields are omitted and custom ``__repr__`` methods are honored. Use
    ``doc()`` for an object's API, or ``truncating_pformat()`` for a hard size cap.
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
    """Print the compact, value-focused representation produced by ``pformat()``."""
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
