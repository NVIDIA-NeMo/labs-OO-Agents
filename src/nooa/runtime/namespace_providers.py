# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The names every cell sees, contributed by the packages that own them.

A *namespace provider* is a name, a loader returning ``{name: object}`` and an
optional prompt note (lines a code strategy shows next to "always available").
The in-process cell namespace (``ActorRuntime.execute_code``) and the sandbox
worker (``runtime/sandbox/worker.py``) both read :func:`cell_namespace_names`,
so the two cannot drift; CodeAct renders :func:`namespace_prompt_notes`.

The loader runs each time a namespace is built, so a provider can import its
objects lazily. This module registers the core provider (``typing`` names,
``doc``/``methods``/``variables``/``help``, ``pprint``); media registers in
:mod:`nooa.media`, the strategy decorator and classes in :mod:`nooa.decorators`.
Each cell also gets ``self`` and ``asyncio`` from the runtime itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "NamespaceProvider",
    "cell_namespace_names",
    "namespace_prompt_notes",
    "register_namespace_provider",
]


@dataclass(frozen=True)
class NamespaceProvider:
    """Names one package adds to every cell, and the note a prompt shows for them."""

    name: str
    names: Callable[[], Mapping[str, Any]]
    prompt_note: tuple[str, ...] = ()


_PROVIDERS: dict[str, NamespaceProvider] = {}


def register_namespace_provider(
    name: str,
    names: Callable[[], Mapping[str, Any]],
    *,
    prompt_note: tuple[str, ...] = (),
) -> NamespaceProvider:
    """Register (or replace) provider ``name``; ``names()`` returns its cell names."""
    provider = NamespaceProvider(name, names, tuple(prompt_note))
    _PROVIDERS[name] = provider
    return provider


def cell_namespace_names() -> dict[str, Any]:
    """Every registered provider's names, in registration order (a later one wins)."""
    merged: dict[str, Any] = {}
    for provider in _PROVIDERS.values():
        merged.update(provider.names())
    return merged


def namespace_prompt_notes() -> tuple[str, ...]:
    """The prompt-note lines of the registered providers, in registration order."""
    return tuple(line for provider in _PROVIDERS.values() for line in provider.prompt_note)


def _core_names() -> dict[str, Any]:
    import typing as _typing

    from nooa.agentdoc import doc
    from nooa.agentdoc.introspect import methods, variables
    from nooa.runtime.pprint import pprint

    return {
        "typing": _typing,
        # Common typing constructs: models use these constantly
        "Annotated": _typing.Annotated,
        "Any": _typing.Any,
        "Literal": _typing.Literal,
        "Optional": _typing.Optional,
        "Union": _typing.Union,
        # agentdoc introspection (doc respects hidden fields)
        "doc": doc,
        "methods": methods,
        "variables": variables,
        "help": doc,  # Shadow built-in help() to prevent blocking on stdin
        # Pretty printing with Rich-compatible API
        "pprint": pprint,
    }


register_namespace_provider("core", _core_names)
