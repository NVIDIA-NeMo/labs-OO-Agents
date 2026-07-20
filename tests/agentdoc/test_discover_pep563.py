# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #260.

Under ``from __future__ import annotations`` (PEP 563) every annotation is a
string. ``discover_referenced_types`` must still resolve custom types that appear
only in a method signature (parameter or return type).

This module deliberately enables PEP 563 so that the annotations below are genuine
strings at runtime. All fixture types are defined at **module level** — under PEP 563
the annotation survives only as a string, so types defined in a local (function) scope
are not resolvable by either ``typing.get_type_hints`` or the eval fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from nooa.agentdoc._discover import discover_referenced_types


@dataclass
class Res:
    x: int


@dataclass
class Req:
    q: str


@dataclass
class Wrapped:
    y: int


@dataclass
class OnlyGeneric:
    """Appears *only* inside list[...] — independent coverage of generic resolution."""

    a: int


@dataclass
class OnlyUnion:
    """Appears *only* inside X | None — independent coverage of union resolution."""

    b: int


class Solo:
    """Plain class (no Agent/LLM dependency) with PEP 563 string annotations."""

    async def go(self) -> Res:
        """do"""
        ...

    def take(self, req: Req) -> None:
        """take a request"""
        ...

    def many(self) -> list[OnlyGeneric]:
        """return many"""
        ...

    def maybe(self) -> OnlyUnion | None:
        """maybe a result"""
        ...

    def annotated(self, w: Annotated[Wrapped, "a wrapped value"]) -> None:
        """annotated param"""
        ...


def standalone(req: Req) -> Res:
    """A module-level function with string annotations."""
    ...


@dataclass
class EagerReq:
    q: str


@dataclass
class EagerRes:
    x: int


def _eager_function(req):
    """Callable whose annotations are live type objects, not strings.

    Even though this module enables PEP 563, assigning ``__annotations__`` directly
    emulates a module *without* ``from __future__ import annotations`` (eager,
    live-object annotations) so we can guard that path against regression.
    """
    ...


_eager_function.__annotations__ = {"req": EagerReq, "return": EagerRes}


def test_solo_return_type_discovered():
    """The issue's exact repro: a return type discovered under PEP 563."""
    names = {t.__name__ for t in discover_referenced_types(Solo)}
    assert "Res" in names, f"Expected Res in {names}"


def test_param_type_discovered():
    """A param-only custom type is discovered under PEP 563."""
    names = {t.__name__ for t in discover_referenced_types(Solo)}
    assert "Req" in names, f"Expected Req in {names}"


def test_generic_and_union_return_discovered():
    """list[OnlyGeneric] and OnlyUnion | None resolve their inner custom types.

    These types appear *only* inside the generic / union, so this independently
    covers generic and union resolution (not the plain-return path).
    """
    names = {t.__name__ for t in discover_referenced_types(Solo)}
    assert "OnlyGeneric" in names, f"Expected OnlyGeneric (from list[...]) in {names}"
    assert "OnlyUnion" in names, f"Expected OnlyUnion (from X | None) in {names}"


def test_annotated_param_discovered():
    """Annotated[Wrapped, ...] param is unwrapped and discovered under PEP 563."""
    names = {t.__name__ for t in discover_referenced_types(Solo)}
    assert "Wrapped" in names, f"Expected Wrapped (from Annotated) in {names}"


def test_standalone_function_discovered():
    """The standalone-callable branch resolves string annotations too."""
    names = {t.__name__ for t in discover_referenced_types(standalone)}
    assert {"Req", "Res"} <= names, f"Expected Req and Res in {names}"


def test_eager_annotations_still_work():
    """Eager (live-object) annotations are discovered, guarding the non-PEP-563 path."""
    names = {t.__name__ for t in discover_referenced_types(_eager_function)}
    assert {"EagerReq", "EagerRes"} <= names, f"Expected EagerReq and EagerRes in {names}"
