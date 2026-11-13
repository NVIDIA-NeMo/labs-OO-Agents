# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataclasses whose PEP 563 string annotations name markers through aliases and calls."""

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Annotated
from typing import Annotated as A

from pydantic import Field

import nooa.agentdoc as agentdoc
from nooa.agentdoc import hidden, spec
from nooa.agentdoc import hidden as H

FLAG = True
HIDE = spec(hidden=True)  # a marker object, not a call in the annotation
SHOW = spec(hidden=False)
BROKEN = spec(hidden=True)
BROKEN.kwargs = {"hidden": "yes"}  # unreadable: not a bool
Secret = Annotated[str, hidden]  # an alias of a whole outermost Annotated
SpecSecret = Annotated[str, spec(hidden=True)]
Documented = Annotated[str, "doc"]

if typing.TYPE_CHECKING:
    from decimal import Decimal  # a name that exists only for type checkers


def make_secret() -> typing.Any:
    """A marker built by a call: only evaluation can tell it hides (CodeRabbit)."""
    return spec(hidden=True)


@dataclasses.dataclass
class Literal:
    secret: Annotated[str, hidden] = "S-literal"
    ok: int = 1


@dataclasses.dataclass
class AliasAnnotated:
    secret: A[str, hidden] = "S-alias-annotated"
    ok: int = 1


@dataclasses.dataclass
class AliasHidden:
    secret: Annotated[str, H] = "S-alias-hidden"
    ok: int = 1


@dataclasses.dataclass
class AliasBoth:
    secret: A[str, H] = "S-alias-both"
    call: A[str, spec(hidden=True)] = "S-alias-call"
    ok: int = 1


@dataclasses.dataclass
class Dotted:
    secret: typing.Annotated[str, agentdoc.hidden] = "S-dotted"
    ok: int = 1


@dataclasses.dataclass
class SpecLiteral:
    secret: Annotated[str, spec(hidden=True)] = "S-spec"
    shown: Annotated[str, spec(hidden=False, description="d")] = "shown"
    ok: int = 1


@dataclasses.dataclass
class SpecUnreadable:
    secret: Annotated[str, spec(hidden=FLAG)] = "S-spec-flag"
    ok: int = 1


@dataclasses.dataclass
class SpecInstance:
    secret: Annotated[str, HIDE] = "S-spec-instance"
    ok: int = 1


@dataclasses.dataclass
class SpecInstanceUnreadable:
    secret: Annotated[str, BROKEN] = "S-spec-broken"
    ok: int = 1


@dataclasses.dataclass
class WholeAlias:
    secret: Secret = "S-whole-alias"
    call: SpecSecret = "S-whole-spec"
    ok: int = 1


@dataclasses.dataclass
class NestedWholeAlias:
    secret: Annotated[Secret, "doc"] = "S-nested-alias"  # Annotated flattens
    ok: int = 1


@dataclasses.dataclass
class CalledMarker:
    secret: Annotated[str, make_secret()] = "S-called"
    ok: int = 1


@dataclasses.dataclass
class PartialMarker:
    secret: Annotated[str, functools.partial(spec, hidden=True)()] = "S-partial"
    ok: int = 1


@dataclasses.dataclass
class UnknownMarker:
    secret: Annotated[str, not_defined_anywhere] = "S-unknown"  # noqa: F821
    ok: int = 1


@dataclasses.dataclass
class TypeCheckingOnly:
    amount: Decimal | None = None  # can't be evaluated at runtime
    ok: int = 1


@dataclasses.dataclass
class Unresolvable:
    mapping: dict[str, Unknown] = dataclasses.field(default_factory=dict)  # noqa: F821
    ok: int = 1


@dataclasses.dataclass
class Visible:
    xs: list[int] = dataclasses.field(default_factory=lambda: [1, 2])
    forward: Later | None = None
    constrained: Annotated[int, Field(gt=0), "doc"] = 3
    nested: list[Annotated[int, "doc"]] = dataclasses.field(default_factory=list)
    shown: Annotated[str, SHOW] = "shown"
    documented: Documented = "doc"
    inner: list[Secret] = dataclasses.field(default_factory=list)  # not outermost


@dataclasses.dataclass
class Later:
    """Defined after its first use: a forward reference that resolves."""

    value: int = 0
