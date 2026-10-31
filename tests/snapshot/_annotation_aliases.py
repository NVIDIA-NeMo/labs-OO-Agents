# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataclasses whose PEP 563 string annotations name markers through aliases."""

from __future__ import annotations

import dataclasses
import typing
from typing import Annotated
from typing import Annotated as A

from pydantic import Field

import nooa.agentdoc as agentdoc
from nooa.agentdoc import hidden, spec
from nooa.agentdoc import hidden as H

FLAG = True


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
class UnknownMarker:
    secret: Annotated[str, not_defined_anywhere] = "S-unknown"  # noqa: F821
    ok: int = 1


@dataclasses.dataclass
class Visible:
    xs: list[int] = dataclasses.field(default_factory=lambda: [1, 2])
    mapping: dict[str, Unknown] = dataclasses.field(default_factory=dict)  # noqa: F821
    forward: Later | None = None  # noqa: F821
    constrained: Annotated[int, Field(gt=0), "doc"] = 3
    nested: list[Annotated[int, "doc"]] = dataclasses.field(default_factory=list)
