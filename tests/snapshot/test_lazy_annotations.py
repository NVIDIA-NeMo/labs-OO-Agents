# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Class annotations read the PEP 649 way: a lazily annotated hidden field stays hidden.

On Python 3.14 a class body without ``from __future__ import annotations``
keeps no ``__annotations__`` dict, only an ``__annotate__`` function; the
field readers must evaluate it (``Format.FORWARDREF``), and fail closed
when they can't. No ``from __future__ import annotations`` in this module:
its classes must be lazily annotated on 3.14.
"""

import dataclasses
import sys
import typing
from typing import Annotated

import pytest

from nooa.agentdoc import hidden, spec
from nooa.snapshot import Object, capture, render_python

lazy_only = pytest.mark.skipif(sys.version_info < (3, 14), reason="PEP 649 lazy annotations")


def _names(value: object) -> list[str]:
    node = capture(value)
    assert type(node) is Object
    return [name for name, _ in node.fields]


class _Lazy:
    secret: Annotated[str, hidden]
    token: Annotated[str, spec(hidden=True)]
    unresolved: Annotated["Undefined", hidden]  # noqa: F821
    label: str = "ok"

    def __init__(self) -> None:
        self.secret = "hidden-1"
        self.token = "hidden-2"
        self.unresolved = "hidden-3"
        self.public = 1


@lazy_only
def test_a_lazily_annotated_class_keeps_no_annotations_dict():
    assert "__annotations__" not in _Lazy.__dict__
    assert "__annotate_func__" in _Lazy.__dict__ or "__annotate__" in _Lazy.__dict__


def test_a_lazily_annotated_hidden_field_stays_hidden():
    assert _names(_Lazy()) == ["label", "public"]
    assert "hidden-" not in render_python(capture(_Lazy()))


@lazy_only
def test_a_hidden_field_whose_type_is_never_defined_stays_hidden():
    class Later:  # 3.14 only: an eager class body would raise NameError
        later: Annotated[NotYetDefined, hidden]  # noqa: F821
        bare: NotYetDefined  # noqa: F821

        def __init__(self) -> None:
            self.later = "hidden-4"
            self.bare = "unknown"
            self.public = 1

    rendered = render_python(capture(Later()))
    assert "hidden-4" not in rendered and "unknown" not in rendered and "public=1" in rendered


class _Unreadable:
    count: int

    def __init__(self) -> None:
        self.count = 1
        self.secret = "maybe-hidden"


def _boom(format: int) -> dict[str, object]:
    raise RuntimeError("annotations are unreadable")


@lazy_only
def test_annotations_that_cannot_be_read_fail_closed():
    cls = type("_Broken", (_Unreadable,), {})
    type.__setattr__(cls, "__annotate__", _boom)
    assert "__annotations__" not in cls.__dict__
    rendered = render_python(capture(cls()))
    assert "maybe-hidden" not in rendered and "count=" not in rendered


@lazy_only
def test_assigned_annotations_win_over_the_class_body():
    class Assigned:
        visible: int

    Assigned.__annotations__ = {"secret": Annotated[str, hidden]}
    value = Assigned()
    value.secret = "hidden-5"
    value.visible = 1
    assert _names(value) == ["visible"]


def test_a_non_dict_annotations_value_fails_closed():
    class Odd:
        __annotations__ = ("secret",)

        def __init__(self) -> None:
            self.secret = "maybe-hidden"

    assert "maybe-hidden" not in render_python(capture(Odd()))


def test_a_forward_reference_annotation_is_resolved_or_fails_closed():
    @dataclasses.dataclass
    class Ref:
        secret: typing.ForwardRef("Annotated[str, hidden]")  # type: ignore[valid-type]
        missing: typing.ForwardRef("Annotated[Undefined, hidden]")  # type: ignore[valid-type]  # noqa: F821
        shown: int = 1

    Ref.__module__ = __name__
    rendered = render_python(capture(Ref("hidden-6", "hidden-7")))
    assert "hidden-" not in rendered and "shown=1" in rendered
