# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard types, plain objects, the repr fallback, ``spec()`` limits and the agentdoc layout."""

from __future__ import annotations

import array
import collections
import datetime
import decimal
import enum
import json
import pathlib
import re
import time
import uuid
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from nooa.agentdoc import hidden, spec
from nooa.snapshot import Limits, Object, Unavailable, capture, render_json, render_python


class Color(enum.Enum):
    RED = 1


class Perm(enum.IntFlag):
    R = 4
    W = 2


Pt = collections.namedtuple("Pt", "x y")


def _sample(a, /, b=2, *args, c, d="x", **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
    return a


@pytest.mark.parametrize(
    ("value", "python"),
    [
        ({1, 2}, "{1, 2}"),
        (set(), "set()"),
        (frozenset({3}), "frozenset({3})"),
        (range(10), "range(0, 10)"),
        (range(0, 10, 2), "range(0, 10, 2)"),
        (1 + 2j, "(1+2j)"),
        (datetime.datetime(2026, 11, 11, 9, 30), "datetime.datetime(2026, 11, 11, 9, 30)"),
        (
            datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
            "datetime.datetime(2026, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)",
        ),
        (datetime.date(2026, 1, 2), "datetime.date(2026, 1, 2)"),
        (datetime.time(9, 30), "datetime.time(9, 30)"),
        (datetime.timedelta(seconds=90), "datetime.timedelta(seconds=90)"),
        (decimal.Decimal("1.50"), "Decimal('1.50')"),
        (uuid.UUID(int=1), "UUID('00000000-0000-0000-0000-000000000001')"),
        (pathlib.PurePosixPath("/tmp/x"), "PurePosixPath('/tmp/x')"),
        (re.compile("a+b"), "re.compile('a+b')"),
        (re.compile("a", re.IGNORECASE), "re.compile('a', re.IGNORECASE)"),
        (ValueError("boom"), "ValueError('boom')"),
        (OSError(2, "gone"), "FileNotFoundError(2, 'gone')"),
        (Color.RED, "<Color.RED: 1>"),
        (Perm.R | Perm.W, "<Perm.R|W: 6>"),
        (Pt(1, 2), "Pt(x=1, y=2)"),
        (collections.Counter("aab"), "Counter({'a': 2, 'b': 1})"),
        (collections.OrderedDict(b=1, a=2), "OrderedDict({'b': 1, 'a': 2})"),
        (collections.defaultdict(int, a=1), "defaultdict({'a': 1})"),
        (
            _sample,
            "<function tests.snapshot.test_standard_types._sample(a, /, b=2, *args, c, d='x', **kw)>",
        ),
        (len, "<built-in function len>"),
        (json, "<module 'json'>"),
        (Color, "<class 'tests.snapshot.test_standard_types.Color'>"),
    ],
)
def test_standard_types_render_in_their_familiar_notation(value: Any, python: str) -> None:
    snapshot = capture(value)

    assert render_python(snapshot) == python
    json.loads(render_json(snapshot, max_chars=1000))  # valid JSON for traces


def test_standard_types_run_no_user_code() -> None:
    calls: list[str] = []

    class Loud(ValueError):
        def __str__(self) -> str:
            calls.append("str")
            return "loud"

        def __repr__(self) -> str:
            calls.append("repr")
            return "loud"

    class Tz(datetime.tzinfo):
        def utcoffset(self, dt: Any) -> datetime.timedelta:
            calls.append("utcoffset")
            return datetime.timedelta(0)

    assert render_python(capture(Loud("x"))) == "Loud('x')"
    # A tzinfo of another type would run its own utcoffset: it is not read natively.
    snapshot = capture(datetime.datetime(2026, 1, 1, tzinfo=Tz()))
    assert isinstance(snapshot, Unavailable)
    assert calls == []


def test_container_subclasses_that_override_iteration_are_not_read_natively() -> None:
    class Lazy(dict):
        def __iter__(self):  # noqa: ANN204
            raise AssertionError("iter ran")

    class Quiet(list):
        pass

    assert isinstance(capture(Lazy(a=1)), Unavailable)
    assert render_python(capture(Quiet([1, 2]))) == "Quiet([1, 2])"


def test_set_cut_keeps_a_head_in_the_items_notation() -> None:
    snapshot = capture(set(range(100)), Limits(max_items=4, tail_share=0.5))

    assert render_python(snapshot) == "set(len=100, items={0, 1, 2, 3, ...})"


# --- plain objects -------------------------------------------------------------------------


class Plain:
    token: Annotated[str, hidden] = ""

    def __init__(self) -> None:
        self.status = 200
        self.url = "https://x"
        self._secret = "s"
        self.token = "t"
        self.callback = print


class Slotted:
    __slots__ = ("a", "_b")

    def __init__(self) -> None:
        self.a = 1
        self._b = 2


def test_plain_objects_show_public_instance_attributes() -> None:
    assert render_python(capture(Plain())) == "Plain(status=200, url='https://x')"
    assert render_python(capture(Slotted())) == "Slotted(a=1)"


def test_plain_object_reader_refuses_proxies_and_repr_only_objects() -> None:
    calls: list[str] = []

    class Proxy:
        def __init__(self) -> None:
            self.a = 1

        def __getattribute__(self, name: str) -> Any:
            calls.append(name)
            return object.__getattribute__(self, name)

    class Handle:
        def __init__(self) -> None:
            self._fd = 3

        def __repr__(self) -> str:
            return "<Handle 3>"

    class Empty:
        pass

    assert isinstance(capture(Proxy()), Unavailable)
    assert isinstance(capture(Handle()), Unavailable)  # nothing public, its own repr
    assert render_python(capture(Empty())) == "Empty()"
    assert calls == []


def test_plain_object_cycles_and_depth() -> None:
    node = Plain()
    node.status = node  # type: ignore[assignment]

    assert render_python(capture(node)) == "Plain(status=<cycle>, url='https://x')"
    assert render_python(capture([[node]], Limits(max_depth=2))) == "[[Plain(...)]]"


# --- the fallback ------------------------------------------------------------------------


class Opaque:
    __slots__ = ()

    def __repr__(self) -> str:
        return "Opaque<" + "x" * 50 + ">"


def test_opaque_label_is_module_qualified() -> None:
    assert render_python(capture(Opaque())) == "<opaque: tests.snapshot.test_standard_types.Opaque>"


def test_repr_fallback_is_bounded_and_explicit() -> None:
    whole = capture(Opaque(), Limits(fallback="repr"))
    cut = capture(Opaque(), Limits(fallback="repr", max_string=10, tail_share=0.5))

    assert isinstance(whole, Object) and whole.kind == "python.repr"
    assert render_python(whole) == "Opaque<" + "x" * 50 + ">"
    assert render_python(cut) == "Opaque(repr_len=58, [:5]='Opaqu', [-5:]='xxxx>')"
    assert json.loads(render_json(whole, max_chars=200)) == {"repr": "Opaque<" + "x" * 50 + ">"}


def test_repr_fallback_that_raises_is_opaque() -> None:
    class Broken:
        __slots__ = ()

        def __repr__(self) -> str:
            raise RuntimeError("no")

    assert isinstance(capture(Broken(), Limits(fallback="repr")), Unavailable)
    with pytest.raises(ValueError):
        capture(1, Limits(fallback="str"))  # type: ignore[arg-type]


# --- spec() limits and hide_empty ------------------------------------------------------------


@spec(hide_empty=True)
class Event(BaseModel):
    text: Annotated[str, spec(max_string=6)] = "x" * 50
    note: str = ""
    full: Annotated[str, spec(max_string=None)] = "y" * 30


@dataclass
class Rows:
    rows: Annotated[list[int], spec(max_length=2)]
    empty: Annotated[list[int], spec(hide_empty=True)]


def test_spec_limits_apply_to_a_field_subtree() -> None:
    limits = Limits(max_string=10, max_items=10, tail_share=0.5)

    assert render_python(capture(Event(), limits)) == (
        "Event(text=str(len=50, [:3]='xxx', [-3:]='xxx'), full='" + "y" * 30 + "')"
    )
    assert render_python(capture(Rows(list(range(9)), []), limits)) == (
        "Rows(rows=list(len=9, [:1]=[0], [-1:]=[8]))"
    )


# --- layout ----------------------------------------------------------------------------------


def test_width_lays_containers_out_like_agentdoc() -> None:
    value = {f"key{i}": "v" * 30 for i in range(4)}

    assert render_python(capture(value), width=120) == (
        "{\n" + "".join(f"    'key{i}': '{'v' * 30}',\n" for i in range(4)) + "}"
    )
    assert render_python(capture([1, [2]]), width=0) == "[\n    1,\n    [\n        2,\n    ],\n]"
    assert render_python(capture([1, 2]), width=120) == "[1, 2]"


# --- node keys and empty containers --------------------------------------------------------


def test_keys_without_a_json_form_are_node_keys() -> None:
    value = {(1, 2): "a", Color.RED: 1, frozenset({1}): 2}
    snapshot = capture(value)

    assert render_python(snapshot) == "{(1, 2): 'a', <Color.RED: 1>: 1, frozenset({1}): 2}"
    assert json.loads(render_json(snapshot, max_chars=500)) == {
        "(1, 2)": "a",
        "<Color.RED: 1>": 1,
        "frozenset({1})": 2,
    }
    # A key capture cannot read, or a number JSON cannot hold, still stops the mapping.
    assert capture({"a": 1, object(): 2}).reason == "key"  # type: ignore[union-attr]
    # A node key whose text an earlier key has collides like any other key.
    assert capture({(1,): 1, "(1,)": 2}).reason == "key"  # type: ignore[union-attr]


def test_empty_containers_at_max_depth_are_whole() -> None:
    snapshot = capture([[], {}, [1]], Limits(max_depth=1))

    assert render_python(snapshot) == "[[], {}, [list: 1 items]]"


_PREFILL = Limits(max_items=25, max_string=2000, max_depth=4, tail_share=0.5, fallback="repr")


@pytest.mark.parametrize(
    ("value", "python"),
    [
        (collections.deque([1, "a"]), "deque([1, 'a'])"),
        (collections.deque(), "deque([])"),
        (array.array("d", [1.5, 2.0]), "array([1.5, 2.0])"),
        (array.array("b"), "array([])"),
    ],
)
def test_deque_and_array_are_native_sequences(value: Any, python: str) -> None:
    for limits in (Limits(), _PREFILL):  # traces and prefill alike: no repr
        assert render_python(capture(value, limits)) == python


@pytest.mark.parametrize(
    "value",
    [collections.deque(range(10_000_000)), array.array("i", range(10_000_000))],
    ids=["deque", "array"],
)
def test_huge_deque_and_array_are_read_from_both_ends_quickly(value: Any) -> None:
    start = time.perf_counter()
    text = render_python(capture(value, _PREFILL))
    assert time.perf_counter() - start < 0.1  # the repr fallback took ~1 s
    name = type(value).__name__
    assert text.startswith(f"{name}(len=10000000, [:13]=[0, 1, 2,")
    assert text.endswith("9999998, 9999999])")


def test_deque_and_array_subclasses_that_override_iteration_are_not_read() -> None:
    class Sneaky(collections.deque):  # type: ignore[type-arg]
        def __iter__(self):  # noqa: ANN204
            raise AssertionError("user iteration must not run")

    class Plain(array.array):
        pass

    assert render_python(capture(Sneaky([1]))).startswith("<opaque: ")
    assert render_python(capture(Plain("i", [1, 2]))) == "Plain([1, 2])"


def test_a_million_attributes_list_the_first_and_say_how_many() -> None:
    class Wide:
        pass

    wide = Wide()
    for i in range(1_000_000):
        setattr(wide, f"a{i}", i)
    for limits in (_PREFILL, Limits()):
        start = time.perf_counter()
        node = capture(wide, limits)
        assert time.perf_counter() - start < 0.05
        assert isinstance(node, Object) and node.length == 1_000_000
        assert node.reason == "max_items"
        assert node.fields[:3] == capture(wide, Limits(max_items=3)).fields
    text = render_python(capture(wide, _PREFILL))
    assert text.startswith("Wide(a0=0, a1=1, a2=2,")
    assert text.endswith(", a24=24, ... of 1,000,000 attributes)")


def test_a_cut_plain_object_says_how_many_attributes_it_has() -> None:
    class Some:
        def __init__(self) -> None:
            self._private = 0
            self.a, self.b, self.c = 1, 2, 3

    assert render_python(capture(Some(), Limits(max_items=2))) == (
        "Some(a=1, b=2, ... of 4 attributes)"
    )
    whole = capture(Some())
    assert isinstance(whole, Object) and whole.length is None
    assert render_python(whole) == "Some(a=1, b=2, c=3)"


@pytest.mark.parametrize(
    ("value", "python"),
    [
        (10**200_000, "int(200,001 digits)"),
        (-(10**5000), "int(5,001 digits, negative)"),
        (1 << 1_000_000, "int(301,030 digits)"),
    ],
    ids=["10**200000", "-10**5000", "2**1000000"],
)
def test_a_huge_int_says_how_many_digits_it_has(value: int, python: str) -> None:
    for limits in (_PREFILL, Limits()):
        start = time.perf_counter()
        node = capture(value, limits)
        assert time.perf_counter() - start < 0.01  # never converted to decimal
        assert isinstance(node, Object) and node.kind == "python.int"
        assert node.reason == "max_length"
        assert render_python(node) == python
    assert json.loads(render_json(capture(value), max_chars=50_000))["digits"] > 4000
    assert render_python(capture(10**30)) == str(10**30)  # a small one stays a number
