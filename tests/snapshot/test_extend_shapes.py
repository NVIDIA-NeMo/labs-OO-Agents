# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The five shapes: their nodes, bounded accessor calls, accessor errors, kinds and the budget."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel

from nooa.snapshot import (
    Limits,
    Mapping,
    Object,
    Scalar,
    Sequence,
    Text,
    Unavailable,
    _capture,
    array_like,
    capture,
    mapping_like,
    object_like,
    render_json,
    render_output,
    render_python,
    text_like,
)
from tests.snapshot._extend_support import _named


class _Counted:
    """A huge lazy collection that counts what capture pulls from it."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.pulled = 0
        self.length_calls = 0

    def length(self) -> int:
        self.length_calls += 1
        return self.size

    def values(self) -> Iterator[int]:
        for value in range(self.size):
            self.pulled += 1
            yield value

    def pairs(self) -> Iterator[tuple[str, int]]:
        for value in self.values():
            yield f"k{value}", value


ARRAY = array_like(lambda v: v.length(), lambda v: v.values())
MAPPING = mapping_like(lambda v: v.length(), lambda v: v.pairs())


def _text(value: str) -> Text:
    return Text(head=value, tail="", length=len(value), unit="chars", reason=None)


# --- array_like ------------------------------------------------------------------------


def test_array_like_without_fields_is_a_sequence(registered) -> None:
    Bag = _named("Bag", qualname="Bag")
    registered(Bag, array_like(lambda v: 3, lambda v: ["a", 1, None]))

    assert capture(Bag()) == Sequence("Bag", 3, (_text("a"), Scalar(1), Scalar(None)))
    assert render_python(capture(Bag())) == "['a', 1, None]"
    assert render_json(capture(Bag()), max_chars=100) == '["a", 1, null]'


def test_array_like_with_fields_is_an_object_with_items(registered) -> None:
    Bag = _named("Bag")
    registered(
        Bag,
        array_like(
            lambda v: 2, lambda v: [1, 2], fields={"maxlen": lambda v: 5}, kind="fake.deque"
        ),
    )
    snapshot = capture(Bag())

    assert snapshot == Object(
        "Bag",
        (("maxlen", Scalar(5)), ("items", Sequence("list", 2, (Scalar(1), Scalar(2))))),
        None,
        "fake.deque",
    )
    assert render_python(snapshot) == "Bag(maxlen=5, items=[1, 2])"


@pytest.mark.parametrize(
    "limits",
    [
        Limits(max_items=7),
        Limits(max_chars=80),
        Limits(max_nodes=30),
        Limits(),
        Limits(max_items=0),
    ],
)
def test_array_like_pulls_at_most_the_kept_items_plus_one(registered, limits) -> None:
    registered(_Counted, ARRAY)
    value = _Counted(10**9)

    snapshot = capture(value, limits)

    assert isinstance(snapshot, Sequence) and snapshot.length == 10**9 and not snapshot.complete
    assert value.length_calls == 1
    assert value.pulled <= len(snapshot.items) + 1
    assert len(render_json(snapshot, max_chars=limits.max_chars)) <= limits.max_chars


def test_array_like_keeps_items_beyond_length_out(registered) -> None:
    Liar = _named("Liar")
    registered(Liar, array_like(lambda v: 2, lambda v: iter(range(100))))

    assert capture(Liar()) == Sequence("Liar", 2, (Scalar(0), Scalar(1)))


def test_array_like_errors(registered) -> None:
    Broken = _named("Broken")

    def fail(value: Any) -> Any:
        raise RuntimeError("boom")

    def two_then_fail(value: Any) -> Iterator[int]:
        yield 1
        yield 2
        raise RuntimeError("boom")

    registered(Broken, array_like(fail, lambda v: []))
    assert capture(Broken()) == Unavailable("error", "Broken")
    for bad_length in (lambda v: -1, lambda v: 2.0, lambda v: True, lambda v: None):
        registered(Broken, array_like(bad_length, lambda v: []))  # type: ignore[arg-type]
        assert capture(Broken()) == Unavailable("error", "Broken")
    registered(Broken, array_like(lambda v: 5, fail))  # no fields: the whole value
    assert capture(Broken()) == Unavailable("error", "Broken")
    registered(Broken, array_like(lambda v: 5, lambda v: 42))  # type: ignore[arg-type]  # not iterable
    assert capture(Broken()) == Unavailable("error", "Broken")
    registered(Broken, array_like(lambda v: 5, two_then_fail))  # the kept items stay
    assert capture(Broken()) == Sequence("Broken", 5, (Scalar(1), Scalar(2)), "error")
    # With fields, only the failing part.
    registered(Broken, array_like(lambda v: 5, fail, fields={"a": lambda v: 1, "b": fail}))
    assert capture(Broken()) == Object(
        "Broken",
        (("a", Scalar(1)), ("b", Unavailable("error")), ("items", Unavailable("error", "list"))),
        None,
    )
    assert render_python(capture(Broken())) == (
        "Broken(a=1, b=<unavailable: error>, items=<unavailable: error>)"
    )


# --- mapping_like ----------------------------------------------------------------------


def test_mapping_like_nodes_and_bounded_pulls(registered) -> None:
    registered(_Counted, MAPPING)
    small = capture(_Counted(2))
    assert small == Mapping("_Counted", 2, (("k0", Scalar(0)), ("k1", Scalar(1))))
    assert render_python(small) == "{'k0': 0, 'k1': 1}"

    value = _Counted(10**9)
    cut = capture(value, Limits(max_items=3))
    assert render_python(cut) == "_Counted(len=1000000000, items={'k0': 0, 'k1': 1, 'k2': 2})"
    assert value.pulled <= 3 + 1 and value.length_calls == 1
    for limits in (Limits(max_chars=60), Limits(max_nodes=20), Limits()):
        value = _Counted(10**9)
        cut = capture(value, limits)
        assert isinstance(cut, Mapping) and value.pulled <= len(cut.items) + 1


def test_mapping_like_with_fields_and_errors(registered) -> None:
    Table = _named("Table")

    def bad_pairs(value: Any) -> Iterator[Any]:
        yield "a", 1
        yield "not a pair"

    registered(
        Table, mapping_like(lambda v: 3, bad_pairs, fields={"name": lambda v: "t"}, kind="k")
    )
    snapshot = capture(Table())
    assert snapshot == Object(
        "Table",
        (("name", _text("t")), ("items", Mapping("dict", 3, (("a", Scalar(1)),), "error"))),
        None,
        "k",
    )
    # A key that is not an exact built-in scalar stops the mapping, as for a dict.
    registered(Table, mapping_like(lambda v: 2, lambda v: [("a", 1), (object(), 2)]))
    stopped = capture(Table())
    assert isinstance(stopped, Mapping) and stopped.reason == "key" and len(stopped.items) == 1


# --- object_like -----------------------------------------------------------------------


def test_object_like_runs_only_the_kept_accessors(registered) -> None:
    calls: list[str] = []

    def field(name: str, value: Any) -> Any:
        def read(instance: Any) -> Any:
            calls.append(name)
            return value

        return read

    Record = _named("Record")
    registered(
        Record,
        object_like({f"f{i}": field(f"f{i}", "y" * 50) for i in range(100)}, kind="fake.record"),
    )

    snapshot = capture(Record(), Limits(max_chars=200))
    assert isinstance(snapshot, Object) and not snapshot.complete and snapshot.kind == "fake.record"
    assert calls == [name for name, _node in snapshot.fields]  # once each, only the kept ones
    calls.clear()
    capped = capture(Record(), Limits(max_items=3))
    assert isinstance(capped, Object) and capped.reason == "max_items"
    assert calls == ["f0", "f1", "f2"]


def test_object_like_greedy_field_leaves_room_for_later_fields(registered) -> None:
    Greedy = _named("Greedy")
    registered(
        Greedy,
        object_like(
            {
                "sample": lambda v: list(range(10_000)),
                "name": lambda v: "abc",
                "count": lambda v: 7,
                "plain": lambda v: [1, 2],
            }
        ),
    )
    for limits in (Limits(max_chars=120), Limits(max_chars=2_000), Limits(max_nodes=80)):
        snapshot = capture(Greedy(), limits)
        assert isinstance(snapshot, Object)
        assert [name for name, _node in snapshot.fields] == ["sample", "name", "count", "plain"]
        sample = snapshot.field("sample")
        assert isinstance(sample, Sequence) and not sample.complete and sample.items
        text = render_json(snapshot, max_chars=limits.max_chars)
        assert len(text) <= limits.max_chars
        assert set(json.loads(text)) == {"sample", "name", "count", "plain"}


def test_object_like_accessor_error_is_that_field(registered) -> None:
    def fail(value: Any) -> Any:
        raise KeyError("x")

    Partly = _named("Partly")
    registered(Partly, object_like({"a": lambda v: 1, "b": fail, "c": lambda v: "z"}))
    assert capture(Partly()) == Object(
        "Partly",
        (("a", Scalar(1)), ("b", Unavailable("error")), ("c", _text("z"))),
        None,
    )


def test_object_like_values_are_captured_normally(registered) -> None:
    Holder = _named("Holder")
    Inner = _named("Inner")
    registered(Inner, object_like({"x": lambda v: 1}))
    registered(Holder, object_like({"inner": lambda v: Inner(), "other": lambda v: object()}))

    snapshot = capture(Holder(), Limits(max_depth=1))
    assert isinstance(snapshot, Object)
    assert snapshot.field("inner") == Object("Inner", (), "max_depth")
    assert snapshot.field("other") == capture(object())
    assert render_python(capture(Holder())) == "Holder(inner=Inner(x=1), other=<opaque: object>)"


def test_shape_metadata_is_copied_to_exact_str(registered) -> None:
    calls: list[str] = []

    class Name(str):
        def __str__(self) -> str:
            calls.append("str")
            raise RuntimeError("live metadata __str__ executed")

        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            raise RuntimeError("live metadata __eq__ executed")

        __hash__ = str.__hash__

    Value = _named("Value")
    registered(Value, object_like({Name("items"): lambda v: [1]}, kind=Name("demo.value")))
    calls.clear()
    node = capture(Value())

    assert isinstance(node, Object)
    assert type(node.kind) is str and node.kind == "demo.value"
    assert type(node.fields[0][0]) is str
    render_python(node)
    render_output(node)
    render_json(node, max_chars=1000)
    assert calls == []


# --- text_like -------------------------------------------------------------------------


class _BigText:
    """Text that is never materialized; records every slice capture asks for."""

    def __init__(self, size: int, unit: str = "chars") -> None:
        self.size = size
        self.unit = unit
        self.slices: list[tuple[int, int]] = []

    def read(self, start: int, stop: int) -> Any:
        self.slices.append((start, stop))
        if self.unit == "chars":
            return "".join("abcdefghij"[i % 10] for i in range(start, stop))
        return bytes(b"\xc3\xa9"[i % 2] for i in range(start, stop))  # "é" repeated


TEXT = text_like(lambda v: v.size, lambda v, start, stop: v.read(start, stop), kind="fake.text")
BYTES = text_like(lambda v: v.size, lambda v, start, stop: v.read(start, stop), unit="bytes")


@pytest.mark.parametrize(
    "limits",
    [
        Limits(),
        Limits(max_string=10, tail_share=0.5),
        Limits(max_chars=7, tail_share=1.0),
        Limits(max_string=0),
    ],
)
def test_text_like_reads_at_most_the_kept_head_and_tail(registered, limits) -> None:
    registered(_BigText, TEXT)
    value = _BigText(10**12)

    snapshot = capture(value, limits)

    assert isinstance(snapshot, Text) and snapshot.unit == "chars" and not snapshot.complete
    assert snapshot.length == 10**12 and snapshot.reason == "max_length"
    assert snapshot.omitted == 10**12 - len(snapshot.head) - len(snapshot.tail)
    assert len(value.slices) <= 2 and (0, 10**12) not in value.slices
    assert sum(stop - start for start, stop in value.slices) <= limits.max_chars
    assert len(render_json(snapshot, max_chars=limits.max_chars)) <= limits.max_chars


def test_text_like_matches_the_str_notation(registered) -> None:
    registered(_BigText, TEXT)
    value = _BigText(1_000)
    plain = "".join("abcdefghij"[i % 10] for i in range(1_000))
    for limits in (Limits(max_string=8, tail_share=0.5), Limits(max_chars=50), Limits()):
        assert capture(value, limits) == capture(plain, limits)
    assert render_python(capture(value, Limits(max_string=4, tail_share=0.5))) == (
        "str(len=1000, [:2]='ab', [-2:]='ij')"
    )


def test_text_like_bytes_cut_on_code_points(registered) -> None:
    registered(_BigText, BYTES)
    value = _BigText(10**9, "bytes")

    snapshot = capture(value, Limits(max_string=7, tail_share=0.5))

    assert snapshot == Text(
        head="éé".encode(),
        tail="é".encode(),
        length=10**9,
        unit="bytes",
        reason="max_length",
    )
    assert len(value.slices) == 2 and (0, 10**9) not in value.slices
    # Same cut as an exact bytes value of that content.
    plain = b"\xc3\xa9" * 500
    small = _BigText(1_000, "bytes")
    for limits in (
        Limits(max_string=7, tail_share=0.5),
        Limits(max_chars=40),
        Limits(max_string=1),
    ):
        assert capture(small, limits) == capture(plain, limits)


class _Rope:
    """A ``str`` or ``bytes`` seen only through ``text_like``."""

    def __init__(self, data: str | bytes) -> None:
        self.data = data


def test_text_like_cuts_exactly_like_str_and_bytes(registered) -> None:
    import random

    registered(_Rope, text_like(lambda v: len(v.data), lambda v, start, stop: v.data[start:stop]))
    alphabet = ["a", "\u00e9", "\u20ac", "\U0001f600", "\n", '"', "\ud800"]
    rng = random.Random(4)
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 60)))
        limits = Limits(
            max_chars=rng.randrange(4, 90),
            max_string=rng.choice([None, 0, 1, 5, 20]),
            tail_share=rng.choice([0, 0.25, 0.5, 1]),
        )
        assert capture(_Rope(text), limits) == capture(text, limits)
    registered(_Rope, text_like(lambda v: len(v.data), lambda v, a, b: v.data[a:b], unit="bytes"))
    for _ in range(300):
        data = "".join(rng.choice(alphabet[:6]) for _ in range(rng.randrange(0, 40))).encode()
        data += rng.choice([b"", b"\xff", b"\xe2\x82"])
        limits = Limits(
            max_chars=rng.randrange(4, 90),
            max_string=rng.choice([None, 0, 1, 5, 20]),
            tail_share=rng.choice([0, 0.25, 0.5, 1]),
        )
        rope, plain = capture(_Rope(data), limits), capture(data, limits)
        if plain == Scalar(data):  # exact bytes kept whole stay a Scalar
            assert rope == Text(head=data, tail=b"", length=len(data), unit="bytes", reason=None)
        else:
            assert rope == plain


def test_text_like_small_text_is_complete(registered) -> None:
    registered(_BigText, BYTES)
    small = _BigText(4, "bytes")
    assert capture(small) == Text(head="éé".encode(), tail=b"", length=4, unit="bytes", reason=None)
    assert small.slices == [(0, 4)]  # kept whole: one read


def test_text_like_errors(registered) -> None:
    Bad = _named("Bad")

    def fail(*args: Any) -> Any:
        raise RuntimeError("boom")

    for shape in (
        text_like(fail, lambda v, a, b: ""),
        text_like(lambda v: 10**6, fail),
        text_like(lambda v: 10**6, lambda v, a, b: b"x" * (b - a)),  # bytes for chars
        text_like(lambda v: 10**6, lambda v, a, b: "x"),  # wrong size
        text_like(lambda v: 10**6, lambda v, a, b: "x" * (b - a), unit="bytes"),
        text_like(lambda v: "10", lambda v, a, b: ""),  # type: ignore[arg-type]
    ):
        registered(Bad, shape)
        assert capture(Bad()) == Unavailable("error", "Bad")


# --- kind and render hooks -------------------------------------------------------------


def test_kind_feeds_the_render_hooks(registered, hooks) -> None:
    registered(_BigText, TEXT)
    Bag = _named("Bag")
    registered(Bag, array_like(lambda v: 1, lambda v: [1], fields={"n": lambda v: 1}, kind="k.bag"))
    hooks(render_python, "k.bag", lambda node, r: f"bag<{r.render(node.field('items'))}>")

    assert render_python(capture([Bag()])) == "[bag<[1]>]"
    assert render_output(capture(Bag())) == "Bag(n=1, items=[1])"  # no hooks: generic
    # A Text has no kind: text_like's kind is for Object-shaped nodes only.
    assert isinstance(capture(_BigText(3)), Text)


# --- budget and determinism ------------------------------------------------------------


@pytest.mark.parametrize("max_chars", [4, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
def test_shapes_never_exceed_the_budget(registered, max_chars) -> None:
    registered(_Counted, ARRAY)
    registered(_BigText, TEXT)
    Budget = _named("Budget")
    registered(
        Budget,
        object_like(
            {
                "seq": lambda v: _Counted(10**6),
                "map": lambda v: {str(i): "v" * i for i in range(1_000)},
                "text": lambda v: _BigText(10**6),
                "inner": lambda v: {"s": "q" * 300, "l": list(range(50))},
                "s": lambda v: "x" * 1_000,
            }
        ),
    )
    snapshot = capture([Budget()], Limits(max_chars=max_chars))
    text = render_json(snapshot, max_chars=max_chars)
    assert len(text) <= max_chars
    json.loads(text)
    assert snapshot == capture([Budget()], Limits(max_chars=max_chars))  # deterministic


def test_starved_structure_parts_say_max_nodes(registered) -> None:
    Fields = _named("Fields")
    registered(
        Fields,
        array_like(lambda v: 1, lambda v: [1], fields={f"f{i}": lambda v: [1] for i in range(50)}),
    )
    snapshot = capture(Fields(), Limits(max_nodes=30))

    assert isinstance(snapshot, Object) and not snapshot.complete
    reasons = {getattr(node, "reason", None) for _name, node in snapshot.fields}
    assert reasons <= {None, "max_nodes"}


# --- Pydantic models and dataclasses are internal field-reader shapes ---------------------


def test_pydantic_and_dataclasses_are_internal_object_shapes() -> None:
    for shape in (_capture._PYDANTIC, _capture._DATACLASS):
        assert type(shape) is _capture._FieldReader

    class Model(BaseModel):
        a: int = 1
        b: list[int] = [1, 2]

    @dataclasses.dataclass
    class Data:
        x: Any
        _private: int = 0

    snapshot = capture([Model(), Data(Model())], Limits(max_items=1))
    assert repr(snapshot) == (
        "Sequence(type_name='list', length=2, items=(Object(type_name='Model', fields=(('a', "
        "Scalar(value=1)),), reason='max_items', kind=None),), reason='max_items', tail=())"
    )
    assert render_python(capture(Data(Model()))) == "Data(x=Model(a=1, b=[1, 2]))"


def test_registered_shape_beats_a_farther_pydantic_view(registered) -> None:
    class Model(BaseModel):
        secret: str = "s"

    class Child(Model):
        pass

    registered(Model, object_like({"shown": lambda v: "redacted"}))
    assert render_python(capture(Child())) == "Child(shown='redacted')"

    @dataclasses.dataclass
    class Data:
        x: int = 1

    @dataclasses.dataclass
    class Nearer(Data):
        y: int = 2

    Base = Data.__mro__[0]
    registered(Base, object_like({}))
    # Nearer declares dataclass fields itself: the dataclass view is nearer than Data's shape.
    assert render_python(capture(Nearer())) == "Nearer(x=1, y=2)"
