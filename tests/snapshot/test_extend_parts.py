# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``out`` parts: ``Object.kind``, deferred single-use parts, slots, sibling reservation and metadata."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import pytest

from nooa.snapshot import (
    Limits,
    Object,
    Opaque,
    Out,
    Scalar,
    Sequence,
    Text,
    Unavailable,
    capture,
    render_json,
    render_output,
    render_python,
)
from tests.snapshot._extend_support import (
    FAKE_MODULE,
    _LazyNdarray,
    _named,
    capture_ndarray,
)

# --- Object.kind ---------------------------------------------------------------------


def test_kind_round_trips_through_the_builder(lazy) -> None:
    lazy(
        FAKE_MODULE,
        "Kinded",
        lambda value, out: out.object({"a": 1, "b": out.capture("x")}, kind="fake.kinded"),
    )
    snapshot = capture(_named("Kinded")())

    assert isinstance(snapshot, Object)
    assert snapshot.kind == "fake.kinded"
    assert snapshot.module == FAKE_MODULE
    assert snapshot.field("a") == Scalar(1)
    assert snapshot.field("b") == Text("x", "", 1, "chars", True, None)
    assert snapshot.field("missing") is None
    assert render_python(snapshot) == "Kinded(a=1, b='x')"
    assert json.loads(render_json(snapshot, max_chars=1_000)) == {"a": 1, "b": "x"}


def test_kind_survives_cuts(lazy) -> None:
    lazy(
        FAKE_MODULE,
        "Cut",
        lambda value, out: out.object(((f"f{i}", "y" * 50) for i in range(100)), kind="k"),
    )
    snapshot = capture(_named("Cut")(), Limits(max_chars=200))
    assert isinstance(snapshot, Object) and not snapshot.complete and snapshot.kind == "k"

    deep = capture([[_named("Cut")()]], Limits(max_depth=1))
    assert isinstance(deep, Sequence)


def test_default_kind_is_none_and_must_be_a_string(eager) -> None:
    class Plain:
        pass

    eager(Plain, lambda value, out: out.object([("a", 1)]))
    snapshot = capture(Plain())
    assert isinstance(snapshot, Object) and snapshot.kind is None

    eager(Plain, lambda value, out: out.object([], kind=3))
    assert capture(Plain()) == Unavailable("error", "Plain")


def test_part_too_large_for_its_slot_is_cut(lazy) -> None:
    def build(value: Any, out: Out):
        big = out.capture("z" * 500)
        return out.object({"big": big, "more": "w" * 500})

    lazy(FAKE_MODULE, "Pre", build)
    snapshot = capture(_named("Pre")(), Limits(max_chars=300))
    text = render_json(snapshot, max_chars=300)
    assert len(text) <= 300
    json.loads(text)


# --- sibling reservation, single-use parts and nesting ----------------------------------------


def test_greedy_first_field_does_not_starve_later_fields(lazy) -> None:
    def build(value: Any, out: Out):
        return out.object(
            {
                "sample": out.sequence(range(10**9)),
                "name": out.capture("abc"),
                "count": out.capture(7),
                "plain": [1, 2],
            }
        )

    lazy(FAKE_MODULE, "Greedy", build)
    for limits in (Limits(max_chars=120), Limits(max_chars=2_000), Limits(max_nodes=50)):
        snapshot = capture(_named("Greedy")(), limits)
        assert isinstance(snapshot, Object)
        assert [name for name, _node in snapshot.fields] == ["sample", "name", "count", "plain"]
        sample = snapshot.field("sample")
        assert isinstance(sample, Sequence) and not sample.complete and sample.items
        # Later fields keep at least their type-oriented fallback.
        assert isinstance(snapshot.field("name"), Text)
        assert isinstance(snapshot.field("count"), (Scalar, Unavailable))
        assert isinstance(snapshot.field("plain"), Sequence)
        text = render_json(snapshot, max_chars=limits.max_chars)
        assert len(text) <= limits.max_chars
        assert set(json.loads(text)) == {"sample", "name", "count", "plain"}


def test_fields_that_fit_are_kept_whole_in_order(lazy) -> None:
    def build(value: Any, out: Out):
        return out.object(
            [("a", out.capture("x" * 20)), ("rest", out.sequence(iter(range(10**6))))]
        )

    lazy(FAKE_MODULE, "Ordered", build)
    snapshot = capture(_named("Ordered")(), Limits(max_chars=100))
    assert isinstance(snapshot, Object)
    assert snapshot.field("a") == Text("x" * 20, "", 20, "chars", True, None)
    rest = snapshot.field("rest")
    assert isinstance(rest, Sequence) and rest.reason == "max_length" and rest.items
    assert len(render_json(snapshot, max_chars=100)) <= 100


def test_nested_object_is_built_in_its_slot(lazy) -> None:
    def build(value: Any, out: Out):
        inner = out.object({"a": "z" * 400, "b": "never"})  # nothing is built yet
        return out.object({"first": out.capture("y" * 100), "inner": inner})

    lazy(FAKE_MODULE, "Nested", build)
    snapshot = capture(_named("Nested")(), Limits(max_chars=300))
    assert isinstance(snapshot, Object)
    assert snapshot.field("first") == Text("y" * 100, "", 100, "chars", True, None)
    inner = snapshot.field("inner")
    # Built in what "first" left, with its own sibling reservation: "b" keeps its fallback.
    assert isinstance(inner, Object) and [name for name, _node in inner.fields] == ["a", "b"]
    cut = inner.field("a")
    assert isinstance(cut, Text) and not cut.complete and cut.head.startswith("zzz")
    assert cut.omitted == 400 - len(cut.head)
    assert inner.field("b") == Text("", "", 5, "chars", False, "max_length", omitted=5)
    text = render_json(snapshot, max_chars=300)
    assert len(text) <= 300 and json.loads(text)["inner"]["a"].startswith("zzz")


def test_object_part_is_deferred_until_its_slot(lazy) -> None:
    pulled: list[str] = []

    def fields() -> Iterator[tuple[str, Any]]:
        pulled.append("x")
        yield "x", 1

    def build(value: Any, out: Out):
        inner = out.object(fields())
        assert pulled == []  # out.object builds nothing when called
        return out.object({"inner": inner})

    lazy(FAKE_MODULE, "Later", build)
    snapshot = capture(_named("Later")())
    assert isinstance(snapshot, Object) and pulled == ["x"]
    assert snapshot.field("inner") == Object("Later", (("x", Scalar(1)),), True, None, FAKE_MODULE)


def test_nested_object_part_takes_its_slot_depth(lazy) -> None:
    lazy(FAKE_MODULE, "Deep", lambda value, out: out.object({"inner": out.object({"x": 1})}))

    snapshot = capture(_named("Deep")(), Limits(max_depth=1))
    assert isinstance(snapshot, Object)
    inner = snapshot.field("inner")
    assert isinstance(inner, Object) and inner.fields == () and inner.reason == "max_depth"


def test_part_used_twice_is_an_error(lazy) -> None:
    def twice(value: Any, out: Out):
        big = out.capture("z" * 500)
        return out.object({"big": big, "again": big})

    def in_a_sequence(value: Any, out: Out):
        part = out.object({"a": 1})
        return out.sequence([part, part])

    def two_calls(value: Any, out: Out):
        return out.object({"big": out.capture("z" * 500), "again": out.capture("z" * 500)})

    lazy(FAKE_MODULE, "Twice", twice)
    lazy(FAKE_MODULE, "Items", in_a_sequence)
    lazy(FAKE_MODULE, "TwoCalls", two_calls)
    # A second use raises ValueError in the capture; the value becomes ``error``.
    assert capture(_named("Twice")(), Limits(max_chars=300)) == Unavailable("error", "Twice")
    assert capture(_named("Items")()) == Unavailable("error", "Items")
    # Two placements: two parts, each built and budgeted in its own slot.
    snapshot = capture(_named("TwoCalls")(), Limits(max_chars=300))
    assert isinstance(snapshot, Object)
    big, again = snapshot.field("big"), snapshot.field("again")
    assert isinstance(big, Text) and isinstance(again, Text)
    assert not big.complete and big.reason == "max_length"
    assert again.head == "" and not again.complete  # only its reserved fallback was left
    assert len(render_json(snapshot, max_chars=300)) <= 300


def test_part_used_twice_raises_a_clear_error(lazy) -> None:
    errors: list[BaseException] = []

    def build(value: Any, out: Out):
        part = out.capture(1)
        root = out.object({"a": part})
        # Build the tree here (as capture would) to see the error itself.
        from nooa.snapshot import _capture

        state = _capture._State(Limits())
        try:
            _capture._build_slot(part, 10, 1, "/a", state, out)
            _capture._build_slot(part, 10, 1, "/b", state, out)
        except ValueError as error:
            errors.append(error)
        return root

    lazy(FAKE_MODULE, "Clear", build)
    capture(_named("Clear")())
    assert len(errors) == 1 and "one slot only" in str(errors[0])


def test_nested_object_sequence_keeps_its_head_and_tail(lazy) -> None:
    def build(value: Any, out: Out):
        numbers = out.object({"n": list(range(200))})
        return out.object({"label": out.capture("w" * 40), "numbers": numbers})

    lazy(FAKE_MODULE, "Tail", build)
    snapshot = capture(_named("Tail")(), Limits(max_chars=200, max_items=20, tail_share=0.5))
    assert isinstance(snapshot, Object)
    numbers = snapshot.field("numbers")
    assert isinstance(numbers, Object)
    n = numbers.field("n")
    assert isinstance(n, Sequence) and not n.complete and n.length == 200
    # Captured in its own slot: a real head and tail split, not a shrunk prefix.
    assert n.items[0] == Scalar(0) and n.tail and n.tail[-1] == Scalar(199)
    assert len(render_json(snapshot, max_chars=200)) <= 200


def test_nested_registered_objects_keep_their_fields(lazy) -> None:
    lazy("fakenumpy_7c1e", "ndarray", capture_ndarray)

    def build(value: Any, out: Out):
        return out.object(
            {
                "values": out.capture(_LazyNdarray(10**9)),
                "index": out.capture(_LazyNdarray(10**6)),
                "name": out.capture("frame"),
            },
            kind="fake.frame",
        )

    lazy(FAKE_MODULE, "Frame", build)
    for limits in (Limits(max_chars=400), Limits(max_chars=5_000), Limits()):
        snapshot = capture(_named("Frame")(), limits)
        assert isinstance(snapshot, Object)
        assert [name for name, _node in snapshot.fields] == ["values", "index", "name"]
        values = snapshot.field("values")
        assert isinstance(values, Object)
        assert values.field("shape") == Sequence(
            "tuple", 2, (Scalar(10**6), Scalar(1000)), True, None
        )
        assert values.field("dtype") is not None and values.field("sample") is not None
        # "values" took the node budget: "index" may be left with its fallback.
        assert isinstance(snapshot.field("index"), (Object, Opaque))
        assert isinstance(snapshot.field("name"), Text)
        text = render_json(snapshot, max_chars=limits.max_chars)
        assert len(text) <= limits.max_chars
        json.loads(text)


@pytest.mark.parametrize("max_chars", [4, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
def test_parts_never_exceed_the_budget(lazy, max_chars) -> None:
    lazy("fakenumpy_7c1e", "ndarray", capture_ndarray)

    def build(value: Any, out: Out):
        inner = out.object({"s": "q" * 300, "l": list(range(50))})
        return out.object(
            {
                "seq": out.sequence(iter(range(10**6))),
                "map": out.mapping((str(i), "v" * i) for i in range(1_000)),
                "arr": out.capture(_LazyNdarray(10**6)),
                "inner": inner,
                "s": out.capture("x" * 1_000),
            }
        )

    lazy(FAKE_MODULE, "Budget", build)
    snapshot = capture([_named("Budget")()], Limits(max_chars=max_chars))
    text = render_json(snapshot, max_chars=max_chars)
    assert len(text) <= max_chars
    json.loads(text)
    assert snapshot == capture([_named("Budget")()], Limits(max_chars=max_chars))  # deterministic


# --- node work of deferred parts ------------------------------------------------------

_CHILD_PARTS = {
    "sequence": lambda out: out.sequence([]),
    "mapping": lambda out: out.mapping([]),
    "object": lambda out: out.object({}),
    "opaque": lambda out: out.opaque(),
    "capture": lambda out: out.capture(1),
}


@pytest.mark.parametrize("make", _CHILD_PARTS.values(), ids=list(_CHILD_PARTS))
def test_every_deferred_part_is_one_node_of_the_shared_budget(lazy, make) -> None:
    # The registered value takes 1 node and its MRO (class, object) 2; the root
    # part shares the value's node, and each child part is one more.
    lazy(FAKE_MODULE, "Parts", lambda value, out: out.sequence([make(out) for _ in range(100)]))
    value = _named("Parts")()

    starved = capture(value, Limits(max_nodes=20))
    assert isinstance(starved, Sequence) and len(starved.items) == 17
    assert not starved.complete and starved.reason == "max_nodes"

    lazy(FAKE_MODULE, "Parts", lambda value, out: out.sequence([make(out) for _ in range(5)]))
    assert capture(value, Limits(max_nodes=3 + 5)).complete  # not charged twice
    short = capture(value, Limits(max_nodes=3 + 4))
    assert isinstance(short, Sequence) and len(short.items) == 4 and short.reason == "max_nodes"


def test_starved_object_field_parts_say_max_nodes(lazy) -> None:
    # A field part with no node work left is its empty, typed fallback.
    lazy(
        FAKE_MODULE,
        "Fields",
        lambda value, out: out.object([(f"f{i}", out.sequence([1])) for i in range(50)]),
    )
    snapshot = capture(_named("Fields")(), Limits(max_nodes=30))

    assert isinstance(snapshot, Object) and not snapshot.complete
    assert len(snapshot.fields) < 50
    reasons = {getattr(node, "reason", None) for _name, node in snapshot.fields}
    assert reasons <= {None, "max_nodes"}


# --- review findings: detached metadata ------------------------------


def test_out_metadata_is_copied_to_exact_str(eager) -> None:
    calls: list[str] = []

    class Name(str):
        def __str__(self) -> str:
            calls.append("str")
            raise RuntimeError("live metadata __str__ executed")

        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            raise RuntimeError("live metadata __eq__ executed")

        __hash__ = str.__hash__

    class Value:
        pass

    def snapshot(value: Any, out: Out):
        return out.object(
            {
                "items": out.sequence([1], type_name=Name("deque")),
                "table": out.mapping([("k", 1)], type_name=Name("OrderedDict")),
            },
            type_name=Name("Value"),
            kind=Name("demo.value"),
        )

    eager(Value, snapshot)
    node = capture(Value())

    assert isinstance(node, Object)
    assert type(node.type_name) is str and node.type_name == "Value"
    assert type(node.kind) is str and node.kind == "demo.value"
    items, table = node.field("items"), node.field("table")
    assert isinstance(items, Sequence) and type(items.type_name) is str
    assert type(cast(Any, table).type_name) is str
    render_python(node)
    render_output(node)
    render_json(node, max_chars=1000)
    assert calls == []


@pytest.mark.parametrize(
    "build",
    [
        lambda out: out.object({}, type_name=1),
        lambda out: out.sequence([], type_name=b"list"),
        lambda out: out.mapping([], type_name=None),
        lambda out: out.sequence([], length=True),
        lambda out: out.mapping([], length=-1),
    ],
)
def test_out_rejects_non_str_metadata(eager, build) -> None:
    class Value:
        pass

    eager(Value, lambda value, out: build(out))

    assert capture(Value()) == Unavailable("error", "Value")
