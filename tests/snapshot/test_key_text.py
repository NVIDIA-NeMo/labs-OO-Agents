# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden tests of the canonical key encoder (``nooa.snapshot._keys.key_text``).

A node key's text is its JSON object key: capture sizes entries and finds
collisions with it, and the renderers print it. It must not move when the
display notation does, so each form is pinned here, byte for byte.
"""

from __future__ import annotations

import collections
import datetime
import decimal
import enum
import json
import pathlib
import re
import uuid

import pytest

from nooa.snapshot import (
    Limits,
    Mapping,
    Object,
    Ref,
    Scalar,
    Sequence,
    Text,
    Unavailable,
    capture,
    render_json,
    render_python,
)
from nooa.snapshot._keys import key_text


class Color(enum.Enum):
    RED = 1


NT = collections.namedtuple("NT", "p q")


def _f(a, b=2):
    return a


GOLDEN = [
    ((1,), "(1,)"),
    ((1, "a", None, True, 1.5), "(1, 'a', None, True, 1.5)"),
    ((), "()"),
    (frozenset(), "frozenset()"),
    (frozenset({2}), "frozenset({2})"),
    (("a\nb",), "('''a\nb''',)"),
    (("x" * 60,), "(str(len=60, [:50]='" + "x" * 50 + "'),)"),
    ((b"\xff" * 3,), "(b'\\xff\\xff\\xff',)"),
    (tuple(range(10)), "tuple(len=10, [:8]=(0, 1, 2, 3, 4, 5, 6, 7))"),
    ((1, (2, (3, (4,)))), "(1, (2, (tuple: 2 items)))"),
    (Color.RED, "<Color.RED: 1>"),
    (datetime.datetime(2026, 1, 2, 3, 4, 5), "datetime.datetime(2026, 1, 2, 3, 4, 5)"),
    (datetime.date(2026, 1, 2), "datetime.date(2026, 1, 2)"),
    (datetime.time(1, 2), "datetime.time(1, 2)"),
    (datetime.timedelta(days=2, seconds=3), "datetime.timedelta(days=2, seconds=3)"),
    (decimal.Decimal("1.50"), "Decimal('1.50')"),
    (uuid.UUID(int=7), "UUID('00000000-0000-0000-0000-000000000007')"),
    (pathlib.PurePosixPath("/a/b"), "PurePosixPath('/a/b')"),
    (re.compile("a+"), "re.compile('a+')"),
    (re.compile("a", re.I), "re.compile('a', re.IGNORECASE)"),
    (range(1, 10, 2), "range(1, 10, 2)"),
    (1 + 2j, "(1+2j)"),
    (NT(1, 2), "NT(p=1, q=2)"),
    (ValueError("boom"), "ValueError('boom')"),
    (len, "<built-in function len>"),
    (_f, "<function tests.snapshot.test_key_text._f(a, b=2)>"),
    (Color, "<class 'tests.snapshot.test_key_text.Color'>"),
    (collections, "<module 'collections'>"),
]


@pytest.mark.parametrize(("key", "text"), GOLDEN, ids=[text for _, text in GOLDEN])
def test_a_live_key_has_its_pinned_text(key, text):
    node = capture({key: 0})
    assert type(node) is Mapping and node.complete
    ((key_node, _),) = node.head
    assert key_text(key_node) == text
    assert render_json(node, max_chars=10_000) == "{" + json.dumps(text) + ": 0}"
    assert render_python(node) == "{" + text + ": 0}"


DETACHED = [
    (Scalar(3), "3"),
    (
        Text(head="ab", tail="yz", length=10, unit="chars", reason="max_length"),
        "str(len=10, [:2]='ab', [-2:]='yz')",
    ),
    (Text(head=b"a", length=4, unit="bytes", reason="max_length"), "bytes(len=4, [:1]=b'a')"),
    (
        Sequence("set", length=3, head=(Scalar(1),), reason="max_items"),
        "set(len=3, items={1, ...})",
    ),
    (Sequence("deque", length=1, head=(Scalar(1),)), "deque([1])"),
    (Sequence("list", length=4, head=(), reason="max_depth"), "[list: 4 items]"),
    (Mapping("dict", length=1, head=(("k", Scalar(1)),)), "{'k': 1}"),
    (Mapping("Counter", length=0, head=()), "Counter({})"),
    (Mapping("dict", length=5, head=(), reason="max_depth"), "{dict: 5 items}"),
    (
        Mapping(
            "dict",
            length=9,
            head=(("a", Scalar(1)),),
            reason="max_items",
            tail=(("z", Scalar(2)),),
        ),
        "dict(len=9, items={'a': 1, ..., 'z': 2})",
    ),
    (
        Object("P", (("x", Scalar(1)),), reason="max_items", length=4),
        "P(x=1, ... of 4 attributes)",
    ),
    (Object("P", (("x", Scalar(1)),), kind="registered.kind"), "P(x=1)"),
    (Ref("/0", "sequence"), "<cycle>"),
    (Unavailable("opaque", "mod.T"), "<opaque: mod.T>"),
    (Unavailable("error", "mod.T"), "<unavailable: error>"),
]


@pytest.mark.parametrize(("node", "text"), DETACHED, ids=[text for _, text in DETACHED])
def test_a_detached_node_key_has_its_pinned_text(node, text):
    assert key_text(node) == text


def test_a_registered_hook_does_not_change_a_key():
    node = Object("P", (("x", Scalar(1)),), kind="golden.point")
    render_python.register("golden.point", lambda n, r: "<point>")
    try:
        assert render_python(node) == "<point>"
        assert key_text(node) == "P(x=1)"
        assert (
            render_json(Mapping("dict", length=1, head=((node, Scalar(0)),)), max_chars=1_000)
            == '{"P(x=1)": 0}'
        )
    finally:
        render_python.unregister("golden.point")


def test_a_node_key_collides_with_the_string_key_of_the_same_text():
    node = capture({(1,): "first", "(1,)": "second"})
    assert type(node) is Mapping and node.reason == "key" and len(node.head) == 1
    assert render_json(node, max_chars=10_000) == '{"(1,)": "first"}'


def test_the_key_text_is_what_capture_sizes():
    value = {(1, "a"): "v", Color.RED: 2}
    node = capture(value, Limits(max_chars=len('{"(1, \'a\')": "v", "<Color.RED: 1>": 2}')))
    assert node.complete
    assert render_json(node, max_chars=10_000) == '{"(1, \'a\')": "v", "<Color.RED: 1>": 2}'
