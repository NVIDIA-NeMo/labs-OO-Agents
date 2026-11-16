# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CodeRabbit's #580 findings on nooa.snapshot, each replayed as the scenario that broke a guarantee."""

from __future__ import annotations

import ast
import datetime
import random
import time
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, ConfigDict

from nooa.agentdoc import spec
from nooa.agentdoc.markers import set_field_metadata
from nooa.snapshot import Limits, Object, capture, render_json, render_python
from nooa.snapshot._codec import encode_text

Q3 = "'" * 3


def test_surrogate_escapes_keep_their_bytes_next_to_a_lone_surrogate() -> None:
    # 4584541146: the backslashreplace retry escaped the surrogateescape byte too.
    assert encode_text("\udc80\ud800") == b"\x80\\ud800"
    assert encode_text("a\udcff") == b"a\xff"
    assert encode_text("\ud800") == b"\\ud800"


def test_hide_empty_runs_no_metaclass_eq() -> None:
    # 4584541156: ``type(value) in _EMPTY_TYPES`` compared with a metaclass ``__eq__``.
    calls: list[object] = []

    class Meta(type):
        def __eq__(cls, other: object) -> bool:
            calls.append(other)
            return False

        __hash__ = type.__hash__

    class Odd(metaclass=Meta):
        pass

    class Model(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        x: Annotated[Any, spec(hide_empty=True)] = None

    model = Model(x=Odd())
    calls.clear()
    capture(model)
    assert calls == []


def test_an_instance_override_past_a_partial_copy_still_hides_its_field() -> None:
    # 4584541163: only the first entries are copied; an override stored after
    # them was not seen, so a field hidden on the instance showed.
    class Plain:
        pass

    big = Plain()
    big.secret = "TOPSECRET"
    for index in range(5_000):
        setattr(big, f"a{index:05d}", index)
    set_field_metadata(big, "secret", hidden=True)
    text = render_python(capture(big, Limits(fallback="repr")))
    assert "TOPSECRET" not in text and "attributes)" in text


def test_a_huge_class_namespace_is_not_copied() -> None:
    # 4584541200: a 200,000-entry class namespace was copied whole under max_nodes=5.
    big = type("Big", (), {f"a{i}": i for i in range(200_000)})
    start = time.perf_counter()
    capture(big(), Limits(max_nodes=5))
    assert time.perf_counter() - start < 0.05  # was ~0.2 s and 24 MB


def test_an_ordinary_class_is_still_read_by_small_nested_captures() -> None:
    class Model(BaseModel):  # BaseModel's own namespace has more than 50 entries
        a: int = 1

    def function(day=datetime.date(2020, 1, 2), model=Model()):  # noqa: B008
        return day, model

    text = render_python(capture(function))
    assert "day=datetime.date(2020, 1, 2)" in text and "model=Model(a=1)" in text


@pytest.mark.parametrize(
    "text", ["a\n'", "a\\nb\n", 'x\n"""y\'', "a\n''", "tab\tcr\r\n\x00", "\\\n\\"]
)
def test_multiline_strings_are_valid_literals_of_the_same_text(text: str) -> None:
    # 4584541205: a quote before the closing delimiter, or a backslash, broke the literal.
    assert ast.literal_eval(render_python(capture(text))) == text


def test_multiline_strings_round_trip_at_random() -> None:
    alphabet = "ab'\"\\\n\t\r\x01é"
    rng = random.Random(580)
    for _ in range(2_000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 12))) + "\n"
        assert ast.literal_eval(render_python(capture(text))) == text


def test_a_readable_multiline_string_stays_triple_quoted() -> None:
    assert render_python(capture("line one\nline two")) == Q3 + "line one\nline two" + Q3


def test_capture_runs_no_render_hook_for_a_default() -> None:
    # 4584541215: the default's text was rendered with the registered hooks.
    calls: list[object] = []

    def hook(node: Object, r: Any) -> str:
        calls.append(node)
        return "HOOKED"

    def function(day=datetime.date(2020, 1, 1)):
        return day

    render_python.register("python.date")(hook)
    try:
        capture(function)
        assert calls == []
    finally:
        render_python.unregister("python.date")


def test_default_storage_is_checked_before_any_truth_test() -> None:
    # 4584541218: ``defaults or ()`` ran a tuple / dict subclass's __bool__.
    calls: list[str] = []

    class Defaults(tuple):
        def __bool__(self) -> bool:
            calls.append("bool")
            return True

        def __len__(self) -> int:
            calls.append("len")
            return tuple.__len__(self)

    class KwDefaults(dict):
        def __bool__(self) -> bool:
            calls.append("dict bool")
            return True

    def positional(a=1):
        return a

    def keyword(*, k=1):
        return k

    positional.__defaults__ = Defaults((1,))
    keyword.__kwdefaults__ = KwDefaults(k=1)
    calls.clear()
    assert render_python(capture(positional)).endswith("positional(...)>")
    assert render_python(capture(keyword)).endswith("keyword(...)>")
    assert calls == []


def test_keyword_default_keys_are_checked_before_lookup() -> None:
    # 4584541224: ``name in kwdefaults`` compared a foreign key with a colliding hash.
    calls: list[object] = []

    class Key:
        def __hash__(self) -> int:
            return hash("x")

        def __eq__(self, other: object) -> bool:
            calls.append(other)
            return False

    def function(*, x=1):
        return x

    function.__kwdefaults__ = {Key(): 1, "x": 2}
    calls.clear()
    assert render_python(capture(function)).endswith("function(...)>")
    assert calls == []


@pytest.mark.parametrize(
    ("value", "max_items"), [(-(10**5000), 1), (10**5000, 0)], ids=["negative", "zero-items"]
)
def test_max_items_keeps_a_huge_ints_digit_count(value: int, max_items: int) -> None:
    # 4584541230: max_items cut the int's structural fields: render_json showed 0.
    snapshot = capture(value, Limits(max_items=max_items))
    assert isinstance(snapshot, Object)
    assert render_json(snapshot, max_chars=200).startswith('{"digits": 5001')
    assert render_python(snapshot).startswith("int(5,001 digits")
