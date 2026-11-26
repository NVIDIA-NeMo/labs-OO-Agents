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


def test_the_key_check_of_a_partial_copy_runs_no_metaclass_hook() -> None:
    # 4584541156 (_with_own_overrides): ``set(map(type, keys))`` hashed each
    # key's type, so a metaclass ``__hash__`` ran on a partial storage copy.
    calls: list[str] = []

    class Meta(type):
        def __hash__(cls) -> int:
            calls.append("hash")
            return 1

        def __eq__(cls, other: object) -> bool:
            calls.append("eq")
            return False

    class Key(metaclass=Meta):
        pass

    class Plain:
        pass

    big = Plain()
    for index in range(5_000):
        setattr(big, f"a{index:05d}", index)
    storage: dict[Any, Any] = big.__dict__
    storage[Key()] = "foreign"
    calls.clear()
    capture(big)
    assert calls == []


def test_a_huge_class_namespace_is_not_copied() -> None:
    # 4584541200: a 200,000-entry class namespace was copied whole under max_nodes=5.
    big = type("Big", (), {f"a{i}": i for i in range(200_000)})
    start = time.perf_counter()
    capture(big(), Limits(max_nodes=5))
    assert time.perf_counter() - start < 0.05  # was ~0.2 s and 24 MB


def test_a_class_namespace_read_is_charged_to_the_work_budget() -> None:
    # 4584541200: a namespace up to a 2,000-entry floor was copied whatever
    # max_nodes was left. The budget is exact: a larger one is not copied.
    from nooa.snapshot._limits import _State
    from nooa.snapshot._raw import _raw_namespace

    big = type("Big", (), {f"a{i}": i for i in range(1_500)})
    state = _State(Limits(max_nodes=100))
    assert _raw_namespace(big, state) is None  # larger than the work left: fail closed
    assert state.nodes_left == 100  # nothing copied, nothing charged
    state = _State(Limits(max_nodes=2_000))
    assert _raw_namespace(big, state) is not None
    assert state.nodes_left == 2_000 - len(big.__dict__)
    small = type("Small", (), {"a": 1})
    state = _State(Limits(max_nodes=100))
    assert _raw_namespace(small, state) is not None
    assert state.nodes_left == 100 - len(small.__dict__)
    assert _raw_namespace(small, state) is not None  # read once per capture
    assert state.nodes_left == 100 - len(small.__dict__)


def test_an_instance_or_module_dict_read_is_charged_to_the_work_budget() -> None:
    # #580 review: ``_safe_dict`` charged a class namespace but copied a plain
    # ``dict`` (an instance or module ``__dict__``) whole, whatever was left.
    from nooa.snapshot._limits import _State
    from nooa.snapshot._raw import _safe_dict

    owner = object()
    big = {f"a{i}": i for i in range(1_500)}
    state = _State(Limits(max_nodes=100))
    assert _safe_dict(owner, big, state) is None  # larger than the work left: fail closed
    assert state.nodes_left == 100  # nothing copied, nothing charged
    state = _State(Limits(max_nodes=2_000))
    assert _safe_dict(owner, big, state) == big
    assert state.nodes_left == 2_000 - len(big)


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


def test_a_function_that_is_its_own_default_stays_bounded(monkeypatch) -> None:
    # 4584541213: each default's text started a fresh capture (its own budget),
    # so a function whose default is itself re-read its signature over and
    # over (143 times under max_nodes=5), bounded by no limit of the capture.
    from nooa.snapshot import _std

    def function(a=None):
        return a

    function.__defaults__ = (function,)
    calls: list[object] = []
    signature = _std._signature

    def counted(*args: Any, **kwargs: Any) -> str:
        calls.append(args[0])
        return signature(*args, **kwargs)

    monkeypatch.setattr(_std, "_signature", counted)
    capture(function, Limits(max_nodes=5))
    assert len(calls) <= 3  # the function, then _MAX_DEFAULT_NESTING (2) defaults
    calls.clear()
    text = render_python(capture(function))
    assert text.startswith("<function ") and len(text) < 2_000
    assert len(calls) <= 3  # the function, then _MAX_DEFAULT_NESTING (2) defaults


def test_a_default_shares_the_enclosing_work_budget() -> None:
    # 4584541213: a default's capture had its own fresh budget of 50 nodes.
    from nooa.snapshot._limits import _State
    from nooa.snapshot._std import _default_text

    state = _State(Limits())
    before = state.nodes_left
    assert _default_text(list(range(10)), state) == "list(len=10, [:3]=[0, 1, 2])"
    assert 0 < before - state.nodes_left < 10
    state.nodes_left = 0
    assert _default_text([1, 2], state) != "[1, 2]"  # no work left: nothing is read


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


def test_a_colliding_key_past_the_copied_prefix_keeps_a_hidden_field_hidden() -> None:
    # 4584541163 follow-up: ``secret`` lies in the copied prefix, its hiding
    # override past it; a non-str key whose type imitates ``str`` (metaclass
    # ``__eq__``/``__hash__``, which fooled ``set(map(type, keys)) - {str}``)
    # and whose own hash and ``__eq__`` collide with the overrides attribute
    # sits next to it. No hook may run and the hidden value must not appear.
    from nooa.snapshot._fields import FIELD_METADATA_ATTR

    calls: list[str] = []
    armed = False

    class LikeStr(type):
        def __hash__(cls) -> int:
            calls.append("meta-hash")
            return hash(str)

        def __eq__(cls, other: object) -> bool:
            calls.append("meta-eq")
            return other is str

    class Colliding(metaclass=LikeStr):
        def __hash__(self) -> int:
            if armed:
                calls.append("hash")
            return hash(FIELD_METADATA_ATTR)

        def __eq__(self, other: object) -> bool:
            if armed:
                calls.append("eq")
            return armed

    class Plain:
        pass

    big = Plain()
    big.secret = "TOPSECRET"  # type: ignore[attr-defined]
    for index in range(5_000):
        setattr(big, f"a{index:05d}", index)
    storage: dict[Any, Any] = big.__dict__
    key = Colliding()
    # Inserted first, with equality off, holding an override map that would
    # unhide ``secret``: an unsafe lookup of the overrides attribute finds it
    # before the real entry once equality is on.
    storage[key] = {"secret": {"hidden": False}}
    set_field_metadata(big, "secret", hidden=True)
    assert dict.get(storage, FIELD_METADATA_ATTR) == {"secret": {"hidden": True}}
    assert any(k is key for k in dict.keys(storage))
    armed = True
    calls.clear()
    text = render_python(capture(big, Limits(fallback="repr")))
    assert "TOPSECRET" not in text
    assert calls == []


def test_a_non_str_key_in_a_huge_dict_does_not_fail_the_whole_object_closed() -> None:
    # 4618578913: the overrides walk skips a key whose type is not exactly
    # ``str`` instead of failing the object closed, so the copied prefix stays
    # listed and the instance's own override still hides its field.
    class Plain:
        pass

    big = Plain()
    big.secret = "TOPSECRET"  # type: ignore[attr-defined]
    for index in range(5_000):
        setattr(big, f"a{index:05d}", index)
    big.__dict__[1234] = "a key that is not a str"
    set_field_metadata(big, "secret", hidden=True)
    text = render_python(capture(big, Limits(fallback="repr")))
    assert "a00000" in text  # the partial copy still lists its first attributes
    assert "TOPSECRET" not in text and "attributes)" in text
