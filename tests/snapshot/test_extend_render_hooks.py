# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-kind render hooks of ``render_python``, ``render_output`` and ``render_json``."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from nooa.snapshot import (
    JsonRenderer,
    Limits,
    Mapping,
    Object,
    OutputRenderer,
    PythonRenderer,
    Scalar,
    Sequence,
    Snapshot,
    Spill,
    Text,
    capture,
    preview_attributes,
    render_json,
    render_output,
    render_python,
)
from tests.snapshot._extend_support import (
    FAKE_MODULE,
)

# --- render hooks --------------------------------------------------------------------

KIND = "fake.thing"


def _thing(**fields: Any) -> Object:
    nodes = tuple((name, capture(value)) for name, value in fields.items())
    return Object("Thing", nodes, True, None, FAKE_MODULE, KIND)


def test_render_python_hook(hooks) -> None:
    seen: list[Any] = []

    def hook(node: Object, r: PythonRenderer) -> str:
        seen.append(r)
        return f"thing<{r.render(node.fields[0][1])}>"

    hooks(render_python, KIND, hook)

    assert render_python([_thing(a="x" * 10)][0]) == "thing<'xxxxxxxxxx'>"
    assert isinstance(seen[0], PythonRenderer)
    nested = Sequence("list", 1, (_thing(a=1),), True, None)
    assert render_python(nested) == "[thing<1>]"


def test_render_python_hook_children_keep_notation(hooks) -> None:
    hooks(render_python, KIND, lambda node, r: f"T({r.render(node.fields[0][1])})")
    cut = capture("abcdefghij", Limits(max_string=4, tail_share=0.5))
    node = Object("Thing", (("s", cut),), True, None, "", KIND)

    assert render_python(node) == "T(str(len=10, [:2]='ab', [-2:]='ij'))"


def _raises(node: Object, r: Any) -> str:
    raise RuntimeError("boom")


@pytest.mark.parametrize("renderer", [render_python, render_output], ids=["python", "output"])
@pytest.mark.parametrize(
    "bad",
    [
        _raises,
        lambda node, r: None,  # not a str
        lambda node, r: 42,  # not a str
        lambda node, r: r.render(node),  # recursion on its own node
    ],
    ids=["raises", "none", "int", "recursion"],
)
def test_bad_text_hook_falls_back_to_generic(hooks, renderer, bad) -> None:
    # The text renderers share their hook contract: a bad hook means generic rendering.
    hooks(renderer, KIND, bad)
    assert renderer(_thing(a=1)) == "Thing(a=1)"


def test_no_hook_for_kind_is_generic() -> None:
    assert render_python(_thing(a=1)) == "Thing(a=1)"
    assert render_output(_thing(a=1)) == "Thing(a=1)"
    assert render_json(_thing(a=1), max_chars=100) == '{"a": 1}'


def test_render_output_hook_uses_output_notation(hooks) -> None:
    seen: list[Any] = []

    def hook(node: Object, r: OutputRenderer) -> str:
        seen.append(r)
        return "log:\n" + r.render(node.fields[0][1])

    hooks(render_output, KIND, hook)
    cut = Text("head", "tail", 1_000, "bytes", False, "max_length", omitted=992)
    node = Object("Log", (("text", cut),), True, None, "", KIND)

    assert render_output(node) == "log:\nhead\n[... 992 bytes omitted; full output not kept]\ntail"
    assert isinstance(seen[0], OutputRenderer)
    # Not registered for render_python: the Python notation is unaffected.
    assert render_python(node) == "Log(text=bytes(len=1000, [:4]=b'head', [-4:]=b'tail'))"


def test_render_output_text_is_unchanged() -> None:
    text = Text("a", "b", 10, "chars", False, "max_length", omitted=8)
    assert render_output(text) == "a\n[... 8 chars omitted; full output not kept]\nb"


def _projection(**entries: Snapshot) -> Mapping:
    """A ``render_json`` hook's detached data: a ``dict`` node of ``entries``."""
    return Mapping("dict", len(entries), tuple(entries.items()), True, None)


def test_render_json_hook(hooks) -> None:
    seen: list[Any] = []

    def hook(node: Object, r: JsonRenderer) -> Snapshot:
        seen.append(r)
        return _projection(thing=node.fields[0][1])

    hooks(render_json, KIND, hook)
    out = render_json(Sequence("list", 1, (_thing(a=[1, 2]),), True, None), max_chars=100)
    assert json.loads(out) == [{"thing": [1, 2]}]
    assert isinstance(seen[0], JsonRenderer) and seen[0].max_chars == 98  # the list's brackets


def test_render_json_hook_children_are_cut_and_reported(hooks) -> None:
    hooks(render_json, KIND, lambda node, r: _projection(s=node.fields[0][1]))
    node = _thing(s="x" * 1_000)
    text = render_json(node, max_chars=50)

    assert len(text) <= 50
    assert json.loads(text)["s"].startswith("xxx")
    attributes = preview_attributes("output", node, max_chars=50)
    assert attributes["nooa.output.preview.incomplete"] is True
    assert attributes["nooa.output.preview.paths"] == ("/s",)


def test_render_json_hook_projection_is_cut_to_the_budget(hooks) -> None:
    # Detached data larger than the node's budget is cut like any node, not refused.
    big = Text("x" * 5_000, "", 5_000, "chars", True, None)
    hooks(render_json, KIND, lambda node, r: Sequence("list", 2, (big, big), True, None))
    node = _thing(a=1)
    for max_chars in (1, 5, 40, 1_000):
        text = render_json(node, max_chars=max_chars)
        assert len(text) <= max_chars
        json.loads(text)
    assert "/0" in preview_attributes("output", node, max_chars=40)["nooa.output.preview.paths"]


@pytest.mark.parametrize(
    "bad",
    [
        lambda node, r: '{"a": 1}',  # JSON text is not a node
        lambda node, r: 42,  # not a node
        lambda node, r: {"a": 1},  # plain data is not a node either
        lambda node, r: 1 / 0,  # raises
        lambda node, r: Scalar(float("nan")),  # a node that can't be encoded
        lambda node, r: Sequence(
            "list", 1, (1,), True, None
        ),  # a non-node child  # type: ignore[arg-type]
        lambda node, r: Text(["x"], "", 1, "chars", True, None),  # type: ignore[arg-type]
        lambda node, r: Sequence("list", 0, (), False, 3),  # type: ignore[arg-type]
    ],
)
def test_render_json_bad_hook_falls_back_to_generic(hooks, bad) -> None:
    hooks(render_json, KIND, bad)
    node = _thing(a="y" * 200)

    for max_chars in (5, 40, 1_000):
        text = render_json(node, max_chars=max_chars)
        assert len(text) <= max_chars
        value = json.loads(text)
        assert type(value) is dict
        assert text == _generic_json(node, max_chars)
        attributes = preview_attributes("output", node, max_chars=max_chars)
        assert attributes == preview_attributes(
            "output", dataclasses.replace(node, kind=None), max_chars=max_chars
        )


def _generic_json(node: Object, max_chars: int) -> str:
    """``node`` as ``render_json`` renders it without its hook (the same node, no ``kind``)."""
    return render_json(dataclasses.replace(node, kind=None), max_chars=max_chars)


def test_render_json_hook_recursing_on_its_node_is_generic(hooks) -> None:
    hooks(render_json, KIND, lambda node, r: node)
    node = _thing(a=1)
    assert render_json(node, max_chars=100) == '{"a": 1}'

    # Embedding its own node: that node renders generically inside the projection.
    hooks(render_json, KIND, lambda node, r: _projection(wrapped=node))
    assert render_json(node, max_chars=100) == '{"wrapped": {"a": 1}}'


def test_render_json_hook_building_new_nodes_of_its_kind_stays_bounded(hooks) -> None:
    # Each projection is a fresh node of the same kind: nesting stops, generically.
    hooks(render_json, KIND, lambda node, r: dataclasses.replace(node, fields=(("n", node),)))
    node = _thing(a=1)
    for max_chars in (10, 1_000, 100_000):
        text = render_json(node, max_chars=max_chars)
        assert len(text) <= max_chars
        json.loads(text)
    assert preview_attributes("output", node)["nooa.output.preview.incomplete"] is False


def test_trace_preview_runs_a_json_hook_once(hooks) -> None:
    from nooa.snapshot import _render

    calls: list[int] = []

    def hook(node: Object, r: JsonRenderer) -> Snapshot:
        calls.append(1)
        return _projection(s=node.fields[0][1])

    hooks(render_json, KIND, hook)
    spilled = Text("h", "", 9, "chars", False, "max_length", Spill("/tmp/full.txt", 9, 9), 0, 8)
    node = Sequence("list", 2, (_thing(s="x" * 1_000), spilled), True, None)

    text, attributes = _render._preview("output", node, max_chars=60)
    assert calls == [1]  # one walk: text and attributes come from the same hook call
    assert text == render_json(node, max_chars=60)
    assert attributes == preview_attributes("output", node, max_chars=60)
    assert "/0/s" in attributes["nooa.output.preview.paths"]
    assert attributes["nooa.output.preview.full_output"] == ("/tmp/full.txt",)
    assert len(calls) == 3  # the two public calls each walk once
