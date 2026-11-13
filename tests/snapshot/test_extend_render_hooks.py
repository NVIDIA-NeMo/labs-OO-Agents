# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one render-hook registry (``render_python``) and the trace entry point."""

from __future__ import annotations

import dataclasses
import random
import sys
from typing import Any

import pytest

from nooa.snapshot import (
    Limits,
    Mapping,
    Object,
    Overflow,
    PythonRenderer,
    Sequence,
    Snapshot,
    Text,
    capture,
    render_json,
    render_output,
    render_python,
    render_trace,
)
from tests.snapshot._random_trees import random_tree

KIND = "fake.thing"
_HOOK_NESTING = 64  # the bound hook nesting shares with nothing else: deeper nodes are generic


def _thing(**fields: Any) -> Object:
    nodes = tuple((name, capture(value)) for name, value in fields.items())
    return Object("Thing", nodes, None, KIND)


# --- render_python hooks ---------------------------------------------------------------


def test_render_python_hook_dispatches_on_kind(hooks) -> None:
    seen: list[Any] = []

    def hook(node: Object, r: PythonRenderer) -> str:
        seen.append(r)
        return f"thing<{r.render(node.fields[0][1])}>"

    hooks(render_python, KIND, hook)

    assert render_python(_thing(a="x" * 10)) == "thing<'xxxxxxxxxx'>"
    assert isinstance(seen[0], PythonRenderer)
    nested = Sequence("list", 1, (_thing(a=1),))
    assert render_python(nested) == "[thing<1>]"
    # Another kind, or no kind, is not dispatched to it.
    assert render_python(dataclasses.replace(_thing(a=1), kind="other")) == "Thing(a=1)"
    assert render_python(dataclasses.replace(_thing(a=1), kind=None)) == "Thing(a=1)"


def test_render_python_hook_children_keep_notation(hooks) -> None:
    hooks(render_python, KIND, lambda node, r: f"T({r.render(node.fields[0][1])})")
    cut = capture("abcdefghij", Limits(max_string=4, tail_share=0.5))
    node = Object("Thing", (("s", cut),), None, KIND)

    assert render_python(node) == "T(str(len=10, [:2]='ab', [-2:]='ij'))"


def test_render_python_hook_children_use_their_hooks(hooks) -> None:
    hooks(render_python, KIND, lambda node, r: "<" + r.render(node.fields[0][1]) + ">")
    inner = _thing(a=1)
    outer = Object("Thing", (("inner", inner),), None, KIND)
    assert render_python(outer) == "<<1>>"


def test_render_python_hook_generic_renders_the_node_without_its_hook(hooks) -> None:
    hooks(render_python, KIND, lambda node, r: "wrapped:" + r.generic(node))
    assert render_python(_thing(a=1)) == "wrapped:Thing(a=1)"


def _raises(node: Object, r: Any) -> str:
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    "bad",
    [
        _raises,
        lambda node, r: None,  # not a str
        lambda node, r: 42,  # not a str
        lambda node, r: r.render(node),  # renders its own node: generic past the bound
    ],
    ids=["raises", "none", "int", "own-node"],
)
def test_bad_render_python_hook_falls_back_to_generic(hooks, bad) -> None:
    hooks(render_python, KIND, bad)
    assert render_python(_thing(a=1)) == "Thing(a=1)"


def test_recursive_render_python_hook_stops_at_the_nesting_bound(hooks) -> None:
    # Each call builds a fresh node of its own kind: nesting stops at 64, generically,
    # well inside the recursion limit (no RecursionError is raised or swallowed).
    calls: list[int] = []

    def hook(node: Object, r: PythonRenderer) -> str:
        calls.append(1)
        return "T(" + r.render(dataclasses.replace(node)) + ")"

    hooks(render_python, KIND, hook)
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(400)  # about six frames per nesting level: a RecursionError would show
    try:
        rendered = render_python(_thing(a=1))
    finally:
        sys.setrecursionlimit(limit)

    assert len(calls) == _HOOK_NESTING
    assert rendered == "T(" * _HOOK_NESTING + "Thing(a=1)" + ")" * _HOOK_NESTING


def test_nesting_bound_counts_running_hooks_not_nodes(hooks) -> None:
    # Siblings don't add up: each subtree gets the whole bound.
    hooks(render_python, KIND, lambda node, r: "t")
    many = Sequence("list", 200, tuple(_thing(a=n) for n in range(200)))
    assert render_python(many) == "[" + ", ".join(["t"] * 200) + "]"


def test_register_and_unregister() -> None:
    with pytest.raises(TypeError):
        render_python.register("")
    with pytest.raises(TypeError):
        render_python.register(None)  # type: ignore[arg-type]

    @render_python.register(KIND)
    def hook(node: Object, r: PythonRenderer) -> str:
        return "hooked"

    try:
        assert render_python(_thing(a=1)) == "hooked"
        render_python.register(KIND, lambda node, r: "again")  # replaces it
        assert render_python(_thing(a=1)) == "again"
    finally:
        render_python.unregister(KIND)
    assert render_python(_thing(a=1)) == "Thing(a=1)"
    render_python.unregister(KIND)  # absent: no-op


# --- render_output and render_json have no hooks ----------------------------------------


def test_no_hook_for_kind_is_generic() -> None:
    assert render_python(_thing(a=1)) == "Thing(a=1)"
    assert render_output(_thing(a=1)) == "Thing(a=1)"
    assert render_json(_thing(a=1), max_chars=100) == '{"a": 1}'


def test_render_output_and_render_json_ignore_render_python_hooks(hooks) -> None:
    hooks(render_python, KIND, lambda node, r: "hooked")
    cut = Text(
        head=b"head",
        tail=b"tail",
        length=1_000,
        unit="bytes",
        reason="max_length",
    )
    node = Object("Log", (("text", cut),), None, KIND)

    assert render_python(node) == "hooked"
    assert render_output(node) == (
        "Log(text=head\n[... 992 bytes omitted; full output not kept]\ntail)"
    )
    assert render_json(node, max_chars=100) == '{"text": "head"}'
    assert not hasattr(render_output, "register")
    assert not hasattr(render_json, "register")


def test_render_output_text_is_unchanged() -> None:
    text = Text(head="a", tail="b", length=10, unit="chars", reason="max_length")
    assert render_output(text) == "a\n[... 8 chars omitted; full output not kept]\nb"


# --- render_trace: the trace entry point --------------------------------------------------

_BUDGETS = (1, 2, 5, 17, 60, 300, 5_000, sys.maxsize)


def test_render_trace_is_render_json_and_its_attributes() -> None:
    with_file = Text(
        head="h",
        tail="",
        length=9,
        unit="chars",
        reason="max_length",
        full=Overflow("/tmp/full.txt", 9, 9),
    )
    node = Sequence("list", 2, (_thing(s="x" * 1_000), with_file))

    text, attributes = render_trace(node, direction="output", max_chars=60)
    assert text == render_json(node, max_chars=60)
    assert "/0/s" in attributes["nooa.output.preview.paths"]
    assert attributes["nooa.output.preview.full_output"] == ("/tmp/full.txt",)
    assert render_trace(node, direction="input", max_chars=60)[1].keys() == {
        key.replace("nooa.output.", "nooa.input.") for key in attributes
    }
    with pytest.raises(ValueError):
        render_trace(node, direction="output", max_chars=0)


def test_render_trace_on_random_trees() -> None:
    for seed in range(300):
        tree = random_tree(random.Random(seed))
        for max_chars in _BUDGETS:
            text, attributes = render_trace(tree, direction="output", max_chars=max_chars)
            assert text == render_json(tree, max_chars=max_chars)
            assert attributes["nooa.output.preview.incomplete"] is (
                "nooa.output.preview.paths" in attributes
            )


def _every_file(node: Snapshot) -> list[str]:
    """Reference walk: every ``Text.full`` path, depth first (items, then tail), once each."""
    found: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, Text):
            if current.full is not None and current.full.path not in found:
                found.append(current.full.path)
        elif isinstance(current, Sequence):
            stack.extend(reversed(current.head + current.tail))
        elif isinstance(current, Mapping):
            stack.extend(child for _key, child in reversed(current.head + current.tail))
        elif isinstance(current, Object):
            stack.extend(child for _name, child in reversed(current.fields))
    return found


def test_overflow_paths_do_not_depend_on_the_budget() -> None:
    # Overflow files collected in their own walk equal those the JSON walk collected
    # (cut items, tails, wrappers), for every budget, order included.
    with_files = 0
    for seed in range(3_000):
        tree = random_tree(random.Random(seed))
        expected = tuple(_every_file(tree))
        with_files += bool(expected)
        for max_chars in (1, 40, sys.maxsize):
            attributes = render_trace(tree, direction="output", max_chars=max_chars)[1]
            assert attributes.get("nooa.output.preview.full_output", ()) == expected
    assert with_files > 1_000  # the corpus exercises overflow files


def test_render_trace_matches_its_golden_digests() -> None:
    # Golden digests of render_trace's text and attributes, and of the overflow files,
    # on 1,000 random trees: any change to the trace preview notation shows up here.
    import hashlib
    import json

    previews, files = hashlib.sha256(), hashlib.sha256()
    for seed in range(1_000):
        tree = random_tree(random.Random(seed))
        for max_chars in (1, 5, 60, 5_000, sys.maxsize):
            text, attributes = render_trace(tree, direction="output", max_chars=max_chars)
            items = sorted(
                (key, list(value) if isinstance(value, tuple) else value)
                for key, value in attributes.items()
            )
            previews.update(json.dumps([text, items]).encode())
        files.update(json.dumps(_every_file(tree)).encode())
    assert previews.hexdigest() == (
        "81c79d0b7e3fb82c91c7e468e8c0b49eddb8926f337d0cbcafe1aae7cec3c2fd"
    )
    assert files.hexdigest() == ("96e3f384395e50e459e7fda10cefd6727c795fab81b086f521f0e830cb8c6b06")
