# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""render_output and render_python: the output notation and agentdoc's Python notation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from nooa.agentdoc import pformat
from nooa.snapshot import (
    Limits,
    Object,
    Opaque,
    Ref,
    Scalar,
    Sequence,
    Spill,
    Text,
    Unavailable,
    capture,
    render_output,
    render_python,
)

# --- render_output -------------------------------------------------------------------


def _cut(unit: str = "bytes", full: Spill | None = None, reason: Any = "max_length") -> Text:
    return Text("HEAD", "TAIL", 1_234_575, unit, False, reason, full, omitted=1_234_567)  # type: ignore[arg-type]


def test_render_output_complete_passes_through() -> None:
    assert render_output(Text("all of it\n", "", 10, "bytes", True, None)) == "all of it\n"


def test_render_output_full_output() -> None:
    text = _cut(full=Spill("/tmp/out", 1_234_575, 1_234_575))

    assert render_output(text) == (
        "HEAD\n[... 1,234,567 bytes omitted; full output: /tmp/out]\nTAIL"
    )


def test_render_output_file_limit() -> None:
    text = _cut(full=Spill("/tmp/out", 65_536, 1_234_575), reason="file_limit")

    assert render_output(text) == (
        "HEAD\n[... 1,234,567 bytes omitted; /tmp/out holds the first 65,536 of 1,234,575 bytes]"
        "\nTAIL"
    )


def test_render_output_not_kept() -> None:
    assert render_output(_cut()) == (
        "HEAD\n[... 1,234,567 bytes omitted; full output not kept]\nTAIL"
    )


def test_render_output_chars_unit() -> None:
    assert render_output(_cut("chars")) == (
        "HEAD\n[... 1,234,567 chars omitted; full output not kept]\nTAIL"
    )


def test_render_output_without_head_or_tail() -> None:
    no_head = Text("", "TAIL", 10, "chars", False, "max_length", omitted=6)
    no_tail = Text("HEAD", "", 10, "chars", False, "max_length", omitted=6)

    assert render_output(no_head) == "[... 6 chars omitted; full output not kept]\nTAIL"
    assert render_output(no_tail) == "HEAD\n[... 6 chars omitted; full output not kept]"


# --- render_python (agentdoc's pformat / prefill notation) -----------------------------

PYTHON_LIMITS = Limits(max_items=4, max_string=8, tail_share=0.5)


@pytest.mark.parametrize(
    ("value", "kwargs"),
    [
        ("x" * 100, {"max_string": 8}),
        ("abcdefghijklmnop", {"max_string": 7}),
        ("short", {"max_string": 8}),
        ("two\nlines", {}),
        ("both ' and \"", {}),
        (list(range(100)), {"max_length": 4}),
        (tuple(range(100)), {"max_length": 4}),
        ({i: i for i in range(100)}, {"max_length": 4}),
        ({"a": [1, 2], "b": None, "c": 1.5, "d": True}, {}),
        ([[[[1]]]], {"max_depth": 2}),
        ({"a": {"b": {"c": 1}}}, {"max_depth": 1}),
        ([], {}),
        ({}, {}),
        ((), {}),
        (b"bytes", {}),
        ({"x" * 60: 1}, {}),
    ],
)
def test_render_python_matches_agentdoc(value: Any, kwargs: dict[str, int]) -> None:
    limits = Limits(
        max_items=kwargs.get("max_length"),
        max_string=kwargs.get("max_string"),
        max_depth=kwargs.get("max_depth", 16),
        tail_share=0.5,
    )

    assert render_python(capture(value, limits)) == pformat(value, **kwargs)


def test_render_python_models_match_agentdoc() -> None:
    class Model(BaseModel):
        a: int = 1
        b: str = "x"

    assert render_python(capture(Model())) == pformat(Model()) == "Model(a=1, b='x')"


def test_render_python_notation_for_other_nodes() -> None:
    assert render_python(Opaque("m", "Widget")) == "<opaque: Widget>"
    assert render_python(Ref("", "sequence")) == "<cycle>"
    assert render_python(Unavailable("error", "Widget")) == "<unavailable: error>"
    assert render_python(Object("M", (("a", Scalar(1)),), False, "max_nodes")) == "M(a=1, ...)"
    assert render_python(Sequence("tuple", 1, (Scalar(1),), True, None)) == "(1,)"
    assert render_python(Sequence("list", None, (Scalar(1),), False, "max_nodes")) == (
        "list(len=?, [:1]=[1])"
    )
    assert render_python(Text("", "end", 10, "chars", False, "max_length", omitted=7)) == (
        "str(len=10, [-3:]='end')"
    )
