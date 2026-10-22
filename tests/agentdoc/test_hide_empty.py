# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``spec(hide_empty=True)``: pformat leaves out None/empty fields; doc() still lists them."""

from typing import Annotated

from pydantic import BaseModel, Field

from nooa.agentdoc import doc, pformat, spec


@spec(hide_empty=True)
class _Update(BaseModel):
    label: str
    done: bool = False
    error: str | None = None
    lines: list[str] = Field(default_factory=list)
    shown: Annotated[list[str], spec(hide_empty=False)] = Field(default_factory=list)


class _Plain(BaseModel):
    a: str = ""
    b: Annotated[str | None, spec(hide_empty=True)] = None


def test_a_class_spec_hides_empty_fields_but_not_false_or_overridden_ones():
    assert pformat(_Update(label="x")) == "_Update(label='x', done=False, shown=[])"


def test_hide_empty_applies_nested_in_a_large_budget_print():
    text = pformat({"jobs": [_Update(label="x", lines=["a"])]}, max_string=100_000, max_length=100)
    assert "_Update(label='x', done=False, lines=['a'], shown=[])" in text
    assert "error" not in text


def test_a_field_spec_alone_hides_only_that_field():
    assert pformat(_Plain()) == "_Plain(a='')"


def test_doc_still_lists_every_field():
    rendered = doc(_Update)
    for name in ("label", "done", "error", "lines", "shown"):
        assert f"    {name}:" in rendered
