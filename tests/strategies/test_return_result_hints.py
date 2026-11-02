# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers behind the return_result keyword alias and the invalid-value call hint."""

from typing import TypedDict

from pydantic import BaseModel

from nooa.interactive import Done, NeedInput, Waiting
from nooa.strategies.codeact_errors import return_call_hint, return_type_field_names


class Scored(BaseModel):
    value: int
    note: str = ""


class Row(TypedDict):
    result: str


def test_field_names_cover_every_union_member_and_typed_dicts():
    assert {"explanation", "result", "question", "on"} <= return_type_field_names(
        Done | NeedInput | Waiting
    )
    assert "value" not in return_type_field_names(Done | NeedInput | Waiting)
    assert return_type_field_names(Scored) == {"value", "note"}
    assert return_type_field_names(Row) == {"result"}
    assert return_type_field_names(int) == set()


def test_call_hint_names_each_model_with_its_required_fields():
    assert return_call_hint(Scored) == "Call it as return_result(Scored(value=...))."
    assert return_call_hint(Done | NeedInput) == (
        "Call it as one of: return_result(Done(explanation=...)), "
        "return_result(NeedInput(question=...))."
    )
    assert return_call_hint(int) is None
    assert return_call_hint(Done | int) is None


def test_call_hint_for_a_direct_tool_call_labels_code_and_gives_json():
    assert return_call_hint(Scored, "execute_python") == (
        "In execute_python code, call it as return_result(Scored(value=...)). "
        'As a direct return_result tool call, pass JSON arguments: {"result": {"value": ...}}.'
    )
    assert return_call_hint(Done | NeedInput, "execute_python") == (
        "In execute_python code, call it as one of: return_result(Done(explanation=...)), "
        "return_result(NeedInput(question=...)). As a direct return_result tool call, pass "
        'JSON arguments: {"result": {"explanation": ...}} or {"result": {"question": ...}}.'
    )


def test_call_hint_for_an_aliased_call_shows_only_the_cell_form():
    assert return_call_hint(Done | NeedInput, "python_cell", cell_only=True) == (
        "Call python_cell with code one of: return_result(Done(explanation=...)), "
        "return_result(NeedInput(question=...))."
    )


def test_alias_note_names_the_value_taken_and_is_bounded():
    from nooa.strategies.codeact_errors import return_alias_note

    assert return_alias_note("python_cell", 42) == (
        "return_result is a function inside python_cell; this call was taken as return_result(42)"
    )
    long = return_alias_note("python_cell", "x" * 1000)
    assert len(long) < 300 and long.endswith("…)")
