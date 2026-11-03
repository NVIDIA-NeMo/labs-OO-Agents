# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers behind the return_result keyword alias and the invalid-value call hint."""

from typing import TypedDict

from pydantic import BaseModel

from nooa.interactive import Done, NeedInput, Waiting
from nooa.strategies.codeact_errors import (
    return_call_hint,
    return_doc_hint,
    return_type_field_names,
)


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
        "To finish, call python_cell with one of: return_result(Done(explanation=...)), "
        "return_result(NeedInput(question=...))."
    )
    assert return_call_hint(Done, "python_cell", cell_only=True) == (
        "To finish, call python_cell with return_result(Done(explanation=...))."
    )


def test_doc_hint_names_every_member_only_when_all_are_bound():
    bound = {"Done": Done, "NeedInput": NeedInput, "Waiting": Waiting, "Scored": Scored}
    assert return_doc_hint(Done | NeedInput | Waiting, bound) == (
        "Fields: print(doc(Done, NeedInput, Waiting))"
    )
    assert return_doc_hint(Done, bound) == "Fields: print(doc(Done))"
    assert return_doc_hint(Scored, bound) == "Fields: print(doc(Scored))"
    # One member unbound, or bound to something else: no line.
    assert return_doc_hint(Done | NeedInput, {"Done": Done}) is None
    assert return_doc_hint(Done, {"Done": object()}) is None
    assert return_doc_hint(Done, None) is None
    assert return_doc_hint(int, {"int": int}) is None  # not a model
    assert return_doc_hint(Row, {"Row": Row}) is None


def test_alias_note_names_the_value_taken_and_is_bounded():
    from nooa.strategies.codeact_errors import return_alias_note

    assert return_alias_note("python_cell", 42) == (
        "return_result isn't a tool. It's a function you call inside python_cell; "
        "this call ran as return_result(42)"
    )
    long = return_alias_note("python_cell", "x" * 1000)
    assert len(long) < 320 and long.endswith("…)")


def test_alias_invalid_intro_says_how_the_call_ran_and_is_bounded():
    from nooa.strategies.codeact_errors import return_alias_invalid_intro

    assert return_alias_invalid_intro("python_cell", "hello") == (
        "return_result isn't a tool. It's a function you call inside python_cell. "
        "I ran your call as python_cell code return_result('hello'), and that value isn't "
        "a valid result:"
    )
    assert "taken as" not in return_alias_invalid_intro("python_cell", "hello")
    assert len(return_alias_invalid_intro("python_cell", "x" * 1000)) < 400
