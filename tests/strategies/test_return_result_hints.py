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
