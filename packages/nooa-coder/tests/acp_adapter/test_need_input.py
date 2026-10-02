# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""need_input_schema: a NeedInput question as a flat ACP elicitation form."""

from typing import Annotated, Literal

import pytest
from nooa_coder.acp.need_input import answer_from_content, need_input_schema
from pydantic import BaseModel, Field

from nooa.interactive import NeedInput


def _json(need: NeedInput):
    schema = need_input_schema(need)
    return (
        None if schema is None else schema.model_dump(mode="json", by_alias=True, exclude_none=True)
    )


def test_options_become_one_enum_property():
    need = NeedInput(question="Which branch?", options=["main", "dev"])
    assert _json(need) == {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "title": "Which branch?", "enum": ["main", "dev"]}
        },
        "required": ["answer"],
    }
    assert answer_from_content(need, {"answer": "dev"}) == "dev"


def test_a_free_text_question_is_one_string_property():
    need = NeedInput(question="What should the title be?")
    assert _json(need) == {
        "type": "object",
        "properties": {"answer": {"type": "string", "title": "What should the title be?"}},
        "required": ["answer"],
    }
    assert answer_from_content(need, {"answer": "Parser"}) == "Parser"


class Deployment(BaseModel):
    target: str = Field(description="Where to deploy")
    replicas: int
    ratio: float = 0.5
    dry_run: bool
    note: str | None = None
    regions: list[Literal["us", "eu"]] = Field(default_factory=list)
    stage: Literal["dev", "prod"] = "dev"


def test_a_flat_answer_type_becomes_one_property_per_field():
    need = NeedInput(question="Deploy how?", answer_type=Deployment)
    assert _json(need) == {
        "type": "object",
        "title": "Deploy how?",
        "properties": {
            "target": {"type": "string", "title": "Target", "description": "Where to deploy"},
            "replicas": {"type": "integer", "title": "Replicas"},
            "ratio": {"type": "number", "title": "Ratio", "default": 0.5},
            "dry_run": {"type": "boolean", "title": "Dry Run"},
            "note": {"type": "string", "title": "Note"},
            "regions": {
                "type": "array",
                "title": "Regions",
                "items": {"type": "string", "enum": ["us", "eu"]},
            },
            "stage": {
                "type": "string",
                "title": "Stage",
                "enum": ["dev", "prod"],
                "default": "dev",
            },
        },
        "required": ["target", "replicas", "dry_run"],
    }
    answer = answer_from_content(
        need, {"target": "staging", "replicas": 2, "dry_run": True, "regions": ["eu"]}
    )
    assert answer == Deployment(target="staging", replicas=2, dry_run=True, regions=["eu"])


class Constrained(BaseModel):
    name: str
    note: Annotated[str, Field(max_length=200)] | None = None
    count: Annotated[int, Field(ge=1, le=5, gt=0)] = 1
    tags: Annotated[list[Literal["a", "b"]], Field(min_length=1)] | None = None


def test_annotated_fields_keep_their_type_and_the_constraints_a_form_can_express():
    """``Annotated[...] | None`` is an ordinary pydantic field; it must not sink the form."""
    need = NeedInput(question="Details?", answer_type=Constrained)
    assert _json(need) == {
        "type": "object",
        "title": "Details?",
        "properties": {
            "name": {"type": "string", "title": "Name"},
            "note": {"type": "string", "title": "Note", "maxLength": 200},
            # gt has no form equivalent and is left out; ge and le are kept.
            "count": {
                "type": "integer",
                "title": "Count",
                "default": 1,
                "minimum": 1,
                "maximum": 5,
            },
            "tags": {
                "type": "array",
                "title": "Tags",
                "items": {"type": "string", "enum": ["a", "b"]},
                "minItems": 1,
            },
        },
        "required": ["name"],
    }


class Tagged(BaseModel):
    tags: list[Literal["a", "b", "c"]] = ["a"]


def test_a_multi_select_keeps_its_default():
    schema = _json(NeedInput(question="Tags?", answer_type=Tagged))
    assert schema is not None
    assert schema["properties"]["tags"] == {
        "type": "array",
        "title": "Tags",
        "items": {"type": "string", "enum": ["a", "b", "c"]},
        "default": ["a"],
    }


class Nested(BaseModel):
    inner: Deployment


class Strings(BaseModel):
    names: list[str]


class Mapping(BaseModel):
    values: dict[str, int]


class Either(BaseModel):
    value: int | str


@pytest.mark.parametrize("answer_type", [Nested, Strings, Mapping, Either])
def test_anything_else_falls_back_to_free_text(answer_type):
    assert need_input_schema(NeedInput(question="?", answer_type=answer_type)) is None


def test_an_invalid_answer_for_a_type_is_kept_as_its_data():
    need = NeedInput(question="Deploy how?", answer_type=Deployment)
    assert answer_from_content(need, {"target": "x"}) == {"target": "x"}
