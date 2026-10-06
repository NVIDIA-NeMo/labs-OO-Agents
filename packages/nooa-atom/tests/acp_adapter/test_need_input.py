# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""need_input_schema: a NeedInput question as a flat ACP elicitation form."""

from typing import Annotated, Literal

import pytest
from nooa_atom.acp.need_input import (
    answer_from_content,
    need_input_schema,
    pool_answer,
    pool_form_schema,
)
from pydantic import BaseModel, Field, ValidationError

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


# ---- Pool forms: string properties only -------------------------------


def test_a_pool_free_text_question_is_one_string_property():
    need = NeedInput(question="Name the release?")
    assert pool_form_schema(need) == {
        "type": "object",
        "properties": {"answer": {"type": "string", "title": "Name the release?"}},
        "required": ["answer"],
    }
    assert pool_answer(need, {"answer": "Aurora"}) == "Aurora"


def test_a_pool_form_declares_every_field_as_a_string_and_says_the_expected_type():
    need = NeedInput(question="Deploy how?", answer_type=Deployment)
    assert pool_form_schema(need) == {
        "type": "object",
        "properties": {
            "target": {"type": "string", "title": "Target", "description": "Where to deploy"},
            "replicas": {"type": "string", "title": "Replicas", "description": "a whole number"},
            "ratio": {"type": "string", "title": "Ratio", "description": "a number"},
            "dry_run": {"type": "string", "title": "Dry Run", "description": "yes or no"},
            "note": {"type": "string", "title": "Note"},
            "regions": {
                "type": "string",
                "title": "Regions",
                "description": "a comma-separated list of: us, eu",
            },
            "stage": {
                "type": "string",
                "title": "Stage",
                "oneOf": [{"const": "dev", "title": "dev"}, {"const": "prod", "title": "prod"}],
            },
        },
        "required": ["target", "replicas", "dry_run"],
    }


def test_pool_answers_are_converted_from_strings():
    need = NeedInput(question="Deploy how?", answer_type=Deployment)
    answer = pool_answer(
        need,
        {
            "target": "staging",
            "replicas": " 2 ",
            "ratio": "",
            "dry_run": "yes",
            "note": "",
            "regions": "us, eu",
            "stage": "prod",
        },
    )
    assert answer == Deployment(
        target="staging", replicas=2, dry_run=True, regions=["us", "eu"], stage="prod"
    )


def test_a_pool_list_of_strings_is_a_comma_separated_list():
    need = NeedInput(question="Names?", answer_type=Strings)
    assert pool_form_schema(need)["properties"]["names"] == {
        "type": "string",
        "title": "Names",
        "description": "a comma-separated list",
    }
    assert pool_answer(need, {"names": "a,b , c"}) == Strings(names=["a", "b", "c"])


def test_a_pool_answer_that_does_not_convert_raises():
    need = NeedInput(question="Deploy how?", answer_type=Deployment)
    with pytest.raises(ValidationError):
        pool_answer(need, {"target": "x", "replicas": "two", "dry_run": "yes"})


@pytest.mark.parametrize("answer_type", [Nested, Mapping, Either])
def test_a_pool_form_needs_simple_fields(answer_type):
    assert pool_form_schema(NeedInput(question="?", answer_type=answer_type)) is None


def test_a_pool_choice_question_is_a_picker_or_free_text():
    need = NeedInput(question="Which branch?", options=["main", "Dev"])
    assert pool_form_schema(need) == {
        "type": "object",
        "properties": {
            "answer": {
                "description": "Which branch?",
                "anyOf": [
                    {
                        "oneOf": [
                            {"const": "main", "title": "main"},
                            {"const": "Dev", "title": "Dev"},
                        ]
                    },
                    {"type": "string"},
                ],
            }
        },
        "required": ["answer"],
    }
    assert pool_answer(need, {"answer": " dev "}) == "Dev"
    assert pool_answer(need, {"answer": "MAIN"}) == "main"


@pytest.mark.parametrize("text", ["release", "  Release Candidate  ", "Ship\nwith tests", "0"])
def test_a_pool_alternative_is_preserved(text):
    need = NeedInput(question="Which branch?", options=["main", "dev"])
    assert pool_answer(need, {"answer": text}) == text


@pytest.mark.parametrize(
    "content",
    [
        None,
        {},
        {"answer": ""},
        {"answer": " \t\n "},
        {"answer": None},
        {"answer": 1},
        {"answer": False},
        {"answer": []},
        {"answer": {"text": "release"}},
        "release",
        1,
        [],
        [["answer", "release"]],
    ],
)
def test_a_pool_options_answer_must_be_a_nonblank_string_in_an_object(content):
    need = NeedInput(question="Which branch?", options=["main", "dev"])
    with pytest.raises(ValueError):
        pool_answer(need, content)


def test_a_yes_no_question_has_no_pool_form():
    assert pool_form_schema(NeedInput(question="Delete it?", options=["Yes", "No"])) is None


class Release(BaseModel):
    channel: Literal["stable", "beta"] = Field(description="Where it goes")
    name: Literal["Aurora", "Borealis"] | str = Field(description="What to call it")
    note: Literal["urgent"] | str | None = None


def test_a_pool_literal_field_is_a_picker_and_literal_or_str_adds_free_text():
    need = NeedInput(question="Release how?", answer_type=Release)
    assert pool_form_schema(need) == {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "title": "Channel",
                "description": "Where it goes",
                "oneOf": [
                    {"const": "stable", "title": "stable"},
                    {"const": "beta", "title": "beta"},
                ],
            },
            "name": {
                "title": "Name",
                "description": "What to call it",
                "anyOf": [
                    {
                        "oneOf": [
                            {"const": "Aurora", "title": "Aurora"},
                            {"const": "Borealis", "title": "Borealis"},
                        ]
                    },
                    {"type": "string"},
                ],
            },
            "note": {
                "title": "Note",
                "anyOf": [{"oneOf": [{"const": "urgent", "title": "urgent"}]}, {"type": "string"}],
            },
        },
        "required": ["channel", "name"],
    }
    answer = pool_answer(need, {"channel": "Beta", "name": "Vega", "note": ""})
    assert answer == Release(channel="beta", name="Vega")
    assert pool_answer(need, {"channel": "stable", "name": "Aurora"}).name == "Aurora"


def test_a_pool_typed_literal_remains_strict():
    need = NeedInput(question="Release how?", answer_type=Release)
    with pytest.raises(ValidationError):
        pool_answer(need, {"channel": "nightly", "name": "Vega"})
