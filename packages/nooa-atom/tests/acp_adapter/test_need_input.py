# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard ACP form schemas for every question type."""

import pytest
from nooa_atom.acp.need_input import need_input_schema
from pydantic import ValidationError

from nooa.interactive import (
    BooleanQuestion,
    FormChoice,
    NeedInput,
    NeedInputForm,
    NumberQuestion,
    PickManyQuestion,
    PickOneQuestion,
    TextQuestion,
)

CHOICES = [
    FormChoice(value="main", title="Main branch"),
    FormChoice(value="dev", title="Development"),
]
OPTIONS = [{"const": "main", "title": "Main branch"}, {"const": "dev", "title": "Development"}]


def form():
    return NeedInputForm(
        heading="Release",
        reason="Safety",
        questions=[
            TextQuestion(id="notes", label="Notes?", help="Free text", required=False),
            PickOneQuestion(
                id="branch",
                label="Branch?",
                help="Choose carefully",
                choices=CHOICES,
                allow_other=False,
            ),
            PickOneQuestion(id="target", label="Target?", choices=CHOICES),
        ],
    )


def _schema(need):
    return need_input_schema(need).model_dump(mode="json", by_alias=True, exclude_none=True)


def test_standard_single_enum_and_explicit_choice_or_text_fallback():
    schema = _schema(form())
    assert schema["type"] == "object"
    assert list(schema["properties"]) == ["notes", "branch", "target"]
    assert schema["required"] == ["branch", "target"]
    assert schema["properties"]["branch"]["oneOf"] == OPTIONS
    assert "enum" not in schema["properties"]["branch"]
    assert "oneOf" not in schema["properties"]["target"]
    assert "main, dev" in schema["properties"]["target"]["description"]
    assert not any(key in str(schema) for key in ("default", "minimum", "maximum", "minLength"))
    assert need_input_schema(NeedInput(question="Why?")) is None


def test_standard_schema_for_every_type_and_constraint():
    need = NeedInputForm(
        heading="All",
        questions=[
            TextQuestion(
                id="name",
                label="Name?",
                min_length=2,
                max_length=8,
                pattern="[a-z]+",
                default="abc",
            ),
            TextQuestion(id="email", label="Email?", format="email", required=False),
            PickOneQuestion(
                id="one", label="One?", choices=CHOICES, allow_other=False, default="dev"
            ),
            PickOneQuestion(id="other", label="Other?", choices=CHOICES, default="custom"),
            PickManyQuestion(
                id="many",
                label="Many?",
                choices=CHOICES,
                min_items=1,
                max_items=2,
                default=["main"],
            ),
            NumberQuestion(id="ratio", label="Ratio?", minimum=0, maximum=1, default=0.5),
            NumberQuestion(
                whole=True, id="count", label="Count?", minimum=1, maximum=10, default=3
            ),
            BooleanQuestion(id="dry", label="Dry run?", help="No changes", default=True),
        ],
    )
    props = _schema(need)["properties"]
    assert props == {
        "name": {
            "type": "string",
            "title": "Name?",
            "minLength": 2,
            "maxLength": 8,
            "pattern": "[a-z]+",
            "default": "abc",
        },
        "email": {"type": "string", "title": "Email?", "format": "email"},
        "one": {"type": "string", "title": "One?", "oneOf": OPTIONS, "default": "dev"},
        "other": {
            "type": "string",
            "title": "Other?",
            "description": "For example: main, dev.",
            "default": "custom",
        },
        "many": {
            "type": "array",
            "title": "Many?",
            "items": {"anyOf": OPTIONS},
            "minItems": 1,
            "maxItems": 2,
            "default": ["main"],
        },
        "ratio": {"type": "number", "title": "Ratio?", "minimum": 0, "maximum": 1, "default": 0.5},
        "count": {"type": "integer", "title": "Count?", "minimum": 1, "maximum": 10, "default": 3},
        "dry": {
            "type": "boolean",
            "title": "Dry run?",
            "description": "No changes",
            "default": True,
        },
    }
    assert _schema(need)["required"] == ["name", "one", "other", "many", "ratio", "count", "dry"]


@pytest.mark.parametrize(
    "questions",
    [
        [],
        [{"kind": "multi_select", "id": "x", "label": "X"}],
        [{"kind": "number", "whole": True, "id": "x", "label": "X", "minimum": 5, "maximum": 1}],
        [{"kind": "text", "id": "x", "label": "X", "default": 3}],
        [{"kind": "text", "id": 1, "label": "X"}],
        [{"kind": "text", "id": "x", "label": "X"}, {"kind": "text", "id": "x", "label": "Y"}],
        [{"kind": "pick_one", "id": "x", "label": "X", "choices": []}],
        [
            {
                "kind": "pick_one",
                "id": "x",
                "label": "X",
                "choices": [{"value": "a", "title": "A"}, {"value": "a", "title": "Other"}],
            }
        ],
    ],
)
def test_invalid_descriptor_types_and_duplicates(questions):
    with pytest.raises(ValidationError):
        NeedInputForm(heading="H", questions=questions)


def test_descriptor_json_roundtrip():
    assert NeedInputForm.model_validate_json(form().model_dump_json()) == form()


def test_pick_one_allows_other_by_default():
    question = PickOneQuestion(id="x", label="X?", choices=[FormChoice(value="a", title="A")])
    assert question.allow_other is True
