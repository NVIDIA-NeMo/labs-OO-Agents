# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit descriptors and honestly supported transport mappings."""

import pytest
from nooa_atom.acp.need_input import (
    answer_from_content,
    need_input_schema,
    pool_answer,
    pool_form_schema,
)
from pydantic import ValidationError

from nooa.interactive import (
    FormChoice,
    FormResponse,
    NeedInput,
    NeedInputForm,
    PickOneOrTextQuestion,
    PickOneQuestion,
    TextQuestion,
)


def form():
    choices = [
        FormChoice(value="main", title="Main branch"),
        FormChoice(value="dev", title="Development"),
    ]
    return NeedInputForm(
        heading="Release",
        reason="Safety",
        questions=[
            TextQuestion(id="notes", label="Notes?", help="Free text", required=False),
            PickOneQuestion(id="branch", label="Branch?", help="Choose carefully", choices=choices),
            PickOneOrTextQuestion(id="target", label="Target?", choices=choices),
        ],
    )


def test_pool_exact_tested_schema_and_distinct_labels():
    assert pool_form_schema(form()) == {
        "type": "object",
        "required": ["branch", "target"],
        "properties": {
            "notes": {"type": "string", "title": "Notes?", "description": "Free text"},
            "branch": {
                "type": "string",
                "title": "Branch?",
                "description": "Choose carefully",
                "oneOf": [
                    {"const": "main", "title": "Main branch"},
                    {"const": "dev", "title": "Development"},
                ],
            },
            "target": {
                "title": "Target?",
                "anyOf": [
                    {
                        "oneOf": [
                            {"const": "main", "title": "Main branch"},
                            {"const": "dev", "title": "Development"},
                        ]
                    },
                    {"type": "string"},
                ],
            },
        },
    }


def test_standard_single_enum_and_explicit_choice_or_text_fallback():
    schema = need_input_schema(form()).model_dump(mode="json", by_alias=True, exclude_none=True)
    assert list(schema["properties"]) == ["notes", "branch", "target"]
    assert schema["required"] == ["branch", "target"]
    assert schema["properties"]["branch"]["enum"] == ["main", "dev"]
    assert "Main branch: main" in schema["properties"]["branch"]["description"]
    assert "enum" not in schema["properties"]["target"]
    assert "suggestions" in schema["properties"]["target"]["description"]
    assert not any(key in str(schema) for key in ("default", "minimum", "maximum", "minLength"))
    assert need_input_schema(NeedInput(question="Why?")) is None
    assert pool_form_schema(NeedInput(question="Why?")) is None


@pytest.mark.parametrize("mapper", [pool_answer, answer_from_content])
def test_string_response_no_conversion_and_explicit_optional_blank(mapper):
    assert mapper(form(), {"branch": "dev", "target": "  custom  "}) == {
        "notes": "",
        "branch": "dev",
        "target": "  custom  ",
    }
    assert mapper(form(), {"notes": "  ", "branch": "main", "target": "2, yes"}) == {
        "notes": "",
        "branch": "main",
        "target": "2, yes",
    }


@pytest.mark.parametrize(
    "content",
    [
        None,
        {},
        {"branch": "DEV", "target": "x"},
        {"branch": "Development", "target": "x"},
        {"branch": " dev ", "target": "x"},
        {"branch": "dev", "target": True},
        {"branch": "dev", "target": ["x"]},
        {"branch": "dev", "target": "x", "unknown": "y"},
    ],
)
def test_invalid_protocol_never_converts(content):
    with pytest.raises(ValueError):
        pool_answer(form(), content)


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_actions(action):
    assert form().validate_response(FormResponse(action=action)) == FormResponse(action=action)


@pytest.mark.parametrize(
    "questions",
    [
        [],
        [{"kind": "integer", "id": "x", "label": "X"}],
        [{"kind": "text", "id": "x", "label": "X", "default": "pretend"}],
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


@pytest.mark.parametrize("kind", [PickOneQuestion, PickOneOrTextQuestion])
def test_optional_picker_rejected_with_explicit_text_alternative(kind):
    with pytest.raises(ValidationError, match="Optional picker questions are unsupported"):
        kind(
            id="choice",
            label="Choice?",
            required=False,
            choices=[FormChoice(value="a", title="A")],
        )
    # Explicit alternative, not an automatic change of picker meaning.
    need = NeedInputForm(
        heading="Optional text",
        questions=[TextQuestion(id="choice", label="Choice?", required=False, help="A: a")],
    )
    assert pool_form_schema(need)["required"] == []
    for content in ({}, {"choice": ""}, {"choice": "   "}):
        assert pool_answer(need, content) == {"choice": ""}
