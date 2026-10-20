# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit descriptors and honestly supported transport mappings."""

import pytest
from nooa_atom.acp.need_input import (
    answer_from_content,
    need_input_schema,
    pool_form_schema,
)
from pydantic import ValidationError

from nooa.interactive import (
    FormChoice,
    FormResponse,
    NeedInput,
    NeedInputForm,
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
            PickOneQuestion(
                id="branch",
                label="Branch?",
                help="Choose carefully",
                choices=choices,
                allow_other=False,
            ),
            PickOneQuestion(id="target", label="Target?", choices=choices),
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
    assert schema["properties"]["branch"]["oneOf"] == [
        {"const": "main", "title": "Main branch"},
        {"const": "dev", "title": "Development"},
    ]
    assert "enum" not in schema["properties"]["branch"]
    assert "oneOf" not in schema["properties"]["target"]
    assert "main, dev" in schema["properties"]["target"]["description"]
    assert not any(key in str(schema) for key in ("default", "minimum", "maximum", "minLength"))
    assert need_input_schema(NeedInput(question="Why?")) is None
    assert pool_form_schema(NeedInput(question="Why?")) is None


def test_string_response_no_conversion_and_explicit_optional_blank():
    mapper = answer_from_content
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
        answer_from_content(form(), content)


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_actions(action):
    assert form().validate_response(FormResponse(action=action)) == FormResponse(action=action)


@pytest.mark.parametrize(
    "questions",
    [
        [],
        [{"kind": "multi_select", "id": "x", "label": "X"}],
        [{"kind": "integer", "id": "x", "label": "X", "minimum": 5, "maximum": 1}],
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


@pytest.mark.parametrize("allow_other", [True, False])
def test_optional_picker_rejected_with_explicit_text_alternative(allow_other):
    with pytest.raises(ValidationError, match="Only text questions can be optional"):
        PickOneQuestion(
            allow_other=allow_other,
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
        assert answer_from_content(need, content) == {"choice": ""}


def test_pick_one_allows_other_by_default():
    question = PickOneQuestion(id="x", label="X?", choices=[FormChoice(value="a", title="A")])
    assert question.allow_other is True
    need = NeedInputForm(heading="H", questions=[question])
    assert answer_from_content(need, {"x": "typed"}) == {"x": "typed"}
    strict = NeedInputForm(
        heading="H", questions=[question.model_copy(update={"allow_other": False})]
    )
    with pytest.raises(ValueError, match="choose a listed value"):
        answer_from_content(strict, {"x": "typed"})


def test_legacy_question_records_migrate():
    from nooa_atom.session.loader import migrate_input_request

    choices = [{"value": "a", "title": "A"}]
    record = migrate_input_request(
        {
            "kind": "form",
            "heading": "H",
            "questions": [
                {"kind": "pick_one", "id": "strict", "label": "S", "choices": choices},
                {"kind": "pick_one_or_text", "id": "other", "label": "O", "choices": choices},
            ],
        },
        form=True,
    )
    need = NeedInputForm.model_validate(record)
    assert [q.allow_other for q in need.questions] == [False, True]
    question = migrate_input_request({"question": "Which?", "options": ["main", "dev"]})
    assert NeedInput.model_validate(question).question == "Which?\n\n- main\n- dev"


def test_integer_and_boolean_questions_map_to_typed_schemas_and_answers():
    from nooa.interactive import BooleanQuestion, IntegerQuestion

    need = NeedInputForm(
        heading="Deploy",
        questions=[
            IntegerQuestion(id="replicas", label="Replicas?", minimum=1, maximum=10),
            BooleanQuestion(id="dry_run", label="Dry run?", help="No changes"),
        ],
    )
    standard = need_input_schema(need).model_dump(mode="json", by_alias=True, exclude_none=True)
    assert standard["properties"] == {
        "replicas": {"type": "integer", "title": "Replicas?", "minimum": 1, "maximum": 10},
        "dry_run": {"type": "boolean", "title": "Dry run?", "description": "No changes"},
    }
    assert pool_form_schema(need)["properties"] == {
        "replicas": {"title": "Replicas?", "type": "integer", "minimum": 1, "maximum": 10},
        "dry_run": {"title": "Dry run?", "description": "No changes", "type": "boolean"},
    }
    # Typed answers, and text-box strings converted for these questions only.
    assert answer_from_content(need, {"replicas": 3, "dry_run": True}) == {
        "replicas": 3,
        "dry_run": True,
    }
    assert answer_from_content(need, {"replicas": " 3 ", "dry_run": "false"}) == {
        "replicas": 3,
        "dry_run": False,
    }
    for content in (
        {"replicas": 11, "dry_run": True},
        {"replicas": 0, "dry_run": True},
        {"replicas": True, "dry_run": True},
        {"replicas": "3.5", "dry_run": True},
        {"replicas": 3, "dry_run": "yes"},
        {"replicas": 3, "dry_run": 1},
        {"replicas": 3},
    ):
        with pytest.raises(ValueError):
            answer_from_content(need, content)


def test_text_and_picker_answers_must_be_strings():
    with pytest.raises(ValueError, match="expected text"):
        answer_from_content(form(), {"branch": "dev", "target": 3})


@pytest.mark.parametrize("kind", ["integer", "boolean"])
def test_typed_questions_must_be_required(kind):
    from nooa.interactive import BooleanQuestion, IntegerQuestion

    model = IntegerQuestion if kind == "integer" else BooleanQuestion
    with pytest.raises(ValidationError, match="Only text questions can be optional"):
        model(id="x", label="X?", required=False)
