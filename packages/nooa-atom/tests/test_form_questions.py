# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Form questions: constraints, defaults, strict checked answers and text rendering."""

import pytest
from pydantic import ValidationError

from nooa.interactive import (
    BooleanQuestion,
    FormChoice,
    FormResponse,
    IntegerQuestion,
    NeedInputForm,
    NumberQuestion,
    PickManyQuestion,
    PickOneQuestion,
    TextQuestion,
    recorded_form_text,
    recorded_item_text,
)

CHOICES = [FormChoice(value="a", title="Alpha"), FormChoice(value="b", title="Beta")]


def _check(question, value):
    need = NeedInputForm(heading="H", questions=[question])
    content = {} if value is None else {question.id: value}
    return need.validate_response(FormResponse(action="accept", content=content)).content[
        question.id
    ]


VALID = [
    (TextQuestion(id="q", label="Q", min_length=2, max_length=3), "ab", "ab"),
    (TextQuestion(id="q", label="Q", pattern="[a-z]+"), "abc", "abc"),
    (TextQuestion(id="q", label="Q", format="email"), "me@x.org", "me@x.org"),
    (TextQuestion(id="q", label="Q", format="uri"), "https://x.org/a", "https://x.org/a"),
    (TextQuestion(id="q", label="Q", format="date"), "2026-01-31", "2026-01-31"),
    (TextQuestion(id="q", label="Q", format="date-time"), "2026-01-31T10:00Z", "2026-01-31T10:00Z"),
    (PickOneQuestion(id="q", label="Q", choices=CHOICES, allow_other=False), "a", "a"),
    (PickOneQuestion(id="q", label="Q", choices=CHOICES), "typed", "typed"),
    (PickManyQuestion(id="q", label="Q", choices=CHOICES, max_items=2), ["b", "a"], ["b", "a"]),
    (NumberQuestion(id="q", label="Q", minimum=0, maximum=1), 1, 1.0),
    (NumberQuestion(id="q", label="Q"), 0.25, 0.25),
    (IntegerQuestion(id="q", label="Q", minimum=1, maximum=10), 10, 10),
    (IntegerQuestion(id="q", label="Q"), 10**400, 10**400),
    (BooleanQuestion(id="q", label="Q"), False, False),
]


@pytest.mark.parametrize("question,value,expected", VALID)
def test_valid_answers(question, value, expected):
    answer = _check(question, value)
    assert answer == expected and type(answer) is type(expected)


INVALID = [
    (TextQuestion(id="q", label="Q", min_length=2), "a", "q: enter at least 2 characters"),
    (TextQuestion(id="q", label="Q", max_length=3), "abcd", "q: enter at most 3 characters"),
    (TextQuestion(id="q", label="Q", pattern="[a-z]+"), "ab1", "q: enter text matching [a-z]+"),
    (TextQuestion(id="q", label="Q", format="email"), "me", "q: enter an email address"),
    (TextQuestion(id="q", label="Q", format="uri"), "not a uri", "q: enter a URI"),
    (TextQuestion(id="q", label="Q", format="date"), "2026-02-30", "q: enter a date"),
    (TextQuestion(id="q", label="Q", format="date-time"), "2026-01-31", "q: enter a date and"),
    (TextQuestion(id="q", label="Q"), 3, "q: enter text"),
    (
        PickOneQuestion(id="q", label="Q", choices=CHOICES, allow_other=False),
        "Alpha",
        "q: choose a listed option",
    ),
    (
        PickManyQuestion(id="q", label="Q", choices=CHOICES),
        ["a", "z"],
        "q: choose only listed options",
    ),
    (
        PickManyQuestion(id="q", label="Q", choices=CHOICES),
        ["a", "a"],
        "q: choose each option at most once",
    ),
    (PickManyQuestion(id="q", label="Q", choices=CHOICES), "a", "q: choose from the list"),
    (
        PickManyQuestion(id="q", label="Q", choices=CHOICES, min_items=2),
        ["a"],
        "q: choose at least 2 options",
    ),
    (
        PickManyQuestion(id="q", label="Q", choices=CHOICES, max_items=1),
        ["a", "b"],
        "q: choose at most 1 option",
    ),
    (NumberQuestion(id="q", label="Q", minimum=0, maximum=1), 1.5, "q: enter a number from 0 to 1"),
    (NumberQuestion(id="q", label="Q"), True, "q: enter a number"),
    (NumberQuestion(id="q", label="Q"), "0.5", "q: enter a number"),
    (NumberQuestion(id="q", label="Q"), float("nan"), "q: enter a number"),
    (NumberQuestion(id="q", label="Q"), 10**400, "q: enter a number"),
    (
        IntegerQuestion(id="q", label="Q", minimum=1, maximum=10),
        10**400,
        "q: enter a whole number from 1 to 10",
    ),
    (
        IntegerQuestion(id="q", label="Q", minimum=1, maximum=10),
        11,
        "q: enter a whole number from 1 to 10",
    ),
    (IntegerQuestion(id="q", label="Q"), 2.0, "q: enter a whole number"),
    (IntegerQuestion(id="q", label="Q"), "3", "q: enter a whole number"),
    (IntegerQuestion(id="q", label="Q"), True, "q: enter a whole number"),
    (BooleanQuestion(id="q", label="Q"), "true", "q: answer yes or no"),
    (BooleanQuestion(id="q", label="Q"), 1, "q: answer yes or no"),
]


@pytest.mark.parametrize("question,value,message", INVALID)
def test_invalid_answers_strict_no_coercion(question, value, message):
    with pytest.raises(
        ValueError, match=message.replace("+", r"\+").replace("[", r"\[").replace("]", r"\]")
    ):
        _check(question, value)


@pytest.mark.parametrize(
    "question",
    [
        PickOneQuestion(id="q", label="Q", choices=CHOICES, required=False),
        PickManyQuestion(id="q", label="Q", choices=CHOICES, required=False),
        NumberQuestion(id="q", label="Q", required=False),
        IntegerQuestion(id="q", label="Q", required=False),
        BooleanQuestion(id="q", label="Q", required=False),
    ],
)
@pytest.mark.parametrize("value", [None, "", []])
def test_optional_non_text_left_empty_is_none(question, value):
    assert _check(question, value) is None


def test_optional_text_left_empty_is_empty_string_and_required_is_enforced():
    assert _check(TextQuestion(id="q", label="Q", required=False), None) == ""
    assert _check(TextQuestion(id="q", label="Q", required=False, min_length=3), " ") == ""
    with pytest.raises(ValueError, match="q: an answer is required"):
        _check(BooleanQuestion(id="q", label="Q"), None)


def test_every_failing_field_is_reported_once():
    need = NeedInputForm(
        heading="H",
        questions=[
            IntegerQuestion(id="replicas", label="R", minimum=1, maximum=10),
            TextQuestion(id="email", label="E", format="email"),
            BooleanQuestion(id="dry", label="D"),
            TextQuestion(id="ok", label="O"),
        ],
    )
    with pytest.raises(ValueError) as error:
        need.validate_response(
            FormResponse(
                action="accept", content={"replicas": 0, "email": "x", "ok": "y", "zz": "1"}
            )
        )
    assert str(error.value) == (
        "unknown answer ids ['zz']; replicas: enter a whole number from 1 to 10; "
        "email: enter an email address; dry: an answer is required"
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: TextQuestion(id="q", label="Q", min_length=3, max_length=2),
        lambda: TextQuestion(id="q", label="Q", pattern="("),
        lambda: TextQuestion(id="q", label="Q", format="phone"),
        lambda: TextQuestion(id="q", label="Q", max_length=2, default="abc"),
        lambda: TextQuestion(id="q", label="Q", format="email", default="nope"),
        lambda: PickOneQuestion(id="q", label="Q", choices=CHOICES, allow_other=False, default="z"),
        lambda: PickManyQuestion(id="q", label="Q", choices=CHOICES, min_items=3),
        lambda: PickManyQuestion(id="q", label="Q", choices=CHOICES, min_items=2, max_items=1),
        lambda: PickManyQuestion(id="q", label="Q", choices=CHOICES, default=["z"]),
        lambda: NumberQuestion(id="q", label="Q", minimum=2, maximum=1),
        lambda: NumberQuestion(id="q", label="Q", maximum=1, default=1.5),
        lambda: IntegerQuestion(id="q", label="Q", minimum=1, default=0),
        lambda: IntegerQuestion(id="q", label="Q", default=True),
        lambda: BooleanQuestion(id="q", label="Q", default="yes"),
    ],
)
def test_bad_constraints_and_defaults_fail_at_construction(build):
    with pytest.raises(ValidationError):
        build()


def test_valid_defaults_and_optional_types_construct():
    PickOneQuestion(id="q", label="Q", choices=CHOICES, default="typed")
    PickManyQuestion(id="q", label="Q", choices=CHOICES, default=["a"], required=False)
    NumberQuestion(id="q", label="Q", minimum=0, default=0)
    BooleanQuestion(id="q", label="Q", default=False, required=False)


def test_as_text_renders_hints():
    need = NeedInputForm(
        heading="Deploy",
        reason="Why",
        questions=[
            TextQuestion(id="name", label="Name?", help="Short", min_length=2, max_length=5),
            TextQuestion(id="mail", label="Mail?", format="email", default="a@b.co"),
            PickOneQuestion(id="one", label="One?", choices=CHOICES, default="b"),
            PickManyQuestion(id="many", label="Many?", choices=CHOICES, min_items=1),
            NumberQuestion(id="r", label="Ratio?", minimum=0, maximum=0.5),
            IntegerQuestion(id="n", label="Count?", minimum=1),
            BooleanQuestion(id="dry", label="Dry run?", default=True),
        ],
    )
    assert need.as_text() == (
        "Deploy\n\nWhy\n\n"
        "- Name? (Short): 2 to 5 characters\n"
        "- Mail?: an email address; default: a@b.co\n"
        "- One?: Alpha, Beta; default: Beta\n"
        "- Many?: choose any of: Alpha, Beta; at least 1 option\n"
        "- Ratio?: a number from 0 to 0.5\n"
        "- Count?: a whole number of at least 1\n"
        "- Dry run?: yes or no; default: yes"
    )


def test_response_as_text():
    response = FormResponse(
        action="accept",
        content={"a": ["x", "y"], "b": None, "c": True, "d": 2.0, "e": 0.5, "f": "", "g": 3},
    )
    assert response.as_text() == "a: x, y\nc: yes\nd: 2\ne: 0.5\ng: 3"
    assert FormResponse(action="accept", content={"b": None}).as_text() == "Submitted the form."
    assert FormResponse(action="decline").as_text() == "Declined the form."


def test_typed_content_roundtrips_json():
    response = FormResponse(
        action="accept", content={"a": ["x"], "b": None, "c": False, "d": 1.5, "e": 2, "f": "s"}
    )
    assert FormResponse.model_validate_json(response.model_dump_json()) == response


def test_legacy_records_still_load():
    record = {
        "request_id": "r1",
        "kind": "form",
        "heading": "H",
        "questions": [
            {
                "kind": "pick_one",
                "id": "s",
                "label": "S",
                "choices": [{"value": "a", "title": "A"}],
            },
            {
                "kind": "pick_one_or_text",
                "id": "o",
                "label": "O",
                "choices": [{"value": "a", "title": "A"}],
            },
            {"kind": "text", "id": "t", "label": "T", "required": False},
            {"kind": "integer", "id": "n", "label": "N", "minimum": 1},
            {"kind": "boolean", "id": "b", "label": "B"},
        ],
    }
    assert recorded_form_text(record) == (
        "H\n\n- S: A\n- O: A\n- T\n- N: a whole number of at least 1\n- B: yes or no"
    )
    item = FormResponse(action="accept", content={"n": 3, "b": False}).model_dump_json()
    assert recorded_item_text(item, "nooa.interactive:FormResponse") == "n: 3\nb: no"
