# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Form questions: constraints, defaults and text rendering."""

import pytest
from nooa_atom.session.forms import (
    BooleanQuestion,
    FormChoice,
    FormResponse,
    NeedInputForm,
    NumberQuestion,
    PickManyQuestion,
    PickOneQuestion,
    TextQuestion,
)
from pydantic import ValidationError

CHOICES = [FormChoice(value="a", title="Alpha"), FormChoice(value="b", title="Beta")]


@pytest.mark.parametrize(
    "build",
    [
        lambda: TextQuestion(id="q", label="Q", min_length=3, max_length=2),
        lambda: TextQuestion(id="q", label="Q", pattern="("),
        lambda: TextQuestion(id="q", label="Q", format="phone"),
        lambda: PickManyQuestion(id="q", label="Q", choices=CHOICES, min_items=3),
        lambda: PickManyQuestion(id="q", label="Q", choices=CHOICES, min_items=2, max_items=1),
        lambda: NumberQuestion(id="q", label="Q", minimum=2, maximum=1),
        lambda: NumberQuestion(whole=True, id="q", label="Q", default=True),
        lambda: BooleanQuestion(id="q", label="Q", default="yes"),
    ],
)
def test_bad_constraints_and_default_types_fail_at_construction(build):
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
            NumberQuestion(whole=True, id="n", label="Count?", minimum=1),
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
