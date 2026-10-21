# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool adapter: downgraded schemas for every question type ."""

from nooa_atom.acp.pool import POOL_FORM_METHOD, pool_form_params, pool_form_schema

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
ONE_OF = {
    "oneOf": [{"const": "main", "title": "Main branch"}, {"const": "dev", "title": "Development"}]
}
YES_NO = [{"const": "true", "title": "Yes"}, {"const": "false", "title": "No"}]
SENT_KEYS = {"title", "description", "type", "oneOf"}


def _props(*questions):
    schema = pool_form_schema(NeedInputForm(heading="H", questions=list(questions)))
    for prop in schema["properties"].values():
        assert set(prop) <= SENT_KEYS
        assert prop.get("type", "string") == "string"
    return schema["properties"]


def test_tested_shapes_and_required():
    need = NeedInputForm(
        heading="Release",
        questions=[
            TextQuestion(id="notes", label="Notes?", help="Free text", required=False),
            PickOneQuestion(
                id="branch", label="Branch?", help="Choose", choices=CHOICES, allow_other=False
            ),
            PickOneQuestion(id="target", label="Target?", choices=CHOICES),
        ],
    )
    assert pool_form_schema(need) == {
        "type": "object",
        "required": ["branch", "target"],
        "properties": {
            "notes": {"type": "string", "title": "Notes?", "description": "Free text"},
            "branch": {"type": "string", "title": "Branch?", "description": "Choose", **ONE_OF},
            "target": {
                "type": "string",
                "title": "Target?",
                "description": "For example: Main branch, Development.",
            },
        },
    }
    assert pool_form_schema(NeedInput(question="Why?")) is None


def test_params_carry_message_and_field_order():
    need = NeedInputForm(
        heading="H",
        reason="R",
        questions=[TextQuestion(id="a", label="A"), TextQuestion(id="b", label="B")],
    )
    params = pool_form_params("sid", need)
    assert POOL_FORM_METHOD == "poolside/elicitation"
    assert params == {
        "sessionId": "sid",
        "mode": "form",
        "message": "H\n\nR",
        "requestedSchema": pool_form_schema(need),
        "_meta": {"poolside/field_order": ["a", "b"]},
    }
    single = NeedInputForm(heading="H", questions=[TextQuestion(id="a", label="A")])
    assert "_meta" not in pool_form_params("sid", single)


def test_text_constraints_and_default_become_hints():
    props = _props(
        TextQuestion(
            id="name", label="Name?", help="Yours", min_length=2, max_length=5, pattern="[A-Z]+"
        ),
        TextQuestion(id="mail", label="Mail?", format="email", default="a@b.co", required=False),
        TextQuestion(id="day", label="Day?", format="date"),
    )
    assert props["name"]["description"] == "Yours\n2 to 5 characters; matching [A-Z]+."
    assert props["mail"]["description"] == "An email address.\nDefault: a@b.co."
    assert props["day"]["description"] == "A date (YYYY-MM-DD)."


def test_numbers_become_strings_with_a_range():
    props = _props(
        NumberQuestion(whole=True, id="n", label="N?", minimum=1, maximum=10, default=3),
        NumberQuestion(whole=True, id="m", label="M?", minimum=0),
        NumberQuestion(id="r", label="R?", minimum=0, maximum=1),
        NumberQuestion(id="x", label="X?", required=False),
    )
    assert props["n"] == {
        "title": "N?",
        "type": "string",
        "description": "A whole number from 1 to 10.\nDefault: 3.",
    }
    assert props["m"]["description"] == "A whole number of at least 0."
    assert props["r"]["description"] == "A number from 0 to 1."
    assert props["x"]["description"] == "A number."


def test_booleans_required_picker_optional_text():
    props = _props(
        BooleanQuestion(id="a", label="A?"),
        BooleanQuestion(id="b", label="B?", required=False, default=False),
    )
    assert props["a"] == {"title": "A?", "type": "string", "oneOf": YES_NO}
    assert props["b"] == {
        "title": "B?",
        "type": "string",
        "description": "Yes or no.\nDefault: no.",
    }


def test_optional_pickers_and_pick_many_become_strings():
    props = _props(
        PickOneQuestion(id="s", label="S?", choices=CHOICES, allow_other=False, required=False),
        PickOneQuestion(id="o", label="O?", choices=CHOICES, required=False, default="dev"),
        PickManyQuestion(id="m", label="M?", choices=CHOICES, max_items=1),
        PickManyQuestion(id="k", label="K?", choices=CHOICES, default=["main", "dev"]),
    )
    assert props["s"] == {
        "title": "S?",
        "type": "string",
        "description": "One of: Main branch, Development.",
    }
    assert props["o"]["description"] == (
        "For example: Main branch, Development.\nDefault: Development."
    )
    assert props["m"]["description"] == (
        "Comma-separated; choose from: Main branch, Development; at most 1 option."
    )
    assert props["k"]["description"] == (
        "Comma-separated; choose from: Main branch, Development.\n"
        "Default: Main branch, Development."
    )


def test_required_picker_default_is_a_hint():
    props = _props(
        PickOneQuestion(id="s", label="S?", choices=CHOICES, allow_other=False, default="dev")
    )
    assert props["s"] == {
        "title": "S?",
        "type": "string",
        **ONE_OF,
        "description": "Default: Development.",
    }


ALL = NeedInputForm(
    heading="H",
    questions=[
        TextQuestion(id="t", label="T", required=False),
        PickOneQuestion(id="p", label="P", choices=CHOICES, allow_other=False, required=False),
        PickManyQuestion(id="m", label="M", choices=CHOICES, required=False),
        NumberQuestion(id="r", label="R", required=False),
        NumberQuestion(whole=True, id="i", label="I", required=False, default=7),
        BooleanQuestion(id="b", label="B", required=False),
    ],
)
