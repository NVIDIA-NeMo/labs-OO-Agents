# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every restricted field/option, native validation and real SDK serialization."""

import time

import pytest
from nooa_atom.acp.need_input import need_input_schema, pool_form_schema, standard_form_response
from pydantic import ValidationError

from nooa.interactive import (
    BooleanQuestion,
    ChoiceQuestion,
    FormChoice,
    FormResponse,
    IntegerQuestion,
    MultiChoiceQuestion,
    NeedInput,
    NeedInputForm,
    NumberQuestion,
    TextQuestion,
)

CHOICES = [
    FormChoice(value="main", title="Main branch"),
    FormChoice(value="dev", title="Development"),
]


def form():
    return NeedInputForm(
        heading="Release",
        reason="Safety",
        questions=[
            TextQuestion(id="notes", label="Notes?", help="Free text", required=False),
            ChoiceQuestion(id="branch", label="Branch?", help="Choose carefully", choices=CHOICES),
            ChoiceQuestion(id="target", label="Target?", choices=CHOICES),
        ],
    )


def one(question):
    return NeedInputForm(heading="H", questions=[question])


def answer(need, content):
    return need.validate_response(FormResponse(action="accept", content=content)).content


def wire(need, content):
    return standard_form_response(need, {"action": "accept", "content": content}).content


def test_pool_equivalent_choice_or_text_and_optional_text():
    schema = pool_form_schema(form())
    assert schema["required"] == ["branch", "target"]
    assert schema["properties"]["notes"] == {
        "type": "string",
        "title": "Notes?",
        "description": "Free text",
    }
    assert schema["properties"]["branch"] == {
        "title": "Branch?",
        "description": "Choose carefully",
        "anyOf": [
            {
                "oneOf": [
                    {"const": "main", "title": "Main branch"},
                    {"const": "dev", "title": "Development"},
                ]
            },
            {"type": "string"},
        ],
    }
    assert pool_form_schema(NeedInput(question="Why?")) is None
    assert need_input_schema(NeedInput(question="Why?")) is None


def test_standard_titled_choices_and_hidden_other():
    schema = need_input_schema(form()).model_dump(mode="json", by_alias=True, exclude_none=True)
    assert schema["type"] == "object"
    assert list(schema["properties"]) == [
        "notes",
        "branch",
        "__acp_other_1",
        "target",
        "__acp_other_2",
    ]
    assert schema["required"] == []  # either-field cannot be required in restricted ACP
    assert schema["properties"]["branch"]["oneOf"] == [
        {"const": "main", "title": "Main branch"},
        {"const": "dev", "title": "Development"},
    ]
    assert schema["properties"]["__acp_other_1"] == {"type": "string", "title": "Branch? — Other"}


@pytest.mark.parametrize(
    "content,expected",
    [
        ({"branch": "dev", "target": " custom "}, {"branch": "dev", "target": " custom "}),
        (
            {"notes": "   ", "branch": "Development", "target": ""},
            {"notes": "   ", "branch": "Development", "target": ""},
        ),
    ],
)
def test_owner_answers_are_one_native_value_without_normalization(content, expected):
    assert answer(form(), content) == expected


@pytest.mark.parametrize(
    "content",
    [
        None,
        {},
        {"branch": True, "target": "x"},
        {"branch": "dev", "target": ["x"]},
        {"branch": "dev", "target": "x", "unknown": "y"},
    ],
)
def test_invalid_protocol_never_converts(content):
    with pytest.raises(ValueError):
        answer(form(), content)


@pytest.mark.parametrize("action", ["decline", "cancel"])
@pytest.mark.parametrize("content", [None, {}, {"bogus": 4}, ["invalid"]])
def test_nonaccept_wire_content_is_ignored(action, content):
    assert standard_form_response(form(), {"action": action, "content": content}) == FormResponse(
        action=action
    )


@pytest.mark.parametrize(
    "question,value,prop",
    [
        (
            TextQuestion(
                id="q",
                label="Q",
                help="Help",
                default="abc",
                min_length=2,
                max_length=4,
                pattern="^[a-z]+$",
                format="unknown-format",
            ),
            "abc",
            {
                "type": "string",
                "title": "Q",
                "description": "Help",
                "default": "abc",
                "minLength": 2,
                "maxLength": 4,
                "pattern": "^[a-z]+$",
                "format": "unknown-format",
            },
        ),
        (
            IntegerQuestion(id="q", label="Q", minimum=-2, maximum=3, default=0),
            2,
            {"type": "integer", "title": "Q", "minimum": -2, "maximum": 3, "default": 0},
        ),
        (
            NumberQuestion(id="q", label="Q", minimum=-0.5, maximum=3, default=0.5),
            1.25,
            {"type": "number", "title": "Q", "minimum": -0.5, "maximum": 3, "default": 0.5},
        ),
        (
            BooleanQuestion(id="q", label="Q", default=False),
            True,
            {"type": "boolean", "title": "Q", "default": False},
        ),
        (
            ChoiceQuestion(
                id="q",
                label="Q",
                choices=[FormChoice(value="a"), FormChoice(value="b")],
                default="a",
            ),
            "a",
            {"type": "string", "title": "Q", "enum": ["a", "b"], "default": "a"},
        ),
        (
            ChoiceQuestion(
                id="q", label="Q", choices=[FormChoice(value="a", title="A", description="help")]
            ),
            "a",
            {
                "type": "string",
                "title": "Q",
                "oneOf": [{"const": "a", "title": "A", "description": "help"}],
            },
        ),
        (
            MultiChoiceQuestion(
                id="q",
                label="Q",
                choices=[FormChoice(value="a"), FormChoice(value="b")],
                min_items=1,
                max_items=3,
                default=["a"],
            ),
            ["a", "b"],
            {
                "type": "array",
                "title": "Q",
                "items": {"type": "string", "enum": ["a", "b"]},
                "minItems": 1,
                "maxItems": 3,
                "default": ["a"],
            },
        ),
        (
            MultiChoiceQuestion(
                id="q", label="Q", choices=[FormChoice(value="a", title="A", description="help")]
            ),
            ["a"],
            {
                "type": "array",
                "title": "Q",
                "items": {"anyOf": [{"const": "a", "title": "A", "description": "help"}]},
            },
        ),
    ],
)
async def test_all_types_actual_sdk_wire_and_native_response(question, value, prop):
    from acp.agent.connection import AgentSideConnection
    from acp.schema import ElicitationFormSessionMode

    class RPC:
        async def send_request(self, method, params):
            self.method, self.params = method, params
            return {"action": "accept", "content": {"q": value}}

    need = one(question)
    conn = object.__new__(AgentSideConnection)
    conn._conn = rpc = RPC()
    response = await conn.create_elicitation(
        message="H",
        mode=ElicitationFormSessionMode(session_id="s", requested_schema=need_input_schema(need)),
    )
    assert rpc.method == "elicitation/create"
    assert rpc.params["sessionId"] == "s" and rpc.params["mode"] == "form"
    assert rpc.params["requestedSchema"]["properties"]["q"] == prop
    result = standard_form_response(need, response.model_dump(mode="json", exclude_none=True))
    assert result.content == {"q": value}
    assert type(result.content["q"]) is type(value)
    assert NeedInputForm.model_validate_json(need.model_dump_json()) == need
    assert FormResponse.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "question", [TextQuestion, IntegerQuestion, NumberQuestion, BooleanQuestion]
)
def test_optional_omissions_remain_absent_and_required_means_presence(question):
    need = one(question(id="q", label="Q", required=False))
    assert answer(need, {}) == {} and wire(need, None) == {}
    with pytest.raises(ValueError, match="required"):
        answer(one(question(id="q", label="Q")), {})
    assert answer(one(TextQuestion(id="q", label="Q")), {"q": ""}) == {"q": ""}


@pytest.mark.parametrize(
    "question,bad",
    [
        (IntegerQuestion, [True, False, 1.0, "1", None, [], float("inf")]),
        (NumberQuestion, [True, False, "1", None, [], float("nan")]),
        (BooleanQuestion, [0, 1, "true", None, []]),
        (TextQuestion, [0, True, [], None]),
    ],
)
def test_strict_native_types(question, bad):
    need = one(question(id="q", label="Q"))
    for value in bad:
        with pytest.raises(ValueError):
            answer(need, {"q": value})


@pytest.mark.parametrize("question", [IntegerQuestion, NumberQuestion])
def test_inclusive_numeric_bounds(question):
    need = one(question(id="q", label="Q", minimum=1, maximum=2))
    for value in (1, 2):
        assert answer(need, {"q": value}) == {"q": value}
    for value in (0, 3):
        with pytest.raises(ValueError):
            answer(need, {"q": value})


@pytest.mark.parametrize("fmt", ["email", "uri", "date", "date-time", "future-format"])
def test_formats_are_preserved_annotations(fmt):
    need = one(TextQuestion(id="q", label="Q", format=fmt))
    assert answer(need, {"q": "not-a-format-value"}) == {"q": "not-a-format-value"}
    assert need_input_schema(need).properties["q"].format == fmt


def test_text_lengths_pattern_and_bounded_pathology():
    need = one(TextQuestion(id="q", label="Q", min_length=1, max_length=3, pattern="^a+$"))
    assert answer(need, {"q": "aaa"}) == {"q": "aaa"}
    for value in ("", "aaaa", "b"):
        with pytest.raises(ValueError):
            answer(need, {"q": value})
    need = one(TextQuestion(id="q", label="Q", pattern="^(a+)+$"))
    start = time.monotonic()
    with pytest.raises(ValueError, match="time budget"):
        answer(need, {"q": "a" * 10000 + "!"})
    assert time.monotonic() - start < 5


@pytest.mark.parametrize(
    "question",
    [
        lambda: TextQuestion(id="q", label="Q", min_length=3, max_length=2),
        lambda: TextQuestion(id="q", label="Q", min_length=2, default="a"),
        lambda: TextQuestion(id="q", label="Q", pattern="["),
        lambda: IntegerQuestion(id="q", label="Q", minimum=2, maximum=1),
        lambda: IntegerQuestion(id="q", label="Q", default=True),
        lambda: IntegerQuestion(id="q", label="Q", minimum=0, default=-1),
        lambda: NumberQuestion(id="q", label="Q", maximum=float("inf")),
        lambda: BooleanQuestion(id="q", label="Q", default=0),
        lambda: ChoiceQuestion(id="q", label="Q", choices=[]),
        lambda: ChoiceQuestion(
            id="q", label="Q", choices=[FormChoice(value="a"), FormChoice(value="a")]
        ),
        lambda: ChoiceQuestion(
            id="q", label="Q", choices=[FormChoice(value="a"), FormChoice(value="b", title="B")]
        ),
        lambda: MultiChoiceQuestion(id="q", label="Q", choices=CHOICES, min_items=2, max_items=1),
        lambda: MultiChoiceQuestion(id="q", label="Q", choices=CHOICES, min_items=2, default=[]),
        lambda: MultiChoiceQuestion(id="q", label="Q", choices=CHOICES, default=["x", "y"]),
    ],
)
def test_invalid_bounds_defaults_choices(question):
    with pytest.raises(ValueError):
        question()


@pytest.mark.parametrize("kind", [ChoiceQuestion, MultiChoiceQuestion])
def test_optional_choice_is_authorable_but_pool_falls_back(kind):
    need = one(kind(id="q", label="Q", choices=CHOICES, required=False))
    assert wire(need, {}) == {}
    assert pool_form_schema(need) is None
    assert need_input_schema(need).required == []


@pytest.mark.parametrize(
    "content,expected",
    [
        ({"q": "main"}, {"q": "main"}),
        ({"__acp_other_0": "custom"}, {"q": "custom"}),
        ({"q": "dev", "__acp_other_0": "custom"}, {"q": "custom"}),
        ({"q": "dev", "__acp_other_0": ""}, {"q": "dev"}),
        ({"__acp_other_0": " "}, {"q": " "}),
    ],
)
def test_single_other_is_an_alternate_answer_not_note(content, expected):
    assert wire(one(ChoiceQuestion(id="q", label="Q", choices=CHOICES)), content) == expected


@pytest.mark.parametrize(
    "content",
    [
        {},
        {"__acp_other_0": ""},
        {"q": "custom"},
        {"q": "Main branch", "__acp_other_0": "valid other"},
        {"q": 1},
        {"__acp_other_0": ["x"]},
        {"q": "main", "q_custom": "x"},
    ],
)
def test_invalid_wire_choice_and_unknown_companions(content):
    with pytest.raises(ValueError):
        wire(one(ChoiceQuestion(id="q", label="Q", choices=CHOICES)), content)


@pytest.mark.parametrize(
    "content,expected",
    [
        ({"q": ["main", "dev"]}, ["main", "dev"]),
        ({"__acp_other_0": "custom"}, ["custom"]),
        ({"q": ["main"], "__acp_other_0": "custom"}, ["main", "custom"]),
        ({"q": ["main"], "__acp_other_0": "main"}, ["main"]),
        ({"q": []}, []),
    ],
)
def test_multiselect_is_one_native_list(content, expected):
    assert wire(one(MultiChoiceQuestion(id="q", label="Q", choices=CHOICES)), content) == {
        "q": expected
    }


def test_multiselect_bounds_before_and_after_other():
    need = one(MultiChoiceQuestion(id="q", label="Q", choices=CHOICES, min_items=1, max_items=2))
    assert wire(need, {"__acp_other_0": "custom"}) == {"q": ["custom"]}
    for content in (
        {"q": []},
        {"q": ["main", "dev"], "__acp_other_0": "custom"},
        {"q": ["unlisted"]},
        {"q": "main"},
    ):
        with pytest.raises(ValueError):
            wire(need, content)


def test_companion_ids_protect_all_authored_ids_and_public_response_keys():
    need = NeedInputForm(
        heading="H",
        questions=[
            ChoiceQuestion(id="q", label="Q", choices=CHOICES),
            TextQuestion(id="__acp_other_0", label="Real"),
            TextQuestion(id="__acp_other_0_", label="Also real"),
        ],
    )
    assert "__acp_other_0__" in need_input_schema(need).properties
    assert wire(
        need, {"__acp_other_0__": "custom", "__acp_other_0": "a", "__acp_other_0_": "b"}
    ) == {"q": "custom", "__acp_other_0": "a", "__acp_other_0_": "b"}
    with pytest.raises(ValueError):
        answer(need, {"q": "main", "__acp_other_0__": "not public"})


def test_choice_custom_defaults_live_on_other_not_enum():
    for q in (
        ChoiceQuestion(id="q", label="Q", choices=CHOICES, default="custom"),
        MultiChoiceQuestion(id="q", label="Q", choices=CHOICES, default=["main", "custom"]),
    ):
        schema = need_input_schema(one(q)).model_dump(mode="json", by_alias=True, exclude_none=True)
        assert schema["properties"]["__acp_other_0"]["default"] == "custom"
        assert schema["properties"]["q"].get("default") in (None, ["main"])


@pytest.mark.parametrize(
    "q",
    [
        IntegerQuestion(id="q", label="Q"),
        NumberQuestion(id="q", label="Q"),
        BooleanQuestion(id="q", label="Q"),
        MultiChoiceQuestion(id="q", label="Q", choices=CHOICES),
        TextQuestion(id="q", label="Q", default="d"),
        TextQuestion(id="q", label="Q", format="email"),
        TextQuestion(id="q", label="Q", min_length=1),
        ChoiceQuestion(id="q", label="Q", choices=CHOICES, default="main"),
    ],
)
def test_pool_unsupported_controls_options_fall_back_complete_form(q):
    assert (
        pool_form_schema(
            NeedInputForm(heading="H", questions=[TextQuestion(id="text", label="Text"), q])
        )
        is None
    )


def test_new_api_rejects_legacy_tags_and_unrelated_extras():
    for q in (
        {"kind": "pick_one", "id": "q", "label": "Q", "choices": []},
        {"kind": "text", "id": "q", "label": "Q", "arbitrary": 1},
    ):
        with pytest.raises(ValidationError):
            NeedInputForm(heading="H", questions=[q])
    with pytest.raises(ValidationError):
        NeedInputForm(
            heading="H",
            questions=[TextQuestion(id="q", label="Q"), TextQuestion(id="q", label="Again")],
        )


async def test_standard_sdk_emits_flat_form_request_with_titled_choices():
    from acp.agent.connection import AgentSideConnection
    from acp.schema import ElicitationFormSessionMode

    class RPC:
        async def send_request(self, method, params):
            self.method, self.params = method, params
            return {"action": "decline"}

    # Exercise the SDK's actual request builder/serializer with fake transport only.
    conn = object.__new__(AgentSideConnection)
    conn._conn = rpc = RPC()
    need = form().model_copy(update={"questions": form().questions[:2]})
    response = await conn.create_elicitation(
        message="Release",
        mode=ElicitationFormSessionMode(
            session_id="session-1", requested_schema=need_input_schema(need)
        ),
    )
    assert response.action == "decline"
    assert rpc.method == "elicitation/create"
    schema_wire = need_input_schema(need).model_dump(mode="json", by_alias=True, exclude_none=True)
    # ACP 0.12.1 drops this required discriminator via exclude_defaults=True.
    # A separate strict xfail below records the unmet sender requirement.
    schema_wire.pop("type")
    assert rpc.params == {
        "sessionId": "session-1",
        "mode": "form",
        "message": "Release",
        "requestedSchema": schema_wire,
    }
    assert "oneOf" in rpc.params["requestedSchema"]["properties"]["branch"]
    assert "one_of" not in str(rpc.params)


@pytest.mark.xfail(
    reason="ACP SDK 0.12.1 strips required root schema type via exclude_defaults; sender rule unmet",
    strict=True,
)
async def test_standard_sdk_sender_includes_required_root_type():
    from acp.agent.connection import AgentSideConnection
    from acp.schema import ElicitationFormSessionMode

    class RPC:
        async def send_request(self, method, params):
            self.params = params
            return {"action": "decline"}

    conn = object.__new__(AgentSideConnection)
    conn._conn = rpc = RPC()
    need = form().model_copy(update={"questions": form().questions[:2]})
    await conn.create_elicitation(
        message="Release",
        mode=ElicitationFormSessionMode(
            session_id="session-1", requested_schema=need_input_schema(need)
        ),
    )
    assert rpc.params["requestedSchema"]["type"] == "object"


def test_multiselect_custom_only_default_never_emits_invalid_empty_picker_default():
    need = one(
        MultiChoiceQuestion(id="q", label="Q", choices=CHOICES, min_items=1, default=["custom"])
    )
    schema = need_input_schema(need).model_dump(mode="json", by_alias=True, exclude_none=True)
    assert "default" not in schema["properties"]["q"]
    assert schema["properties"]["__acp_other_0"]["default"] == "custom"
    with pytest.raises(ValueError, match="picker minItems"):
        need_input_schema(
            one(
                MultiChoiceQuestion(
                    id="q", label="Q", choices=CHOICES, min_items=2, default=["main", "custom"]
                )
            )
        )


def test_shared_plain_display_preserves_option_help_constraints_defaults_and_other():
    from nooa.forms import question_lines

    need = NeedInputForm(
        heading="H",
        questions=[
            ChoiceQuestion(
                id="q",
                label="Choice?",
                choices=[
                    FormChoice(value="a", title="A", description="Option help"),
                ],
                required=False,
                default="custom",
            ),
            IntegerQuestion(id="i", label="Count?", minimum=1, maximum=3, default=2),
            TextQuestion(
                id="t",
                label="Text?",
                help="Field help",
                min_length=1,
                pattern="a",
                format="unknown",
            ),
        ],
    )
    text = "\n".join(question_lines([q.model_dump() for q in need.questions]))
    assert "A (a) — Option help, or write Other" in text
    assert "(optional)" in text and "default=custom" in text
    assert (
        "(integer)" in text and "minimum=1" in text and "maximum=3" in text and "default=2" in text
    )
    assert (
        "Field help" in text
        and "min_length=1" in text
        and "pattern=a" in text
        and "format=unknown" in text
    )
    assert pool_form_schema(need) is None


@pytest.mark.parametrize("default", [["main"], ["custom"], ["main", "custom"], ["main", "dev"]])
@pytest.mark.parametrize("required", [True, False])
def test_multiselect_default_wire_owner_roundtrip(default, required):
    need = one(
        MultiChoiceQuestion(
            id="q",
            label="Q",
            choices=CHOICES,
            min_items=1,
            max_items=2,
            default=default,
            required=required,
        )
    )
    props = need_input_schema(need).model_dump(mode="json", by_alias=True, exclude_none=True)[
        "properties"
    ]
    values = {key: prop["default"] for key, prop in props.items() if "default" in prop}
    assert wire(need, values) == {"q": default}
    assert answer(need, {"q": default}) == {"q": default}
    if not required:
        assert wire(need, {}) == {} and answer(need, {}) == {}


@pytest.mark.parametrize("kind,default", [(ChoiceQuestion, ""), (MultiChoiceQuestion, [""])])
def test_empty_written_default_is_a_standard_boundary_error_only(kind, default):
    need = one(kind(id="q", label="Q", choices=CHOICES, default=default))
    assert answer(need, {"q": default}) == {"q": default}
    with pytest.raises(ValueError, match="empty written Other default"):
        need_input_schema(need)
    assert (
        need_input_schema(one(TextQuestion(id="q", label="Q", default=""))).properties["q"].default
        == ""
    )


def test_one_other_rule_is_shared_by_native_defaults_and_answers():
    need = one(MultiChoiceQuestion(id="q", label="Q", choices=CHOICES, min_items=2))
    for values in (["x", "y"], ["main", "x", "y"]):
        with pytest.raises(ValueError, match="at most one written Other"):
            answer(need, {"q": values})
        with pytest.raises(ValueError, match="at most one written Other"):
            MultiChoiceQuestion(id="q", label="Q", choices=CHOICES, default=values)
    assert answer(need, {"q": ["main", "custom"]}) == {"q": ["main", "custom"]}
    with pytest.raises(ValueError, match="minItems"):
        wire(need, {"q": ["main"], "__acp_other_0": "custom"})


def test_standard_default_cannot_silently_reorder_native_other_first_list():
    need = one(MultiChoiceQuestion(id="q", label="Q", choices=CHOICES, default=["custom", "main"]))
    assert answer(need, {"q": ["custom", "main"]}) == {"q": ["custom", "main"]}
    with pytest.raises(ValueError, match="written Other must be last"):
        need_input_schema(need)
