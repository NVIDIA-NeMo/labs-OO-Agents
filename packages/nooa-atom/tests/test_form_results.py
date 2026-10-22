# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``NeedInputForm`` as an Atom turn result: models, serialisation, strategies, turn loop.

Moved from the core ``tests/test_interactive_agent.py``: core knows only
``NeedInput``; forms are Atom's extra return type.
"""

import json

import pytest
from nooa_atom.agent.conditions import require_task_result
from nooa_atom.session.forms import (
    FormChoice,
    FormResponse,
    InputRequest,
    NeedInputForm,
    PickOneQuestion,
    TextQuestion,
    recorded_form_text,
    recorded_item_text,
)
from pydantic import BaseModel, ValidationError

from nooa import hidden, strategy
from nooa.context_blocks import ToolCallEvent
from nooa.events import PythonOutput, ResultStatus
from nooa.interactive import Done, InteractiveAgent, NeedInput, Waiting
from nooa.runtime.turn_loop import _settled_from
from nooa.strategies import CodeActStrategy
from nooa.strategy_validation import InvariantError
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class _FormHost(InteractiveAgent, llm=FakeLLMClient()):
    """An InteractiveAgent whose interactive turn also returns forms, as AtomAgent's does."""

    @hidden
    @strategy(CodeActStrategy())
    async def handle(self, notification: dict[str, list]) -> Done | InputRequest | Waiting:
        """Handle one turn."""
        ...


def _cell(code: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))
        ],
        finish_reason="tool_calls",
    )


def _cell_errors(agent: InteractiveAgent) -> list[str]:
    return [
        e.stderr + e.error
        for e in agent.event_manager.values()
        if isinstance(e, PythonOutput) and e.execution_status is ResultStatus.ERROR
    ]


_NOTIFICATION = {"user_messages": ["hi"]}
_FORM_CODE = (
    'NeedInputForm(heading="Details?", questions=[TextQuestion(id="answer", label="Details?")])'
)


def test_form_rejects_obsolete_authoring():
    for kwargs in ({"question": "Why?"}, {"answer_type": BaseModel}, {"options": ["yes"]}):
        with pytest.raises(ValidationError):
            NeedInputForm(
                heading="Why?", questions=[TextQuestion(id="answer", label="Why?")], **kwargs
            )


def test_form_descriptors_serialise_in_tool_events():
    form = NeedInputForm(heading="Details", questions=[TextQuestion(id="name", label="Name?")])
    event = ToolCallEvent(tool_call_id="c1", name="return_result", arguments={"result": form})
    data = json.loads(event.model_dump_json())["arguments"]["result"]
    assert data == form.model_dump(mode="json")
    assert NeedInputForm.model_validate_json(form.model_dump_json()) == form


def test_the_core_turn_loop_settles_a_form_as_an_input_request():
    form = NeedInputForm(heading="Details", questions=[TextQuestion(id="name", label="Name?")])
    settled = _settled_from(form)
    assert settled.kind == "need_input" and settled.result is form
    assert "_turn_kind" not in NeedInputForm.model_fields
    assert NeedInputForm._turn_kind == "need_input"


def test_a_one_shot_worker_may_not_return_a_form():
    class _Port:
        may_ask = False

    class _Worker:
        session = _Port()

    form = NeedInputForm(heading="Details", questions=[TextQuestion(id="name", label="Name?")])
    with pytest.raises(InvariantError):
        require_task_result(_Worker(), form, None)
    with pytest.raises(InvariantError):
        require_task_result(_Worker(), NeedInput(question="Which?"), None)
    _Port.may_ask = True
    require_task_result(_Worker(), form, None)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ('return_result(NeedInput(question="Which branch?"))', NeedInput),
        (f"return_result({_FORM_CODE})", NeedInputForm),
    ],
)
async def test_handle_accepts_each_input_request(code, expected):
    agent = _FormHost(llm=FakeLLMClient([_cell(code, "c1")]))
    assert type(await agent.handle(_NOTIFICATION)) is expected


async def test_handle_batch_rejects_a_form_and_names_allowed_types():
    llm = FakeLLMClient(
        [
            _cell(f"return_result({_FORM_CODE})", "c1"),
            _cell('return_result(Done(explanation="blocked: branch not given"))', "c2"),
        ]
    )
    agent = _FormHost(llm=llm)
    result = await agent.handle_batch(_NOTIFICATION)
    assert result == Done(explanation="blocked: branch not given")
    [error] = _cell_errors(agent)
    assert "return_result validation error" in error
    assert "Done | " in error and "Waiting" in error
    assert "NeedInputForm" in error


def test_explicit_input_roundtrip_and_extra_rejection():
    for need in (
        NeedInput(question="Which?", reason="Safety."),
        NeedInputForm(heading="Which?", questions=[TextQuestion(id="name", label="Name?")]),
    ):
        assert type(need).model_validate_json(need.model_dump_json()) == need
        with pytest.raises(ValidationError):
            type(need).model_validate({**need.model_dump(), "presentation": "auto"})


def test_form_response_actions():
    for action in ("decline", "cancel"):
        assert FormResponse(action=action).action == action
        with pytest.raises(ValidationError):
            FormResponse(action=action, content={"count": "1"})
    with pytest.raises(ValidationError):
        FormResponse(action="accept")


def test_return_union_disambiguates_forms():
    from pydantic import TypeAdapter

    adapter = TypeAdapter(Done | InputRequest | Waiting)
    assert isinstance(
        adapter.validate_python(
            {
                "kind": "form",
                "heading": "Details?",
                "questions": [{"kind": "text", "id": "answer", "label": "Details?"}],
            }
        ),
        NeedInputForm,
    )
    assert isinstance(adapter.validate_python({"kind": "question", "question": "Why?"}), NeedInput)
    assert not issubclass(NeedInputForm, NeedInput)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "form", "question": "Why?", "presentation": "auto"})


def test_untagged_request_dicts_are_not_guessed():
    from pydantic import TypeAdapter

    for value in ({"question": "Why?"}, {"question": "Why?", "answer_type": None}):
        with pytest.raises(ValidationError, match="union_tag_not_found"):
            TypeAdapter(InputRequest).validate_python(value)


@pytest.mark.parametrize("value", [object(), float("nan"), {"opaque": object()}])
def test_form_response_rejects_opaque_content(value):
    with pytest.raises(ValidationError):
        FormResponse(action="accept", content=value)


@pytest.mark.parametrize("strategy_name", ["codeact", "predict"])
def test_strategy_return_wrappers_preserve_explicit_discriminator(strategy_name):
    annotation = Done | InputRequest | Waiting
    if strategy_name == "codeact":
        from nooa.strategies.codeact import CodeActStrategy

        model, validated = CodeActStrategy()._create_return_model(annotation, "handle")
        assert validated
        field = "result"
    else:
        from nooa.strategies.predict import PredictStrategy

        model = PredictStrategy()._create_response_model(annotation, "handle")
        field = "value"
    assert "discriminator" in json.dumps(model.model_json_schema())
    output = model.model_validate(
        {
            field: {
                "kind": "form",
                "heading": "Details?",
                "questions": [{"kind": "text", "id": "answer", "label": "Details?"}],
            }
        }
    )
    assert isinstance(getattr(output, field), NeedInputForm)
    with pytest.raises(ValidationError):
        model.model_validate({field: {"question": "Details?"}})


@pytest.mark.parametrize("as_json", [False, True])
async def test_handle_tagged_form_dictionary_and_json_return(as_json):
    data = {
        "kind": "form",
        "heading": "Details?",
        "questions": [{"kind": "text", "id": "answer", "label": "Details?"}],
    }
    code = f"return_result({json.dumps(data)!r})" if as_json else f"return_result({data!r})"
    agent = _FormHost(llm=FakeLLMClient([_cell(code, "c1")]))
    assert isinstance(await agent.handle(_NOTIFICATION), NeedInputForm)


def test_forms_and_responses_render_as_readable_text():
    form = NeedInputForm(
        heading="Release",
        reason="Pick a target.",
        questions=[
            PickOneQuestion(
                id="color",
                label="Color",
                allow_other=False,
                choices=[
                    FormChoice(value="red", title="Red"),
                    FormChoice(value="blue", title="Blue"),
                ],
            ),
            TextQuestion(id="note", label="Note", help="optional", required=False),
        ],
    )
    assert form.as_text() == "Release\n\nPick a target.\n\n- Color: Red, Blue\n- Note (optional)"
    record = {**form.model_dump(mode="json"), "request_id": "r1", "question_id": "q1"}
    assert recorded_form_text(record) == form.as_text()
    assert recorded_form_text({"kind": "form", "heading": "H", "questions": "bad"}) is None

    accepted = FormResponse(action="accept", content={"color": "red", "note": " "})
    assert accepted.as_text() == "color: red"
    assert FormResponse(action="accept", content={}).as_text() == "Submitted the form."
    assert FormResponse(action="decline").as_text() == "Declined the form."
    assert FormResponse(action="cancel").as_text() == "Cancelled the form."
    assert recorded_item_text(
        accepted.model_dump_json(), "nooa_atom.session.forms:FormResponse"
    ) == ("color: red")
    assert recorded_item_text('{"action": "accept"}', "builtins:dict") is None
