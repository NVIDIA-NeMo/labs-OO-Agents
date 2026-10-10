# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nooa.interactive — the dispatcher-driven agent base.

Deeper behavioral coverage (dispatcher loop, snapshot restore, echo hooks)
lives with the TUI package, whose BaseTUIAgent subclasses this. These tests
pin the core contract the ARC-AGI-3 example and other hosts rely on.
"""

import json

import pytest
from pydantic import BaseModel, ValidationError

from nooa import hidden, strategy
from nooa.context_blocks import ToolCallEvent
from nooa.events import PythonOutput, ResultStatus
from nooa.interactive import (
    AgentMessage,
    AgentVars,
    Done,
    FormResponse,
    InputRequest,
    InteractiveAgent,
    NeedInput,
    NeedInputForm,
    SummarizationConfig,
    TextQuestion,
    Waiting,
    install_summarizer,
)
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class _Host(InteractiveAgent, llm=FakeLLMClient()):
    """Minimal InteractiveAgent subclass standing in for a host-driven agent."""


@pytest.fixture
def agent():
    return _Host(llm=FakeLLMClient())


def test_declares_only_the_user_channel(agent):
    """Being dispatcher-driven implies a human feeding it, and nothing more.

    Hosts declare whatever else they need. slash_commands and system_messages
    are coding-host concepts and live on CodingAgent — see
    packages/nooa-cli/tests/test_coding_agent.py.
    """
    assert agent.queue_manager.channels().keys() == {"user_messages"}
    # Reader facade exposed under the public name; producer side hidden.
    assert agent.user_messages is agent._user_messages_in.reader


async def test_queue_roundtrip(agent):
    agent._user_messages_in.put("hello")
    assert await agent.user_messages.get() == "hello"


def test_persistent_vars_proxy(agent):
    agent.v.cursor = 3
    assert agent.v.cursor == 3
    assert "cursor" in agent.v
    assert agent.vars["cursor"] == 3
    del agent.v.cursor
    with pytest.raises(AttributeError):
        _ = agent.v.cursor
    assert isinstance(agent.v, AgentVars)


def test_message_records_event_and_renders(agent):
    rendered: list[str] = []
    agent._render_message = lambda text, **kw: rendered.append(text)
    agent.message("**hi**")
    assert rendered == ["**hi**"]
    events = [e for e in agent.event_manager.values() if isinstance(e, AgentMessage)]
    assert len(events) == 1
    assert events[0].content == "**hi**"


def test_turn_results_require_their_text():
    assert Done(explanation=" finished ").explanation == "finished"
    assert Waiting(explanation="job ci-42", on=["jobs:ci-42"]).on == ["jobs:ci-42"]
    assert NeedInput(question="Which branch?", options=["main", "dev"]).options == ["main", "dev"]
    with pytest.raises(ValidationError):
        Done(explanation="  ")
    with pytest.raises(ValidationError):
        Waiting(explanation="", on=["jobs"])
    with pytest.raises(ValidationError):
        Waiting(explanation="waiting", on=[])
    with pytest.raises(ValidationError):
        Waiting(explanation="waiting", on=[" "])
    with pytest.raises(ValidationError):
        NeedInput(question=" ")


def test_done_carries_an_optional_reply_and_evidence():
    done = Done(explanation="answered", message="  Here is the answer.  ")
    assert done.message == "Here is the answer."
    assert Done(explanation="answered", message="   ").message is None
    assert Done(explanation="answered").message is None
    first, second = Done(explanation="a"), Done(explanation="b")
    first.evidence.append("pytest tests/x.py: 24 passed")
    assert second.evidence == []
    assert Done(explanation="a", evidence=["ruff: clean"]).evidence == ["ruff: clean"]


def test_waiting_carries_an_optional_user_line():
    waiting = Waiting(message="  Tests are running.  ", explanation="ci", on=["jobs:ci-42"])
    assert waiting.message == "Tests are running."
    assert Waiting(message=" ", explanation="ci", on=["jobs:ci-42"]).message is None
    assert Waiting(explanation="ci", on=["jobs:ci-42"]).message is None


def test_need_input_reason_is_optional():
    assert NeedInput(question="Which branch?").reason is None
    asked = NeedInput(question="Which branch?", reason="Pushing to the wrong one is hard to undo.")
    assert asked.reason == "Pushing to the wrong one is hard to undo."


@pytest.mark.parametrize(
    "build",
    [
        lambda: Done(explanation=" "),
        lambda: Waiting(explanation="waiting", on=["jobs", " "]),
    ],
)
def test_blank_text_is_rejected_with_one_message(build):
    with pytest.raises(ValidationError, match="Value error, must not be blank"):
        build()


def test_form_rejects_obsolete_authoring():
    for kwargs in ({"question": "Why?"}, {"answer_type": BaseModel}, {"options": ["yes"]}):
        with pytest.raises(ValidationError):
            NeedInputForm(
                heading="Why?", questions=[TextQuestion(id="answer", label="Why?")], **kwargs
            )


class _Opaque:
    """An object pydantic cannot turn into JSON."""


def test_done_result_serialises_when_it_holds_an_arbitrary_object():
    """return_result(result="<name>") puts the live object into the tool-call event."""
    event = ToolCallEvent(
        tool_call_id="c1",
        name="return_result",
        arguments={"result": Done(explanation="finished", result=_Opaque())},
    )
    assert "_Opaque" in event.model_dump_json()
    assert json.loads(Done(explanation="x", result={"a": 1}).model_dump_json())["result"] == {
        "a": 1
    }


def test_form_descriptors_serialise_in_tool_events():
    form = NeedInputForm(heading="Details", questions=[TextQuestion(id="name", label="Name?")])
    event = ToolCallEvent(tool_call_id="c1", name="return_result", arguments={"result": form})
    data = json.loads(event.model_dump_json())["arguments"]["result"]
    assert data == form.model_dump(mode="json")
    assert NeedInputForm.model_validate_json(form.model_dump_json()) == form


def test_need_input_options_must_not_be_empty():
    assert NeedInput(question="Anything else?").options is None
    with pytest.raises(ValidationError):
        NeedInput(question="Which branch?", options=[])


def test_need_input_options_must_not_be_blank():
    assert NeedInput(question="Which branch?", options=[" main "]).options == ["main"]
    with pytest.raises(ValidationError):
        NeedInput(question="Which branch?", options=["main", "  "])


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


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ('return_result(Done(explanation="finished"))', Done),
        ('return_result(NeedInput(question="Which branch?"))', NeedInput),
        (
            'return_result(NeedInputForm(heading="Details?", questions=[TextQuestion(id="answer", label="Details?")]))',
            NeedInputForm,
        ),
        ('return_result(Waiting(explanation="waiting for job ci-42", on=["jobs:ci-42"]))', Waiting),
    ],
)
async def test_handle_accepts_each_turn_result(code, expected):
    agent = _Host(llm=FakeLLMClient([_cell(code, "c1")]))
    result = await agent.handle(_NOTIFICATION)
    assert type(result) is expected


async def test_handle_returns_the_reply_in_done():
    code = 'return_result(Done(message="Here it is.", explanation="answered", evidence=["ran it"]))'
    agent = _Host(llm=FakeLLMClient([_cell(code, "c1")]))
    result = await agent.handle(_NOTIFICATION)
    assert result == Done(message="Here it is.", explanation="answered", evidence=["ran it"])
    assert result.message == "Here it is."


async def test_handle_returns_the_user_line_in_waiting():
    code = 'return_result(Waiting(message="Tests are running.", explanation="ci", on=["jobs:ci"]))'
    agent = _Host(llm=FakeLLMClient([_cell(code, "c1")]))
    result = await agent.handle(_NOTIFICATION)
    assert result == Waiting(message="Tests are running.", explanation="ci", on=["jobs:ci"])
    assert result.message == "Tests are running."


async def test_handle_rejects_other_results():
    llm = FakeLLMClient(
        [
            _cell('return_result("just text")', "c1"),
            _cell('return_result(Done(explanation="finished"))', "c2"),
        ]
    )
    agent = _Host(llm=llm)
    assert isinstance(await agent.handle(_NOTIFICATION), Done)
    [error] = _cell_errors(agent)
    assert "return_result validation error" in error


@pytest.mark.parametrize("name", ["NeedInput", "NeedInputForm"])
async def test_handle_batch_rejects_need_input_and_names_allowed_types(name):
    llm = FakeLLMClient(
        [
            _cell(
                "return_result("
                + (
                    'NeedInput(question="Which?")'
                    if name == "NeedInput"
                    else 'NeedInputForm(heading="Which?", questions=[TextQuestion(id="answer", label="Which?")])'
                )
                + ")",
                "c1",
            ),
            _cell('return_result(Done(explanation="blocked: branch not given"))', "c2"),
        ]
    )
    agent = _Host(llm=llm)
    result = await agent.handle_batch(_NOTIFICATION)
    assert result == Done(explanation="blocked: branch not given")
    [error] = _cell_errors(agent)
    assert "return_result validation error" in error
    assert "Done | " in error and "Waiting" in error
    assert name in error  # names what was returned


class _NarrowHost(InteractiveAgent, llm=FakeLLMClient()):
    """Host whose turns always finish."""

    @hidden
    @strategy(CodeActStrategy())
    async def handle(self, notification: dict[str, list]) -> Done:
        """Answer the question in one turn; SUBCLASS_HANDLE_DOC."""
        ...


async def test_model_sees_the_subclass_handle_docstring_and_annotation():
    llm = FakeLLMClient([_cell('return_result(Done(explanation="answered"))', "c1")])
    agent = _NarrowHost(llm=llm)
    assert isinstance(await agent.handle(_NOTIFICATION), Done)

    prompt = "\n".join(str(m.get("content", "")) for m in llm.calls[0].messages)
    assert "SUBCLASS_HANDLE_DOC" in prompt
    assert "Handle one interactive turn." not in prompt
    [return_tool] = [t for t in llm.calls[0].tools or [] if t.name == "return_result"]
    assert "Expected return type: Done." in return_tool.description


async def test_handle_docs_put_the_reply_in_done_message():
    llm = FakeLLMClient([_cell('return_result(Done(explanation="finished"))', "c1")])
    await _Host(llm=llm).handle(_NOTIFICATION)

    prompt = "\n".join(str(m.get("content", "")) for m in llm.calls[0].messages)
    assert "return_result(Done(message=" in prompt
    assert "exactly one terminal result" in prompt
    assert "return_result(Waiting(message=" in prompt


async def test_model_sees_the_base_handle_batch_annotation():
    llm = FakeLLMClient([_cell('return_result(Done(explanation="finished"))', "c1")])
    agent = _Host(llm=llm)
    await agent.handle_batch(_NOTIFICATION)

    prompt = "\n".join(str(m.get("content", "")) for m in llm.calls[0].messages)
    assert "Handle one unattended turn." in prompt
    assert "Waiting(message=..., explanation=..., on=[...])" in prompt
    [return_tool] = [t for t in llm.calls[0].tools or [] if t.name == "return_result"]
    assert "Expected return type: Done | Waiting." in return_tool.description


def test_install_summarizer_none_policy_is_noop(agent):
    install_summarizer(SummarizationConfig(policy="none"), agent=agent)
    assert not getattr(agent, "_summarizers", [])


def test_install_summarizer_attaches(agent):
    install_summarizer(SummarizationConfig(max_tokens=50_000), agent=agent)
    summarizers = getattr(agent, "_summarizers", [])
    assert len(summarizers) == 1
    assert summarizers[0].config.max_tokens == 50_000


@pytest.mark.parametrize("fraction", [0, -0.1, 1, 1.1])
def test_summarization_threshold_fraction_must_be_between_zero_and_one(fraction):
    with pytest.raises(ValidationError):
        SummarizationConfig(threshold_fraction=fraction)


def test_explicit_input_roundtrip_and_extra_rejection():
    for need in (
        NeedInput(question="Which?", options=["main", "dev"]),
        NeedInputForm(heading="Which?", questions=[TextQuestion(id="name", label="Name?")]),
    ):
        assert type(need).model_validate_json(need.model_dump_json()) == need
        with pytest.raises(ValidationError):
            type(need).model_validate({**need.model_dump(), "presentation": "auto"})


def test_question_rejects_typed_answer_and_form_response_actions():
    with pytest.raises(ValidationError):
        NeedInput(question="How many?", answer_type=BaseModel)
    form = NeedInputForm(heading="How many?", questions=[TextQuestion(id="count", label="Count?")])
    assert form.validate_response(
        FormResponse(action="accept", content={"count": "bad"})
    ).content == {"count": "bad"}
    for action in ("decline", "cancel"):
        assert form.validate_response(FormResponse(action=action)).action == action
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


@pytest.mark.parametrize("value", [1, {}, ["answer"], {"answer": 2}])
def test_form_rejects_invalid_content(value):
    with pytest.raises(ValueError):
        NeedInputForm(
            heading="Name?", questions=[TextQuestion(id="answer", label="Name?")]
        ).validate_response(FormResponse(action="accept", content=value))


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
    agent = _Host(llm=FakeLLMClient([_cell(code, "c1")]))
    assert isinstance(await agent.handle(_NOTIFICATION), NeedInputForm)
