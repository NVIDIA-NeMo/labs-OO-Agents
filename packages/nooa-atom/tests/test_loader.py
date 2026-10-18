# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loading agent classes from a module:Class spec, and building one for a session."""

import pytest
from atom_test_agents import EchoAgent
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.loader import AgentSpecError, load_agent_class
from nooa_atom.session.options import SessionOptions

from nooa.storage import InMemoryStorageManager
from nooa.unifiedllm import FakeLLMClient


def test_loads_an_interactive_agent_class():
    assert load_agent_class("atom_test_agents:EchoAgent") is EchoAgent


@pytest.mark.parametrize(
    "spec",
    [
        "atom_test_agents",  # no class
        "atom_test_agents:",  # empty class
        ":EchoAgent",  # empty module
        "no_such_module_for_nooa_atom:Agent",
        "atom_test_agents:NoSuchAgent",
        "atom_test_agents:NotAnAgent",  # not an InteractiveAgent
    ],
)
def test_bad_specs_are_rejected(spec):
    with pytest.raises(AgentSpecError):
        load_agent_class(spec)


def test_the_session_factory_passes_storage_and_llm(tmp_path):
    storage = InMemoryStorageManager()
    llm = FakeLLMClient()
    options = SessionOptions(workspace=tmp_path, agent_spec="atom_test_agents:EchoAgent", llm=llm)
    agent = create_session_agent(options, storage)
    assert isinstance(agent, EchoAgent)
    assert agent.llm is llm
    assert agent._storage is storage


_ANNOTATED_AGENT_FILE = """\
from __future__ import annotations

from nooa.interactive import InteractiveAgent
from nooa.unifiedllm import FakeLLMClient


class {helper}:
    pass


class {name}(InteractiveAgent, llm=FakeLLMClient()):
    helper: {helper} | None = None
"""


def _resolve_own_annotation(cls: type, name: str) -> str:
    """Resolve one string annotation the way typing does: via sys.modules[cls.__module__]."""
    import sys
    import typing

    hint = eval(cls.__dict__["__annotations__"][name], vars(sys.modules[cls.__module__]))
    (helper,) = [arg for arg in typing.get_args(hint) if arg is not type(None)]
    return f"{helper.__module__}.{helper.__qualname__}"


def test_file_agents_get_distinct_modules_and_keep_resolving_annotations(tmp_path):
    """A second file-based agent must not replace the first one's sys.modules entry.

    Postponed annotations resolve through ``sys.modules[cls.__module__]``; with
    one shared module name the first class's annotations pointed at the second
    file's namespace and ``get_type_hints`` failed. (From atom/3-engine's
    test_coding_factory.py, now against the one loader.)
    """
    import sys

    first = tmp_path / "one" / "agent.py"
    second = tmp_path / "two" / "agent.py"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(_ANNOTATED_AGENT_FILE.format(name="FirstAgent", helper="FirstHelper"))
    second.write_text(_ANNOTATED_AGENT_FILE.format(name="SecondAgent", helper="SecondHelper"))

    first_cls = load_agent_class(f"{first}:FirstAgent")
    second_cls = load_agent_class(f"{second}:SecondAgent")

    assert first_cls.__module__ != second_cls.__module__
    assert sys.modules[first_cls.__module__].FirstAgent is first_cls
    assert _resolve_own_annotation(first_cls, "helper") == first_cls.__module__ + ".FirstHelper"
    assert _resolve_own_annotation(second_cls, "helper") == second_cls.__module__ + ".SecondHelper"
    # The same unchanged file loads once, so repeated specs share one class.
    assert load_agent_class(f"{first}:FirstAgent") is first_cls


def test_a_relative_file_spec_resolves_against_the_given_base(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "agents").mkdir(parents=True)
    (workspace / "agents" / "mine.py").write_text(
        _ANNOTATED_AGENT_FILE.format(name="MyAgent", helper="MyHelper")
    )
    cls = load_agent_class("./agents/mine.py:MyAgent", base=workspace)
    assert cls.__name__ == "MyAgent"


@pytest.mark.parametrize(
    "body",
    [
        None,  # the file does not exist
        "class Other:\n    pass\n",  # no such class
        "from nooa import Agent\n\nclass Plain(Agent):\n    pass\n",  # not interactive
    ],
)
def test_bad_file_specs_are_rejected(tmp_path, body):
    path = tmp_path / "agent_file.py"
    if body is not None:
        path.write_text(body)
    with pytest.raises(AgentSpecError):
        load_agent_class(f"{path}:Plain")


@pytest.mark.parametrize("kind", [None, "question"])
def test_durable_conversational_options_migration_preserves_question_and_reason(kind):
    from nooa_atom.session.loader import load_typed
    from pydantic import ValidationError

    from nooa.interactive import NeedInput

    old = {"question": "Which branch?", "reason": "Safety.", "options": ["main", "dev"]}
    if kind:
        old["kind"] = kind
    original = dict(old)
    with pytest.raises(ValidationError):
        NeedInput.model_validate(old)
    loaded = load_typed("nooa.interactive:NeedInput", old)
    assert loaded == NeedInput(question="Which branch?", reason="Safety.")
    assert old == original
    assert "options" not in loaded.model_dump()
    # Only the removed field is migrated, not unrelated new/unknown authoring fields.
    assert isinstance(load_typed("nooa.interactive:NeedInput", {**old, "unknown": True}), dict)


def test_durable_question_migration_does_not_reinterpret_forms_or_unknown_kinds():
    from nooa_atom.session.loader import load_typed

    form = {
        "kind": "form",
        "heading": "Release",
        "options": ["main"],
        "questions": [{"kind": "text", "id": "name", "label": "Name?"}],
    }
    assert load_typed("nooa.interactive:NeedInputForm", form) == form
    unknown = {"kind": "unknown", "question": "Why?", "options": ["yes"]}
    assert load_typed("nooa.interactive:NeedInput", unknown) == unknown


def test_durable_child_question_drops_conversational_options():
    from nooa_atom.session.items import ChildQuestion
    from nooa_atom.session.loader import load_typed

    old = {
        "child": {"id": "child", "name": "helper", "depth": 1, "status": "retained"},
        "question": "Which branch?",
        "reason": "Safety.",
        "options": ["main", "dev"],
    }
    question = load_typed("nooa_atom.session.items:ChildQuestion", old)
    assert isinstance(question, ChildQuestion)
    assert question.question == "Which branch?" and question.reason == "Safety."
    assert question.request_kind == "question"
    assert "options" not in question.model_dump()
    assert "options" not in ChildQuestion.model_json_schema()["properties"]


def test_child_question_direct_options_rejected_and_old_null_form_carrier_migrates():
    from nooa_atom.session.items import ChildQuestion
    from nooa_atom.session.loader import load_typed
    from pydantic import ValidationError

    child = {"id": "child", "name": "helper", "depth": 1, "status": "retained"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ChildQuestion(child=child, question="Which?", options=["main"])
    record = {
        "child": child,
        "question": "Release",
        "request_kind": "form",
        "options": None,
        "questions": [{"kind": "text", "id": "name", "label": "Name?"}],
        "request_id": "owned-request",
    }
    loaded = load_typed("nooa_atom.session.items:ChildQuestion", record)
    assert isinstance(loaded, ChildQuestion)
    assert loaded.request_kind == "form" and loaded.request_id == "owned-request"
    assert loaded.questions[0].id == "name"


def test_old_child_form_choices_are_not_conversationally_migrated():
    from nooa_atom.session.loader import load_typed

    record = {
        "child": {"id": "child", "name": "helper", "depth": 1, "status": "retained"},
        "question": "Which branch?",
        "presentation": "form",
        "options": ["main", "dev"],
    }
    loaded = load_typed("nooa_atom.session.items:ChildQuestion", record)
    assert isinstance(loaded, dict)
    assert loaded["request_kind"] == "form" and loaded["options"] == ["main", "dev"]


@pytest.mark.parametrize("old_kind", ["pick_one", "pick_one_or_text"])
def test_legacy_descriptors_migrate_only_on_durable_decoding(old_kind):
    from nooa_atom.session.loader import load_typed, migrate_input_request

    from nooa.interactive import ChoiceQuestion, FormResponse, NeedInputForm

    descriptor = {
        "kind": old_kind,
        "id": "__acp_other_0",
        "label": "Old?",
        "choices": [{"value": "a", "title": "A"}],
    }
    record = {
        "kind": "form",
        "heading": "Old",
        "request_id": "owner-token",
        "questions": [descriptor],
    }
    decoded = load_typed("nooa.interactive:NeedInputForm", record)
    assert isinstance(decoded, NeedInputForm)
    assert decoded.questions[0].kind == "choice"
    assert decoded.questions[0].id == "__acp_other_0"
    assert decoded.validate_response(
        FormResponse(action="accept", content={"__acp_other_0": "written"})
    ).content == {"__acp_other_0": "written"}
    assert record["questions"][0]["kind"] == old_kind
    migrated = migrate_input_request(record, form=True)
    assert migrated["questions"][0]["choices"] == descriptor["choices"]
    qualifier = "PickOneQuestion" if old_kind == "pick_one" else "PickOneOrTextQuestion"
    assert isinstance(load_typed("nooa.interactive:" + qualifier, descriptor), ChoiceQuestion)
    assert load_typed("nooa.forms:NeedInputForm", decoded.model_dump()) == decoded
    with pytest.raises(ValueError):
        NeedInputForm.model_validate(record)


def test_native_form_response_durable_roundtrip():
    from nooa_atom.session.loader import load_typed

    from nooa.interactive import FormResponse

    response = FormResponse(
        action="accept",
        content={"i": 2, "n": 0.5, "b": False, "a": ["x", "custom"], "s": "  text  "},
    )
    for name in ("nooa.interactive:FormResponse", "nooa.forms:FormResponse"):
        restored = load_typed(name, response.model_dump_json())
        assert restored == response
        for key, value in response.content.items():
            assert type(restored.content[key]) is type(value)


def test_legacy_choice_class_does_not_reinterpret_unrelated_kind_or_corrupt_data():
    from nooa_atom.session.loader import load_typed

    record = {"kind": "number", "id": "q", "label": "Q", "choices": [{"value": "a"}]}
    assert load_typed("nooa.interactive:PickOneQuestion", record) == record
    corrupt = {"kind": "pick_one", "id": "q", "label": "Q", "choices": 2}
    assert load_typed("nooa.interactive:PickOneQuestion", corrupt) == corrupt
