# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data crossing the Session boundary, and session options."""

import types
import typing
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest
from nooa_atom.session import items
from nooa_atom.session.items import (
    ChildFailed,
    ChildQuestion,
    ChildRef,
    ChildResult,
    Receipt,
    SessionInfo,
    TaskResult,
    TranscriptEntry,
    TurnCancelledOutcome,
    Usage,
)
from nooa_atom.session.options import SessionOptions
from pydantic import BaseModel, ValidationError

from nooa.interactive import Done, NeedInput

_CHILD = ChildRef(id="c1", name="Review auth", depth=1, status="running")


class _Answer(BaseModel):
    branch: str
    force: bool = False


def test_models_validate():
    assert Receipt(session_id="s", channel="user_messages", item_id="i", delivered="queued")
    with pytest.raises(ValidationError):
        Receipt(session_id="s", channel="c", item_id="i", delivered="lost")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ChildRef(id="c", name="n", depth=1, status="sleeping")  # type: ignore[arg-type]
    result = TaskResult(solution_description="d", evidence="e", how_to_verify="v")
    assert result.report == ""
    child_result = ChildResult(child=_CHILD, done=Done(explanation="ok", result=result))
    assert ChildResult.model_validate_json(child_result.model_dump_json()).child == _CHILD
    assert ChildFailed(child=_CHILD, error="boom").error == "boom"
    assert TurnCancelledOutcome(by="user").by == "user"
    entry = TranscriptEntry(role="question", content="Which?", item_id=None, timestamp=1.0)
    assert entry.role == "question"
    with pytest.raises(ValidationError):
        TranscriptEntry(role="system", content="x", item_id=None, timestamp=1.0)  # type: ignore[arg-type]
    info = SessionInfo(id="s")
    assert info.usage == Usage() and info.status == "on_disk"


def test_child_question_from_need_input_carries_the_json_schema():
    question = ChildQuestion.from_need_input(
        _CHILD, NeedInput(question="Which branch?", answer_type=_Answer)
    )
    assert question.answer_schema == _Answer.model_json_schema()
    assert question.options is None
    assert ChildQuestion.model_validate_json(question.model_dump_json()) == question

    choice = ChildQuestion.from_need_input(
        _CHILD, NeedInput(question="Which?", options=["main", "dev"])
    )
    assert (choice.options, choice.answer_schema) == (["main", "dev"], None)


_LEAVES = {str, int, float, bool, type(None), typing.Any}


def _check_annotation(annotation: object, seen: set[type], where: str) -> None:
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return
    if origin is typing.Annotated:
        _check_annotation(typing.get_args(annotation)[0], seen, where)
        return
    if origin in (list, dict, tuple, typing.Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if arg is not Ellipsis:
                _check_annotation(arg, seen, where)
        return
    if annotation in _LEAVES:
        return
    if isinstance(annotation, type) and issubclass(annotation, (Enum, datetime)):
        return
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        _walk(annotation, seen)
        return
    raise AssertionError(f"{where}: {annotation!r} is not plain data")


def _walk(model: type[BaseModel], seen: set[type]) -> None:
    if model in seen:
        return
    seen.add(model)
    for name, field in model.model_fields.items():
        _check_annotation(field.annotation, seen, f"{model.__name__}.{name}")


def test_items_hold_only_plain_data():
    """No field of any boundary model can hold a callable, a class or a live object."""
    models = [
        value
        for value in vars(items).values()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value.__module__ == items.__name__
    ]
    assert ChildQuestion in models and SessionInfo in models
    seen: set[type] = set()
    for model in models:
        _walk(model, seen)


def test_options_inherit_for_a_child(tmp_path):
    llm = object()
    parent = SessionOptions(
        workspace=tmp_path,
        agent_spec="pkg:Agent",
        model="alias",
        llm=llm,
        permission_mode="auto",
        max_depth=3,
        retain=True,
        name="root",
        host="acp",
        sessions_dir=tmp_path / "s",
    )
    child = parent.inherit(name="Review auth")
    assert (child.workspace, child.agent_spec, child.model, child.llm) == (
        tmp_path,
        "pkg:Agent",
        "alias",
        llm,
    )
    assert (child.permission_mode, child.max_depth, child.host, child.sessions_dir) == (
        "auto",
        3,
        "acp",
        tmp_path / "s",
    )
    assert (child.name, child.retain) == ("Review auth", False)

    other = parent.inherit(model="other-alias", retain=True, turn_method="handle_batch")
    assert (other.model, other.llm, other.retain, other.turn_method) == (
        "other-alias",
        None,
        True,
        "handle_batch",
    )
    assert parent.inherit(model=None).llm is llm
    with pytest.raises(ValidationError):
        parent.inherit(turn_method="run")


def test_options_require_an_agent_and_do_not_serialise_the_llm(tmp_path):
    with pytest.raises(ValidationError):
        SessionOptions(workspace=tmp_path)  # type: ignore[call-arg]
    options = SessionOptions(workspace=tmp_path, agent_spec="pkg:Agent", llm=object())
    assert "llm" not in options.model_dump()
    assert Path(options.model_dump(mode="json")["workspace"]) == tmp_path
