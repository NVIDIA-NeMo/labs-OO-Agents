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
        kept=True,
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
    assert (child.name, child.kept) == ("Review auth", False)

    other = parent.inherit(model="other-alias", kept=True, turn_method="handle_work")
    assert (other.model, other.llm, other.kept, other.turn_method) == (
        "other-alias",
        None,
        True,
        "handle_work",
    )
    assert parent.inherit(model=None).llm is llm


@pytest.mark.parametrize("removed", ["retain", "keep", "mode", "child_mode"])
def test_a_removed_child_option_raises_instead_of_being_ignored(tmp_path, removed):
    """Finding 7: ``retain=`` (and the short-lived ``keep=``/``mode=``) must not vanish silently."""
    parent = SessionOptions(workspace=tmp_path, agent_spec="pkg:Agent")
    with pytest.raises(TypeError, match=f"{removed}=, which was removed"):
        parent.inherit(name="kid", **{removed: True})
    with pytest.raises(TypeError, match=f"{removed}=, which was removed"):
        SessionOptions(workspace=tmp_path, agent_spec="pkg:Agent", **{removed: True})


def test_session_options_refuse_unknown_fields(tmp_path):
    from pydantic import ValidationError

    parent = SessionOptions(workspace=tmp_path, agent_spec="pkg:Agent")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SessionOptions(workspace=tmp_path, agent_spec="pkg:Agent", kep=True)
    with pytest.raises(ValidationError):
        parent.inherit(turn_method="run")


def test_options_require_an_agent_and_do_not_serialise_the_llm(tmp_path):
    with pytest.raises(ValidationError):
        SessionOptions(workspace=tmp_path)  # type: ignore[call-arg]
    options = SessionOptions(workspace=tmp_path, agent_spec="pkg:Agent", llm=object())
    assert "llm" not in options.model_dump()
    assert Path(options.model_dump(mode="json")["workspace"]) == tmp_path


# ---- review-2 finding 6: a ref whose child the registry no longer knows ----


class _GonePort:
    """A parent port whose registry no longer knows the child (e.g. after delete)."""

    id = "parent1"

    def __init__(self):
        self.calls = 0

    def child_status(self, child_id):
        self.calls += 1
        raise items.ChildFailedError(f"Child {child_id!r} does not exist")

    def pending_question(self, child_id):
        raise items.ChildFailedError(f"Child {child_id!r} does not exist")


def test_a_child_ref_snapshots_its_stored_status_and_never_raises():
    from nooa_atom.session.port import current_port

    from nooa.storage.serialization import deserialize, serialize
    from nooa.storage.snapshot_vars import SnapshotVars

    ref = ChildRef(id="c1", name="worker", depth=1, status="idle", parent_id="parent1")
    port = _GonePort()
    token = current_port.set(port)
    try:
        blob, allowlist = serialize(ref)
        dumped = ref.model_dump()
        assert port.calls == 0  # snapshots and dumps never ask the port
        assert blob["data"]["status"] == "idle" and dumped["status"] == "idle"
        assert deserialize(blob, allowlist) == ref
        v = SnapshotVars()
        v["child"] = ref  # self.v assignment serializes; it must not raise
        assert "child" in v
        assert "status='idle'" in repr(ref)  # repr falls back to the stored value
        assert ref.status == "idle"  # so does a live read
    finally:
        current_port.reset(token)


def test_a_child_ref_reads_status_live_in_its_parents_turn():
    from nooa_atom.session.port import current_port

    class LivePort:
        id = "parent1"

        def child_status(self, child_id):
            return "running"

        def pending_question(self, child_id):
            return None

    ref = ChildRef(id="c1", name="worker", depth=1, status="idle", parent_id="parent1")
    token = current_port.set(LivePort())
    try:
        assert ref.status == "running"
        assert ref.model_dump()["status"] == "idle"
    finally:
        current_port.reset(token)
