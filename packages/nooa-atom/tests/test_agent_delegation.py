# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Atom agent's flat delegation methods, run through real child sessions."""

import asyncio

import pytest
from atom_test_agents import ATOM_SPEC, AtomModels, cell, done
from nooa_atom.session.items import TaskResult
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done

TIMEOUT = 30

CHILD_RESULT = (
    "return_result(Done(explanation='fixed', result=TaskResult("
    "solution_description='patched parse()', evidence='tests pass', "
    "how_to_verify='pytest tests/test_parser.py', report='REPORT: off-by-one fixed')))"
)


@pytest.fixture
def atom_models():
    return AtomModels()


@pytest.fixture
async def atom_registry(sessions_dir, atom_models):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=atom_models)
    yield registry
    await registry.close_all()


@pytest.fixture
def atom_options(tmp_path, sessions_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, sessions_dir=sessions_dir)


def _messages(models, name, call=0):
    return str(models.llms[name].calls[call].messages)


async def test_delegate_returns_the_childs_done_with_its_task_result(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "done = await self.delegate('Fix parser', 'CHILD-PROMPT: fix the parser')\n"
            "self.v.result = done.result\n"
            "return_result(Done(explanation=done.result.report))"
        )
    ]
    atom_models.scripts["Fix parser"] = [cell(CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("fix it"), TIMEOUT)
    assert outcome == Done(explanation="REPORT: off-by-one fixed")
    assert isinstance(root._agent.v.result, TaskResult)
    assert "CHILD-PROMPT: fix the parser" in _messages(atom_models, "Fix parser")
    # The child is a session of its own, with its own record.
    [child] = atom_registry.children(root.id)
    assert (child.name, child.turn_method, child.retained) == ("Fix parser", "handle_batch", False)
    assert child.agent == ATOM_SPEC
    assert atom_registry.store.load_rows(child.id, frozenset({"TurnStarted"}))


async def test_a_spawned_childs_result_arrives_on_delegates_and_wakes_the_parent(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "ref = await self.spawn('Review', 'review the diff')\n"
            "self.v.child = ref\n"
            "return_result(Waiting(explanation='review running', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "assert isinstance(item, ChildResult)\n"
            "assert item.child.id == self.v.child.id\n"
            "return_result(Done(explanation='review: ' + item.done.result.report))"
        ),
    ]
    atom_models.scripts["Review"] = [cell(CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("review"), TIMEOUT)
    assert outcome == Done(explanation="review: REPORT: off-by-one fixed")
    assert [ref.name for ref in root._agent.children()] == ["Review"]


CHILD_DICT_RESULT = (
    "return_result(Done(explanation='fixed', result={"
    "'solution_description': 'patched parse()', 'evidence': 'tests pass', "
    "'how_to_verify': 'pytest tests/test_parser.py', 'report': 'REPORT: dict result'}))"
)


async def test_a_spawned_childs_dict_result_arrives_as_a_task_result(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "await self.spawn('Review', 'review the diff')\n"
            "return_result(Waiting(explanation='review running', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "return_result(Done(explanation='review: ' + item.done.result.report))"
        ),
    ]
    atom_models.scripts["Review"] = [cell(CHILD_DICT_RESULT)]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("review"), TIMEOUT)
    assert outcome == Done(explanation="review: REPORT: dict result")


async def test_delegate_coerces_a_dict_result_too(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        cell(
            "done = await self.delegate('Fix parser', 'fix the parser')\n"
            "return_result(Done(explanation=done.result.report))"
        )
    ]
    atom_models.scripts["Fix parser"] = [cell(CHILD_DICT_RESULT)]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("fix it"), TIMEOUT)
    assert outcome == Done(explanation="REPORT: dict result")


def test_a_child_result_rebuilt_from_json_carries_a_task_result():
    """A queued ChildResult reloaded from the record keeps its TaskResult."""
    from nooa_atom.session.items import ChildRef, ChildResult

    result = TaskResult(solution_description="s", evidence="e", how_to_verify="h", report="r")
    item = ChildResult(
        child=ChildRef(id="c", name="c", depth=1, status="idle"),
        done=Done(explanation="x", result=result),
    )
    reloaded = ChildResult.model_validate_json(item.model_dump_json())
    assert reloaded.done.result == result
    # A result that is not a TaskResult stays as it was.
    other = ChildResult.model_validate(
        {"child": item.child.model_dump(), "done": {"explanation": "x", "result": {"a": 1}}}
    )
    assert other.done.result == {"a": 1}


async def test_a_todo_objective_goes_as_text_and_gets_the_report(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "todo = self.todo.add('Fix the parser', description='parse() drops the last line')\n"
            "self.todo.comment(todo, 'suspect the loop bound')\n"
            "done = await self.delegate('Parser', todo)\n"
            "return_result(Done(explanation='delegated'))"
        )
    ]
    atom_models.scripts["Parser"] = [cell(CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    prompt = _messages(atom_models, "Parser")
    for text in ("Fix the parser", "parse() drops the last line", "suspect the loop bound"):
        assert text in prompt
    [todo] = root._agent.todo.list_todos()
    assert [c.body for c in todo.comments][0] == "suspect the loop bound"
    assert "REPORT: off-by-one fixed" in todo.comments[-1].body


async def test_delegating_past_the_depth_limit_fails_in_the_cell(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell("await self.delegate('Too deep', 'anything')"),
        done("could not delegate"),
    ]
    root = await atom_registry.create(atom_options.model_copy(update={"max_depth": 0}))
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="could not delegate")
    assert "DepthLimitError" in _messages(atom_models, None, 1)
    assert atom_registry.children(root.id) == []


async def test_a_failed_child_raises_in_the_parents_cell(atom_registry, atom_options, atom_models):
    """The child's strict model has no script, so its turn fails."""
    atom_models.scripts[None] = [
        cell(
            "try:\n"
            "    await self.delegate('Broken', 'anything')\n"
            "except ChildFailedError as exc:\n"
            "    return_result(Done(explanation='child failed'))"
        )
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="child failed")


async def test_rename_session_sets_the_title_through_the_port(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "title = await self.rename_session('  \"Parser   fix\" ')\n"
            "return_result(Done(explanation=title))"
        )
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="Parser fix")
    assert root.info.title == "Parser fix"
    assert atom_registry.store.get(root.id).title == "Parser fix"


async def test_a_title_request_submitted_by_the_host_is_recorded_and_answered(
    atom_registry, atom_options, atom_models
):
    """The host admits the request through the Session, so it is recorded like any item."""
    from nooa_atom.agent.agent import session_title_request

    atom_models.scripts[None] = [
        cell(
            "[request] = notification['system_messages']\n"
            "assert request.startswith('[session-title]')\n"
            "await self.rename_session('Parser fix')\n"
            "return_result(Done(explanation='titled'))"
        )
    ]
    root = await atom_registry.create(atom_options)
    receipt = await root.submit(
        session_title_request("fix the parser"), channel="system_messages", source="host"
    )
    assert await asyncio.wait_for(root.outcome(receipt.item_id), TIMEOUT) == Done(
        explanation="titled"
    )
    assert root.info.title == "Parser fix"
    [admitted] = atom_registry.store.load_rows(root.id, frozenset({"ItemAdmitted"}))
    assert (admitted[1]["channel"], admitted[1]["source"]) == ("system_messages", "host")
