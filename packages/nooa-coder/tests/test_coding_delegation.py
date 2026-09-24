# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The coding agent's flat delegation methods, run through real child sessions."""

import asyncio

import pytest
from coder_test_agents import CODER_SPEC, CoderModels, cell, done
from nooa_coder.session.items import TaskResult
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.store import SessionStore

from nooa.interactive import Done

TIMEOUT = 30

CHILD_RESULT = (
    "return_result(Done(explanation='fixed', result=TaskResult("
    "solution_description='patched parse()', evidence='tests pass', "
    "how_to_verify='pytest tests/test_parser.py', report='REPORT: off-by-one fixed')))"
)


@pytest.fixture
def coder_models():
    return CoderModels()


@pytest.fixture
async def coder_registry(sessions_dir, coder_models):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=coder_models)
    yield registry
    await registry.close_all()


@pytest.fixture
def coder_options(tmp_path, sessions_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SessionOptions(workspace=workspace, agent_spec=CODER_SPEC, sessions_dir=sessions_dir)


def _messages(models, name, call=0):
    return str(models.llms[name].calls[call].messages)


async def test_delegate_returns_the_childs_done_with_its_task_result(
    coder_registry, coder_options, coder_models
):
    coder_models.scripts[None] = [
        cell(
            "done = await self.delegate('Fix parser', 'CHILD-PROMPT: fix the parser')\n"
            "self.v.result = done.result\n"
            "return_result(Done(explanation=done.result.report))"
        )
    ]
    coder_models.scripts["Fix parser"] = [cell(CHILD_RESULT)]
    root = await coder_registry.create(coder_options)
    outcome = await asyncio.wait_for(root.prompt("fix it"), TIMEOUT)
    assert outcome == Done(explanation="REPORT: off-by-one fixed")
    assert isinstance(root.agent.v.result, TaskResult)
    assert "CHILD-PROMPT: fix the parser" in _messages(coder_models, "Fix parser")
    # The child is a session of its own, with its own record.
    [child] = coder_registry.children(root.id)
    assert (child.name, child.turn_method, child.retained) == ("Fix parser", "handle_batch", False)
    assert child.agent == CODER_SPEC
    assert coder_registry.store.load_rows(child.id, frozenset({"TurnStarted"}))


async def test_a_spawned_childs_result_arrives_on_delegates_and_wakes_the_parent(
    coder_registry, coder_options, coder_models
):
    coder_models.scripts[None] = [
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
    coder_models.scripts["Review"] = [cell(CHILD_RESULT)]
    root = await coder_registry.create(coder_options)
    outcome = await asyncio.wait_for(root.prompt("review"), TIMEOUT)
    assert outcome == Done(explanation="review: REPORT: off-by-one fixed")
    assert [ref.name for ref in root.agent.children()] == ["Review"]


async def test_a_todo_objective_goes_as_text_and_gets_the_report(
    coder_registry, coder_options, coder_models
):
    coder_models.scripts[None] = [
        cell(
            "todo = self.todo.add('Fix the parser', description='parse() drops the last line')\n"
            "self.todo.comment(todo, 'suspect the loop bound')\n"
            "done = await self.delegate('Parser', todo)\n"
            "return_result(Done(explanation='delegated'))"
        )
    ]
    coder_models.scripts["Parser"] = [cell(CHILD_RESULT)]
    root = await coder_registry.create(coder_options)
    await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    prompt = _messages(coder_models, "Parser")
    for text in ("Fix the parser", "parse() drops the last line", "suspect the loop bound"):
        assert text in prompt
    [todo] = root.agent.todo.list_todos()
    assert [c.body for c in todo.comments][0] == "suspect the loop bound"
    assert "REPORT: off-by-one fixed" in todo.comments[-1].body


async def test_delegating_past_the_depth_limit_fails_in_the_cell(
    coder_registry, coder_options, coder_models
):
    coder_models.scripts[None] = [
        cell("await self.delegate('Too deep', 'anything')"),
        done("could not delegate"),
    ]
    root = await coder_registry.create(coder_options.model_copy(update={"max_depth": 0}))
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="could not delegate")
    assert "DepthLimitError" in _messages(coder_models, None, 1)
    assert coder_registry.children(root.id) == []


async def test_a_failed_child_raises_in_the_parents_cell(
    coder_registry, coder_options, coder_models
):
    """The child's strict model has no script, so its turn fails."""
    coder_models.scripts[None] = [
        cell(
            "try:\n"
            "    await self.delegate('Broken', 'anything')\n"
            "except ChildFailedError as exc:\n"
            "    return_result(Done(explanation='child failed'))"
        )
    ]
    root = await coder_registry.create(coder_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="child failed")


async def test_rename_session_sets_the_title_through_the_port(
    coder_registry, coder_options, coder_models
):
    coder_models.scripts[None] = [
        cell(
            "title = await self.rename_session('  \"Parser   fix\" ')\n"
            "return_result(Done(explanation=title))"
        )
    ]
    root = await coder_registry.create(coder_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="Parser fix")
    assert root.info.title == "Parser fix"
    assert coder_registry.store.get(root.id).title == "Parser fix"


async def test_a_title_request_submitted_by_the_host_is_recorded_and_answered(
    coder_registry, coder_options, coder_models
):
    """The host admits the request through the Session, so it is recorded like any item."""
    from nooa_coder.coding.agent import session_title_request

    coder_models.scripts[None] = [
        cell(
            "[request] = notification['system_messages']\n"
            "assert request.startswith('[session-title]')\n"
            "await self.rename_session('Parser fix')\n"
            "return_result(Done(explanation='titled'))"
        )
    ]
    root = await coder_registry.create(coder_options)
    receipt = await root.submit(
        session_title_request("fix the parser"), channel="system_messages", source="host"
    )
    assert await asyncio.wait_for(root.outcome(receipt.item_id), TIMEOUT) == Done(
        explanation="titled"
    )
    assert root.info.title == "Parser fix"
    [admitted] = coder_registry.store.load_rows(root.id, frozenset({"ItemAdmitted"}))
    assert (admitted[1]["channel"], admitted[1]["source"]) == ("system_messages", "host")
