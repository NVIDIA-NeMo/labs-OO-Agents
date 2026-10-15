# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The child-session interface: worker and conversation modes, lifetime, status."""

import asyncio

import pytest
from atom_test_agents import ATOM_SPEC, AtomModels, cell, done, until
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done

TIMEOUT = 30

CHILD_RESULT = (
    "return_result(Done(explanation='fixed', result=TaskResult("
    "solution_description='patched parse()', evidence='tests pass', "
    "how_to_verify='pytest', report='REPORT: off-by-one fixed')))"
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


def _wait_then(code):
    """A parent cell that spawns with ``code`` and ends the turn waiting on delegates."""
    return cell(f"{code}\nreturn_result(Waiting(explanation='child working', on=['delegates']))")


# ---- D1/D2: worker mode ----------------------------------------------------


async def test_a_worker_must_return_a_task_result_and_the_parent_gets_it(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Fix parser', 'fix the parser')"),
        cell(
            "[item] = notification['delegates']\n"
            "assert isinstance(item, ChildResult)\n"
            "assert isinstance(item.done.result, TaskResult)\n"
            "return_result(Done(explanation=item.done.result.report))"
        ),
    ]
    atom_models.scripts["Fix parser"] = [
        # A worker's Done without a TaskResult is rejected and it tries again.
        cell("return_result(Done(explanation='fixed'))"),
        cell(CHILD_RESULT),
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="REPORT: off-by-one fixed")
    assert "A worker turn must end with Done" in _messages(atom_models, "Fix parser", 1)
    [child] = atom_registry.children(root.id)
    assert (child.turn_method, child.retained) == ("handle_work", False)


async def test_a_worker_result_that_is_not_a_task_result_is_rejected(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "done = await self.delegate('Fix', 'fix it')\n"
            "return_result(Done(explanation=type(done.result).__name__))"
        )
    ]
    atom_models.scripts["Fix"] = [
        cell("return_result(Done(explanation='fixed', result='just a string'))"),
        cell(CHILD_RESULT),
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="TaskResult")


async def test_the_worker_prompt_is_written_for_a_parent(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        cell("await self.delegate('W', 'w')\nreturn_result(Done(explanation='ok'))")
    ]
    atom_models.scripts["W"] = [cell(CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    prompt = _messages(atom_models, "W")
    assert "Work on a task for your parent agent and report back to it." in prompt
    assert "remove scratch files" in prompt
    assert "only one editing this checkout" in prompt


# ---- D1/D7: lifetime -------------------------------------------------------


async def test_keep_false_closes_after_the_result_and_send_then_raises(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Once', 'do it once')"),
        cell(
            "[item] = notification['delegates']\n"
            "try:\n"
            "    await self.v.ref.send('more work')\n"
            "except ChildFailedError as exc:\n"
            "    return_result(Done(explanation=f'{self.v.ref.status}: {exc}'))"
        ),
    ]
    atom_models.scripts["Once"] = [cell(CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    # Closed for good as soon as its result is delivered, even before the close finishes.
    assert outcome.explanation.startswith("closed: Child 'Once' is closed")
    [child] = atom_registry.children(root.id)
    await until(lambda: atom_registry.get(child.id) is None)
    assert atom_registry.children(root.id)[0].status == "closed"


async def test_keep_true_persists_and_takes_follow_ups(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Kept', 'first task', keep=True)"),
        cell(
            "[item] = notification['delegates']\n"
            "self.v.first = item.done.result.report\n"
            "assert self.v.ref.status == 'retained'\n"
            "await self.v.ref.send('follow-up')\n"
            "return_result(Waiting(explanation='follow-up', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "return_result(Done(explanation=self.v.first + ' / ' + item.done.result.report))"
        ),
    ]
    atom_models.scripts["Kept"] = [
        cell(CHILD_RESULT),
        cell(
            "[m] = notification['user_messages']\n"
            "return_result(Done(explanation='again', result=TaskResult("
            "solution_description=m, evidence='e', how_to_verify='h', report='SECOND')))"
        ),
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="REPORT: off-by-one fixed / SECOND")
    [child] = atom_registry.children(root.id)
    assert atom_registry.get(child.id) is not None
    assert (child.status, child.turn_method, child.retained) == ("retained", "handle_work", True)


async def test_closing_a_kept_child_is_final(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Kept', 'task', keep=True)"),
        cell(
            "await self.v.ref.close()\n"
            "try:\n"
            "    await self.v.ref.steer('hello?')\n"
            "except ChildFailedError:\n"
            "    return_result(Done(explanation=self.v.ref.status))"
        ),
    ]
    atom_models.scripts["Kept"] = [cell(CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="closed")


async def test_retain_is_removed_with_a_clear_error(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        cell(
            "errors = []\n"
            "try:\n"
            "    await self.spawn('X', 'x', retain=True)\n"
            "except TypeError as exc:\n"
            "    errors.append(str(exc))\n"
            "try:\n"
            "    await self.session.delegate('X', 'x', retain=True)\n"
            "except TypeError as exc:\n"
            "    errors.append(str(exc))\n"
            "self.v.errors = errors\n"
            "return_result(Done(explanation=str(len(errors))))"
        )
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="2")
    for error in root._agent.v.errors:
        assert "retain=" in error and "mode=" in error and "keep=" in error
    assert atom_registry.children(root.id) == []


def test_spawn_shows_mode_and_keep_not_retain_or_kwargs():
    import inspect

    from nooa_atom.agent.agent import AtomAgent

    params = inspect.signature(AtomAgent.spawn).parameters
    assert {"mode", "keep"} <= set(params)
    assert "retain" not in params
    assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_a_child_ref_restores_from_old_and_new_snapshots():
    from nooa_atom.session.items import ChildRef

    from nooa.storage.serialization import deserialize, serialize

    ref = ChildRef(id="c", name="kid", depth=1, status="retained", parent_id="p", todo_id="t")
    blob, allowlist = serialize(ref)
    assert deserialize(blob, allowlist) == ref
    # A snapshot written before the live status stored the field as ``status``.
    data = blob["data"]
    data["status"] = data.pop("status_when_made")
    del data["todo_id"]
    old = deserialize(blob, allowlist)
    assert (old.status, old.todo_id) == ("retained", None)


# ---- conversation mode -----------------------------------------------------


async def test_conversation_mode_is_the_interactive_handle(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then("await self.spawn('Chat', 'talk to me', mode='conversation', keep=True)"),
        cell(
            "[item] = notification['delegates']\n"
            "return_result(Done(explanation=f'{item.done.explanation} {item.done.result}'))"
        ),
    ]
    # No TaskResult needed in a conversation turn.
    atom_models.scripts["Chat"] = [done("hello")]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="hello None")
    [child] = atom_registry.children(root.id)
    assert child.turn_method == "handle"
    assert "Handle one interactive turn" in _messages(atom_models, "Chat")


# ---- D7: a Waiting child reads "working" and sends nothing -----------------


async def test_a_waiting_child_is_working_and_sends_no_item(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Slow', 'start a job', keep=True)"),
    ]
    atom_models.scripts["Slow"] = [
        cell("return_result(Waiting(explanation='job running', on=['jobs']))")
    ]
    root = await atom_registry.create(atom_options)
    await root.submit("go")  # the root's turn ends Waiting, so prompt() would stay open
    await until(lambda: len(atom_registry.children(root.id)) == 1, TIMEOUT)
    [info] = atom_registry.children(root.id)
    await until(lambda: atom_registry.children(root.id)[0].status == "working", TIMEOUT)
    assert root._agent.queue_manager.get_channel("delegates").qsize() == 0
    assert atom_registry.child_status(root, info.id) == "working"
