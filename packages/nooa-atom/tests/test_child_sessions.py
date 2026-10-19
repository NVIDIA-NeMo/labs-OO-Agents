# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The child-session interface: one-shot ``delegate()``, kept ``spawn()``, lifetime, status.

The tests marked "finding N" are regressions for the code review of PR #451
at 5b84477a7; each one failed before its fix.
"""

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest
from atom_test_agents import ATOM_SPEC, AtomModels, cell, done, fresh_events, until
from nooa_atom.session.items import ChildRef
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done
from nooa.tools.todo import TodoManager

TIMEOUT = 30

CHILD_RESULT = (
    "return_result(Done(explanation='fixed', result=TaskResult("
    "solution_description='patched parse()', evidence='tests pass', "
    "how_to_verify='pytest', report='REPORT: off-by-one fixed')))"
)
ASK = "return_result(NeedInput(question='Which branch?', options=['main', 'dev']))"
# Atom cells run with the agent module's globals, so they reach the test's
# events through the test-agents module.
BLOCKING = (
    "import atom_test_agents as _t\n_t.STARTED.set()\nawait _t.BLOCK.wait()\nprint('cell released')"
)


def _result(report):
    return cell(
        "return_result(Done(explanation='ok', result=TaskResult("
        f"solution_description='s', evidence='e', how_to_verify='h', report={report!r})))"
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
    """A parent cell that runs ``code`` and ends the turn waiting on delegates."""
    return cell(f"{code}\nreturn_result(Waiting(explanation='child working', on=['delegates']))")


def _admitted_delegates(registry, session_id):
    rows = registry.store.load_rows(session_id, frozenset({"ItemAdmitted"}))
    return [
        str(raw["item_type"]).rsplit(".", 1)[-1].rsplit(":", 1)[-1]
        for _t, raw in rows
        if raw.get("channel") == "delegates"
    ]


# ---- the two verbs: delegate() is one-shot, spawn() is kept ------------------


async def test_a_spawned_child_must_return_a_task_result_and_the_parent_gets_it(
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
    assert (child.turn_method, child.retained) == ("handle_work", True)


async def test_a_delegated_result_that_is_not_a_task_result_is_rejected(
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
    [child] = atom_registry.children(root.id)
    assert (child.turn_method, child.retained) == ("handle_work", False)


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


async def test_a_delegated_child_is_closed_after_its_result_and_send_then_raises(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "await self.delegate('Once', 'do it once')\n"
            "[ref] = self.children()\n"
            "try:\n"
            "    await ref.send('more work')\n"
            "except ChildFailedError as exc:\n"
            "    return_result(Done(explanation=f'{ref.status}: {exc}'))"
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


async def test_a_spawned_child_persists_and_takes_follow_ups(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Kept', 'first task')"),
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


async def test_send_steers_a_running_child(atom_registry, atom_options, atom_models):
    """``send()`` folds in ``steer()``: a running turn reads the text before its next model call."""
    started, block = fresh_events()
    atom_models.scripts[None] = [
        cell(
            "ref = await self.spawn('Busy', 'work')\n"
            "import atom_test_agents as _t\n"
            "await _t.STARTED.wait()\n"
            "receipt = await ref.send('STEERED-TEXT')\n"
            "_t.BLOCK.set()\n"
            "r = await ref.wait()\n"
            "return_result(Done(explanation=f'{receipt.delivered}: {r.done.result.report}'))"
        )
    ]
    atom_models.scripts["Busy"] = [cell(BLOCKING), _result("steered")]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="steered: steered")
    assert "STEERED-TEXT" in _messages(atom_models, "Busy", 1)
    assert not hasattr(ChildRef, "steer")


async def test_closing_a_kept_child_is_final(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Kept', 'task')"),
        cell(
            "await self.v.ref.close()\n"
            "try:\n"
            "    await self.v.ref.send('hello?')\n"
            "except ChildFailedError:\n"
            "    return_result(Done(explanation=self.v.ref.status))"
        ),
    ]
    atom_models.scripts["Kept"] = [cell(CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="closed")


async def test_removed_keywords_raise_a_one_line_type_error(
    atom_registry, atom_options, atom_models
):
    calls = [
        "self.spawn('X', 'x', retain=True)",
        "self.spawn('X', 'x', mode='conversation')",
        "self.spawn('X', 'x', keep=True)",
        "self.spawn('X', 'x', context={'a': 1})",
        "self.delegate('X', 'x', keep=False)",
        "self.delegate('X', 'x', context={'a': 1})",
        "self.session.delegate('X', 'x', retain=True)",
        "self.session.delegate('X', 'x', keep=True)",
    ]
    code = "errors = []\n" + "".join(
        f"try:\n    await {call}\nexcept TypeError as exc:\n    errors.append(str(exc))\n"
        for call in calls
    )
    atom_models.scripts[None] = [
        cell(code + "self.v.errors = errors\nreturn_result(Done(explanation=str(len(errors))))")
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation=str(len(calls)))
    for error in root._agent.v.errors:
        assert "\n" not in error
        assert "which was removed" in error or "unexpected keyword argument 'context'" in error
    assert atom_registry.children(root.id) == []


def test_the_verbs_take_only_description_prompt_brief_and_model():
    from nooa_atom.agent.agent import AtomAgent

    for method in (AtomAgent.spawn, AtomAgent.delegate):
        params = inspect.signature(method).parameters
        assert list(params) == ["self", "description", "prompt", "brief", "model"]
    assert {name for name in dir(ChildRef) if not name.startswith("_")} >= {"wait", "send", "close"}
    for removed in ("steer", "info", "pending_question"):
        assert not hasattr(ChildRef, removed)


def test_only_child_failed_error_and_depth_limit_error_remain():
    import nooa_atom
    from nooa_atom.session import items

    for name in ("ChildIdleError", "ChildQuestionError"):
        assert not hasattr(items, name) and name not in nooa_atom.__all__
    assert "ChildFailedError" in nooa_atom.__all__ and "DepthLimitError" in nooa_atom.__all__


def test_a_child_ref_restores_from_old_and_new_snapshots():
    from nooa.storage.serialization import deserialize, serialize

    ref = ChildRef(id="c", name="kid", depth=1, status="retained", parent_id="p", todo_id="t")
    blob, allowlist = serialize(ref)
    assert deserialize(blob, allowlist) == ref
    # A snapshot of the PR's first version stored the field as ``status_when_made``.
    data = blob["data"]
    data["status_when_made"] = data.pop("status")
    del data["todo_id"]
    old = deserialize(blob, allowlist)
    assert (old.status, old.todo_id) == ("retained", None)


def test_a_child_ref_dumps_status_without_serialize_by_alias():
    """Finding 8: ``serialize_by_alias`` needs pydantic 2.11; the packages allow 2.5."""
    from nooa_atom.session.items import ChildQuestion
    from nooa_atom.session.session import item_to_json

    assert "serialize_by_alias" not in ChildRef.model_config
    assert "status_when_made" not in ChildRef.model_fields
    ref = ChildRef(id="c", name="kid", depth=1, status="retained", parent_id="p")
    keys = {"id", "name", "depth", "status", "parent_id", "todo_id"}
    assert set(ref.model_dump()) == keys
    assert set(ref.model_dump(mode="json")) == keys
    assert set(ref.model_dump(by_alias=True)) == keys
    question = ChildQuestion(child=ref, question="q")
    assert set(question.model_dump()["child"]) == keys
    assert set(json.loads(item_to_json(question))["child"]) == keys
    assert ChildRef.model_validate(ref.model_dump()) == ref


def test_session_options_carry_one_child_flag():
    """Finding 9: ``kept`` is the one source of truth; there is no dead ``child_mode``."""
    fields = set(SessionOptions.model_fields)
    assert "kept" in fields
    assert not fields & {"child_mode", "keep", "retain"}


# ---- a Waiting child reads "working" and sends nothing ---------------------


async def test_a_waiting_child_is_working_and_sends_no_item(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Slow', 'start a job')"),
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


# ---- the Todo as a one-way brief, the brief= text, the parent line ---------


async def test_a_todo_and_a_brief_both_reach_a_spawned_child_with_the_parent_line(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Fix the parser', description='REQ: parse() keeps the last line')\n"
            "self.todo.comment(t, 'NOTE: suspect the loop bound')\n"
            "self.v.todo_id = t.id\n"
            "self.v.ref = await self.spawn('Parser', t, brief='BRIEF: start from parse_header()')"
        ),
        done("received"),
    ]
    atom_models.scripts["Parser"] = [cell(CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    todo_id = root._agent.v.todo_id
    prompt = _messages(atom_models, "Parser")
    assert (
        f"You are working for parent root ({root.id}) on Todo {todo_id}: Fix the parser. "
        "You can ask your parent questions and may get follow-ups." in prompt
    )
    assert f"Task [{todo_id}]: Fix the parser" in prompt
    for text in ("REQ: parse() keeps the last line", "NOTE: suspect the loop bound"):
        assert text in prompt
    assert "BRIEF: start from parse_header()" in prompt


async def test_a_delegated_text_prompt_says_it_is_one_shot(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "await self.delegate('W', 'TEXT-TASK', brief='TEXT-BRIEF')\n"
            "return_result(Done(explanation='ok'))"
        )
    ]
    atom_models.scripts["W"] = [cell(CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    prompt = _messages(atom_models, "W")
    assert f"You are working for parent root ({root.id}). One-shot task: you can" in prompt
    assert "TEXT-TASK" in prompt and "TEXT-BRIEF" in prompt


async def test_the_child_works_on_a_local_copy_and_nothing_flows_back(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Fix the parser', description='REQ', owner='root')\n"
            "self.todo.comment(t, 'parent note')\n"
            "self.v.todo_id = t.id\n"
            "self.v.before = t.model_dump(mode='json')\n"
            "self.v.ref = await self.spawn('Parser', t)"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "assert isinstance(item, ChildResult)\n"
            "assert item.child.todo_id == self.v.todo_id == self.v.ref.todo_id\n"
            "assert self.todo.get(self.v.todo_id).model_dump(mode='json') == self.v.before\n"
            "return_result(Done(explanation=item.done.result.report))"
        ),
    ]
    atom_models.scripts["Parser"] = [
        cell(
            "t = self.todo.active()\n"
            "self.v.seen = (t.id, t.title, t.description, [c.body for c in t.comments])\n"
            "self.todo.comment(t, 'child evidence: tests pass')\n"
            "self.todo.set_var(t, 'patch', 'parser.diff')\n"
            "self.todo.update(t, description='child rewrote the requirements')\n"
            "self.todo.complete(t)\n" + CHILD_RESULT
        )
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="REPORT: off-by-one fixed")
    todo_id = root._agent.v.todo_id
    [child] = atom_registry.children(root.id)
    assert child.todo_id == todo_id
    # The parent's Todo is unchanged: no status, comments, vars or edits came back.
    todo = root._agent.todo.get(todo_id)
    assert todo.model_dump(mode="json") == root._agent.v.before
    assert (todo.status, todo.description, dict(todo.vars)) == ("open", "REQ", {"owner": "root"})
    assert [c.body for c in todo.comments] == ["parent note"]
    # The child had its own active copy, with the brief's fields.
    kid = atom_registry.get(child.id) or await atom_registry.open_child(root, child.id)
    assert kid._agent.v.seen == (todo_id, "Fix the parser", "REQ", ["parent note"])
    mine = kid._agent.todo.get(todo_id)
    assert mine is not todo and mine.status == "done"
    assert mine.vars["patch"] == "parser.diff"
    assert [c.body for c in mine.comments] == ["parent note", "child evidence: tests pass"]


async def test_a_kept_childs_results_never_touch_the_parents_todo(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Task')\n"
            "self.v.todo_id = t.id\n"
            "self.v.ref = await self.spawn('Kept', t)"
        ),
        cell(
            "self.todo.comment(self.v.todo_id, 'parent review: one more check')\n"
            "await self.v.ref.send('run the last check')\n"
            "return_result(Waiting(explanation='follow-up', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "self.todo.comment(self.v.todo_id, 'verified: ' + item.done.result.report)\n"
            "self.todo.complete(self.v.todo_id)\n"
            "return_result(Done(explanation='finished'))"
        ),
    ]
    atom_models.scripts["Kept"] = [
        cell("self.todo.comment(self.todo.active(), 'round 1')\n" + CHILD_RESULT),
        cell(
            "t = self.todo.list_todos()[0]\n"
            "self.todo.comment(t, 'round 2')\n"
            "self.todo.complete(t)\n" + CHILD_RESULT
        ),
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="finished")
    todo = root._agent.todo.get(root._agent.v.todo_id)
    # Only what the parent itself recorded after verifying.
    assert [c.body for c in todo.comments] == [
        "parent review: one more check",
        "verified: REPORT: off-by-one fixed",
    ]
    assert todo.status == "done" and "child" not in todo.vars
    [child] = atom_registry.children(root.id)
    kid = atom_registry.get(child.id) or await atom_registry.open_child(root, child.id)
    assert [c.body for c in kid._agent.todo.get(todo.id).comments] == ["round 1", "round 2"]


async def test_a_hand_built_todo_works_as_a_brief(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        _wait_then(
            "from nooa.tools.todo import Todo as _Todo\n"
            "t = _Todo(title='stray', description='REQ', deps=['not-yet'])\n"
            "self.v.todo_id = t.id\n"
            "import atom_test_agents as _t\n"
            "_t.HELD = t\n"
            "self.v.ref = await self.spawn('X', t)\n"
            "assert self.todo.get(t.id) is None  # read, not adopted"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "import atom_test_agents as _t\n"
            "assert self.todo.get(self.v.todo_id) is None\n"
            "return_result(Done(explanation=f'{_t.HELD.status} {len(_t.HELD.comments)}'))"
        ),
    ]
    atom_models.scripts["X"] = [
        cell(
            "t = self.todo.active()\n"
            "assert (t.title, t.description, t.deps) == ('stray', 'REQ', ['not-yet'])\n"
            "self.todo.comment(t, 'child evidence')\n"
            "self.todo.complete(t)\n" + CHILD_RESULT
        )
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="open 0")
    todo_id = root._agent.v.todo_id
    assert f"Task [{todo_id}]: stray" in _messages(atom_models, "X")
    [child] = atom_registry.children(root.id)
    assert child.todo_id == todo_id


async def test_a_second_object_with_a_managed_id_works_as_a_brief(
    atom_registry, atom_options, atom_models
):
    """Review-2 finding 3: after a reload ``self.v`` and ``self.todo`` hold different objects."""
    atom_models.scripts[None] = [
        cell(
            "kept = self.todo.add('recon task', description='keep me')\n"
            "self.v.todo_id = kept.id\n"
            "copy = kept.model_copy(deep=True)\n"
            "copy.description = 'the copy'\n"
            "done = await self.delegate('X', copy)\n"
            "assert self.todo.get(kept.id) is kept and kept.description == 'keep me'\n"
            "return_result(Done(explanation=kept.status))"
        )
    ]
    atom_models.scripts["X"] = [
        cell(
            "assert self.todo.active().description == 'the copy'\n"
            "self.todo.complete(self.todo.active())\n" + CHILD_RESULT
        )
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="open")
    todo = root._agent.todo.get(root._agent.v.todo_id)
    # Review-3 finding 3: delegate() writes nothing to the parent's Todo.
    assert todo.comments == [] and todo.status == "open"


def test_receive_delegated_todo_activates_a_local_copy():
    from nooa_atom.agent.agent import AtomAgent

    from nooa.tools.todo import Todo

    child = SimpleNamespace(todo=TodoManager())
    other = child.todo.add("already mine")
    brief = Todo(title="brief", description="REQ", status="open")
    AtomAgent.receive_delegated_todo(child, brief)
    mine = child.todo.get(brief.id)
    assert mine is not brief and mine == brief
    assert child.todo.active() is mine
    assert [t.id for t in child.todo.list_todos()] == [other.id, brief.id]
    child.todo.comment(mine, "only mine")
    child.todo.complete(mine)
    assert (brief.status, brief.comments) == ("open", [])


# ---- wait() returns an outcome and never hangs; questions -------------------


async def test_wait_returns_a_question_then_the_result(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        cell(
            "ref = await self.spawn('Pusher', 'push the branch')\n"
            "q = await ref.wait()\n"
            "assert isinstance(q, ChildQuestion) and q.options == ['main', 'dev']\n"
            "assert f'pending_question=\\'[{q.id}] Which branch?' in repr(self.children()[0])\n"
            "try:\n"
            "    await ref.wait()\n"
            "except ChildFailedError as exc:\n"
            "    self.v.waiting = str(exc)\n"
            "await q.child.send('main')\n"
            "assert 'pending_question' not in repr(self.children()[0])\n"
            "r = await ref.wait()\n"
            "return_result(Done(explanation=f'{type(r).__name__}: {r.done.result.report}'))"
        )
    ]
    atom_models.scripts["Pusher"] = [
        cell(ASK),
        cell(
            "[a] = notification['user_messages']\n"
            "return_result(Done(explanation='pushed', result=TaskResult("
            "solution_description=a, evidence='e', how_to_verify='h', report='pushed ' + a)))"
        ),
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="ChildResult: pushed main")
    assert "is waiting for your answer" in root._agent.v.waiting
    # Both outcomes went to wait(), none to delegates: one place only.
    assert _admitted_delegates(atom_registry, root.id) == []


async def test_wait_fails_fast_on_an_idle_kept_child(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Kept', 'task')"),
        cell(
            "[item] = notification['delegates']\n"
            "try:\n"
            "    await self.v.ref.wait()\n"
            "except ChildFailedError as exc:\n"
            "    return_result(Done(explanation=str(exc)))"
        ),
    ]
    atom_models.scripts["Kept"] = [cell(CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), 10)
    assert "is idle with nothing queued" in outcome.explanation


async def test_wait_withdraws_a_queued_question(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        cell(
            "ref = await self.spawn('Asker', 'ask')\n"
            "delegates = self.queue_manager.get_channel('delegates')\n"
            "async with asyncio.timeout(10):\n"
            "    while delegates.qsize() == 0:\n"
            "        await asyncio.sleep(0.01)\n"
            "q = await ref.wait()\n"
            "return_result(Done(explanation=f'{type(q).__name__}: {q.question}'))"
        )
    ]
    atom_models.scripts["Asker"] = [cell(ASK)]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="ChildQuestion: Which branch?")
    # Withdrawn from delegates by wait(): delivered once, not twice.
    assert root._agent.queue_manager.get_channel("delegates").qsize() == 0
    assert len(atom_registry.store.load_rows(root.id, frozenset({"ItemWithdrawn"}))) == 1


async def test_a_message_does_not_end_wait(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        cell(
            "ref = await self.spawn('Chatty', 'work')\n"
            "r = await ref.wait()\n"
            "return_result(Done(explanation=type(r).__name__))"
        ),
        done("saw the message"),
    ]
    atom_models.scripts["Chatty"] = [cell("self.message('progress note')\n" + CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ChildResult")
    assert _admitted_delegates(atom_registry, root.id) == ["ChildMessage"]


async def test_an_unanswered_question_is_pending_and_does_not_block_done(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Push')\n"
            "self.v.todo_id = t.id\n"
            "self.v.ref = await self.spawn('Asker', t)"
        ),
        cell(
            "[q] = notification['delegates']\n"
            "assert isinstance(q, ChildQuestion) and q.id\n"
            "assert q.id in repr(self.v.ref)\n"
            "assert self.v.ref.status == 'retained'\n"
            "return_result(Done(explanation=f'asked {q.id}'))"
        ),
    ]
    atom_models.scripts["Asker"] = [
        cell("self.todo.comment(self.todo.active(), 'found two branches')\n" + ASK)
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    [child] = atom_registry.children(root.id)
    question = atom_registry.pending_question(root, child.id)
    assert outcome.explanation == f"asked {question.id}"
    assert atom_registry.get(child.id) is not None
    # A question brings nothing back into the parent's Todo either.
    todo = root._agent.todo.get(root._agent.v.todo_id)
    assert todo.comments == []


async def test_send_answers_and_clears_the_pending_question(
    atom_registry, atom_options, atom_models
):
    """``ChildQuestion`` is data: the reply is ``child.send()``, which clears it."""
    atom_models.scripts[None] = [
        cell(
            "ref = await self.spawn('Asker', 'ask')\n"
            "q = await ref.wait()\n"
            "assert not hasattr(q, 'answer')\n"
            "assert self.session.pending_question(ref.id).id == q.id\n"
            "await q.child.send('dev')\n"
            "assert self.session.pending_question(ref.id) is None\n"
            "assert 'pending_question' not in repr(ref)\n"
            "r = await ref.wait()\n"
            "return_result(Done(explanation=r.done.result.report))"
        )
    ]
    atom_models.scripts["Asker"] = [
        cell(ASK),
        cell(
            "[a] = notification['user_messages']\n"
            "return_result(Done(explanation='pushed', result=TaskResult("
            "solution_description=a, evidence='e', how_to_verify='h', report='pushed ' + a)))"
        ),
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="pushed dev")


# ---- finding 4: a one-shot worker cannot ask, and is always closed ---------


async def test_a_delegated_worker_that_asks_is_told_to_decide(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "done = await self.delegate('Pusher', 'push the branch')\n"
            "return_result(Done(explanation=done.result.report))"
        )
    ]
    atom_models.scripts["Pusher"] = [cell(ASK), _result("assumed main")]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="assumed main")
    assert "answer a one-shot worker" in _messages(atom_models, "Pusher", 1)
    [child] = atom_registry.children(root.id)
    await until(lambda: atom_registry.get(child.id) is None)
    assert atom_registry.children(root.id)[0].status == "closed"
    assert _admitted_delegates(atom_registry, root.id) == []


async def test_a_one_shot_question_from_an_agent_without_the_check_fails_and_closes(
    registry, root_options, models
):
    """The backstop: an agent class whose turn lets a one-shot child ask anyway."""
    root = await registry.create(root_options)
    models.scripts["Asker"] = [cell("return_result(NeedInput(question='which?'))")]
    child = await registry.create(
        root.options.inherit(name="Asker", turn_method="handle"),
        parent_id=root.id,
        initial_items=[("user_messages", "go")],
    )
    port = root._agent.session
    with pytest.raises(Exception, match="one-shot worker cannot ask"):
        await asyncio.wait_for(port.wait_child(child.id), TIMEOUT)
    await until(lambda: registry.get(child.id) is None)
    assert registry.is_closed(root.id, child.id)
    assert registry.pending_question(root, child.id) is None


def test_no_doc_tells_the_model_to_catch_a_question_as_a_failure():
    from nooa_atom.session.loader import load_agent_class

    cls = load_agent_class(ATOM_SPEC)
    texts = [cls.__doc__, cls.delegate.__doc__, cls.spawn.__doc__, cls.handle_work.__doc__]
    for text in texts:
        assert "ChildQuestionError" not in text and "ChildIdleError" not in text
    assert "can't ask" in " ".join((cls.delegate.__doc__ or "").split()) or "cannot ask" in (
        cls.delegate.__doc__ or ""
    )


# ---- finding 3: wait() checks busy first; any new turn clears the question --


async def test_a_question_answered_by_the_host_is_cleared_and_wait_waits(
    atom_registry, atom_options, atom_models
):
    started, block = fresh_events()
    root = await atom_registry.create(atom_options)
    port = root._agent.session
    atom_models.scripts["Asker"] = [cell(ASK), cell(BLOCKING), _result("r")]
    ref = await port.delegate("Asker", "ask", kept=True)
    child = atom_registry.get(ref.id)
    await until(lambda: port.pending_question(ref.id) is not None, TIMEOUT)
    # A person answers through the host, not through the parent.
    await child.submit("main")
    await asyncio.wait_for(started.wait(), TIMEOUT)
    assert port.child_status(ref.id) == "running"
    assert port.pending_question(ref.id) is None
    waiting = asyncio.ensure_future(port.wait_child(ref.id))
    await asyncio.sleep(0.05)
    assert not waiting.done()
    block.set()
    outcome = await asyncio.wait_for(waiting, TIMEOUT)
    assert outcome.done.result.report == "r"


# ---- finding 5: after a reload, state comes from the record ----------------


async def test_a_pending_question_survives_a_reload_with_its_id(
    atom_options, atom_models, sessions_dir
):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=atom_models)
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Asker', 'ask')"),
        cell(
            "[q] = notification['delegates']\n"
            "self.v.question_id = q.id\n"
            "return_result(Done(explanation='not answering yet'))"
        ),
    ]
    atom_models.scripts["Asker"] = [cell(ASK)]
    root = await registry.create(atom_options)
    await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    question_id = root._agent.v.question_id
    await root.wait_for_checkpoint()
    await registry.close_all()

    later = AtomModels(
        {
            None: [
                cell(
                    "assert self.v.question_id in repr(self.children()[0])\n"
                    "try:\n"
                    "    await self.v.ref.wait()\n"
                    "except ChildFailedError as exc:\n"
                    "    self.v.waiting = str(exc)\n"
                    "await self.v.ref.send('dev')\n"
                    "assert 'pending_question' not in repr(self.v.ref)\n"
                    "r = await self.v.ref.wait()\n"
                    "return_result(Done(explanation=r.done.result.report))"
                )
            ],
            "Asker": [
                cell(
                    "[a] = notification['user_messages']\n"
                    "return_result(Done(explanation='pushed', result=TaskResult("
                    "solution_description=a, evidence='e', how_to_verify='h', report='pushed ' + a)))"
                )
            ],
        }
    )
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=later)
    try:
        loaded = await fresh.load(root.id)
        [child] = fresh.children(root.id)
        question = fresh.pending_question(loaded, child.id)
        assert question is not None and question.id == question_id
        outcome = await asyncio.wait_for(loaded.prompt("answer it"), TIMEOUT)
        assert outcome == Done(explanation="pushed dev")
        assert "is waiting for your answer" in loaded._agent.v.waiting
    finally:
        await fresh.close_all()


async def test_a_queued_outcome_is_withdrawn_by_wait_after_a_reload(
    atom_options, atom_models, sessions_dir
):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=atom_models)
    root = await registry.create(atom_options)
    atom_models.scripts["Kid"] = [_result("QUEUED")]
    # The parent is busy in a blocked turn, so the result stays queued on delegates.
    started, _block = fresh_events()
    atom_models.llms[None]._response_queue.append(cell(BLOCKING))
    await root.submit("block")
    await asyncio.wait_for(started.wait(), TIMEOUT)
    ref = await root._agent.session.delegate("Kid", "work", kept=True)
    await until(lambda: root._agent.queue_manager.get_channel("delegates").qsize() == 1, TIMEOUT)
    await registry.close_all()

    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    try:
        loaded = await fresh.load(root.id)
        # Right away, before the loaded parent's loop takes the re-queued item.
        outcome = await asyncio.wait_for(loaded._agent.session.wait_child(ref.id), TIMEOUT)
        assert outcome.done.result.report == "QUEUED"
        assert loaded._agent.queue_manager.get_channel("delegates").qsize() == 0
    finally:
        await fresh.close_all()


async def test_a_waiting_child_still_reads_working_after_a_reload(
    atom_options, atom_models, sessions_dir
):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=atom_models)
    root = await registry.create(atom_options)
    atom_models.scripts["Slow"] = [
        cell("return_result(Waiting(explanation='job running', on=['jobs']))")
    ]
    ref = await root._agent.session.delegate("Slow", "start a job", kept=True)
    await until(lambda: registry.child_status(root, ref.id) == "working", TIMEOUT)
    await registry.close_all()

    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    try:
        loaded = await fresh.load(root.id)
        await fresh.load(ref.id)
        port = loaded._agent.session
        assert port.child_status(ref.id) == "working"
        waiting = asyncio.ensure_future(port.wait_child(ref.id))
        await asyncio.sleep(0.05)
        assert not waiting.done()  # its job still runs: wait() waits, it does not fail
        waiting.cancel()
    finally:
        await fresh.close_all()


# ---- finding 6: tombstones are read once per parent ------------------------


async def test_closed_children_are_read_once_per_parent(atom_registry, atom_options, monkeypatch):
    root = await atom_registry.create(atom_options)
    kids = [
        await atom_registry.create(
            root.options.inherit(name=f"kid{i}", kept=True), parent_id=root.id
        )
        for i in range(3)
    ]
    reads = []
    load_rows = atom_registry.store.load_rows

    def counting(session_id, event_types=None):
        if event_types and "ChildClosed" in event_types:
            reads.append(session_id)
        return load_rows(session_id, event_types)

    monkeypatch.setattr(atom_registry.store, "load_rows", counting)
    port = root._agent.session
    for _ in range(3):
        atom_registry.children(root.id)
        for kid in kids:
            port.child_status(kid.id)
            repr(atom_registry.child_ref(kid))
    await port.close_child(kids[0].id)
    assert atom_registry.is_closed(root.id, kids[0].id)
    assert port.child_status(kids[0].id) == "closed"
    assert len(reads) <= 1


# ---- finding 10: a Done's message is marked on the event -------------------


async def test_a_done_message_also_sent_with_message_is_delivered_once(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then("await self.spawn('Kid', 'work')"),
        done("got it"),
    ]
    atom_models.scripts["Kid"] = [
        cell(
            "self.message('final words')\n"
            "return_result(Done(explanation='ok', message='final words', result=TaskResult("
            "solution_description='s', evidence='e', how_to_verify='h')))"
        )
    ]
    root = await atom_registry.create(atom_options)
    await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert _admitted_delegates(atom_registry, root.id) == ["ChildMessage", "ChildResult"]


async def test_a_result_message_handled_late_is_not_relayed(registry, root_options):
    """The marker is on the event itself, so deferred handling cannot misroute it."""
    from nooa_atom.session.items import AgentEventUpdate

    from nooa.interactive import AgentMessage

    root = await registry.create(root_options)
    child = await registry.create(root.options.inherit(name="Kid", kept=True), parent_id=root.id)
    late = AgentMessage(content="the Done's text", is_result_text=True)
    registry._deliver(child, AgentEventUpdate(session_id=child.id, event=late))
    registry._deliver(
        child, AgentEventUpdate(session_id=child.id, event=AgentMessage(content="a note"))
    )
    delegates = root._agent.queue_manager.get_channel("delegates")
    await until(lambda: delegates.qsize() == 1)
    assert _admitted_delegates(registry, root.id) == ["ChildMessage"]


# ---- model-facing guidance ---------------------------------------------------


def test_the_parent_docs_say_what_to_put_in_the_todo_and_who_edits():
    from nooa_atom.session.loader import load_agent_class

    cls = load_agent_class(ATOM_SPEC)
    text = " ".join((cls.__doc__ or "").split())
    assert "requirements and its rules or constraints" in text
    assert "one editing child at a time" in text
    for removed in ("retain", "keep=", "mode=", "steer"):
        assert removed not in text
    work = " ".join((cls.handle_work.__doc__ or "").split())
    assert "remove scratch files" in work and "TaskResult" in work
    assert "context" not in work


# ---- send(todo): a Todo crosses sessions as data ------------------------------


def test_a_todo_survives_the_session_record_and_the_send_copy():
    from nooa_atom.session.session import as_data, item_to_json

    from nooa.tools.todo import Todo

    todos = TodoManager()
    empty = todos.add("no vars")
    withvars = todos.add("with vars", description="D")
    todos.set_var(withvars, "child", "abc123")
    todos.set_var(withvars, "n", 3)
    todos.comment(withvars, "note")
    for todo in (empty, withvars):
        record = json.loads(item_to_json(todo))
        assert record["vars"] == dict(todo.vars)
        copy = as_data(todo)
        assert isinstance(copy, Todo) and copy is not todo and copy.vars is not todo.vars
        assert copy.model_dump(mode="json") == todo.model_dump(mode="json")
        assert type(copy.vars).__name__ == "SnapshotVars"
        assert Todo.model_validate_json(item_to_json(todo)) == copy


def test_a_todo_with_a_non_json_var_is_refused_when_sent_naming_the_key():
    from nooa_atom.session.session import as_data, item_to_json
    from pydantic_core import PydanticSerializationError

    todos = TodoManager()
    todo = todos.add("with a tuple")
    todos.set_var(todo, "pair", (1, 2))
    for send_copy in (as_data, item_to_json):
        with pytest.raises(PydanticSerializationError, match="var 'pair' holds a tuple"):
            send_copy(todo)


async def test_send_hands_a_todo_to_a_kept_child(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Kept', 'first task')"),
        cell(
            "a = self.todo.add('Empty vars')\n"
            "b = self.todo.add('With vars', description='REQ', owner='root')\n"
            "self.todo.set_var(b, 'n', 3)\n"
            "await self.v.ref.send(a)\n"
            "await self.v.ref.send(b)\n"
            "return_result(Waiting(explanation='sent', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "return_result(Done(explanation=item.done.result.report))"
        ),
    ]
    atom_models.scripts["Kept"] = [
        cell(CHILD_RESULT),
        cell(
            "items = notification['user_messages']\n"
            "seen = '; '.join(f'{type(t).__name__} {t.title} {sorted(t.vars.items())}' "
            "for t in items)\n"
            "return_result(Done(explanation='got', result=TaskResult("
            "solution_description='s', evidence='e', how_to_verify='h', report=seen)))"
        ),
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(
        explanation="Todo Empty vars []; Todo With vars [('n', 3), ('owner', 'root')]"
    )


# ---- send(todo) is an ordinary item; old records still load ---------------


async def test_sending_the_briefed_todo_is_an_ordinary_item(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Task', description='REQ v1')\n"
            "self.v.todo_id = t.id\n"
            "self.v.ref = await self.spawn('Kept', t)"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "t = self.todo.get(self.v.todo_id)\n"
            "self.todo.update(t, status='open', description='REQ v2')\n"
            "await self.v.ref.send(t)\n"
            "await self.v.ref.send('just text')\n"
            "return_result(Waiting(explanation='sent', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "return_result(Done(explanation=item.done.result.report))"
        ),
    ]
    atom_models.scripts["Kept"] = [
        cell("self.todo.complete(self.todo.active())\n" + CHILD_RESULT),
        cell(
            "items = notification['user_messages']\n"
            "mine = self.todo.list_todos()\n"
            "report = (f'{[type(i).__name__ for i in items]} {items[0].description} '\n"
            "    f'{[(t.status, t.description) for t in mine]}')\n"
            "return_result(Done(explanation='ok', result=TaskResult("
            "solution_description='s', evidence='e', how_to_verify='h', report=report)))"
        ),
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    # The Todo arrives as data; the child's own copy is untouched until it acts.
    assert outcome == Done(explanation="['Todo', 'str'] REQ v2 [('done', 'REQ v1')]")
    todo = root._agent.todo.get(root._agent.v.todo_id)
    assert (todo.status, todo.description, todo.comments) == ("open", "REQ v2", [])


async def test_an_old_record_with_a_child_todo_base_row_reloads(
    atom_options, atom_models, sessions_dir
):
    """Records written while #451 merged Todos back hold ``ChildTodoBase`` rows; they are ignored."""
    from typing import ClassVar

    from nooa.context_blocks import Metadata
    from nooa.context_blocks.models import Role

    class ChildTodoBase(Metadata):  # the removed event type, as an older build defined it
        _role: ClassVar[Role] = Role.METADATA
        child_id: str = ""
        todo_json: str = ""

    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=atom_models)
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Task')\n"
            "self.v.todo_id = t.id\n"
            "self.v.ref = await self.spawn('Kept', t)"
        ),
        done("first"),
    ]
    atom_models.scripts["Kept"] = [cell(CHILD_RESULT)]
    try:
        root = await registry.create(atom_options)
        assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="first")
        [child] = registry.children(root.id)
        # The row an older build wrote when it handed the child its Todo.
        old_row = ChildTodoBase(
            child_id=child.id, todo_json=json.dumps({"blob": {}, "allowlist": []})
        )
        root.handle.events.add(old_row)
        await root.wait_for_checkpoint()
        root_id, child_id = root.id, child.id
    finally:
        await registry.close_all()
    assert [
        kind
        for kind, _ in SessionStore(sessions_dir).load_rows(root_id, frozenset({"ChildTodoBase"}))
    ] == ["ChildTodoBase"]

    later = AtomModels(
        {
            None: [
                _wait_then("await self.v.ref.send('one more')"),
                cell(
                    "[item] = notification['delegates']\n"
                    "return_result(Done(explanation=item.done.result.report))"
                ),
            ],
            "Kept": [cell(CHILD_RESULT)],
        }
    )
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=later)
    try:
        loaded = await fresh.load(root_id)
        assert await asyncio.wait_for(loaded.prompt("follow up"), TIMEOUT) == Done(
            explanation="REPORT: off-by-one fixed"
        )
        assert [c.id for c in fresh.children(root_id)] == [child_id]
        assert "ChildTodoBase" not in _messages(later, None)
    finally:
        await fresh.close_all()


async def test_spawn_and_send_refuse_the_same_non_json_todo_var(
    atom_registry, atom_options, atom_models
):
    """Review-3 finding 2: the in-process hand-over applies send()'s JSON check."""
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Kept', 'first task')"),
        cell(
            "from pydantic_core import PydanticSerializationError\n"
            "t = self.todo.add('With a tuple')\n"
            "self.todo.set_var(t, 'pair', (1, 2))\n"
            "errors = []\n"
            "try:\n"
            "    await self.spawn('S', t)\n"
            "except PydanticSerializationError as exc:\n"
            "    errors.append(\"var 'pair' holds a tuple\" in str(exc))\n"
            "try:\n"
            "    await self.delegate('D', t)\n"
            "except PydanticSerializationError as exc:\n"
            "    errors.append(\"var 'pair' holds a tuple\" in str(exc))\n"
            "try:\n"
            "    await self.v.ref.send(t)\n"
            "except PydanticSerializationError as exc:\n"
            "    errors.append(\"var 'pair' holds a tuple\" in str(exc))\n"
            "return_result(Done(explanation=f'{errors} {len(self.children())}'))"
        ),
    ]
    atom_models.scripts["Kept"] = [cell(CHILD_RESULT)]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    # All three refuse naming the key, and no second child was started.
    assert outcome == Done(explanation="[True, True, True] 1")
