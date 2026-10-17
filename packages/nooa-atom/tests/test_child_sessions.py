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


# ---- the Todo as the shared contract, the brief, the parent line -----------


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


async def test_the_child_works_on_a_copy_that_merges_back(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Fix the parser', description='REQ', owner='root')\n"
            "self.todo.comment(t, 'parent note')\n"
            "self.v.todo_id = t.id\n"
            "self.v.ref = await self.spawn('Parser', t)"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "assert isinstance(item, ChildResult)\n"
            "assert item.child.todo_id == self.v.todo_id == self.v.ref.todo_id\n"
            "return_result(Done(explanation='merged'))"
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
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="merged")
    todo_id = root._agent.v.todo_id
    [child] = atom_registry.children(root.id)
    assert child.todo_id == todo_id
    kid = atom_registry.get(child.id)
    assert kid is None or kid._agent.todo.get(todo_id) is not root._agent.todo.get(todo_id)
    todo = root._agent.todo.get(todo_id)
    assert [c.body for c in todo.comments] == ["parent note", "child evidence: tests pass"]
    assert todo.status == "done"
    assert (todo.vars["patch"], todo.vars["owner"], todo.vars["child"]) == (
        "parser.diff",
        "root",
        child.id,
    )
    # The requirements are the parent's: the child's edit is not merged.
    assert todo.description == "REQ"


async def test_a_kept_childs_todo_merges_on_each_result(atom_registry, atom_options, atom_models):
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
        done("finished"),
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
    assert [c.body for c in todo.comments] == [
        "round 1",
        "parent review: one more check",
        "round 2",
    ]
    assert todo.status == "done"


async def test_a_hand_built_todo_is_adopted_and_its_merge_back_reaches_the_callers_object(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then(
            "from nooa.tools.todo import Todo as _Todo\n"
            "t = _Todo(title='stray', description='REQ', deps=['not-yet'])\n"
            "self.v.todo_id = t.id\n"
            "import atom_test_agents as _t\n"
            "_t.HELD = t\n"
            "self.v.ref = await self.spawn('X', t)\n"
            "assert self.todo.get(t.id) is t"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "import atom_test_agents as _t\n"
            "assert self.todo.get(self.v.todo_id) is _t.HELD\n"
            "return_result(Done(explanation=_t.HELD.status + ' ' + _t.HELD.comments[-1].body))"
        ),
    ]
    atom_models.scripts["X"] = [
        cell(
            "t = self.todo.active()\n"
            "assert t.description == 'REQ'\n"
            "self.todo.comment(t, 'child evidence')\n"
            "self.todo.complete(t)\n" + CHILD_RESULT
        )
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(
        explanation="done child evidence"
    )
    [child] = atom_registry.children(root.id)
    todo = root._agent.todo.get(root._agent.v.todo_id)
    assert child.todo_id == todo.id and todo.vars["child"] == child.id
    assert todo.deps == ["not-yet"]


async def test_a_different_todo_reusing_a_managed_id_is_refused(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "from nooa.tools.todo import Todo as _Todo\n"
            "self.todo.add('first').id\n"
            "kept = self.todo.add('recon task')\n"
            "self.todo.update(kept, description='keep me')\n"
            "kept_id = kept.id\n"
            "try:\n"
            "    await self.spawn('X', _Todo(id=kept_id, title='other'))\n"
            "except ValueError as exc:\n"
            "    assert self.todo.get(kept_id) is kept and kept.description == 'keep me'\n"
            "    return_result(Done(explanation=str(exc).replace(kept_id, 'ID')))"
        )
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(
        explanation="a different Todo with id 'ID' is already managed; "
        "pass self.todo.get('ID') or drop id="
    )
    assert atom_registry.children(root.id) == []


def test_adopt_keeps_the_object_and_refuses_only_a_different_one():
    from nooa.tools.todo import Todo

    todos = TodoManager()
    managed = todos.add("managed")
    assert todos.adopt(managed) is managed
    loose = Todo(id="recon", title="loose", deps=["unknown"])
    assert todos.adopt(loose) is loose and todos.get("recon") is loose
    assert [t.id for t in todos.list_todos()] == [managed.id, "recon"]
    with pytest.raises(ValueError, match="a different Todo with id 'recon' is already managed"):
        todos.adopt(Todo(id="recon", title="again"))
    assert todos.get("recon") is loose
    copy = todos.copy_todo(loose)
    assert copy == loose and copy is not loose


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
            "await q.answer('main')\n"
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
    # The child's Todo comments merged in when the question was delivered.
    todo = root._agent.todo.get(root._agent.v.todo_id)
    assert [c.body for c in todo.comments] == ["found two branches"]


async def test_answering_a_stale_question_id_is_refused(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        cell(
            "ref = await self.spawn('Asker', 'ask')\n"
            "q = await ref.wait()\n"
            "stale = q.model_copy(update={'id': 'stale000'})\n"
            "try:\n"
            "    await stale.answer('x')\n"
            "except ChildFailedError as exc:\n"
            "    return_result(Done(explanation=str(exc)))"
        )
    ]
    atom_models.scripts["Asker"] = [cell(ASK)]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert "is not the child's pending question" in outcome.explanation


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


@pytest.mark.parametrize("spec", [ATOM_SPEC, "nooa_atom.agent:ExperimentalAtomAgent"])
def test_no_doc_tells_the_model_to_catch_a_question_as_a_failure(spec):
    from nooa_atom.session.loader import load_agent_class

    cls = load_agent_class(spec)
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
                    "from nooa_atom.session.items import ChildQuestion as _Q\n"
                    "q = _Q(id=self.v.question_id, child=self.v.ref, question='Which branch?')\n"
                    "try:\n"
                    "    await q.model_copy(update={'id': 'stale000'}).answer('x')\n"
                    "except ChildFailedError as exc:\n"
                    "    self.v.stale = str(exc)\n"
                    "await q.answer('dev')\n"
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
        assert "is not the child's pending question" in loaded._agent.v.stale
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


# ---- findings 1 and 2: the Todo merge base and partial merges --------------


def _handed(phase="1"):
    parent = SimpleNamespace(todo=TodoManager())
    todo = parent.todo.add("task", description="d", phase=phase)
    handed = parent.todo.copy_todo(todo)
    child = TodoManager()
    child.receive_todo(handed)
    return parent, todo, handed.model_copy(deep=True), child, child.get(todo.id)


def _merge(parent, updated, base):
    from nooa_atom.agent.agent import AtomAgent

    return AtomAgent.merge_delegated_todo(parent, "kid", updated, base)


def test_a_merge_without_a_base_takes_only_comments_and_done():
    """Finding 1: never use the parent's current Todo as the base."""
    parent, todo, _base, child, copy = _handed()
    child.comment(copy, "child progress")
    child.complete(copy)
    parent.todo.set_var(todo, "phase", "2")
    assert _merge(parent, copy, None) is None
    assert (todo.status, todo.vars["phase"]) == ("done", "2")
    bodies = [c.body for c in todo.comments]
    assert bodies[0] == "child progress"
    assert "no merge base was recorded" in bodies[1] and "done status" in bodies[1]


def test_a_merge_without_a_base_does_not_reopen_a_finished_todo():
    parent, todo, _base, child, copy = _handed()
    child.comment(copy, "child progress")
    parent.todo.set_var(todo, "phase", "2")
    parent.todo.complete(todo)
    _merge(parent, copy, None)
    assert (todo.status, todo.vars["phase"]) == ("done", "2")


def test_a_conflict_merges_the_rest_and_is_retried():
    """Finding 2: one conflicting field no longer rejects comments and status."""
    parent, todo, base, child, copy = _handed()
    child.comment(copy, "evidence A")
    child.set_var(copy, "x", "child")
    child.complete(copy)
    parent.todo.set_var(todo, "x", "parent")
    new_base = _merge(parent, copy, base)
    assert todo.status == "done"
    assert todo.vars["x"] == "parent"
    bodies = [c.body for c in todo.comments]
    assert bodies[0] == "evidence A"
    [note] = bodies[1:]
    assert "not merged" in note and "'x'" in note
    # The base advanced for status and comments, not for the conflicting var.
    assert new_base.status == "done" and "x" not in new_base.vars
    # Once the parent agrees (here: removes its value), the child's change applies.
    parent.todo.set_var(todo, "x", None)
    del todo.vars["x"]
    child.comment(copy, "evidence B")
    _merge(parent, copy, new_base)
    assert todo.vars["x"] == "child"
    assert [c.body for c in todo.comments][-1] == "evidence B"


def test_a_repeated_conflict_is_noted_once_and_again_when_its_values_change():
    """Each distinct conflict (child, field, values) gets one note, across merges and a reload."""
    parent, todo, base, child, copy = _handed()
    child.set_var(copy, "x", "child")
    parent.todo.set_var(todo, "x", "parent")

    def notes(t):
        return [c.body for c in t.comments if "not merged" in c.body]

    new_base = _merge(parent, copy, base)
    new_base = _merge(parent, copy, new_base)
    new_base = _merge(parent, copy, new_base)
    [note] = notes(todo)
    assert "var 'x'" in note and "'child'" in note and "'parent'" in note
    # The child changes its value: the conflict is a new one and gets a second note.
    child.set_var(copy, "x", "child 2")
    new_base = _merge(parent, copy, new_base)
    new_base = _merge(parent, copy, new_base)
    assert len(notes(todo)) == 2 and "'child 2'" in notes(todo)[1]
    # After a reload (the Todo and base rebuilt from their saved data), the same conflict adds none.
    from nooa.storage.serialization import deserialize, serialize

    def saved(value):
        blob, allowlist = serialize(value)
        return deserialize(blob, allowlist)

    reloaded = SimpleNamespace(todo=TodoManager())
    restored = reloaded.todo.receive_todo(saved(todo))
    _merge(reloaded, copy, saved(new_base))
    assert notes(restored) == notes(todo)
    # Another child's identical conflict is its own.
    from nooa_atom.agent.agent import AtomAgent

    AtomAgent.merge_delegated_todo(reloaded, "other", copy, new_base)
    assert len(notes(restored)) == 3


def test_done_does_not_override_a_status_the_parent_changed():
    parent, todo, base, child, copy = _handed()
    parent.todo.complete(todo)
    todo.status = "open"  # the parent reopened it itself after the hand-over
    base.status = "done"
    child.complete(copy)
    copy.status = "open"
    _merge(parent, copy, base)
    assert todo.status == "open"


async def test_the_merge_base_survives_a_parent_reload(atom_options, atom_models, sessions_dir):
    """Finding 1: a kept child delivering after a restart must not reopen finished work."""
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=atom_models)
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Task', phase='1')\n"
            "self.v.todo_id = t.id\n"
            "self.v.ref = await self.spawn('Kept', t)"
        ),
        cell(
            "self.todo.set_var(self.v.todo_id, 'phase', '2')\n"
            "self.todo.complete(self.v.todo_id)\n"
            "return_result(Done(explanation='moved on'))"
        ),
    ]
    atom_models.scripts["Kept"] = [
        cell("self.todo.comment(self.todo.active(), 'one')\n" + CHILD_RESULT)
    ]
    root = await registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="moved on")
    await root.wait_for_checkpoint()
    await registry.close_all()

    later = AtomModels(
        {
            None: [
                _wait_then("await self.v.ref.send('one more')"),
                done("merged"),
            ],
            "Kept": [
                cell("t = self.todo.list_todos()[0]\nself.todo.comment(t, 'two')\n" + CHILD_RESULT)
            ],
        }
    )
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=later)
    try:
        loaded = await fresh.load(root.id)
        assert await asyncio.wait_for(loaded.prompt("follow up"), TIMEOUT) == Done(
            explanation="merged"
        )
        todo = loaded._agent.todo.get(loaded._agent.v.todo_id)
        assert (todo.status, todo.vars["phase"]) == ("done", "2")
        assert [c.body for c in todo.comments] == ["one", "two"]
    finally:
        await fresh.close_all()


async def test_the_session_port_records_the_base_too(atom_registry, atom_options, atom_models):
    """Finding 1: ``self.session.delegate(todo=...)`` records the base, not only ``spawn()``."""
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Task', phase='1')\n"
            "self.v.todo_id = t.id\n"
            "await self.session.delegate('Kid', 'work', todo=self.todo.copy_todo(t), kept=True)\n"
            "self.todo.set_var(t, 'phase', '2')"
        ),
        done("merged"),
    ]
    atom_models.scripts["Kid"] = [
        cell("self.todo.comment(self.todo.active(), 'progress')\n" + CHILD_RESULT)
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="merged")
    todo = root._agent.todo.get(root._agent.v.todo_id)
    assert todo.vars["phase"] == "2"
    assert [c.body for c in todo.comments] == ["progress"]
    rows = atom_registry.store.load_rows(root.id, frozenset({"ChildTodoBase"}))
    assert len(rows) == 2  # the hand-over, then the base after the merge


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


@pytest.mark.parametrize("spec", [ATOM_SPEC, "nooa_atom.agent:ExperimentalAtomAgent"])
def test_the_parent_docs_say_what_to_put_in_the_todo_and_who_edits(spec):
    from nooa_atom.session.loader import load_agent_class

    cls = load_agent_class(spec)
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


# ---- send(todo) with the handed Todo: the parent-to-child merge ------------


async def test_a_reopened_todo_sent_again_is_completed_and_merged_once_more(
    atom_registry, atom_options, atom_models
):
    """vuls-7e91: the parent reopens a merged Todo and sends it; the child's re-completion counts."""
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Fix scanner')\n"
            "self.v.todo_id = t.id\n"
            "self.v.ref = await self.spawn('Kept', t)"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "assert self.todo.get(self.v.todo_id).status == 'done'\n"
            "self.todo.update(self.v.todo_id, status='open')\n"
            "await self.v.ref.send(self.todo.get(self.v.todo_id))\n"
            "return_result(Waiting(explanation='reopened', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "return_result(Done(explanation=self.todo.get(self.v.todo_id).status))"
        ),
    ]
    atom_models.scripts["Kept"] = [
        cell(
            "self.todo.comment(self.todo.active(), 'round 1')\nself.todo.complete(self.todo.active())\n"
            + CHILD_RESULT
        ),
        cell(
            "[line] = notification['user_messages']\n"
            "t = self.todo.list_todos()[0]\n"
            "assert self.todo.active() is t and t.status == 'open', (line, t)\n"
            "self.todo.comment(t, 'round 2: ' + line)\n"
            "self.todo.complete(t)\n" + CHILD_RESULT
        ),
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="done")
    todo = root._agent.todo.get(root._agent.v.todo_id)
    assert [c.body for c in todo.comments] == [
        "round 1",
        # ``vars``: the parent's Todo records the child's id after the hand-over.
        f"round 2: Your parent updated Todo {todo.id}: status open; vars changed.",
    ]
    assert "Your parent updated Todo" in _messages(atom_models, "Kept", 1)


async def test_a_sent_todo_carries_the_parents_changes_and_keeps_both_sides_comments(
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
            "self.todo.update(t, description='REQ v2')\n"
            "self.todo.comment(t, 'parent: also cover the CLI')\n"
            "self.todo.set_var(t, 'scope', 'cli')\n"
            "await self.v.ref.send(t)\n"
            "return_result(Waiting(explanation='updated', on=['delegates']))"
        ),
        cell("return_result(Done(explanation='finished'))"),
    ]
    atom_models.scripts["Kept"] = [
        cell("self.todo.comment(self.todo.active(), 'child: round 1')\n" + CHILD_RESULT),
        cell(
            "[line] = notification['user_messages']\n"
            "t = self.todo.list_todos()[0]\n"
            "self.v.seen = (line, t.description, [c.body for c in t.comments], t.vars.get('scope'))\n"
            "self.todo.comment(t, 'child: round 2')\n" + CHILD_RESULT
        ),
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="finished")
    todo = root._agent.todo.get(root._agent.v.todo_id)
    [child] = atom_registry.children(root.id)
    kid = (
        await atom_registry.open_child(root, child.id)
        if atom_registry.get(child.id) is None
        else atom_registry.get(child.id)
    )
    line, description, comments, scope = kid._agent.v.seen
    assert line == (
        f"Your parent updated Todo {todo.id}: description changed; vars changed; 1 new comment."
    )
    assert (description, scope) == ("REQ v2", "cli")
    assert comments == ["child: round 1", "parent: also cover the CLI"]
    assert [c.body for c in todo.comments] == [
        "child: round 1",
        "parent: also cover the CLI",
        "child: round 2",
    ]
    assert todo.description == "REQ v2"


async def test_the_base_a_sent_todo_records_survives_a_parent_reload(
    atom_options, atom_models, sessions_dir
):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=atom_models)
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Task')\n"
            "self.v.todo_id = t.id\n"
            "self.v.ref = await self.spawn('Kept', t)"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "self.todo.update(self.v.todo_id, status='open', description='again')\n"
            "await self.v.ref.send(self.todo.get(self.v.todo_id))\n"
            "return_result(Done(explanation='sent'))"
        ),
    ]
    atom_models.scripts["Kept"] = [
        cell("self.todo.complete(self.todo.active())\n" + CHILD_RESULT),
        cell("return_result(Waiting(explanation='parked', on=['jobs']))"),
    ]
    try:
        root = await registry.create(atom_options)
        assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="sent")
        [child] = registry.children(root.id)
        root_id, child_id = root.id, child.id
        todo_id = root._agent.v.todo_id
    finally:
        await registry.close_all()
    reloaded = SessionRegistry(SessionStore(sessions_dir), agent_factory=atom_models)
    try:
        base = reloaded._todo_bases_of(root_id)[child_id]
        assert (base.id, base.status, base.description) == (todo_id, "open", "again")
    finally:
        await reloaded.close_all()


async def test_a_todo_with_another_id_and_a_string_stay_ordinary_items(
    atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        _wait_then(
            "t = self.todo.add('Task')\n"
            "self.v.todo_id = t.id\n"
            "self.v.ref = await self.spawn('Kept', t)"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "self.todo.update(self.v.todo_id, status='open')\n"
            "await self.v.ref.send(self.todo.add('Unrelated'))\n"
            "await self.v.ref.send('just text')\n"
            "return_result(Waiting(explanation='sent', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\nreturn_result(Done(explanation=item.done.result.report))"
        ),
    ]
    atom_models.scripts["Kept"] = [
        cell("self.todo.complete(self.todo.active())\n" + CHILD_RESULT),
        cell(
            "items = notification['user_messages']\n"
            "mine = [t for t in self.todo.list_todos()]\n"
            "report = f'{[type(i).__name__ for i in items]} {[t.status for t in mine]}'\n"
            "return_result(Done(explanation='ok', result=TaskResult("
            "solution_description='s', evidence='e', how_to_verify='h', report=report)))"
        ),
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    # The child's copy is untouched (still done) and both items arrive as sent.
    assert outcome == Done(explanation="['Todo', 'str'] ['done']")
