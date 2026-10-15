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
    assert {"mode", "keep", "brief"} <= set(params)
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


# ---- D3/D4: the Todo as the shared contract, the brief, the parent line ----


async def test_a_todo_and_a_brief_both_reach_the_child_with_the_parent_line(
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
        f"You are working for parent root ({root.id}) on Todo {todo_id}: Fix the parser" in prompt
    )
    assert f"Task [{todo_id}]: Fix the parser" in prompt
    for text in ("REQ: parse() keeps the last line", "NOTE: suspect the loop bound"):
        assert text in prompt
    assert "BRIEF: start from parse_header()" in prompt


async def test_a_text_prompt_gets_the_parent_line_and_the_brief(
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
    assert f"You are working for parent root ({root.id})." in prompt
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
    # The child saw its own copy, as its active Todo.
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
            "self.v.ref = await self.spawn('Kept', t, keep=True)"
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


async def test_a_todo_not_in_the_parents_list_is_refused(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        cell(
            "from nooa.tools.todo import Todo as _Todo\n"
            "try:\n"
            "    await self.spawn('X', _Todo(title='stray'))\n"
            "except ValueError as exc:\n"
            "    return_result(Done(explanation='refused'))"
        )
    ]
    root = await atom_registry.create(atom_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="refused")
    assert atom_registry.children(root.id) == []


# ---- D5: wait() returns an outcome and never hangs; tracked questions -------


ASK = "return_result(NeedInput(question='Which branch?', options=['main', 'dev']))"


def _admitted_delegates(registry, session_id):
    rows = registry.store.load_rows(session_id, frozenset({"ItemAdmitted"}))
    return [
        str(raw["item_type"]).rsplit(".", 1)[-1].rsplit(":", 1)[-1]
        for _t, raw in rows
        if raw.get("channel") == "delegates"
    ]


async def test_wait_returns_a_question_then_the_result(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        cell(
            "ref = await self.spawn('Pusher', 'push the branch')\n"
            "q = await ref.wait()\n"
            "assert isinstance(q, ChildQuestion) and q.options == ['main', 'dev']\n"
            "assert ref.pending_question.id == q.id\n"
            "assert f'pending_question=\\'[{q.id}] Which branch?' in repr(self.children()[0])\n"
            "try:\n"
            "    await ref.wait()\n"
            "except ChildIdleError as exc:\n"
            "    self.v.idle = str(exc)\n"
            "await q.answer('main')\n"
            "assert ref.pending_question is None\n"
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
    assert "waits for your answer" in root._agent.v.idle
    # Both outcomes went to wait(), none to delegates: one place only.
    assert _admitted_delegates(atom_registry, root.id) == []


async def test_wait_fails_fast_on_an_idle_kept_child(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        _wait_then("self.v.ref = await self.spawn('Kept', 'task', keep=True)"),
        cell(
            "[item] = notification['delegates']\n"
            "try:\n"
            "    await self.v.ref.wait()\n"
            "except ChildIdleError as exc:\n"
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
            "assert self.v.ref.pending_question == q\n"
            "assert self.v.ref.status == 'idle'\n"
            "return_result(Done(explanation=f'asked {q.id}'))"
        ),
    ]
    atom_models.scripts["Asker"] = [
        cell("self.todo.comment(self.todo.active(), 'found two branches')\n" + ASK)
    ]
    root = await atom_registry.create(atom_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome.explanation.startswith("asked ")
    [child] = atom_registry.children(root.id)
    question = atom_registry.pending_question(root, child.id)
    assert outcome.explanation == f"asked {question.id}"
    # A question is not a result: a keep=False worker stays open for the answer.
    assert atom_registry.get(child.id) is not None
    # The child's Todo comments merged in when the question was delivered.
    todo = root._agent.todo.get(root._agent.v.todo_id)
    assert [c.body for c in todo.comments] == ["found two branches"]


async def test_delegate_raises_with_the_childs_question(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        cell(
            "try:\n"
            "    await self.delegate('Asker', 'push the branch')\n"
            "except ChildQuestionError as exc:\n"
            "    await exc.question.answer('dev')\n"
            "    r = await exc.question.child.wait()\n"
            "    return_result(Done(explanation=r.done.result.report))"
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
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="pushed dev")


async def test_answering_a_stale_question_id_is_refused(atom_registry, atom_options, atom_models):
    atom_models.scripts[None] = [
        cell(
            "ref = await self.spawn('Asker', 'ask', keep=True)\n"
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


# ---- D4/D6: model-facing guidance -------------------------------------------


@pytest.mark.parametrize("spec", [ATOM_SPEC, "nooa_atom.agent:ExperimentalAtomAgent"])
def test_the_parent_docs_say_what_to_put_in_the_todo_and_who_edits(spec):
    from nooa_atom.session.loader import load_agent_class

    cls = load_agent_class(spec)
    text = " ".join((cls.__doc__ or "").split())
    assert "requirements and its rules or constraints" in text
    assert "one editing child at a time" in text
    assert "retain" not in text
    work = " ".join((cls.handle_work.__doc__ or "").split())
    assert "remove scratch files" in work and "TaskResult" in work
