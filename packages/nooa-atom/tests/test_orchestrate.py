# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``handle_orchestrate``: an unattended root that plans, delegates and verifies."""

import asyncio
import hashlib

import pytest
from atom_test_agents import ATOM_SPEC, AtomModels, CellLLM, cell
from nooa_atom import SessionOptions, SessionStore, run_task
from nooa_atom.agent import AtomAgent
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import sessions_root
from pydantic import ValidationError

from nooa.agentdoc import doc

TIMEOUT = 30

# Phrases only the orchestrator prompt has.
ORCHESTRATOR_TEXT = (
    "You plan, dispatch and verify. Do not edit files or implement the",
    "Put each piece of work in a Todo",
    "await self.spawn(description, todo, retain=True)",
    "await child.send(feedback)",
    "acceptance checks",
)

# The interactive and batch prompts this change must leave as they are.
# Changing either prompt on purpose means updating its digest here.
UNCHANGED_PROMPTS = {
    "handle": "080869ff86083cf63ef079eecf34c48d13a696050dfc2cb8048b97c9e1ee176b",
    "handle_batch": "9124c2874abd30f61a26f313abacbdc925f02a20efac8b4d7e3319a22ab545f0",
}

RESULT = (
    "return_result(Done(explanation='verified', result=TaskResult("
    "solution_description='the worker fixed parse()', evidence='pytest passed', "
    "how_to_verify='pytest', report='VERIFIED')))"
)


def test_handle_orchestrate_has_the_orchestrator_prompt():
    from nooa.strategies import CodeActV2

    prompt = AtomAgent.handle_orchestrate.__doc__
    for phrase in ORCHESTRATOR_TEXT:
        assert phrase in prompt
    assert isinstance(AtomAgent.handle_orchestrate._plan_strategy, CodeActV2)


@pytest.mark.parametrize("method", list(UNCHANGED_PROMPTS))
def test_the_interactive_and_batch_prompts_are_unchanged(method):
    prompt = getattr(AtomAgent, method).__doc__
    assert hashlib.sha256(prompt.encode()).hexdigest() == UNCHANGED_PROMPTS[method]
    for phrase in ORCHESTRATOR_TEXT:
        assert phrase not in prompt


async def test_handle_orchestrate_is_hidden_from_the_model(tmp_path):
    """Hosts call it as a turn method; the model never sees it listed."""
    assert "handle_orchestrate" not in doc(AtomAgent, concise=True)
    assert "handle_orchestrate" not in doc(AtomAgent)
    agent = AtomAgent(llm=CellLLM([cell("return_result(Done(explanation='x'))")]), cwd=tmp_path)
    try:
        await agent.handle({"user_messages": ["hi"]})
        call = agent.llm.calls[0]
        # The workspace path carries this test's name; leave it out.
        rendered = str(call.messages).replace(str(tmp_path), "<workspace>")
        assert "handle_orchestrate" not in rendered
        assert "handle_orchestrate" not in str(call.tools)
    finally:
        await agent.aclose()


def test_the_turn_method_comes_from_a_known_set(tmp_path):
    SessionOptions(workspace=tmp_path, agent_spec=ATOM_SPEC, turn_method="handle_orchestrate")
    with pytest.raises(ValidationError):
        SessionOptions(workspace=tmp_path, agent_spec=ATOM_SPEC, turn_method="handle_orchestrat")


def _no_llm(alias, workspace):
    """No real client: AtomModels gives each session its scripted fake model."""
    return None


def _options(tmp_path, sessions_dir, **update):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return SessionOptions(
        workspace=workspace, agent_spec=ATOM_SPEC, sessions_dir=sessions_dir, **update
    )


async def test_run_task_runs_handle_batch_by_default(tmp_path, sessions_dir):
    models = AtomModels({None: [cell(RESULT)]})
    options = _options(tmp_path, sessions_dir)
    run = await asyncio.wait_for(
        run_task(options, "fix it", agent_factory=models, llm_factory=_no_llm), TIMEOUT
    )
    assert run.result is not None
    info = SessionStore(sessions_root(options.workspace, sessions_dir)).get(run.session_id)
    assert info.turn_method == "handle_batch"
    prompt = str(models.llms[None].calls[0].messages)
    assert "Do the task completely, verify it" in prompt
    assert ORCHESTRATOR_TEXT[0] not in prompt


async def test_an_orchestrating_root_spawns_reviews_sends_feedback_and_closes(
    tmp_path, sessions_dir
):
    """The flow the prompt asks for, through run_task: each delegates item starts a turn.

    The retained worker runs ``handle`` turns (SessionPort.delegate picks
    ``handle`` for ``retain=True``), so a follow-up can be sent after its Done.
    """
    models = AtomModels()
    models.scripts[None] = [
        cell(
            "todo = self.todo.add('Fix parse()', description='Goal: fix it. Check: pytest')\n"
            "self.v.todo_id = todo.id\n"
            "self.v.child = await self.spawn('Fix parser', todo, retain=True)\n"
            "return_result(Waiting(explanation='worker running', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "assert isinstance(item, ChildResult) and item.child.id == self.v.child.id\n"
            "self.todo.comment(self.v.todo_id, 'first try: ' + item.done.explanation)\n"
            "await self.v.child.send('FEEDBACK: the empty-input case still fails')\n"
            "return_result(Waiting(explanation='follow-up running', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "assert item.done.explanation == 'fixed the empty case'\n"
            "await self.v.child.close()\n" + RESULT
        ),
    ]
    models.scripts["Fix parser"] = [
        cell("return_result(Done(explanation='first try'))"),
        cell("return_result(Done(explanation='fixed the empty case'))"),
    ]
    options = _options(tmp_path, sessions_dir, turn_method="handle_orchestrate")

    run = await asyncio.wait_for(
        run_task(options, "fix it", agent_factory=models, llm_factory=_no_llm), TIMEOUT
    )

    assert run.stopped is None, run.stopped
    assert run.result is not None and run.result.report == "VERIFIED"
    assert run.turns == 3
    store = SessionStore(sessions_root(options.workspace, sessions_dir))
    assert store.get(run.session_id).turn_method == "handle_orchestrate"
    [child] = [info for info in store.list(roots_only=False) if info.parent_id == run.session_id]
    assert (child.name, child.turn_method, child.retained) == ("Fix parser", "handle", True)
    root_prompt = str(models.llms[None].calls[0].messages)
    child_prompts = str([call.messages for call in models.llms["Fix parser"].calls])
    assert ORCHESTRATOR_TEXT[0] in root_prompt
    assert ORCHESTRATOR_TEXT[0] not in child_prompts
    # The worker got the Todo as its prompt, then the feedback.
    assert "Goal: fix it. Check: pytest" in child_prompts
    assert "FEEDBACK: the empty-input case still fails" in child_prompts


async def test_an_orchestrating_roots_unretained_children_run_handle_batch(tmp_path, sessions_dir):
    """delegate()/spawn() without retain still give children handle_batch: no recursion."""
    child_result = (
        "return_result(Done(explanation='fixed', result=TaskResult("
        "solution_description='s', evidence='e', how_to_verify='h', report='CHILD')))"
    )
    models = AtomModels()
    models.scripts[None] = [
        cell("done = await self.delegate('Worker', 'fix it')\n" + RESULT),
    ]
    models.scripts["Worker"] = [cell(child_result)]
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(
            _options(tmp_path, sessions_dir, turn_method="handle_orchestrate")
        )
        await asyncio.wait_for(root.prompt("fix it"), TIMEOUT)
        [child] = registry.children(root.id)
        assert child.turn_method == "handle_batch"
        assert ORCHESTRATOR_TEXT[0] not in str(models.llms["Worker"].calls[0].messages)
    finally:
        await registry.close_all()
