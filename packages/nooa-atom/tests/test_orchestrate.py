# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``handle_orchestrate``: an unattended root that plans, delegates and verifies."""

import asyncio
import hashlib

import pytest
from atom_test_agents import ATOM_SPEC, AtomModels, CellLLM, cell
from nooa_atom import SessionOptions, SessionStore, run_task
from nooa_atom.agent.base import AtomAgentBase
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import sessions_root
from pydantic import ValidationError

from nooa.agentdoc import doc

TIMEOUT = 30

# Phrases only the orchestrator prompt has.
ORCHESTRATOR_TEXT = (
    "You are the orchestrator. You decide how the work is split and who does",
    "Write the requirements: how success will be measured.",
    "await self.spawn(description, todo, name=...)",
    "task's requirements and its rules or constraints in it.",
    "run one editing child at a time",
    "a ``ChildMessage`` (``item.text``) is a progress",
    "Its report is the TaskResult; record what you verified on",
    "Before ``Done``, leave the tree clean: no scratch files.",
    "Run every acceptance check yourself; a report is",
    "Leave children open; the host closes them after grading.",
)

# The interactive and batch prompts this change must leave as they are.
# Changing either prompt on purpose means updating its digest here.
UNCHANGED_PROMPTS = {
    # handle() as on dev/atom after #447 and #451 (#451's "or from your parent
    # agent if you are a child" line and #447's NeedInput free-text wording),
    # plus the regular progress-update guidance, and NeedInput as a plain-text
    # question (no options). Then the stable turn instructions moved into the
    # class doc: handle() keeps three sentences, and the locals paragraph left
    # every turn method's doc.
    "handle": "f4558376a3b74ecd3f6685a44b534581336a1e1ea701e3e71f55afac858d72ad",
    "handle_batch": "9eef8eb5f2d8067d02c0eb742c4f6b8006ae8fff001130277e119a8510f6027d",
}

RESULT = (
    "return_result(Done(explanation='verified', result=TaskResult("
    "solution_description='the worker fixed parse()', evidence='pytest passed', "
    "how_to_verify='pytest', report='VERIFIED')))"
)


def test_handle_orchestrate_has_the_orchestrator_prompt():
    from nooa.strategies import CodeActV2

    prompt = AtomAgentBase.handle_orchestrate.__doc__
    assert prompt.startswith("You are a recursive language model agent.")
    for phrase in ORCHESTRATOR_TEXT:
        assert phrase in prompt
    # The return contract is kept from the earlier prompt.
    assert "result=TaskResult(" in prompt
    assert "If something blocks you, still return ``Done`` with a ``TaskResult``" in prompt
    assert "Locals last one method call" in AtomAgentBase.__doc__  # said once, in the class doc
    # The orchestrator no longer closes its children; the host does.
    assert "child.close()" not in prompt
    # Nobody can answer a question, so the prompt never offers NeedInput.
    assert "NeedInput" not in prompt
    assert isinstance(AtomAgentBase.handle_orchestrate._plan_strategy, CodeActV2)


def test_handle_orchestrate_returns_only_done_or_waiting():
    import typing

    from nooa.interactive import Done, NeedInput, Waiting

    def returns(method):
        return typing.get_type_hints(getattr(AtomAgentBase, method))["return"]

    assert returns("handle_orchestrate") == Done | Waiting
    # handle() and handle_batch() keep their return types.
    assert returns("handle") == Done | NeedInput | Waiting
    assert returns("handle_batch") == Done | Waiting


async def test_handle_orchestrate_rejects_need_input(tmp_path):
    """Return-type validation turns a NeedInput into feedback the model sees."""
    llm = CellLLM(
        [
            cell("return_result(NeedInput(question='which one: a or b?'))"),
            cell(RESULT),
        ]
    )
    agent = AtomAgentBase(llm=llm, cwd=tmp_path)
    try:
        outcome = await agent.handle_orchestrate({"user_messages": ["fix it"]})
        assert outcome.result.report == "VERIFIED"
        feedback = str(llm.calls[1].messages)
        assert "return_result validation error" in feedback
        assert "Expected return type: nooa.interactive.Done | nooa.interactive.Waiting" in feedback
        assert "Got: NeedInput(question=" in feedback
    finally:
        await agent.aclose()


@pytest.mark.parametrize("method", list(UNCHANGED_PROMPTS))
def test_the_interactive_and_batch_prompts_are_unchanged(method):
    prompt = getattr(AtomAgentBase, method).__doc__
    assert hashlib.sha256(prompt.encode()).hexdigest() == UNCHANGED_PROMPTS[method]
    for phrase in ORCHESTRATOR_TEXT:
        assert phrase not in prompt


async def test_handle_orchestrate_is_hidden_from_the_model(tmp_path):
    """Hosts call it as a turn method; the model never sees it listed."""
    assert "handle_orchestrate" not in doc(AtomAgentBase, concise=True)
    assert "handle_orchestrate" not in doc(AtomAgentBase)
    agent = AtomAgentBase(llm=CellLLM([cell("return_result(Done(explanation='x'))")]), cwd=tmp_path)
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


async def test_an_orchestrating_root_leaves_children_open_for_the_host_to_close(
    tmp_path, sessions_dir, monkeypatch
):
    """The flow the prompt asks for, through run_task: each delegates item starts a turn.

    The kept worker (``spawn()``) runs ``handle_work`` turns and stays open
    after its Done, so a follow-up can be sent with ``send()``.
    The root does not close the worker before Done; run_task's registry
    closes it at the end (close_all).
    """
    close_all_calls = []
    original_close_all = SessionRegistry.close_all

    async def spy_close_all(registry):
        before = [info.status for info in registry.children(run_id[0])]
        await original_close_all(registry)
        after = [info.status for info in registry.children(run_id[0])]
        close_all_calls.append((before, after))

    monkeypatch.setattr(SessionRegistry, "close_all", spy_close_all)
    run_id = []
    models = AtomModels()
    models.scripts[None] = [
        cell(
            "todo = self.todo.add('Fix parse()', description='Goal: fix it. Check: pytest')\n"
            "self.v.todo_id = todo.id\n"
            "self.v.child = await self.spawn('fix_parser', todo, name='fix_parser')\n"
            "return_result(Waiting(explanation='worker running', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "assert isinstance(item, ChildResult) and item.child.id == self.v.child.id\n"
            "self.todo.comment(self.v.todo_id, 'first try: ' + item.summary)\n"
            "await self.v.child.send('FEEDBACK: the empty-input case still fails')\n"
            "return_result(Waiting(explanation='follow-up running', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "assert item.summary == 'fixed the empty case'\n"
            "[ref] = self.children()\n"
            "assert ref.status != 'closed', ref.status\n" + RESULT
        ),
    ]
    models.scripts["fix_parser"] = [
        cell(
            "return_result(Done(explanation='first try', result=TaskResult("
            "solution_description='s', evidence='e', how_to_verify='h', report='first try')))"
        ),
        cell(
            "return_result(Done(explanation='fixed the empty case', result=TaskResult("
            "solution_description='s', evidence='e', how_to_verify='h', report='fixed the empty case')))"
        ),
    ]
    options = _options(tmp_path, sessions_dir, turn_method="handle_orchestrate")
    original_create = SessionRegistry.create

    async def record_root(registry, *args, **kwargs):
        session = await original_create(registry, *args, **kwargs)
        if not run_id:
            run_id.append(session.id)
        return session

    monkeypatch.setattr(SessionRegistry, "create", record_root)

    run = await asyncio.wait_for(
        run_task(options, "fix it", agent_factory=models, llm_factory=_no_llm), TIMEOUT
    )

    assert run.stopped is None, run.stopped
    assert run.result is not None and run.result.report == "VERIFIED"
    assert run.turns == 3
    # The worker was still open (retained) when run_task reached close_all,
    # and close_all closed it.
    assert close_all_calls == [(["retained"], ["on_disk"])]
    store = SessionStore(sessions_root(options.workspace, sessions_dir))
    assert store.get(run.session_id).turn_method == "handle_orchestrate"
    [child] = [info for info in store.list(roots_only=False) if info.parent_id == run.session_id]
    assert (child.name, child.turn_method, child.retained) == ("fix_parser", "handle_work", True)
    root_prompt = str(models.llms[None].calls[0].messages)
    child_prompts = str([call.messages for call in models.llms["fix_parser"].calls])
    assert ORCHESTRATOR_TEXT[0] in root_prompt
    assert ORCHESTRATOR_TEXT[0] not in child_prompts
    # The worker got the Todo as its prompt, then the feedback.
    assert "Goal: fix it. Check: pytest" in child_prompts
    assert "FEEDBACK: the empty-input case still fails" in child_prompts


async def test_an_orchestrating_roots_one_shot_children_run_handle_work(tmp_path, sessions_dir):
    """A delegate() worker runs handle_work, not handle_orchestrate: no recursion."""
    child_result = (
        "return_result(Done(explanation='fixed', result=TaskResult("
        "solution_description='s', evidence='e', how_to_verify='h', report='CHILD')))"
    )
    models = AtomModels()
    models.scripts[None] = [
        cell("done = await self.delegate('worker', 'fix it', name='worker')\n" + RESULT),
    ]
    models.scripts["worker"] = [cell(child_result)]
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(
            _options(tmp_path, sessions_dir, turn_method="handle_orchestrate")
        )
        await asyncio.wait_for(root.prompt("fix it"), TIMEOUT)
        [child] = registry.children(root.id)
        assert child.turn_method == "handle_work"
        assert ORCHESTRATOR_TEXT[0] not in str(models.llms["worker"].calls[0].messages)
    finally:
        await registry.close_all()
