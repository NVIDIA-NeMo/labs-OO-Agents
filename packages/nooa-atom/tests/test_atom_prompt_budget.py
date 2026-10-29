# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-size guards for the Atom agent's first model call.

Installed ``nooa.skills`` entry points are loaded into the agent, and the
user's skill directories under the home directory are discovered, so both
change the prompt with whatever this machine has. Discovery is patched to
nothing and ``HOME`` points at an empty directory, which leaves the agent's
own tools only.

The limits are the sizes measured when this guard was added (2026-09-24),
rounded up to the next 100 characters, with the workspace path (which
varies by machine) counted as a fixed placeholder. A change that grows the
prompt must raise a limit here on purpose.
"""

import asyncio
import json

import pytest
from atom_test_agents import python_cell
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.unifiedllm import FakeLLMClient

ATOM = "nooa_atom.agent:AtomAgent"
RESULT = (
    "return_result(Done(explanation='x', result=TaskResult("
    "solution_description='a', evidence='b', how_to_verify='c')))"
)

# (spec, turn method): (system prompt chars, all message chars)
# Measured after PR 0, then raised on tree/4-router: the state block names the
# repository root, the class says ``cd`` moves shell and repo tools, the turn
# methods say locals last one method call, and the repo tools take ``cwd``.
# Lowered when AtomAgent moved to CodeActV2 (2026-09-25; was 19,827 / 23,935
# for handle): measured values plus about 2% headroom.
# Lowered when one <skills> block replaced the <skills> and <mcp> blocks
# (2026-09-28): measured values plus about 2% headroom.
# Raised when python_cell_tools added the TodoManager API (2026-10-02, about
# 2,570 characters): measured values plus about 2% headroom.
# Raised after main's #415 rendered import lines from the declared module
# (about 430 characters): measured values plus about 2% headroom.
# Raised for the child-session interface (2026-10-15): the class doc tells the
# model to put a task's requirements and rules in the Todo it hands a child and
# to run one editing child at a time, spawn() takes brief=, and
# ChildRef/ChildQuestion carry todo_id, a live status and question ids (about
# 770 characters over the 2026-10-02 values, ChildMessage docs included). The
# handle_work rows are the new worker turn. Measured values plus about 2%.
# Lowered when the child API was trimmed (2026-10-16): delegate() is one-shot and
# spawn() kept, with no mode=/keep=/context= parameters and no
# ChildRef.steer()/info/pending_question (about 150 characters less than at
# 5b84477a7). Measured values plus about 2%.
# Re-measured when InteractiveAgent gained the base handle_work and TaskResult
# moved to nooa.interactive (2026-10-16): unchanged for every row (the Atom
# agents override handle_work), so the limits stay.
# Lowered when agent.py's internal _without_removed helper was hidden from the
# execution-context stub (2026-10-16): 168 characters less for every row (its
# signature and docstring, which named the removed retain=/mode=/keep=
# options). Measured values plus about 2%.
# Re-measured after the #451 review fixes (2026-10-17): SnapshotVars got a
# pydantic schema, spawn()/delegate() adopt a hand-built Todo,
# TodoManager.status() caps the active Todo's description at 600 characters,
# and send() of the handed Todo updates the child's copy (one sentence in the
# spawn() and ChildRef.send() docs, past the summary line the prompt shows).
# Unchanged for every row (the first call has no active Todo), so the limits
# stay. A worker's todo_status block for a 5,000-character description falls
# from 2,000 characters (cut before Other Todos and Hint) to 870, all kept.
# Added handle_orchestrate (#446, the v2 orchestrator prompt as arm D
# evaluated it, on the #451 child API with the D4/D6 guidance) (2026-10-17):
# the other rows are unchanged; measured values plus about 2% headroom.
# Re-measured when ChildQuestion.answer() was removed (2026-10-18): the class
# docs say to reply with ``await item.child.send(...)``, 15 characters more for
# every row (system and total; was 14,669 / 14,467 system). Within the limits,
# so they stay.
# On dev/atom (#452): raised when AtomAgent kept the storage guidance from the
# removed experimental prompt (2026-10-16, about 190 characters): measured values plus about 2% headroom.
# Re-measured after merging dev/atom into #451 (2026-10-18): #452 removed
# ExperimentalAtomAgent (its rows are gone) and added the storage-guidance
# sentence to the AtomAgent class doc, 141 characters more for every row
# (system and total; was 14,684 system). Measured values plus about 2% headroom.
# Re-measured for the one-way Todo brief (2026-10-18): the class doc says the
# brief is one-way and to update your own Todo after verifying, 84 characters
# more in the system prompt (was 14,825). The handle_work prompt's merge-back
# lines became one sentence (51 characters less in total, was 17,981), and
# handle_orchestrate's merge-back sentence became the one-way one (53 more,
# was 19,357). Within the limits, so they stay.
# Re-measured after the #451 review-3 fixes (2026-10-19): handle_work says "a
# Todo in your list" instead of "the Todo in your list", 2 characters less in
# total (was 17,930). delegate() no longer comments on the parent's Todo and
# copy_todo/merge_todo/with_todo are gone (insert_copy() is hidden), but
# those docs are past what the prompt shows: every other row is unchanged.
# Within the limits, so they stay.
# From dev/atom (#447, merged 2026-10-19):
# Raised for truthful API text (2026-10-14: replace() forms and real parameter
# names, sync/async labels in <skills>, NeedInput free-text guidance, the show()
# line; about 290 characters): measured values plus about 2% headroom.
# Raised for #447 review (2026-10-14: replace() split into replace() and
# replace_match(), self.web removed, show() names nooa.media.Image; about 300
# characters net): measured values plus about 2% headroom.
# Raised when AtomAgent kept the storage guidance from the removed experimental
# prompt (2026-10-16, about 140 characters): measured values plus about 2% headroom.
# Re-measured after merging dev/atom with #447 into #451 (2026-10-19): #447's
# truthful API text plus the child-session docs, 273 characters more in the system
# prompt for every row (was 14,909). Measured values plus about 2% headroom.
# Raised for self.jobs (2026-10-19): the class doc says to run long work with
# self.jobs.run(...) and wait on its name, and the self doc lists the
# jobs attribute: 224 characters more in the system prompt and in total for
# every row (handle measured 18,475 total before, not 18,240). Measured values
# plus about 2% headroom.
# Raised for path names (feat/atom-jobs-skill): spawn()/delegate() take a
# required name= (in the self doc's signatures and the class doc's examples),
# ChildRef has path, and the class doc says jobs and children share one set of
# names: about 110 characters more in the system prompt and in total for every
# row. Measured values plus about 2% headroom.
# Re-measured when the stable turn instructions moved from handle() into the
# class doc (feat/atom-jobs-reland): before, 15,922 system for every row and
# 19,285 / 18,465 / 18,980 / 20,460 total. The handle() doc fell from 2,005 to
# 290 characters, so its total (repeated every turn) fell by 1,557; the class
# doc grew by 158, which put the system prompt over 16,000: the system limit
# was raised to the measured 16,080 rounded up to the next 100, and handle's
# total limit lowered to its measured value plus about 2%.
# Raised when the class doc gave self.jobs.run's full signature (name=, not
# label=) and said results arrive in notification["jobs"] (feat/atom-jobs-reland):
# 199 characters more in the system prompt and in total for every row (before,
# 16,080 system and 18,150 / 18,409 / 18,924 / 20,404 total). Limits: measured
# values plus about 2%, rounded up to the next 100 (system: next 100).
# Raised when self.jobs (JobsApi with JobInfo/JobUpdate) joined the
# python_cell_tools block and the class doc went back to the short self.jobs
# sentence (feat/atom-jobs-reland): system 16,279 -> 19,737 (+3,458), still
# under BENCH_SYSTEM_LIMIT. Limits: measured totals plus about 2%, rounded up
# to the next 100 (system: next 100).
# Session-namespace budgets already enforced before it became unconditional.
# Removing the selector does not relax these limits or the hard system ceiling.
# Raised when Done(message=...) became the documented reply (2026-10-29,
# PR503 follow-up, user-approved: prompt budget is not a constraint here): the
# class doc shows the Done/Waiting forms with message=, says what is shown as
# the reply and who receives replies and notes (user or parent), and the
# interactive and child roles say the same. System before/after: handle
# 19,377 -> 19,689, handle_batch 19,605 -> 19,884, handle_work 19,714 ->
# 20,052, handle_orchestrate 19,908 -> 20,187. Limits: measured values plus
# about 2%, rounded up to the next 100; BENCH_SYSTEM_LIMIT raised with them.
LIMITS = {
    (ATOM, "handle"): (20_100, 20_900),
    (ATOM, "handle_batch"): (20_300, 21_100),
    (ATOM, "handle_work"): (20_500, 21_300),
    (ATOM, "handle_orchestrate"): (20_600, 21_500),
}
# The bench guard's system-prompt ceiling; every agent stays under it.
# Raised from 20,000 for the Done(message=...) guidance (2026-10-29, see above).
BENCH_SYSTEM_LIMIT = 20_700

# Separate CHAR/JSON-size guards, not token counts or a full provider-wire payload.
# Effective restriction guidance grows default tool descriptions 2,069 -> 2,839;
# canonical function-tool JSON 2,259 -> 3,032 (+773); schema JSON stays 79.
TOOL_DESCRIPTION_CHAR_LIMIT = 2_900
TOOL_SCHEMA_JSON_CHAR_LIMIT = 100
TOOL_PAYLOAD_JSON_CHAR_LIMIT = 3_100


@pytest.fixture(autouse=True)
def _no_installed_skills(monkeypatch, tmp_path):
    monkeypatch.setattr("nooa.skill_registry.entry_points", lambda *, group: [])
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    # User skill directories (~/.claude/skills and the like) join the prompt.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


async def _first_call(spec, method, tmp_path, sessions_dir):
    """Run one turn and return the first call's messages, workspace, and provider tools."""
    code = RESULT if method != "handle" else "return_result(Done(explanation='x'))"
    response = python_cell(code, "call_1")
    llm = FakeLLMClient([response], strict_exhaustion=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        root = await registry.create(
            SessionOptions(
                workspace=workspace,
                agent_spec=spec,
                llm=llm,
                turn_method=method,
                sessions_dir=sessions_dir,
            )
        )
        await asyncio.wait_for(root.prompt("hello"), 30)
    finally:
        await registry.close_all()
    return llm.calls[0].messages, workspace, llm.last_tools


def _message_sizes(messages, workspace):
    system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    rendered = "\n".join(str(m.get("content", "")) for m in messages)
    path = str(workspace.resolve())
    return {
        "raw_system": len(system),
        "raw_messages": len(rendered),
        "system": len(system.replace(path, "<workspace>")),
        "messages": len(rendered.replace(path, "<workspace>")),
    }


async def test_prompt_sizes_normalize_workspace_in_system_and_messages(tmp_path, sessions_dir):
    # Real requests, not synthetic strings: workspace appears in the static system block.
    sizes = []
    for name in ("short", "longer_workspace_path"):
        root = tmp_path / name
        root.mkdir()
        messages, workspace, _ = await _first_call(ATOM, "handle", root, sessions_dir)
        sizes.append(_message_sizes(messages, workspace))
    short, long = sizes
    assert short["raw_system"] < long["raw_system"]
    assert short["raw_messages"] < long["raw_messages"]
    assert short["system"] == long["system"]
    assert short["messages"] == long["messages"]


@pytest.mark.parametrize(("spec", "method"), list(LIMITS))
async def test_first_call_prompt_stays_within_budget(spec, method, tmp_path, sessions_dir):
    messages, workspace, tools = await _first_call(spec, method, tmp_path, sessions_dir)
    tools_block = "\n".join(str(m.get("content", "")) for m in messages)
    tools_block = tools_block.split("<python_cell_tools>", 1)[1].split("</python_cell_tools>", 1)[0]
    from nooa.agentdoc import doc
    from nooa.runtime.pprint import pprint

    assert doc(doc, pprint, concise=True, inline_depth=0) in tools_block
    assert tools_block.count("def doc(*objs:") == 1
    assert tools_block.count("def pprint(_object:") == 1
    assert "Return API documentation as str; display with print(doc(obj))" in tools_block
    assert "Print (return None): max_length=items/container" in tools_block
    assert "max_string=string chars, max_depth=nesting; None=unlimited" in tools_block
    assert "get(self, id_or_title: str)" in tools_block
    assert "def read(self, name: str)" not in tools_block
    assert "_progress:" not in tools_block and "_ProgressSink" not in tools_block
    manager_doc = tools_block.split("class SkillManager", 1)[1].split("# Work:", 1)[0]
    assert "source_dir" not in manager_doc
    sizes = _message_sizes(messages, workspace)
    descriptions = sum(len(tool.description) for tool in tools)
    schemas = [tool.get_parameter_schema() for tool in tools]
    schema_chars = len(json.dumps(schemas, sort_keys=True, separators=(",", ":")))
    payload = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            },
        }
        for tool, schema in zip(tools, schemas, strict=True)
    ]
    tool_chars = len(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    print(f"provider tools: description={descriptions} schema={schema_chars} json={tool_chars}")
    system_limit, total_limit = LIMITS[(spec, method)]
    print(f"{spec} {method}: {sizes}")
    rendered = "\n".join(str(message.get("content", "")) for message in messages)
    assert "<experimental_namespace>" not in rendered
    assert "<python_cell_lifetime>" in rendered
    assert "job.output / JobEvent.output" in rendered
    assert "Cell variables survive Done, Waiting and NeedInput" in rendered
    assert "cell locals last one method call" not in rendered
    assert sizes["system"] <= system_limit
    assert sizes["messages"] <= total_limit
    assert sizes["system"] < BENCH_SYSTEM_LIMIT
    assert descriptions <= TOOL_DESCRIPTION_CHAR_LIMIT
    assert schema_chars <= TOOL_SCHEMA_JSON_CHAR_LIMIT
    assert tool_chars <= TOOL_PAYLOAD_JSON_CHAR_LIMIT


@pytest.mark.parametrize(("spec", "method"), list(LIMITS))
async def test_execution_context_hides_internal_helpers(spec, method, tmp_path, sessions_dir):
    """agent.py's internal helpers stay out of the ``<python_cell_context>`` stub.

    ``_without_removed`` once leaked there with a docstring naming the removed
    ``retain=``/``mode=``/``keep=`` options, putting them in front of the model.
    """
    messages, _, _ = await _first_call(spec, method, tmp_path, sessions_dir)
    rendered = "\n".join(str(m.get("content", "")) for m in messages)
    start = rendered.index("<python_cell_context>")
    stub = rendered[start : rendered.index("</python_cell_context>", start)]
    for leaked in ("_without_removed", "retain=", "keep=", "mode="):
        assert leaked not in stub


@pytest.mark.parametrize(("spec", "method"), list(LIMITS))
async def test_prompts_reply_to_a_child_question_with_send(spec, method, tmp_path, sessions_dir):
    """``ChildQuestion`` is data with no ``answer()``; prompts say ``child.send()``."""
    from nooa_atom.session.items import ChildQuestion
    from nooa_atom.session.port import SessionPort

    assert not hasattr(ChildQuestion, "answer")
    assert not hasattr(SessionPort, "answer_child")
    messages, _, _ = await _first_call(spec, method, tmp_path, sessions_dir)
    rendered = "\n".join(str(m.get("content", "")) for m in messages)
    assert ".answer(" not in rendered
    assert "answer_child" not in rendered


@pytest.mark.parametrize(("spec", "method"), list(LIMITS))
async def test_prompts_make_done_message_the_reply(spec, method, tmp_path, sessions_dir):
    """Done(message=...) is the documented reply; the role says who receives it."""
    messages, _, _ = await _first_call(spec, method, tmp_path, sessions_dir)
    rendered = " ".join("\n".join(str(m.get("content", "")) for m in messages).split())
    assert 'Done(message="your reply", explanation="short status")' in rendered
    assert "Waiting(explanation=..., on=[names], message=...)" in rendered
    assert (
        "``message`` and ``question`` are shown as your reply; ``explanation`` is a short "
        "status line, not the reply; use ``self.message()`` for notes while you work."
    ) in rendered
    assert (
        "Replies and notes go to the user, or to your parent if you are a subagent; unattended, "
        "only your result is read."
    ) in rendered
    assert "the reply with self.message()" not in rendered
    role = rendered.split("<role>", 1)[1].split("</role>", 1)[0]
    if method == "handle_work":
        assert "not for the user: everything you send goes to your parent" in role
        assert "the user," not in role
    else:
        assert "parent" not in role
    if method == "handle":
        assert "You are working with a person, the user. Reply with Done(message=...)" in role


class _InstalledSkill:
    """An installed ``nooa.skills`` entry point with a unique description."""

    def __init__(self, index: int) -> None:
        from nooa.skill import Skill

        self.name = f"budget.helper_{index:02d}"
        self.value = f"budget_skills:Helper{index:02d}"
        self.dist = None
        self.description = f"Budget helper number {index:02d} with a unique description"
        self._skill = type(f"Helper{index:02d}", (Skill,), {"__doc__": self.description})

    def load(self):
        return self._skill


async def test_installed_skills_are_counted_not_listed(monkeypatch, tmp_path, sessions_dir):
    """With this machine's installed skills and 40 more, none is named in the prompt.

    The other tests here turn installed skills off, which hid how the prompt
    grows with them. Each skill is found through ``self.skills.search()``;
    none is named or described in the prompt until it is activated.
    """
    import importlib.metadata

    extra = [_InstalledSkill(index) for index in range(40)]
    monkeypatch.setattr(
        "nooa.skill_registry.entry_points",
        lambda *, group: [*importlib.metadata.entry_points(group=group), *extra],
    )
    llm = FakeLLMClient(
        [python_cell("return_result(Done(explanation='x'))", "call_1")], strict_exhaustion=True
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        root = await registry.create(
            SessionOptions(workspace=workspace, agent_spec=ATOM, llm=llm, sessions_dir=sessions_dir)
        )
        await asyncio.wait_for(root.prompt("hello"), 30)
    finally:
        await registry.close_all()
    rendered = "\n".join(str(m.get("content", "")) for m in llm.calls[0].messages)
    # The Atom agent shows no dynamic <skills> inventory: SkillManager's docs
    # are in the static python_cell_tools block, and status() is on demand.
    assert "<skills" not in rendered
    assert "class SkillManager" in rendered and "def status(self) -> str:" in rendered
    for skill in extra:
        assert skill.description not in rendered
        assert skill.name.split(".")[-1] not in rendered
