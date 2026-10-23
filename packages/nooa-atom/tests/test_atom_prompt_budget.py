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

import pytest
from atom_test_agents import python_cell
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.unifiedllm import FakeLLMClient

ATOM = "nooa_atom.agent:AtomAgent"
UNIFIED = "nooa_atom.agent.unified:UnifiedAtomAgent"
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
LIMITS = {
    (ATOM, "handle"): (19_800, 22_300),  # measured 19,737 / 21,807
    (ATOM, "handle_batch"): (19_800, 22_600),  # measured 19,737 / 22,066
    (ATOM, "handle_work"): (19_800, 23_100),  # measured 19,737 / 22,581
    (ATOM, "handle_orchestrate"): (19_800, 24_600),  # measured 19,737 / 24,061
    # The experimental unified-work agent (spike/atom-unified-work): one
    # self.jobs vocabulary (WorkManager, Job, Shell, Subagent) documented in
    # python_cell_tools, the stable mode guidance in the class doc, and turn
    # docs under 300 characters. Measured values plus about 2%.
    (UNIFIED, "handle"): (18_600, 20_200),  # measured 18,269 / 19,865
    (UNIFIED, "handle_batch"): (18_600, 20_200),  # measured 18,269 / 19,885
    (UNIFIED, "handle_work"): (18_600, 20_200),  # measured 18,269 / 19,882
    (UNIFIED, "handle_orchestrate"): (18_600, 20_300),  # measured 18,269 / 19,922
}
# The bench guard's system-prompt ceiling; every agent stays under it.
BENCH_SYSTEM_LIMIT = 20_000


@pytest.fixture(autouse=True)
def _no_installed_skills(monkeypatch, tmp_path):
    monkeypatch.setattr("nooa.skill_registry.entry_points", lambda *, group: [])
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    # User skill directories (~/.claude/skills and the like) join the prompt.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


async def _first_call(spec, method, tmp_path, sessions_dir):
    """Run one turn of ``method`` and return the first call's messages and workspace."""
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
    return llm.calls[0].messages, workspace


@pytest.mark.parametrize(("spec", "method"), list(LIMITS))
async def test_first_call_prompt_stays_within_budget(spec, method, tmp_path, sessions_dir):
    messages, workspace = await _first_call(spec, method, tmp_path, sessions_dir)
    system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    rendered = "\n".join(str(m.get("content", "")) for m in messages)
    # The workspace path appears in the prompt; count it as a fixed token.
    rendered = rendered.replace(str(workspace.resolve()), "<workspace>")
    system_limit, total_limit = LIMITS[(spec, method)]
    print(f"{spec} {method}: system={len(system)} total={len(rendered)}")
    assert len(system) <= system_limit
    assert len(rendered) <= total_limit
    assert len(system) < BENCH_SYSTEM_LIMIT


@pytest.mark.parametrize(("spec", "method"), list(LIMITS))
async def test_execution_context_hides_internal_helpers(spec, method, tmp_path, sessions_dir):
    """agent.py's internal helpers stay out of the ``<python_cell_context>`` stub.

    ``_without_removed`` once leaked there with a docstring naming the removed
    ``retain=``/``mode=``/``keep=`` options, putting them in front of the model.
    """
    messages, _ = await _first_call(spec, method, tmp_path, sessions_dir)
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
    messages, _ = await _first_call(spec, method, tmp_path, sessions_dir)
    rendered = "\n".join(str(m.get("content", "")) for m in messages)
    assert ".answer(" not in rendered
    assert "answer_child" not in rendered


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
    """With this machine's installed skills and 40 more, the ``<skills>`` block stays small.

    The other tests here turn installed skills off, which hid how the prompt
    grows with them. Each skill is found through ``self.skills.search()``;
    none is named or described in the prompt until it is activated.
    """
    import importlib.metadata
    import re

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
    [block] = re.findall(r"<skills[^>]*>\n(.*?)\n</skills>", rendered, re.DOTALL)
    print(block)
    assert len(block) < 400
    assert re.search(r"^\d+ more \(", block.splitlines()[-1])
    for skill in extra:
        assert skill.description not in rendered
        assert skill.name.split(".")[-1] not in rendered
