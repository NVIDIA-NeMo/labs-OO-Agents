# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-size guards for the Atom agents' first model call.

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
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from test_experimental_agent import python_cell

from nooa.unifiedllm import FakeLLMClient

ATOM = "nooa_atom.agent:AtomAgent"
EXPERIMENTAL = "nooa_atom.agent:ExperimentalAtomAgent"
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
LIMITS = {
    (ATOM, "handle"): (15_000, 18_000),  # measured 14,669 / 17,620
    (ATOM, "handle_batch"): (15_000, 17_600),  # measured 14,669 / 17,218
    (ATOM, "handle_work"): (15_000, 18_200),  # measured 14,669 / 17,825
    (ATOM, "handle_orchestrate"): (15_000, 19_600),  # measured 14,669 / 19,197
    (EXPERIMENTAL, "handle"): (14_800, 16_600),  # measured 14,467 / 16,237
    (EXPERIMENTAL, "handle_batch"): (14_800, 16_950),  # measured 14,467 / 16,590
    (EXPERIMENTAL, "handle_work"): (14_800, 17_250),  # measured 14,467 / 16,902
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
