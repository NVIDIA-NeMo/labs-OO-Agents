# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control contract: actual legacy root/child calls, not just option readback."""

import asyncio
from types import SimpleNamespace

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from nooa_atom.tools.todo_list import TodoList

from nooa.interactive import Done, NeedInput

CHILD_DONE = "return_result(Done(explanation='child', result=TaskResult(solution_description='s', evidence='e', how_to_verify='v', report='fresh')))"


@pytest.mark.parametrize("selection", [None, "legacy"], ids=["source_default", "explicit_legacy"])
async def test_actual_legacy_root_and_kept_child_do_not_retain_handle_locals(tmp_path, selection):
    child_first = "child_local=22\nself.v.fact='durable-child'\n" + CHILD_DONE
    child_second = (
        "try:\n child_local\nexcept NameError:\n pass\nelse:\n raise AssertionError('child local survived')\n"
        "assert self.v.fact == 'durable-child'\n" + CHILD_DONE
    )
    scripts = {
        None: CellLLM(
            [
                cell(
                    "root_local=11\nself.v.fact='durable-root'\n"
                    "child=self.jobs.launch(Subagent('control child'), name='kid')\n"
                    "one=await child.send('first')\ntwo=await child.send('second')\n"
                    "assert one.report == two.report == 'fresh'\n"
                    "await child.cancel()\nreturn_result(Done(explanation='one'))"
                ),
                cell(
                    "try:\n root_local\nexcept NameError:\n pass\nelse:\n raise AssertionError('root local survived')\n"
                    "assert self.v.fact == 'durable-root'\nsecond_local=33\n"
                    "return_result(NeedInput(question='continue'))"
                ),
                cell(
                    "try:\n second_local\nexcept NameError:\n pass\nelse:\n raise AssertionError('NeedInput retained locals')\n"
                    "return_result(Done(explanation='fresh'))"
                ),
            ]
        ),
        "kid": CellLLM([cell(child_first), cell(child_second)]),
    }
    made = []

    def factory(options, storage):
        options = options.model_copy(update={"llm": scripts[options.name]})
        agent = create_session_agent(options, storage)
        made.append((options, agent))
        return agent

    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=factory)
    settings = {} if selection is None else {"experimental_namespace": selection}
    try:
        root = await registry.create(
            SessionOptions(workspace=tmp_path, agent_spec="nooa_atom.agent:AtomAgent", **settings)
        )
        assert await asyncio.wait_for(root.prompt("one"), 20) == Done(explanation="one")
        assert await asyncio.wait_for(root.prompt("two"), 20) == NeedInput(question="continue")
        assert await asyncio.wait_for(root.prompt("three"), 20) == Done(explanation="fresh")
        assert len(made) == 2
        assert [len(scripts[name].calls) for name in (None, "kid")] == [3, 2]
        for options, agent in made:
            assert options.experimental_namespace == agent._namespace_lifetime == "legacy"
            assert options.experimental_done_reflection is agent._done_reflection is False
            assert (
                agent._execution_namespace_for(SimpleNamespace(method_name="handle", kwargs={}))
                is None
            )
            assert agent._handle_namespace is None and agent.jobs._capture_factory is None
            assert "experimental_namespace" not in agent.context
            assert all(job.output is None for job in agent.jobs.list())
            assert all(job._task is None for job in agent.jobs.list())
    finally:
        await registry.close_all()


def test_control_todo_is_exact_id_lookup_not_treatment_title_lookup():
    todos = TodoList()
    todo = todos.add("distinct title")
    assert todos.get(todo.id) is todo
    with pytest.raises(ValueError, match="no item"):
        todos.get(todo.title)


# The common capture schema is the ONLY model-facing control addition to f497.
# Keep the original test_atom_prompt_budget.py limits and its four red comparisons
# untouched; this separate control-specific check removes exactly these two lines.
_CAPTURE_SCHEMA_LINES = (
    "    output: Any = None  # Bounded stdout/stderr/media once finished\n",
    "    output: Any = None  # Bounded stdout/stderr/media for experimental owned jobs\n",
)


@pytest.mark.parametrize("method", ["handle", "handle_batch", "handle_work", "handle_orchestrate"])
async def test_control_prompt_budget_allows_only_shared_capture_schema(
    tmp_path, sessions_dir, monkeypatch, method
):
    from test_atom_prompt_budget import ATOM, BENCH_SYSTEM_LIMIT, LIMITS, _first_call

    monkeypatch.setattr("nooa.skill_registry.entry_points", lambda *, group: [])
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    messages, workspace = await _first_call(ATOM, method, tmp_path, sessions_dir)
    system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    rendered = "\n".join(str(m.get("content", "")) for m in messages)
    rendered = rendered.replace(str(workspace.resolve()), "<workspace>")
    baseline_system, baseline_rendered = system, rendered
    for line in _CAPTURE_SCHEMA_LINES:
        assert system.count(line) == rendered.count(line) == 1
        baseline_system = baseline_system.replace(line, "")
        baseline_rendered = baseline_rendered.replace(line, "")
    assert len(system) - len(baseline_system) == 150
    assert len(rendered) - len(baseline_rendered) == 150
    system_limit, total_limit = LIMITS[(ATOM, method)]
    print(
        f"{method}: raw system={len(system)} total={len(rendered)}; "
        f"minus exactly shared schema={len(baseline_system)}/{len(baseline_rendered)}; "
        f"original ceilings={system_limit}/{total_limit}; hard system cap={BENCH_SYSTEM_LIMIT}"
    )
    assert len(baseline_system) <= system_limit
    assert len(baseline_rendered) <= total_limit
    assert len(system) < BENCH_SYSTEM_LIMIT  # unchanged hard cap, no schema exemption
