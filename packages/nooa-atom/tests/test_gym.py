# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa_atom.gym: Atom as a NeMo Gym harness, driven by a scripted model."""

from __future__ import annotations

import json

import pytest
from atom_test_agents import CellLLM, python_cell
from nooa_atom.gym import TURN_METHODS, AtomHarness
from nooa_atom.session.session import TurnFailedError

from nooa.interactive import TaskResult
from nooa.runtime.hooks import hooks_scope

FINISH = (
    "open('hello.txt', 'w').write('hello gym')\n"
    "return_result(Done(explanation='wrote the file', result=TaskResult("
    "solution_description='created hello.txt', evidence='file exists', how_to_verify='cat hello.txt', "
    "report='done')))"
)


class _Hooks:
    """A harness observer: records code executions the way a Gym observer would."""

    def __init__(self) -> None:
        self.cells: list[str] = []

    def before_code_execution(self, *, code: str, **_: object) -> dict:
        self.cells.append(code)
        return {}

    def after_code_execution(self, **_: object) -> None: ...

    def before_agent_call(self, **_: object) -> None: ...

    def after_agent_call(self, **_: object) -> None: ...

    def before_tool_execution(self, **_: object) -> None: ...

    def after_tool_execution(self, **_: object) -> None: ...

    def before_generation(self, **_: object) -> None: ...

    def after_generation(self, **_: object) -> None: ...

    def before_method_invocation(self, **_: object) -> None: ...

    def after_method_invocation(self, **_: object) -> None: ...

    def on_messages_built(self, **_: object) -> None: ...


@pytest.mark.parametrize("turn_method", TURN_METHODS)
async def test_harness_runs_one_atom_session_to_its_result(tmp_path, monkeypatch, turn_method):
    workdir = tmp_path / "task"
    workdir.mkdir()
    monkeypatch.chdir(
        workdir
    )  # Gym's worker runs with cwd = task directory; cells resolve paths against it
    llm = CellLLM([python_cell(FINISH, "call-1")])
    harness = AtomHarness(
        llm, turn_method=turn_method, workdir=workdir, sessions_dir=tmp_path / "sessions"
    )
    hooks = _Hooks()
    with hooks_scope(hooks):
        result = await harness.solve("Create hello.txt containing 'hello gym', then finish.")
    assert isinstance(result, TaskResult) and result.solution_description == "created hello.txt"
    assert (workdir / "hello.txt").read_text() == "hello gym"  # cells run in the task directory
    # The observer active around solve() saw the cell Atom executed inside its session loop.
    assert any("hello.txt" in cell for cell in hooks.cells), (
        hooks.cells
    )  # Atom prefills an inspection cell first
    assert (
        llm._response_queue == type(llm._response_queue)()
    )  # exactly the scripted calls were made


async def test_harness_refuses_an_unknown_turn_method(tmp_path):
    with pytest.raises(ValueError, match="turn_method"):
        AtomHarness(CellLLM([]), turn_method="handle_work", workdir=tmp_path)


async def test_harness_refuses_model_aliases(tmp_path, monkeypatch):
    """A Gym rollout may use only the injected model; Atom must not resolve another one."""
    from nooa_atom import gym as gym_module

    seen: dict = {}
    original = gym_module.SessionOptions

    class Options(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            seen["llm"] = kwargs["llm"]

    monkeypatch.setattr(gym_module, "SessionOptions", Options)
    llm = CellLLM([python_cell(FINISH, "call-1")])
    harness = AtomHarness(
        llm, turn_method="handle_batch", workdir=tmp_path, sessions_dir=tmp_path / "s"
    )
    await harness.solve("task")
    assert seen["llm"] is llm


def test_harness_is_an_agent_without_model_facing_methods():
    """Gym adapters construct any NOOA agent class; the harness exposes nothing to a model."""
    from nooa.agentdoc import doc

    text = doc(AtomHarness)
    assert "def solve" not in text and "python_cell" not in text
    assert json.dumps(TURN_METHODS) == '["handle_batch", "handle_orchestrate"]'


async def test_harness_removes_the_session_directory_it_created_and_keeps_a_callers(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    owned = AtomHarness(
        CellLLM([python_cell(FINISH, "call-1")]), turn_method="handle_batch", workdir=tmp_path
    )
    root = owned._owned_root
    assert root is not None and root.is_dir()
    await owned.solve("task")
    assert not root.exists()  # nothing of the rollout's session records stays behind in /tmp

    given = tmp_path / "keep"
    kept = AtomHarness(
        CellLLM([python_cell(FINISH, "call-2")]),
        turn_method="handle_batch",
        workdir=tmp_path,
        sessions_dir=given,
    )
    assert kept._owned_root is None
    await kept.solve("task")
    assert given.is_dir() and any(
        given.rglob("*")
    )  # the caller's directory and its records are untouched


async def test_harness_removes_its_session_directory_when_solve_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    harness = AtomHarness(
        CellLLM([]), turn_method="handle_batch", workdir=tmp_path
    )  # no scripted reply: the turn fails
    root = harness._owned_root
    with pytest.raises(TurnFailedError):
        await harness.solve("task")
    assert not root.exists()


async def test_harness_refuses_a_done_without_a_task_result(tmp_path, monkeypatch):
    """Done.result may carry anything; only a TaskResult is a successful rollout for Gym."""
    monkeypatch.chdir(tmp_path)
    llm = CellLLM(
        [python_cell("return_result(Done(explanation='finished', result='done'))", "call-1")]
    )
    harness = AtomHarness(
        llm, turn_method="handle_batch", workdir=tmp_path, sessions_dir=tmp_path / "s"
    )
    with pytest.raises(RuntimeError, match="without a TaskResult.*Done carrying str"):
        await harness.solve("task")


async def test_solve_cancels_and_rejects_children_still_executing_when_the_root_answers():
    """Session.close will not close an agent mid-turn, so a root Done over a running child would leak
    the child's loop; the harness cancels such children and refuses the outcome."""
    from types import SimpleNamespace

    cancelled: list[str] = []

    def session(session_id: str, status: str):
        async def cancel(*, by: str) -> bool:
            cancelled.append(f"{session_id}:{by}")
            return True

        return SimpleNamespace(id=session_id, status=status, cancel=cancel)

    root = session("root", "idle")
    busy = session("child-1", "running")
    done = session("child-2", "idle")
    registry = SimpleNamespace(
        sessions={"root": root, "child-1": busy, "child-2": done},
        live_info=lambda s: SimpleNamespace(status=s.status),
    )
    with pytest.raises(RuntimeError, match="1 child turn\\(s\\) were still executing: child-1"):
        await AtomHarness._settle_children(registry, root)
    assert cancelled == ["child-1:harness"]  # the idle child and the root are left alone

    cancelled.clear()
    await AtomHarness._settle_children(
        SimpleNamespace(sessions={"root": root}, live_info=registry.live_info), root
    )
    assert cancelled == []
