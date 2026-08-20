# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generic BenchAgent with structured TaskResult output."""

from __future__ import annotations

import pytest
from nooa_bench import bench_agent as bench_agent_module
from nooa_bench.bench_agent import BenchAgent, RLMBenchAgent, TaskResult
from nooa_cli.coding import delegation as delegation_module

from nooa.agentdoc import doc
from nooa.unifiedllm import FakeLLMClient


class _FakeShell:
    def __init__(self, cwd: str, init_command: str | None = None) -> None:
        self.cwd = cwd
        self.init_command = init_command
        self.commands: list[str] = []
        self._session = object()

    @property
    def session(self) -> object:
        return self._session

    async def run(self, command: str):
        self.commands.append(command)
        return None


class _FakeRepo:
    def __init__(self, root: str, session: object | None = None) -> None:
        self.root = root
        self.session = session


def test_task_result_model():
    """TaskResult validates required fields with solution_description."""
    r = TaskResult(
        solution_description="Fixed missing URL-encoding in auth.py with quote_plus().",
        evidence="pytest tests/ passed: 5 passed in 1.2s",
        command_to_verify="pytest tests/ -x",
    )
    assert "URL-encoding" in r.solution_description
    assert "pytest" in r.command_to_verify


def test_bench_agent_has_no_verify():
    """BenchAgent does not expose a verify() method."""
    assert not hasattr(BenchAgent, "verify")


def test_bench_agent_has_private_solve_task():
    """BenchAgent uses _solve_task (private) directly; no public solve_task wrapper."""
    assert hasattr(BenchAgent, "_solve_task")


def test_bench_agent_class_exists():
    """BenchAgent can be imported and has expected methods."""
    assert BenchAgent.__name__ == "BenchAgent"
    assert hasattr(BenchAgent, "_run_evaluation")


def test_bench_agent_context_is_minimal_and_automatic():
    """Only actionable live context is exposed; compaction is automatic."""
    agent = BenchAgent(llm=FakeLLMClient())

    keys = list(agent.context_manager.keys())

    assert "todo_status" in keys
    assert "python_tools" in keys
    assert "task" not in keys
    assert "todo" not in keys
    assert "context_usage" not in keys
    assert agent._summarizers


def test_task_text_is_not_retained_in_agent_state():
    """The method argument is the sole task copy; state must not duplicate it."""
    agent = BenchAgent(llm=FakeLLMClient())

    assert not hasattr(agent, "problem_statement")


def test_bench_agent_hides_manual_context_maintenance_apis():
    """The model should solve the task, not manually rewrite its prompt history."""
    agent = BenchAgent(llm=FakeLLMClient())

    agent_doc = doc(agent)

    assert "context:" not in agent_doc
    assert "events:" not in agent_doc


@pytest.mark.asyncio
async def test_run_evaluation_returns_structured_task_result(monkeypatch, tmp_path):
    shells: list[_FakeShell] = []

    def fake_make_shell(cwd: str, init_command=None):
        shell = _FakeShell(cwd)
        shells.append(shell)
        return shell

    async def fake_solve_task(description: str):
        assert description == "fix the bug"
        return TaskResult(
            solution_description="Fixed the bug.",
            evidence="pytest passed",
            command_to_verify="pytest -q",
        )

    monkeypatch.setattr(bench_agent_module, "ShellTools", fake_make_shell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    agent = BenchAgent(llm=FakeLLMClient())
    monkeypatch.setattr(agent, "_solve_task", fake_solve_task)

    result = await agent._run_evaluation(
        {"problem_statement": "fix the bug", "working_dir": str(tmp_path)}
    )

    assert result == {
        "response": "pytest -q",
        "success": True,
        "result": {
            "solution_description": "Fixed the bug.",
            "evidence": "pytest passed",
            "command_to_verify": "pytest -q",
        },
    }
    assert shells[-1].cwd == str(tmp_path)


@pytest.mark.asyncio
async def test_run_evaluation_returns_failure_on_exception(monkeypatch, tmp_path):
    def fake_make_shell(cwd: str, init_command=None):
        return _FakeShell(cwd)

    async def fake_solve_task(description: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(bench_agent_module, "ShellTools", fake_make_shell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    agent = BenchAgent(llm=FakeLLMClient())
    monkeypatch.setattr(agent, "_solve_task", fake_solve_task)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(tmp_path)}
    )

    assert result == {"response": "", "success": False, "error": "boom"}


@pytest.mark.asyncio
async def test_run_evaluation_clears_optional_context_between_tasks(monkeypatch, tmp_path):
    """Absent per-task metadata must not leak from an earlier evaluation."""

    def fake_make_shell(cwd: str, init_command=None):
        return _FakeShell(cwd)

    async def fake_solve_task(description: str):
        return TaskResult(
            solution_description="Fixed.", evidence="check passed", command_to_verify="true"
        )

    monkeypatch.setattr(bench_agent_module, "ShellTools", fake_make_shell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    agent = BenchAgent(llm=FakeLLMClient())
    monkeypatch.setattr(agent, "_solve_task", fake_solve_task)

    await agent._run_evaluation(
        {
            "problem_statement": "first",
            "working_dir": str(tmp_path),
            "instructions": "first-only constraint",
            "initial_observation": "first-only state",
        }
    )
    await agent._run_evaluation({"problem_statement": "second", "working_dir": str(tmp_path)})

    assert "instructions" not in agent.context_manager
    assert "initial_observation" not in agent.context_manager


@pytest.mark.asyncio
async def test_run_evaluation_requires_problem_statement(monkeypatch, tmp_path):
    """BenchAgent rejects tasks without a usable task description."""

    def fake_make_shell(cwd: str, init_command=None):
        return _FakeShell(cwd)

    monkeypatch.setattr(bench_agent_module, "ShellTools", fake_make_shell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    agent = BenchAgent(llm=FakeLLMClient())

    with pytest.raises(ValueError, match="user_message, problem_statement, or task_description"):
        await agent._run_evaluation({"working_dir": str(tmp_path)})


def test_bench_agent_uses_narrow_python_tools_context():
    """The prompt documents primary tools without duplicating the todo API."""
    agent = BenchAgent(llm=FakeLLMClient())

    keys = list(agent.context_manager.keys())
    assert "python_tools" in keys
    assert "todo_status" in keys
    assert "todo" not in keys

    python_tools_doc = agent.context_manager["python_tools"]
    assert "class RepoTools" in python_tools_doc
    assert "def symbols(" in python_tools_doc
    assert "class ShellTools" in python_tools_doc
    assert "def run(" in python_tools_doc


def test_bench_agent_wires_repo_to_shell_session():
    """BenchAgent gives RepoTools the same root/session as ShellTools."""

    agent = BenchAgent(llm=FakeLLMClient())

    assert agent.repo.root == agent.shell.cwd
    assert agent.repo.session is agent.shell.session


def test_tool_repr_shows_state():
    """pprint()/repr expose held tool state instead of object addresses."""

    agent = BenchAgent(llm=FakeLLMClient())

    assert repr(agent.shell) == f"ShellTools(cwd={agent.shell.cwd!s})"
    assert repr(agent.repo) == (
        f"RepoTools(root={str(agent.repo.root)!r}, session=shared, has_rg=None)"
    )


def test_solve_task_prompt_is_compact_and_non_ritualized():
    """Prompt keeps core engineering invariants without mandatory planning theater."""
    prompt = BenchAgent._solve_task.__doc__ or ""

    assert "Inspect before editing" in prompt
    assert "minimum sufficient change" in prompt
    assert "Plan with ``self.todo`` only when useful" in prompt
    assert "1. Explore" not in prompt


def test_bench_agent_does_not_preseed_todos():
    """Simple tasks start without an artificial planning obligation."""
    agent = BenchAgent(llm=FakeLLMClient())

    assert agent.todo.list_todos() == []


@pytest.mark.asyncio
async def test_run_evaluation_clears_stale_todos(monkeypatch, tmp_path):
    """Per-task reset clears prior state without adding a ritual todo."""

    def fake_make_shell(cwd: str, init_command=None):
        return _FakeShell(cwd)

    async def fake_solve_task(description: str):
        assert agent.todo.list_todos() == []
        return TaskResult(
            solution_description="Fixed.", evidence="check passed", command_to_verify="true"
        )

    monkeypatch.setattr(bench_agent_module, "ShellTools", fake_make_shell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    agent = BenchAgent(llm=FakeLLMClient())
    agent.todo.add("stale todo")
    monkeypatch.setattr(agent, "_solve_task", fake_solve_task)

    result = await agent._run_evaluation(
        {"problem_statement": "fix the bug", "working_dir": str(tmp_path)}
    )

    assert result["success"] is True


def test_bounded_worker_has_no_duplicate_plan_or_persistent_vars():
    """Worker isolation excludes controller plan and durable session state."""
    from nooa_cli.coding.delegation import CodingWorker

    assert not hasattr(CodingWorker, "todo")
    assert not hasattr(CodingWorker, "v")


def test_rlm_variant_is_registered_and_documents_delegation():
    from nooa_bench import AGENT_CLASSES

    assert AGENT_CLASSES["rlm"] == "nooa_bench.bench_agent:RLMBenchAgent"
    assert "delegate" in (RLMBenchAgent._solve_task.__doc__ or "")


@pytest.mark.asyncio
async def test_rlm_delegate_builds_isolated_worker(monkeypatch, tmp_path):
    observed = {}

    class FakeWorker:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        async def investigate(self, objective: str, supplied_context=None) -> str:
            observed.update(objective=objective, supplied_context=supplied_context)
            return "concise report"

        async def close(self) -> None:
            observed["closed"] = True

    monkeypatch.setattr(bench_agent_module, "ShellTools", _FakeShell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    monkeypatch.setattr(delegation_module, "CodingWorker", FakeWorker)
    llm = FakeLLMClient()
    agent = RLMBenchAgent(llm=llm)
    agent._install_python_tools(str(tmp_path))

    todo = agent.todo.add("Investigate empty parser input")
    result = await agent.delegate("inspect parser", todo)

    assert result == "concise report"
    assert observed["objective"] == "inspect parser"
    assert observed["supplied_context"] is todo
    assert observed["llm"] is llm
    assert observed["cwd"] == str(tmp_path)
    assert observed["init_command"] == bench_agent_module._OPTIONAL_TESTBED_ACTIVATE
    assert observed["closed"] is True


def test_problem_statement_skips_blank_primary_field():
    """Blank higher-priority fields do not block fallback task text."""

    assert (
        bench_agent_module._problem_statement(
            {"user_message": "   ", "problem_statement": " use this "}
        )
        == "use this"
    )
