# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generic BenchAgent with structured TaskResult output."""

from __future__ import annotations

import pytest
from nooa_bench import bench_agent as bench_agent_module
from nooa_bench.bench_agent import BenchAgent, TaskResult

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


def test_bench_agent_installs_context_usage_dynamic_block():
    """BenchAgent exposes live context-window usage to the LLM."""
    agent = BenchAgent(llm=FakeLLMClient())

    keys = list(agent.context_manager.keys())

    assert "context_usage" in keys


def test_bench_agent_exposes_context_and_events_apis():
    """BenchAgent exposes context and events APIs so the LLM can act on context-usage hints."""
    agent = BenchAgent(llm=FakeLLMClient())

    agent_doc = doc(agent)

    assert "context:" in agent_doc
    assert "events:" in agent_doc


def test_context_usage_block_includes_collapse_hint():
    """Context usage tells agents how to compact old event history."""
    from nooa.context_blocks.models import ContextWindowStats

    agent = BenchAgent(llm=FakeLLMClient())
    agent.runtime._last_context_stats = ContextWindowStats(
        context_blocks_count=2,
        events_count=12,
        prompt_tokens=1000,
        context_blocks_chars=100,
        events_chars=900,
        max_context_tokens=1000,
        model_context_window=1000,
    )

    block = agent._context_usage_block()

    assert "Context usage:" in block
    assert "self.events.collapse(start_tag, end_tag, summary_text)" in block


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
async def test_run_evaluation_requires_problem_statement(monkeypatch, tmp_path):
    """BenchAgent rejects tasks without a usable task description."""

    def fake_make_shell(cwd: str, init_command=None):
        return _FakeShell(cwd)

    monkeypatch.setattr(bench_agent_module, "ShellTools", fake_make_shell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    agent = BenchAgent(llm=FakeLLMClient())

    with pytest.raises(ValueError, match="user_message, problem_statement, or task_description"):
        await agent._run_evaluation({"working_dir": str(tmp_path)})


def test_bench_agent_uses_python_tools_and_todo_context_blocks():
    """BenchAgent renders Python tool and todo docs as static context blocks."""

    agent = BenchAgent(llm=FakeLLMClient())

    keys = list(agent.context_manager.keys())

    assert "python_tools" in keys
    assert "todo" in keys
    assert "shell" not in keys
    assert "self.shell" not in keys

    python_tools_doc = agent.context_manager["python_tools"]
    assert "class RepoTools" in python_tools_doc
    assert "def symbols(" in python_tools_doc
    assert "def refs(" in python_tools_doc
    assert "class ShellTools" in python_tools_doc
    assert "def run(" in python_tools_doc

    todo_doc = agent.context_manager["todo"]
    assert "def add(" in todo_doc
    assert "def done(" in todo_doc


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


def test_solve_task_prompt_uses_todo_plan_workflow():
    """The task prompt asks the agent to make a todo-based plan."""

    doc = BenchAgent._solve_task.__doc__ or ""

    assert "2. Write a plan based on todos" in doc
    assert "Use ``doc(self)`` to see all available tools and methods." not in doc


def test_bench_agent_preseeds_planning_todo():
    """BenchAgent starts each task with an explicit planning todo."""

    agent = BenchAgent(llm=FakeLLMClient())

    todos = agent.todo.list_todos()

    assert [t.title for t in todos] == ["Create a todo-based plan with clear dependencies"]


@pytest.mark.asyncio
async def test_run_evaluation_reseeds_planning_todo_after_clear(monkeypatch, tmp_path):
    """The planning todo is restored after per-task todo reset."""

    def fake_make_shell(cwd: str, init_command=None):
        return _FakeShell(cwd)

    async def fake_solve_task(description: str):
        titles = [t.title for t in agent.todo.list_todos()]
        assert titles == ["Create a todo-based plan with clear dependencies"]
        return TaskResult(
            solution_description="Planned and fixed.",
            evidence="pytest passed",
            command_to_verify="pytest -q",
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


def test_problem_statement_skips_blank_primary_field():
    """Blank higher-priority fields do not block fallback task text."""

    assert (
        bench_agent_module._problem_statement(
            {"user_message": "   ", "problem_statement": " use this "}
        )
        == "use this"
    )
