# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generic BenchAgent with structured TaskResult output."""

from __future__ import annotations

import json

import pytest
from nooa_bench import bench_agent as bench_agent_module
from nooa_bench import runner
from nooa_bench.bench_agent import BenchAgent, TaskResult
from nooa_bench.rlm_bench_agent import RLMBenchAgent

from nooa.agentdoc import doc
from nooa.runtime.event_manager import EventManager
from nooa.unifiedllm import AssistantReasoning, AssistantText, FakeLLMClient, LLMResponse, ToolCall


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

    async def close(self) -> None:
        self.closed = True


class _FakeRepo:
    def __init__(self, root: str, session: object | None = None) -> None:
        self.root = root
        self.session = session


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [BenchAgent, RLMBenchAgent])
async def test_working_directory_context_is_untraced(agent_type, monkeypatch, tmp_path):
    monkeypatch.setattr(bench_agent_module, "ShellTools", _FakeShell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    agent = agent_type(llm=FakeLLMClient(), working_dir=str(tmp_path))
    try:
        assert getattr(agent_type._working_directory_context, "_no_trace", False)
        before = list(agent.event_manager.all_events())
        assert agent._working_directory_context() == f"Working directory for self.shell: {tmp_path}"
        assert list(agent.event_manager.all_events()) == before
    finally:
        await agent.aclose()


def test_trajectory_excludes_opaque_provider_state(monkeypatch, tmp_path):
    response = LLMResponse(
        parts=(
            AssistantText(text="public answer"),
            AssistantReasoning(
                text="portable reasoning", native={"encrypted_content": "provider-secret"}
            ),
        ),
    )
    manager = EventManager()
    manager.add(response)
    agent = type("Agent", (), {"event_manager": manager})()
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)

    runner._write_trajectory(agent)

    payload = (tmp_path / "trajectory.json").read_text()
    assert "public answer" in payload
    assert "portable reasoning" in payload
    assert "provider-secret" not in payload
    assert "llm_state" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [BenchAgent, RLMBenchAgent])
async def test_delegate_uses_framework_value_formatting(agent_type, monkeypatch, tmp_path):
    monkeypatch.setattr(bench_agent_module, "ShellTools", _FakeShell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    code = (
        "return_result(TaskResult(solution_description=description, "
        "evidence=supplied_context['password'], how_to_verify='check'))"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(id="cell", name="python_cell", arguments=json.dumps({"code": code}))
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    agent = agent_type(llm=llm, working_dir=str(tmp_path))
    try:
        value = {"password": "synthetic-example", "numbers": list(range(100))}
        result = await agent.delegate("inspect", value)
        assert result.solution_description == "inspect"
        assert result.evidence == "synthetic-example"
        rendered = str(llm.last_messages)
        assert "supplied_context" in rendered
        assert "synthetic-example" in rendered
        assert "[REDACTED]" not in rendered
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "how_to_verify", ["pytest tests/ -x", "Compare the totals in the report with the source table."]
)
def test_task_result_model(how_to_verify):
    """TaskResult validates required fields with solution_description."""
    r = TaskResult(
        solution_description="Fixed missing URL-encoding in auth.py with quote_plus().",
        evidence="pytest tests/ passed: 5 passed in 1.2s",
        how_to_verify=how_to_verify,
    )
    assert "URL-encoding" in r.solution_description
    assert r.how_to_verify == how_to_verify
    properties = TaskResult.model_json_schema()["properties"]
    assert properties["how_to_verify"]["title"] == "How to Verify"
    assert "command_to_verify" not in properties


def test_trajectory_preserves_nested_json_without_private_state(monkeypatch, tmp_path):
    from pydantic import BaseModel, Field

    from nooa.context_blocks.events import ToolCallEvent, ToolResult
    from nooa.events import PythonOutput

    class Payload(BaseModel):
        answer: str = "visible"
        hidden: str = Field(default="hidden-secret", repr=False)
        excluded: str = Field(default="excluded-secret", exclude=True)

    response = LLMResponse(
        parts=(
            AssistantText(text="public answer"),
            AssistantReasoning(
                text="readable thought", native={"encrypted_content": "provider-secret"}
            ),
        )
    )
    call = ToolCallEvent(
        tool_call_id="c1",
        name="lookup",
        arguments={},
        result=ToolResult(tool_call_id="c1", content="actual result"),
    )
    nested = PythonOutput(
        tool_call_id="c1",
        execution_status="complete",
        execution_count=1,
        value={"responses": [response], "payload": Payload()},
    )
    manager = EventManager()
    manager.add(call)
    manager.add(nested)
    agent = type("Agent", (), {"event_manager": manager})()
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    runner._write_trajectory(agent)
    encoded = (tmp_path / "trajectory.json").read_text()
    exported = json.loads(encoded)
    assert exported[0]["result"]["content"] == "actual result"
    assert exported[0]["result"]["tool_call_id"] == "c1"
    assert exported[1]["value"]["responses"][0]["content"] == "public answer"
    assert exported[1]["value"]["responses"][0]["reasoning"] == "readable thought"
    assert exported[1]["value"]["payload"] == {"answer": "visible"}
    assert "secret" not in encoded


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


def test_bench_agent_close_is_hidden_from_model_docs():
    agent = BenchAgent(llm=FakeLLMClient())

    assert "def close(" not in doc(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_class", [BenchAgent, RLMBenchAgent])
async def test_merge_error_is_not_advertised_in_python_cell_context(agent_class):
    """Recovery exceptions remain importable but are not up-front capabilities."""
    from nooa.strategies import CodeActV2

    agent = agent_class(llm=FakeLLMClient())
    runtime = type("Runtime", (), {"agent": agent})()
    try:
        rendered = await CodeActV2().python_cell_context(runtime)
        assert "DelegationMergeError" not in rendered
        assert "TaskResult" in rendered
        assert issubclass(bench_agent_module.DelegationMergeError, ValueError)
    finally:
        await agent.aclose()


@pytest.mark.parametrize("agent_type", [BenchAgent, RLMBenchAgent])
def test_bench_agent_context_is_minimal_and_automatic(agent_type):
    """Only actionable live context is exposed; compaction is automatic."""
    agent = agent_type(llm=FakeLLMClient())

    keys = list(agent.context_manager.keys())

    assert "todo_status" in keys
    assert "python_cell_tools" in keys
    assert "task" not in keys
    assert "todo" not in keys
    assert "context_usage" in keys
    assert getattr(agent, "_summarizers", [])


def test_task_text_is_not_retained_in_agent_state():
    """The method argument is the sole task copy; state must not duplicate it."""
    agent = BenchAgent(llm=FakeLLMClient())

    assert not hasattr(agent, "problem_statement")


def test_bench_agent_hides_manual_context_maintenance_apis():
    """The model should solve the task, not manually rewrite its prompt history."""
    agent = BenchAgent(llm=FakeLLMClient())

    agent_doc = doc(agent)

    assert "    context:" not in agent_doc
    assert "events:" not in agent_doc


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["plain answer", None, 42])
async def test_run_evaluation_handles_non_task_result(monkeypatch, tmp_path, value):
    monkeypatch.setattr(bench_agent_module, "ShellTools", _FakeShell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    agent = BenchAgent(llm=FakeLLMClient(), working_dir=str(tmp_path))

    async def solve(_description):
        return value

    monkeypatch.setattr(agent, "_solve_task", solve)
    try:
        result = await agent._run_evaluation(
            {"problem_statement": "task", "working_dir": str(tmp_path)}
        )
        assert result == {
            "response": str(value) if value is not None else "",
            "success": True,
            "result": value,
        }
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "how_to_verify", ["pytest -q", "Compare the report totals with the source table."]
)
async def test_run_evaluation_returns_structured_task_result(monkeypatch, tmp_path, how_to_verify):
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
            how_to_verify=how_to_verify,
        )

    monkeypatch.setattr(bench_agent_module, "ShellTools", fake_make_shell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    agent = BenchAgent(llm=FakeLLMClient())
    monkeypatch.setattr(agent, "_solve_task", fake_solve_task)

    result = await agent._run_evaluation(
        {"problem_statement": "fix the bug", "working_dir": str(tmp_path)}
    )

    assert result == {
        "response": how_to_verify,
        "success": True,
        "result": {
            "solution_description": "Fixed the bug.",
            "evidence": "pytest passed",
            "how_to_verify": how_to_verify,
        },
    }
    assert shells[-1].cwd == str(tmp_path)
    assert shells[0].closed is True


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
            solution_description="Fixed.", evidence="check passed", how_to_verify="true"
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


def test_bench_agent_python_tools_follow_agent_attribute_order():
    """Python tool docs follow the model-facing shell, repo, todo order."""
    agent = BenchAgent(llm=FakeLLMClient())

    keys = list(agent.context_manager.keys())
    assert "python_cell_tools" in keys
    assert "todo_status" in keys
    assert "todo" not in keys

    python_tools_doc = agent.context_manager["python_cell_tools"]
    assert "class ShellTools" in python_tools_doc
    assert "def run(" in python_tools_doc
    assert "class RepoTools" in python_tools_doc
    assert "def symbols(" in python_tools_doc
    assert "class TodoManager" in python_tools_doc
    assert "class MethodWriting" in python_tools_doc
    assert "@strategy(PredictStrategy())" in python_tools_doc
    assert "asyncio.gather" in python_tools_doc
    assert agent.methodwriting._agent is agent
    assert python_tools_doc.index("class ShellTools") < python_tools_doc.index("class RepoTools")
    assert python_tools_doc.index("class RepoTools") < python_tools_doc.index("class TodoManager")
    assert python_tools_doc.index("class TodoManager") < python_tools_doc.index(
        "class MethodWriting"
    )


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
            solution_description="Fixed.", evidence="check passed", how_to_verify="true"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [BenchAgent, RLMBenchAgent])
async def test_bench_workers_start_with_task_local_state(agent_type, tmp_path):
    worker = agent_type(llm=FakeLLMClient(), working_dir=str(tmp_path), delegation_depth=1)
    try:
        assert worker.todo.list_todos() == []
        assert not hasattr(worker, "v")
        assert worker._delegation_depth == 1
    finally:
        await worker.close()


def test_variants_share_identity_and_document_delegation_hierarchy():
    from nooa_bench import AGENT_CLASSES

    assert AGENT_CLASSES["rlm"] == "nooa_bench.rlm_bench_agent:RLMBenchAgent"
    for agent_type in (BenchAgent, RLMBenchAgent):
        prompt = doc(agent_type)
        assert "You are an autonomous software engineering agent." in prompt
        assert "Recursive same-kind delegation is bounded" in prompt
        assert "Inspect and integrate each result" in prompt


def test_rlm_identity_is_normalized_independently_of_python_docstring_dedent():
    import inspect

    prompt = RLMBenchAgent.__doc__
    assert prompt == inspect.cleandoc(prompt)
    assert "\nUse context-isolated subagents" in prompt
    assert prompt.startswith(inspect.cleandoc(BenchAgent.__doc__))


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [BenchAgent, RLMBenchAgent])
async def test_delegate_input_failure_closes_worker(agent_type, monkeypatch, tmp_path):
    shells = []

    class CountingShell(_FakeShell):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            shells.append(self)

    async def fail_solve(self, description, supplied_context=None):
        raise ValueError("invalid supplied context")

    monkeypatch.setattr(bench_agent_module, "ShellTools", CountingShell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    monkeypatch.setattr(agent_type, "_solve_task", fail_solve)
    agent = agent_type(llm=FakeLLMClient(), working_dir=str(tmp_path))
    try:
        with pytest.raises(ValueError, match="invalid supplied context"):
            await agent.delegate("inspect", {"reference": "data"})
        assert len(shells) == 2
        assert shells[1].closed
        assert not getattr(agent.shell, "closed", False)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [BenchAgent, RLMBenchAgent])
async def test_delegate_launches_isolated_subagent_of_same_type(agent_type, monkeypatch, tmp_path):
    observed = {}
    expected = TaskResult(
        solution_description="Inspected parser.",
        evidence="Focused check passed.",
        how_to_verify="pytest -q tests/test_parser.py",
    )

    async def fake_solve(self, description: str, supplied_context=None):
        observed.update(
            child_type=type(self),
            child=self,
            description=description,
            supplied_context=supplied_context,
            cwd=str(self.shell.cwd),
            depth=self._delegation_depth,
            max_depth=self._max_delegation_depth,
        )
        return expected

    async def fake_close(self):
        observed["closed"] = True

    monkeypatch.setattr(bench_agent_module, "ShellTools", _FakeShell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    monkeypatch.setattr(agent_type, "_solve_task", fake_solve)
    monkeypatch.setattr(_FakeShell, "close", fake_close, raising=False)
    llm = FakeLLMClient()
    from nooa.interactive import SummarizationConfig

    config = SummarizationConfig(policy="none")
    agent = agent_type(llm=llm, working_dir=str(tmp_path), summarization=config)

    todo = agent.todo.add("Investigate empty parser input")
    result = await agent.delegate("inspect parser", todo)

    assert result == expected
    assert observed["child_type"] is agent_type
    assert observed["child"] is not agent
    assert observed["child"].llm is llm
    assert observed["child"]._summarization is config
    assert observed["description"] == "inspect parser"
    assert observed["supplied_context"] is todo
    assert observed["cwd"] == str(tmp_path)
    assert observed["depth"] == 1
    assert observed["max_depth"] == 4
    assert observed["child"].shell.init_command == bench_agent_module._OPTIONAL_TESTBED_ACTIVATE
    assert observed["closed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [BenchAgent, RLMBenchAgent])
async def test_delegate_todo_merges_worker_description(agent_type, monkeypatch, tmp_path):
    expected = TaskResult(
        solution_description="Inspected parser.",
        evidence="Focused check passed.",
        how_to_verify="pytest -q tests/test_parser.py",
    )

    async def fake_solve(self, description: str, supplied_context=None):
        delegated = self.todo.list_todos()[0]
        assert delegated is not task
        assert description.startswith(f"{task.title}\n\nWork on active todo {task.id}.")
        assert "Record material findings" in description
        self.todo.comment(delegated, "worker finding")
        self.todo.set_var(delegated, "path", "parser.py")
        return expected

    async def fake_close(self):
        pass

    monkeypatch.setattr(bench_agent_module, "ShellTools", _FakeShell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    monkeypatch.setattr(agent_type, "_solve_task", fake_solve)
    monkeypatch.setattr(_FakeShell, "close", fake_close, raising=False)
    agent = agent_type(llm=FakeLLMClient(), working_dir=str(tmp_path))
    task = agent.todo.add("Inspect parser", description="focus on errors")

    result = await agent.delegate(task)

    assert result == expected
    assert [comment.body for comment in task.comments] == ["worker finding"]
    assert task.v.path == "parser.py"


@pytest.mark.asyncio
async def test_delegate_todo_does_not_merge_when_close_fails(monkeypatch, tmp_path):
    expected = TaskResult(
        solution_description="Inspected parser.",
        evidence="Focused check passed.",
        how_to_verify="pytest -q tests/test_parser.py",
    )

    async def fake_solve(self, description: str, supplied_context=None):
        self.todo.comment(self.todo.list_todos()[0], "worker finding")
        return expected

    async def fake_close(self):
        raise RuntimeError("close failed")

    monkeypatch.setattr(bench_agent_module, "ShellTools", _FakeShell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    monkeypatch.setattr(BenchAgent, "_solve_task", fake_solve)
    monkeypatch.setattr(_FakeShell, "close", fake_close, raising=False)
    agent = BenchAgent(llm=FakeLLMClient(), working_dir=str(tmp_path))
    task = agent.todo.add("Inspect parser")

    with pytest.raises(RuntimeError, match="close failed"):
        await agent.delegate(task)

    assert task.comments == []


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [BenchAgent, RLMBenchAgent])
@pytest.mark.parametrize("depth,limit", [(2, 2), (5, None)])
async def test_delegate_rejects_unbounded_recursion(
    agent_type, monkeypatch, tmp_path, depth, limit
):
    monkeypatch.setattr(bench_agent_module, "ShellTools", _FakeShell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    agent = agent_type(
        llm=FakeLLMClient(),
        working_dir=str(tmp_path),
        delegation_depth=depth,
        **({"max_delegation_depth": limit} if limit is not None else {}),
    )

    with pytest.raises(RuntimeError, match="maximum delegation depth"):
        await agent.delegate("delegate again")


def test_problem_statement_skips_blank_primary_field():
    """Blank higher-priority fields do not block fallback task text."""

    assert (
        bench_agent_module._problem_statement(
            {"user_message": "   ", "problem_statement": " use this "}
        )
        == "use this"
    )


def test_capability_and_delegation_examples_are_host_independent():
    from nooa.tools.method_writing_lib import MethodWriting

    assert "doc(self.methodwriting)" not in MethodWriting.__doc__
    assert "await self.delegate(objective, supplied_context)" in RLMBenchAgent._solve_task.__doc__


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [BenchAgent, RLMBenchAgent])
async def test_solve_task_uses_v2_single_tool_contract(agent_type, tmp_path):
    """Bench agents share CodeActV2's context contract.

    The single python_cell tool stays; duplicated framework blocks (state,
    execution_context, strategy prompt) are suppressed; the
    class docs render once, concisely, as the self block. The namespace context
    remains available without the automatic cell-state inventory.
    """
    code = (
        "return_result(TaskResult(solution_description='done', evidence='ran true', "
        "how_to_verify='true'))"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="python_cell", arguments=json.dumps({"code": code}))
                ],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": ""},
            )
        ]
    )
    agent = agent_type(llm=llm, working_dir=str(tmp_path))
    from nooa.context_blocks.models import ContextWindowStats

    llm.config["max_tokens"] = 8000
    agent.runtime._last_context_stats = ContextWindowStats(
        context_blocks_count=5,
        events_count=12,
        prompt_tokens=24000,
        context_blocks_chars=10000,
        events_chars=30000,
        model_context_window=128000,
        reserved_output_tokens=8000,
    )
    try:
        result = await agent._solve_task("solve the supplied task")
        assert result.solution_description == "done"

        assert [tool.name for tool in llm.last_tools or []] == ["python_cell"]
        assert "doc(self.delegate)" in llm.last_tools[0].description
        assert "asyncio.gather" in llm.last_tools[0].description
        assert "PredictStrategy" in llm.last_tools[0].description
        system_prompt = "\n".join(
            str(message.get("content", ""))
            for message in llm.last_messages
            if message.get("role") == "system"
        )
        rendered = "\n".join(str(m.get("content", "")) for m in llm.last_messages)

        assert "<state" not in system_prompt
        assert "<execution_context" not in rendered
        assert "<context_usage" in rendered
        assert "Context: 24,000 / 120,000 usable tokens (20.0%)" in rendered
        assert "Compact history: self.events.collapse" in rendered
        assert "doc(self.events)" in rendered
        assert "doc(self.context)" in rendered
        assert "<strategy_prompt" not in rendered
        assert "<python_cell_tools" in system_prompt
        assert "<python_cell_context" in system_prompt
        assert "<python_cell_state" not in rendered
        assert "<self" in system_prompt
        assert "You are an autonomous software engineering agent." in system_prompt
        assert "Solve the supplied task completely." in rendered
        assert "MethodWriting" in rendered
        assert "doc(self.writing)" not in rendered
        assert "supplied_context" in rendered
        # The prefix uses concise docs; doc(self.delegate) expands the guidance.
        assert "ordinary method argument" in doc(agent.delegate)
        assert len(system_prompt) < 20_000
    finally:
        await agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", ["title", "dependency", "removed"])
async def test_delegation_merge_failure_keeps_result_and_worker_state(
    monkeypatch, tmp_path, conflict
):
    from nooa_bench.bench_agent import DelegationMergeError

    expected = TaskResult(solution_description="done", evidence="passed", how_to_verify="true")

    async def fake_solve(self, description, supplied_context=None):
        delegated = self.todo.list_todos()[0]
        self.todo.comment(delegated, "useful finding")
        if conflict == "title":
            agent.todo.update(task, title="parent edit")
            self.todo.update(delegated, title="worker edit")
        elif conflict == "dependency":
            child = self.todo.add("worker dependency")
            self.todo.add_dep(delegated, child)
        else:
            self.todo.remove(delegated)
        return expected

    monkeypatch.setattr(bench_agent_module, "ShellTools", _FakeShell)
    monkeypatch.setattr(bench_agent_module, "RepoTools", _FakeRepo)
    monkeypatch.setattr(BenchAgent, "_solve_task", fake_solve)
    agent = BenchAgent(llm=FakeLLMClient(), working_dir=str(tmp_path))
    task = agent.todo.add("work")
    try:
        with pytest.raises(DelegationMergeError) as caught:
            await agent.delegate(task)
        assert caught.value.result == expected
        restored = bench_agent_module.TodoManager(caught.value.worker_state)
        if conflict == "removed":
            assert restored.get(task.id) is None
            assert "disappeared" in str(caught.value)
        else:
            assert restored.get(task.id).comments[0].body == "useful finding"
        assert task.comments == []
        assert task.deps == []
        if conflict == "title":
            assert task.title == "parent edit"
        elif conflict == "dependency":
            assert restored.get(task.id).deps[0] == restored.list_todos()[1].id
    finally:
        await agent.close()


def test_bench_import_does_not_import_coding_application():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from nooa_bench import bench_agent
assert bench_agent.BenchAgent
for name in ('agent', 'activity', 'slash_commands', 'settings'):
    assert 'nooa_cli.coding.' + name not in sys.modules, name
""",
        ],
        check=True,
    )


@pytest.mark.asyncio
async def test_original_task_remains_after_prefill_compaction(tmp_path):
    class CompactingLLM(FakeLLMClient):
        async def acall(self, messages, **kwargs):
            if not self.compacted:
                tags = list(agent.event_manager.keys())
                agent.event_manager.collapse(tags[0], tags[-1], summary_text="no task text here")
                self.compacted = True
            return await super().acall(messages, **kwargs)

    def response(code, call_id):
        return LLMResponse(
            parts=(ToolCall(id=call_id, name="python_cell", arguments=json.dumps({"code": code})),),
            finish_reason="tool_calls",
        )

    llm = CompactingLLM(
        scripted_responses=[
            response("pass", "one"),
            response(
                "assert Todo is not None and TodoManager is not None and ShellTools is not None "
                "and RepoTools is not None and MethodWriting is not None\n"
                "return_result(TaskResult(solution_description='done', evidence='ok', how_to_verify='true'))",
                "two",
            ),
        ]
    )
    llm.compacted = False
    agent = BenchAgent(llm=llm, working_dir=str(tmp_path))
    try:
        assert (await agent._solve_task("UNIQUE-ORIGINAL-TASK")).solution_description == "done"
        assert llm.compacted
        rendered = str(llm.last_messages)
        assert "UNIQUE-ORIGINAL-TASK" in rendered
        assert "TaskResult" in rendered
    finally:
        await agent.close()
