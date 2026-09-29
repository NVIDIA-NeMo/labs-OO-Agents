# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generic benchmark agent for code and system tasks.

A single, non-specialized CodeAct agent that works on any task requiring shell
access inside a container. Not tuned for any particular benchmark -- the same
agent handles SWE-bench, Terminal-Bench, or any Harbor-compatible task.

Core contract:
- ``self.shell`` for persistent shell access (run/read/replace/write_file)
- ``self.repo`` for code navigation that returns ShellTools Match anchors
- ``self.todo`` for optional structured progress tracking
- Structured return: the agent must declare solution_description, evidence,
  and how_to_verify when finishing -- forcing reflection before return.
"""

from __future__ import annotations

from nooa_cli.tools.repo_tools import RepoTools

from nooa import hidden as _hidden
from nooa.tools.method_writing_lib import MethodWriting
from nooa.tools.shell_tools import ShellTools
from nooa.tools.todo import Todo, TodoManager

_agentdoc_hidden_names = {"_hidden"}

with _hidden:
    import logging
    import os
    from typing import TYPE_CHECKING, Any

    from pydantic import BaseModel, Field

    from nooa import Agent, Context, no_trace, strategy
    from nooa.agentdoc import doc
    from nooa.config import CodeActConfig
    from nooa.interactive import SummarizationConfig, install_summarizer
    from nooa.strategies import CodeActV2
    from nooa.unifiedllm import FakeLLMClient

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM

_logger = logging.getLogger(__name__)

_OPTIONAL_TESTBED_ACTIVATE = (
    "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then "
    "[ -d /opt/harbor/cpython312/bin ] && export PATH=/opt/harbor/cpython312/bin:$PATH; "
    "source /opt/miniconda3/etc/profile.d/conda.sh; "
    "conda env list | awk '{print $1}' | grep -qx testbed && conda activate testbed || true; "
    "fi"
)

_SOLVE_STRATEGY = CodeActV2(config=CodeActConfig(max_retries=10, cell_timeout=1800.0))
_SOLVE_CONTEXT = {
    "state": None,
    "execution_context": None,
    "python_cell_state": None,
    "self": Context(expr="doc(type(self), concise=True)", prefix=True),
    # Method inputs remain live even if their prefill events are summarized.
    # Reuse the framework's bounded parameter rendering rather than a raw copy.
    "task": Context(
        expr="runtime.current_call.format_parameters_as_code(tc=runtime.truncation_config)",
        prefix=True,
    ),
}


class TaskResult(BaseModel):
    """Structured result the agent must return when finishing a task."""

    solution_description: str = Field(
        description="What you did and why it solves the problem. Describe root cause and fix."
    )
    evidence: str = Field(
        description=(
            "Concrete evidence that the task is done: what tests passed, "
            "what output was produced, what behavior changed. Not a guess -- "
            "cite the actual results you observed."
        )
    )
    how_to_verify: str = Field(
        title="How to Verify",
        description=(
            "How a verifier can confirm correctness: concrete checks or steps and their "
            "expected results. Include commands when appropriate; a shell command is not required."
        ),
    )


@_hidden
class DelegationMergeError(ValueError):
    """Worker completed, but Todo changes could not be merged safely.

    ``result`` is the completed TaskResult; ``worker_state`` holds all worker
    todos, including local dependencies. Inspect these and reconcile explicitly.
    Parent state is unchanged. The completed worker need not be run again.
    """

    def __init__(self, message: str, result: TaskResult, worker_state: dict):
        super().__init__(message)
        self.result = result
        self.worker_state = worker_state


@_hidden
def _problem_statement(task_input: dict) -> str:
    """Extract the task text from supported Harbor/benchmark field names."""
    for key in ("user_message", "problem_statement", "task_description"):
        value = task_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        "task_input must include a non-empty user_message, problem_statement, or task_description"
    )


class BenchAgent(
    Agent,
    llm=FakeLLMClient(),
    context={
        "todo_status": Context(expr="self.todo.status()"),
        "context_usage": Context(expr="self.context_stats.format() if self.context_stats else ''"),
    },
):
    """You are an autonomous software engineering agent.

    Understand the task and inspect relevant inputs before acting. Preserve unrelated
    work. Define how success will be verified; for code changes, reproduce the failure
    or add a failing test first. Make the smallest sufficient change. Never claim a
    task is complete without verifying that it meets the requested requirements:
    run the relevant checks and inspect their results. If verification is blocked,
    report the blocker rather than claiming completion. Use todos only
    when they clarify multi-step work. Keep an active Todo's title and description
    aligned with the current understanding, and comment material findings, decisions,
    completed steps, and verification—not routine narration. Finish with ``TaskResult``.
    """

    shell: ShellTools
    repo: RepoTools
    todo: TodoManager
    methodwriting: MethodWriting

    def __init__(
        self,
        llm: UnifiedLLM | None = None,
        *,
        summarization: SummarizationConfig | None = None,
        working_dir: str | None = None,
        delegation_depth: int = 0,
        max_delegation_depth: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**({"llm": llm} if llm is not None else {}), **kwargs)
        cwd = working_dir or next(
            (d for d in ("/testbed", "/app") if os.path.isdir(d)), os.getcwd()
        )
        self._delegation_depth = delegation_depth
        self._max_delegation_depth = max_delegation_depth
        self._summarization = summarization or SummarizationConfig()
        self._install_python_tools(cwd)
        self.todo = TodoManager()
        self.methodwriting = MethodWriting()
        self.methodwriting.attach(self)
        self.context_manager["python_cell_tools"] = Context(
            doc(ShellTools, RepoTools, TodoManager, MethodWriting), prefix=True
        )
        self.context_manager["working_directory"] = Context(
            expr="self._working_directory_context()"
        )
        install_summarizer(self._summarization, self)

    @no_trace
    def _working_directory_context(self) -> str:
        """Render the application's shell location as a bounded context label."""
        from html import escape

        path = str(self.shell.cwd).replace("\n", "\\n").replace("\r", "\\r")
        return "Working directory for self.shell: " + escape(path[:160], quote=False)

    def _install_python_tools(self, cwd: str) -> None:
        """Install shell/repo tools rooted at the same working directory."""
        self.shell = ShellTools(
            cwd=cwd,
            init_command=_OPTIONAL_TESTBED_ACTIVATE,
        )
        self.repo = RepoTools(root=cwd, session=self.shell.session)

    @_hidden
    async def close(self) -> None:
        """Compatibility alias for the standard async cleanup contract."""
        await self.aclose()

    @_hidden
    async def aclose(self) -> None:
        """Drain background summaries and close the shell, leaving the LLM to its owner."""
        try:
            await super().aclose()
        finally:
            await self.shell.close()

    async def _run_evaluation(self, task_input: dict) -> dict:
        """Entry point called by the Harbor runner."""
        description = _problem_statement(task_input)
        instructions = task_input.get("system_prompt") or task_input.get("instructions") or ""
        initial_obs = task_input.get("initial_observation") or ""

        self.context["instructions"] = instructions or None
        self.context["initial_observation"] = initial_obs or None

        cwd = task_input.get("working_dir")
        if cwd:
            if not os.path.isdir(cwd):
                raise ValueError(f"working_dir does not exist: {cwd!r}")
        else:
            cwd = next((d for d in ("/testbed", "/app") if os.path.isdir(d)), os.getcwd())
        old_shell = self.shell
        await old_shell.close()
        self._install_python_tools(cwd)
        self.todo.clear()

        try:
            result = await self._solve_task(description)
            if isinstance(result, TaskResult):
                return {
                    "response": result.how_to_verify,
                    "success": bool(result.solution_description),
                    "result": result.model_dump(),
                }
            result_str = str(result) if result is not None else ""
            return {"response": result_str, "success": True, "result": result}
        except Exception as e:
            _logger.error("BenchAgent failed: %s", e)
            return {"response": "", "success": False, "error": str(e)}

    async def delegate(self, objective: str | Todo, supplied_context: Any = None) -> TaskResult:
        """Ask an isolated subagent to complete a bounded objective.

        Pass a Todo as the first argument to make it the subagent's task. It receives an
        independent task copy and can record comments or variables with ``self.todo``;
        after successful execution and cleanup, changes are merged into the parent.
        A string objective is used as the task text verbatim.

        ``supplied_context`` is passed as an ordinary method argument to the worker;
        NOOA's standard parameter formatting displays it to the model.
        To delegate a Todo, pass it as ``objective``, not ``supplied_context``.

        Conflicting edits, new worker-only dependencies, or removal of the delegated
        Todo raise DelegationMergeError;
        its ``result`` and ``worker_state`` preserve the completed work for recovery.
        A failed worker or failed cleanup does not merge partial Todo changes.

        Use delegation when isolated context helps exploration, diagnosis, review, or
        implementation. Recursive same-kind delegation is bounded by
        ``max_delegation_depth`` (default 4). Independent calls may run concurrently
        with ``asyncio.gather``. Inspect and integrate each result; you retain final
        verification ownership.
        """
        if self._delegation_depth >= self._max_delegation_depth:
            raise RuntimeError(f"maximum delegation depth ({self._max_delegation_depth}) reached")
        todo_base = self.todo.copy_todo(objective) if isinstance(objective, Todo) else None
        if todo_base is not None:
            description = (
                f"{todo_base.title}\n\nWork on active todo {todo_base.id}. Keep its title and "
                "description aligned with the current understanding. Record material findings, "
                "decisions, completed steps, and verification with self.todo.comment(...), not "
                "routine narration; use self.todo.set_var(...) for structured artifacts."
            )
        else:
            description = str(objective)
        updated: Todo | None = None
        worker_state: dict = {}
        subagent = type(self)(
            llm=self.llm,
            working_dir=str(self.shell.cwd),
            delegation_depth=self._delegation_depth + 1,
            max_delegation_depth=self._max_delegation_depth,
            summarization=self._summarization,
        )
        try:
            if todo_base is not None:
                subagent.todo = TodoManager.with_todo(todo_base)
            result = await subagent._solve_task(description, supplied_context=supplied_context)
            updated = subagent.todo.get(todo_base) if todo_base is not None else None
            if todo_base is not None:
                worker_state = subagent.todo.to_dict()
            if todo_base is not None and updated is None:
                raise DelegationMergeError(
                    f"delegated todo {todo_base.id!r} disappeared", result, worker_state
                )
        finally:
            await subagent.close()
        if todo_base is not None and updated is not None:
            try:
                self.todo.merge_todo(updated, base=todo_base)
            except ValueError as exc:
                raise DelegationMergeError(str(exc), result, worker_state) from exc
        return result

    @_hidden
    @strategy(
        _SOLVE_STRATEGY,
        context=_SOLVE_CONTEXT,
    )
    async def _solve_task(self, description: str, supplied_context: Any = None) -> TaskResult:
        """Solve the supplied task completely.

        Inspect before editing. Plan with ``self.todo`` only when useful. Make the
        minimum sufficient change, preserve unrelated work, and run relevant tests.
        Then call ``return_result(TaskResult(...))`` with the root cause and fix,
        concrete observed evidence, and how to verify the result.
        """
        ...
