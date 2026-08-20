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
  and command_to_verify when finishing -- forcing reflection before return.
"""

from __future__ import annotations

from nooa import hidden as _hidden

_agentdoc_hidden_names = {"_hidden"}

with _hidden:
    import logging
    import os
    from typing import TYPE_CHECKING, Any

    from nooa_cli.coding.delegation import CodingDelegationMixin
    from nooa_cli.tools.repo_tools import RepoTools
    from pydantic import BaseModel, Field

    from nooa import Agent, Context, strategy
    from nooa.agentdoc import doc
    from nooa.config import CodeActConfig
    from nooa.interactive import SummarizationConfig, install_summarizer
    from nooa.strategies import CodeActStrategy
    from nooa.tools.shell_tools import ShellTools
    from nooa.tools.todo import TodoManager
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


class TaskResult(BaseModel):
    """Structured result the agent must return when finishing a task."""

    solution_description: str = Field(
        description="What you did and why it solves the problem. Describe root cause and fix."
    )
    evidence: str = Field(
        description=(
            "Concrete evidence that the task is done: what tests passed, "
            "what output was produced, what behavior changed. Not a guess -- "
            "cite the actual shell output you observed."
        )
    )
    command_to_verify: str = Field(
        description="A shell command a verifier can run to confirm correctness (exit 0 on success)."
    )


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
    context={"todo_status": Context(expr="self.todo.status()")},
):
    """Compact benchmark agent for code and system tasks.

    Read relevant code before editing, preserve unrelated work, make the smallest
    sufficient change, and verify with an observed command result. Use todos only
    when they clarify multi-step work. Finish with ``TaskResult``.
    """

    shell: ShellTools
    repo: RepoTools
    todo: TodoManager

    def __init__(
        self,
        llm: UnifiedLLM | None = None,
        *,
        summarization: SummarizationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, **kwargs)
        cwd = next((d for d in ("/testbed", "/app") if os.path.isdir(d)), os.getcwd())
        self._install_python_tools(cwd)
        self.todo = TodoManager()
        self.context_manager["python_tools"] = Context(doc(RepoTools, ShellTools), prefix=True)
        install_summarizer(summarization or SummarizationConfig(), self)

    def _install_python_tools(self, cwd: str) -> None:
        """Install shell/repo tools rooted at the same working directory."""
        self.shell = ShellTools(cwd=cwd, init_command=_OPTIONAL_TESTBED_ACTIVATE)
        self.repo = RepoTools(root=cwd, session=self.shell.session)

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
        self._install_python_tools(cwd)
        self.todo.clear()

        try:
            result = await self._solve_task(description)
            if isinstance(result, TaskResult):
                return {
                    "response": result.command_to_verify,
                    "success": bool(result.solution_description),
                    "result": result.model_dump(),
                }
            result_str = str(result) if result is not None else ""
            return {"response": result_str, "success": True, "result": result}
        except Exception as e:
            _logger.error("BenchAgent failed: %s", e)
            return {"response": "", "success": False, "error": str(e)}

    @strategy(
        CodeActStrategy(
            config=CodeActConfig(
                max_iterations=300, max_retries=10, text_only_stop_behavior="synthetic_comment"
            )
        )
    )
    async def _solve_task(self, description: str) -> TaskResult:
        """Solve this task completely: {description}

        Inspect before editing. Plan with ``self.todo`` only when useful. Make the
        minimum sufficient change, preserve unrelated work, and run relevant tests.
        Then call ``return_result(TaskResult(...))`` with the root cause and fix,
        concrete observed evidence, and one verifier command that exits zero.
        """
        ...


class RLMBenchAgent(CodingDelegationMixin, BenchAgent):
    """Benchmark controller with explicit context-isolated worker delegation.

    Delegate bounded, context-heavy exploration, diagnosis, review, or independent
    implementation. Keep planning, integration, final verification, and the final
    ``TaskResult`` in the controller. Independent calls may use ``asyncio.gather``;
    dependent calls must be sequential.
    """

    _worker_init_command = _OPTIONAL_TESTBED_ACTIVATE

    @strategy(
        CodeActStrategy(
            config=CodeActConfig(
                max_iterations=300, max_retries=10, text_only_stop_behavior="synthetic_comment"
            )
        )
    )
    async def _solve_task(self, description: str) -> TaskResult:
        """Solve this task completely: {description}

        Inspect before editing. Use ``delegate(objective, supplied_context)`` only
        for bounded work whose isolated context is an advantage; give each worker a
        self-contained request and inspect its report. The controller owns the plan,
        integration, final tests, and ``TaskResult``. Make the minimum sufficient
        change and cite only verification you observed.
        """
        ...
