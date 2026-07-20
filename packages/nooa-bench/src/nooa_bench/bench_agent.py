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

    from nooa_cli.tools.repo_tools import RepoTools
    from pydantic import BaseModel, Field

    from nooa import Agent, CodeActStrategy, strategy
    from nooa.agentdoc import doc, spec
    from nooa.config import CodeActConfig
    from nooa.context_blocks import DynamicContext
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
    context={
        "context_usage": DynamicContext(expr="self._context_usage_block()"),
        "todo_status": DynamicContext(expr="self.todo.status()"),
        "task": DynamicContext(expr="self.problem_statement"),
    },
):
    """Generic agent for code and system tasks in containers.

    ## Tools

    ```python
    r = await self.shell.run("command")                     # persistent shell
    r = await self.shell.read("file.py", lines=(1,50))      # view -> Match
    defs = await self.repo.symbols("src/", query="Handler") # definitions -> Match anchors
    refs = await self.repo.refs("Handler", path="src/")     # usages -> Match anchors
    await self.shell.replace(defs[0], new_code)              # edit at Match
    await self.shell.write_file("file.py", content)         # create/overwrite
    ```

    ## Workflow

    1. **Understand** -- explore the codebase/environment, reproduce the issue
    2. **Plan** -- write a plan based on todos
    3. **Implement and verify** -- make the fix and run relevant tests
    4. **Return** -- ``return_result(TaskResult(...))`` with evidence

    ## Return format

    When you are done, you MUST return a ``TaskResult``:

    ```python
    return_result(TaskResult(
        solution_description="Root cause: missing URL-encoding in auth.py. Fixed with quote_plus().",
        evidence="pytest tests/test_login.py passed (3 passed in 0.4s)",
        command_to_verify="pytest tests/test_login.py -x",
    ))
    ```

    Use ``self.todo`` to track progress on multi-step tasks.
    Mark todos done as you complete them.
    """

    shell: ShellTools
    repo: RepoTools

    def _context_usage_block(self) -> str:
        """Return context-window usage plus a benchmark-agent compaction hint."""
        if not self.context_stats:
            return ""
        return (
            f"{self.context_stats.format()}\n"
            "If event history is taking too much space, summarize older work "
            "and call self.events.collapse(start_tag, end_tag, summary_text) "
            "to replace it with a compact summary while keeping details accessible."
        )

    def __init__(self, llm: UnifiedLLM | None = None, **kwargs: Any) -> None:
        super().__init__(llm=llm, **kwargs)
        cwd = next((d for d in ("/testbed", "/app") if os.path.isdir(d)), os.getcwd())
        self._install_python_tools(cwd)
        self.todo = TodoManager()
        self._seed_todos()
        self.problem_statement = ""
        # Base Agent hides context/events by default; BenchAgent's context_usage
        # hint references them, so expose both APIs to the LLM here.
        spec(self, "context", hidden=False)
        spec(self, "events", hidden=False)
        from nooa import Context

        self.context_manager["python_tools"] = Context(doc(RepoTools, ShellTools), prefix=True)
        self.context_manager["todo"] = Context(doc(type(self.todo)), prefix=True)

    def _install_python_tools(self, cwd: str) -> None:
        """Install shell/repo tools rooted at the same working directory."""
        self.shell = ShellTools(cwd=cwd, init_command=_OPTIONAL_TESTBED_ACTIVATE)
        self.repo = RepoTools(root=cwd, session=self.shell.session)

    def _seed_todos(self) -> None:
        """Preload the planning todo every benchmark task should start from."""
        self.todo.add("Create a todo-based plan with clear dependencies")

    async def _run_evaluation(self, task_input: dict) -> dict:
        """Entry point called by the Harbor runner."""
        # Read task fields generically (Harbor adapters vary in field names).
        self.problem_statement = _problem_statement(task_input)
        instructions = task_input.get("system_prompt") or task_input.get("instructions") or ""
        initial_obs = task_input.get("initial_observation") or ""

        if instructions:
            self.context["instructions"] = instructions
        if initial_obs:
            self.context["initial_observation"] = initial_obs

        # Reset shell to the task working dir.
        cwd = task_input.get("working_dir")
        if cwd:
            if not os.path.isdir(cwd):
                raise ValueError(f"working_dir does not exist: {cwd!r}")
        else:
            cwd = next((d for d in ("/testbed", "/app") if os.path.isdir(d)), os.getcwd())
        self._install_python_tools(cwd)
        from nooa import Context

        self.context_manager["todo"] = Context(doc(type(self.todo)), prefix=True)
        self.todo.clear()
        self._seed_todos()

        try:
            result = await self._solve_task(self.problem_statement)
            if isinstance(result, TaskResult):
                return {
                    "response": result.command_to_verify,
                    "success": bool(result.solution_description),
                    "result": result.model_dump(),
                }
            # Fallback for non-structured returns
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
        """Solve the task.

        You are an expert software engineer and system administrator working
        inside a Linux container. Solve the task described below.

        ## Task
        {description}

        ## Instructions
        - Use ``await self.shell.run("command")`` to run shell commands.
        - Use ``await self.shell.read("path")`` to view files.
        - Use ``await self.repo.symbols(path, query="...")`` to find definitions.
        - Use ``await self.repo.refs(name, path=".")`` to find usages.
        - Use ``await self.shell.replace(...)`` to edit files or RepoTools matches.
        - Use ``await self.shell.write_file(path, content)`` to create files.
        - Use ``self.todo`` to track progress on multi-step work.
        - You have root access; install packages as needed.
        - Read task instructions carefully -- grading is strict and automated.

        ## Verification & Return

        Before finishing, run the relevant tests to confirm your work is correct.
        Then return a structured result explaining WHY you believe the task is done:

        ```python
        return_result(TaskResult(
            solution_description="The login handler didn't escape special chars in emails. Fixed by adding quote_plus() in auth.py:42.",
            evidence="pytest tests/test_login.py -x passed: 5 passed in 1.2s",
            command_to_verify="pytest tests/test_login.py -x",
        ))
        ```

        ## Workflow

        1. Explore and understand the task/codebase
        2. Write a plan based on todos
        3. Implement the solution and run tests to verify
        4. Return ``TaskResult(...)`` with concrete evidence
        """
        ...
