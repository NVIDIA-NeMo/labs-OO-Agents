# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal benchmark agent example for Harbor-compatible tasks."""

from pydantic import BaseModel, Field

from nooa import Agent, CodeActStrategy, hidden, strategy
from nooa.config import CodeActConfig
from nooa.tools.shell_tools import ShellTools
from nooa.tools.todo import TodoManager


class TaskResult(BaseModel):
    """Structured result the agent must return when finishing a task."""

    solution_description: str = Field(description="What you did and why.")
    command_to_verify: str = Field(description="Shell command to confirm correctness.")


class BenchAgent(Agent):
    """Generic benchmark agent for code and system tasks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Both tools own mutable state and must be local to this benchmark run.
        self.shell = ShellTools(cwd=".")
        self.todo = TodoManager()

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=30)))
    async def solve(self, task_input: dict) -> TaskResult:
        """Solve the task and propose a command that verifies the completed work."""
        ...

    @hidden
    async def run(self, task_input: dict) -> dict:
        """Entry point called by Harbor runner."""
        problem = task_input.get("user_message") or task_input.get("problem_statement", "")
        result = await self.solve({"problem": problem})
        verification = await self.shell.run(result.command_to_verify)
        if not verification.success:
            raise RuntimeError(
                "Verification failed "
                f"(exit {verification.returncode}): {verification.stderr.strip()}"
            )
        evidence = verification.stdout.strip() or (
            f"{result.command_to_verify!r} exited with code {verification.returncode}"
        )
        return {
            "solution": result.solution_description,
            "evidence": evidence,
        }
