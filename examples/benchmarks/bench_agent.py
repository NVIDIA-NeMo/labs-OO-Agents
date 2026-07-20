# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal benchmark agent example for Harbor-compatible tasks."""

from pydantic import BaseModel, Field

from nooa import Agent, CodeActStrategy, strategy
from nooa.tools.shell_tools import ShellTools
from nooa.tools.todo import TodoManager


class TaskResult(BaseModel):
    """Structured result the agent must return when finishing a task."""

    solution_description: str = Field(description="What you did and why.")
    evidence: str = Field(description="Concrete evidence the task is done.")
    command_to_verify: str = Field(description="Shell command to confirm correctness.")


class BenchAgent(Agent):
    """Generic benchmark agent for code and system tasks."""

    shell: ShellTools
    todo: TodoManager

    @strategy(CodeActStrategy(max_turns=30))
    async def solve(self, task_input: dict) -> TaskResult:
        """Solve the given task. Problem: {problem}"""
        ...

    async def run(self, task_input: dict) -> dict:
        """Entry point called by Harbor runner."""
        problem = task_input.get("user_message") or task_input.get("problem_statement", "")
        result = await self.solve({"problem": problem})
        return {"solution": result.solution_description, "evidence": result.evidence}
