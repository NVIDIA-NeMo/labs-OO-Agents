# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real nested NOOA method calls for the optional execution-tree demo."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from nooa_cli.coding.activity import ActivityShellTools

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

TREE_FILE = "nooa_tree_demo.py"
TREE_MARKER = "# NOOA AionUi execution-tree demo artifact"


class DemoVerifier(Agent):
    """Use real method instrumentation to demonstrate a failed check and recovery."""

    def check_total(self, values: list[int], expected: int) -> int:
        """Check the result; failures remain visible in the execution tree."""
        actual = sum(values)
        if actual != expected:
            raise ValueError(f"Expected {expected}, got {actual}")
        return actual

    def verify(self) -> str:
        """Demonstrate one explicitly handled validation failure, then a passing check."""
        try:
            self.check_total([2, 3, 5], expected=11)
        except ValueError as error:
            # This failure is deliberate and reported, rather than silently ignored.
            print(f"Expected demo failure: {error}; retrying with expected=10")
        else:
            raise AssertionError("The deliberate failing check unexpectedly passed")
        total = self.check_total([2, 3, 5], expected=10)
        return f"Verified total: {total}; deliberate failed check was recovered"


class DemoWorkflow(Agent):
    """Deterministic methods subclass Agent here to exercise its tracing hooks."""

    async def write_program(self, shell: ActivityShellTools, cwd: Path, turn: int) -> str:
        """Write a small program through the real NOOA file tool."""
        path = cwd / TREE_FILE
        if path.exists() and not path.read_text().startswith(TREE_MARKER):
            raise FileExistsError(f"Refusing to overwrite a file not owned by this demo: {path}")
        program = (
            f"{TREE_MARKER}\n"
            f"DEMO_TURN = {turn}\n"
            "assert sum([2, 3, 5]) == 10\n"
            "print('NOOA_EXECUTION_TREE_OK')\n"
        )
        await shell.write_file(TREE_FILE, program)
        return TREE_FILE

    async def run_program(self, shell: ActivityShellTools, filename: str) -> str:
        """Execute the artifact and verify its exit status and stdout."""
        result = await shell.run(
            f"{shlex.quote(sys.executable)} {shlex.quote(filename)}", timeout=10
        )
        if result.returncode != 0 or result.timed_out:
            raise AssertionError(f"Demo program failed: {result!r}")
        if result.stdout.strip() != "NOOA_EXECUTION_TREE_OK":
            raise AssertionError(f"Unexpected demo output: {result.stdout!r}")
        return result.stdout.strip()

    async def run(self, shell: ActivityShellTools, cwd: Path, turn: int) -> str:
        """Run a nested worker, write the program, and execute it."""
        reviewer = DemoVerifier(llm=FakeLLMClient())
        verification = reviewer.verify()
        filename = await self.write_program(shell, cwd, turn)
        output = await self.run_program(shell, filename)
        return f"{verification}. Program output: {output}"
