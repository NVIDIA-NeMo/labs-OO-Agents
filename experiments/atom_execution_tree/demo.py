# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Atom's real ACP server with scripted calls, nested methods and real tools."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from nooa_atom.acp.cli import command
from nooa_atom.agent.activity import ActivityShellTools

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

MARKER = "# Atom execution-tree demo artifact\n"


class DemoVerifier(Agent, llm=FakeLLMClient()):
    def check(self, expected: int) -> int:
        """A failed call is visible even though its parent handles the error."""
        actual = sum([2, 3, 5])
        if actual != expected:
            raise ValueError(f"Expected {expected}, got {actual}")
        return actual

    def verify(self) -> int:
        """Recover from one deliberate failure, then pass."""
        try:
            self.check(11)
        except ValueError:
            return self.check(10)
        raise AssertionError("The deliberately failing check unexpectedly passed")


class DemoWorkflow(Agent, llm=FakeLLMClient()):
    async def run(self, shell: ActivityShellTools) -> str:
        """Validate, write a program and execute it through Atom's actual tools."""
        assert DemoVerifier().verify() == 10
        existing = await shell.run("test ! -e atom_tree_demo.py || head -1 atom_tree_demo.py")
        if existing.stdout and not existing.stdout.startswith(MARKER.rstrip()):
            raise FileExistsError("Refusing to overwrite a file not owned by the demo")
        await shell.write_file("atom_tree_demo.py", MARKER + "print('ATOM_TREE_OK')\n")
        result = await shell.run(f"{shlex.quote(sys.executable)} atom_tree_demo.py")
        assert result.returncode == 0 and result.stdout.strip() == "ATOM_TREE_OK"
        return "ATOM_TREE_OK: deliberate child failure recovered; program verified."


def model_factory(alias: str | None, workspace: Path) -> FakeLLMClient:
    """No credentials, model traffic or paid inference; the agent/tools are real."""
    code = (
        "from demo import DemoWorkflow\n"
        "report = await DemoWorkflow().run(self.shell)\n"
        "self.message(report)\n"
        "return_result(Done(explanation=report))"
    )
    response = LLMResponse(
        tool_calls=[
            ToolCall(id="demo-cell", name="python_cell", arguments=json.dumps({"code": code}))
        ],
        finish_reason="tool_calls",
    )
    return FakeLLMClient([response] * 20, strict_exhaustion=True)


if __name__ == "__main__":
    command.main(obj={"llm_factory": model_factory})
