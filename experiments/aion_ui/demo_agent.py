# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scripted NOOA ACP demo: real agent/tools, deterministic responses, no provider calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from typing import Any
from uuid import uuid4

from nooa_acp.server import serve
from pydantic import BaseModel

from nooa.unifiedllm import FakeLLMClient, LLMResponse, Tool, ToolCall

DEMO_FILE = "nooa_aion_demo.py"
MARKER = "# NOOA AionUi scripted spike artifact"


class DemoLLM(FakeLLMClient):
    """Return one complete, fresh CodeAct execution per prompt."""

    def __init__(self, *, blocking: bool = False) -> None:
        super().__init__()
        self._blocking = blocking

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del output_model, kwargs
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = tools
        if self._blocking and self.call_count == 1:
            code = (
                "self.message('Scripted cancellation demo: waiting until you stop this turn.')\n"
                "await asyncio.Event().wait()"
            )
        else:
            program = (
                f"{MARKER}\n"
                f"DEMO_TURN = {self.call_count}\n"
                "assert sum([1, 2, 3]) == 6\n"
                "print('NOOA_AION_ASSERTION_PASSED')\n"
            )
            command = f"{shlex.quote(sys.executable)} {shlex.quote(DEMO_FILE)}"
            code = (
                "from pathlib import Path\n"
                "self.message('Scripted NOOA demo — no live model call. '"
                "'I will write a small program and run its assertion.')\n"
                f"demo_path = self.cwd / {DEMO_FILE!r}\n"
                f"assert not demo_path.exists() or demo_path.read_text().startswith({MARKER!r}), "
                "'Refusing to overwrite a file not owned by this demo'\n"
                f"await self.shell.write_file({DEMO_FILE!r}, {program!r})\n"
                f"result = await self.shell.run({command!r}, timeout=10)\n"
                "assert result.returncode == 0 and not result.timed_out, repr(result)\n"
                "assert 'NOOA_AION_ASSERTION_PASSED' in result.stdout, repr(result)\n"
                "self.message('Verified: the demo program printed `NOOA_AION_ASSERTION_PASSED` '"
                "'and exited successfully. This used scripted responses and real NOOA tools.')\n"
                "return_result(RespondReason.DONE, explanation='Wrote and executed demo assertion')"
            )
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"demo-{uuid4()}",
                    name="execute_python",
                    arguments=json.dumps({"code": code}),
                )
            ],
            finish_reason="tool_calls",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocking", action="store_true", help="First prompt waits for cancellation."
    )
    args = parser.parse_args()
    asyncio.run(serve(lambda: DemoLLM(blocking=args.blocking)))


if __name__ == "__main__":
    main()
