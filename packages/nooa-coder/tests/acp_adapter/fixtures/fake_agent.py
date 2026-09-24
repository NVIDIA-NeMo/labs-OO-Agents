# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic nooa-coder ACP subprocess for the protocol tests.

Runs the real server (``nooa_coder.acp.cli.run``) and the real coding
agent with a fake model. Flags pick the model's script: ``--blocking``
(a cell that never ends), ``--shell`` (a cell blocked in a shell command),
``--question`` (asks which branch, then finishes), ``--noisy`` (prints to
stdout while building each session's model, as tracing does); default: one message
and ``Done``. ``--tee PATH`` turns on the in-process tee.
"""

import json
import sys
import uuid
from pathlib import Path

# The ACP subprocess launcher drops PYTHONPATH. Pin this checkout's sources
# so the wire tests cannot run an editable install from another worktree.
_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "packages" / "nooa-coder" / "src")]

from nooa_coder.acp.cli import run  # noqa: E402

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall  # noqa: E402


def _cell(code: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name="execute_python",
                arguments=json.dumps({"code": code}),
            )
        ],
        finish_reason="tool_calls",
    )


def llm_factory(alias, workspace) -> FakeLLMClient:
    if "--noisy" in sys.argv:
        # What nooa.tracing does when it finds an endpoint: print to stdout.
        print("OTel tracing enabled: noise on stdout", flush=True)
    if "--shell" in sys.argv:
        # Blocks inside a real shell command, so cancellation exercises
        # ActivityShellTools.run rather than a bare asyncio wait.
        script = [_cell("await self.shell.run('sleep 30', timeout=30)")]
    elif "--blocking" in sys.argv:
        script = [_cell("await asyncio.Event().wait()")]
    elif "--question" in sys.argv:
        script = [
            _cell("return_result(NeedInput(question='Which branch?', options=['main', 'dev']))"),
            _cell("self.message('Using the answer.')\nreturn_result(Done(explanation='answered'))"),
        ]
    else:
        script = [
            _cell(
                "self.message('NOOA ACP smoke test passed.')\n"
                "return_result(Done(explanation='smoke test complete'))"
            )
        ]
    # A few spare turns, so tests that prompt again (or a re-queued item) finish.
    script += [_cell("return_result(Done(explanation='again'))") for _ in range(5)]
    return FakeLLMClient(script)


tee = sys.argv[sys.argv.index("--tee") + 1] if "--tee" in sys.argv else None
run(llm_factory=llm_factory, model="fake", tee=Path(tee) if tee else None)
