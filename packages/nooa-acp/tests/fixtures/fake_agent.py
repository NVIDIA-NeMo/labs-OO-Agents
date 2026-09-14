# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic ACP subprocess used by protocol smoke tests."""

import asyncio
import sys
from pathlib import Path

# The ACP subprocess launcher deliberately drops PYTHONPATH. Pin the source
# checkout so these wire tests cannot silently exercise an editable install
# from a different worktree that happens to share the Python environment.
_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(_ROOT / "src"),
    str(_ROOT / "packages" / "nooa-cli" / "src"),
    str(_ROOT / "packages" / "nooa-acp" / "src"),
    str(_ROOT / "packages" / "nooa-memory" / "src"),
]

from nooa_acp.server import serve  # noqa: E402

from nooa.unifiedllm import FakeLLMClient  # noqa: E402


def llm_factory() -> FakeLLMClient:
    if "--shell" in sys.argv:
        # Blocks inside a real shell command, so cancellation exercises
        # ActivityShellTools.run rather than a bare asyncio wait.
        return FakeLLMClient.with_tool_call(
            "python_cell",
            {"code": "await self.shell.run('sleep 30', timeout=30)"},
        )
    if "--blocking" in sys.argv:
        return FakeLLMClient.with_tool_call(
            "python_cell",
            {"code": "await asyncio.Event().wait()"},
        )
    return FakeLLMClient.with_tool_call(
        "python_cell",
        {
            "code": (
                "self.message('NOOA ACP smoke test passed.')\n"
                "return_result(RespondReason.DONE, explanation='smoke test complete')"
            )
        },
    )


asyncio.run(serve(llm_factory))
