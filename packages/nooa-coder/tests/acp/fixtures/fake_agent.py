# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic ACP subprocess used by protocol smoke tests."""

import asyncio
import sys

from nooa_acp.server import serve

from nooa.unifiedllm import FakeLLMClient


def llm_factory() -> FakeLLMClient:
    if "--shell" in sys.argv:
        # Blocks inside a real shell command, so cancellation exercises
        # ActivityShellTools.run rather than a bare asyncio wait.
        return FakeLLMClient.with_tool_call(
            "execute_python",
            {"code": "await self.shell.run('sleep 30', timeout=30)"},
        )
    if "--blocking" in sys.argv:
        return FakeLLMClient.with_tool_call(
            "execute_python",
            {"code": "await asyncio.Event().wait()"},
        )
    return FakeLLMClient.with_tool_call(
        "execute_python",
        {
            "code": (
                "self.message('NOOA ACP smoke test passed.')\n"
                "return_result(RespondReason.DONE, explanation='smoke test complete')"
            )
        },
    )


asyncio.run(serve(llm_factory))
