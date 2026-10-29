# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python core smoke test, run on native Windows CI or under scripts/winsim.py.

Exits 0 and prints ``SMOKE OK`` when: ``import nooa`` works, a scripted
(FakeLLMClient) agent runs with PredictStrategy and with in-process CodeAct,
and every POSIX-only feature raises PosixOnlyFeatureError naming WSL 2.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path


def main() -> None:
    import nooa
    import nooa.context_blocks  # noqa: F401
    import nooa.decisions  # noqa: F401
    import nooa.media  # noqa: F401
    from nooa import Agent
    from nooa._platform import PosixOnlyFeatureError
    from nooa.decorators import strategy
    from nooa.strategies.predict import PredictStrategy
    from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

    print("platform:", sys.platform, "nooa", getattr(nooa, "__version__", "?"))

    # 1. PredictStrategy with a scripted LLM
    predict_llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None, content='{"value": "hello"}', tool_calls=[], finish_reason="stop"
            )
        ]
    )

    class Greeter(Agent, llm=predict_llm):
        @strategy(PredictStrategy())
        async def greet(self, name: str) -> str:
            """Greet someone."""
            ...

    out = asyncio.run(Greeter().greet("Ada"))
    assert out == "hello", out
    print("predict:", out)

    # 2. In-process CodeAct (default strategy) with a scripted LLM
    codeact_llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1", name="execute_python", arguments=json.dumps({"code": "x = 2 + 3"})
                    ),
                    ToolCall(id="c2", name="return_result", arguments=json.dumps({"result": 5})),
                ],
                finish_reason="tool_calls",
            )
        ]
    )

    class Adder(Agent, llm=codeact_llm):
        async def add(self) -> int:
            """Return 2 + 3."""
            ...

    total = asyncio.run(Adder().add())
    assert total == 5, total
    print("codeact:", total)

    # 3. POSIX-only features fail clearly, at the point of use
    if sys.platform == "win32":
        checks = {}

        def _sqlite() -> None:
            from nooa.storage import SQLiteStorageManager

            with tempfile.TemporaryDirectory() as tmp:
                SQLiteStorageManager(str(Path(tmp) / "s.db"))

        def _bash() -> None:
            from nooa.tools._bash_session import BashSession

            asyncio.run(BashSession().start())

        def _monitor() -> None:
            from nooa.runtime.producers import monitor

            async def first() -> None:
                async for _ in monitor("echo hi"):
                    break

            asyncio.run(first())

        def _sandbox() -> None:
            from nooa.runtime.sandbox.config import SandboxConfig
            from nooa.runtime.sandbox.executor import SandboxedExecutor

            SandboxedExecutor(Adder(), SandboxConfig(), cell_timeout=5)

        for name, fn in {
            "sqlite": _sqlite,
            "bash": _bash,
            "monitor": _monitor,
            "sandbox": _sandbox,
        }.items():
            try:
                fn()
            except PosixOnlyFeatureError as exc:
                assert "WSL 2" in str(exc), exc
                checks[name] = str(exc)
            else:
                raise AssertionError(f"{name} did not raise PosixOnlyFeatureError")
        for name, msg in checks.items():
            print(f"clear error [{name}]: {msg}")

        from nooa.storage import SQLiteStorageManager

        with SQLiteStorageManager(":memory:"):
            print("sqlite :memory: ok")

    print("SMOKE OK")


if __name__ == "__main__":
    main()
