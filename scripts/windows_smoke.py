# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python core smoke test, run on native Windows CI or under scripts/winsim.py.

Native Windows is refused at import unless NOOA_SKIP_PLATFORM_CHECK=1; this
script opts in for itself. Exits 0 and prints ``SMOKE OK`` when: ``import nooa``
works, a scripted (FakeLLMClient) agent runs with PredictStrategy and with
in-process CodeAct, every POSIX-only feature raises PosixOnlyFeatureError naming
WSL 2, and (on real Windows) the refusal without the override is clean and the
real ``nooa connect`` saves with its output redirected.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SKIP = "NOOA_SKIP_PLATFORM_CHECK"
# CI sets NOOA_SMOKE_STRICT=1 so a missing nooa.exe or nooa-cli fails instead of skipping.
STRICT = os.environ.get("NOOA_SMOKE_STRICT") == "1"
CONNECT_ARGS = [
    "fake-model",
    "--provider",
    "custom",
    "--endpoint",
    "http://127.0.0.1:9/v1",
    "--api-style",
    "chat",
    "--api-key-env",
    "NOOA_SMOKE_FAKE_KEY",
    "--as",
    "fake",
    "--no-probe",
    "--no-catalogue",
    "--yes",
]


def _child_env(*, skip: str | None) -> dict[str, str]:
    """The environment of a fresh Windows user: no UTF-8 overrides, opt-in as given."""
    env = {k: v for k, v in os.environ.items() if k not in {SKIP, "PYTHONUTF8", "PYTHONIOENCODING"}}
    if skip is not None:
        env[SKIP] = skip
    return env


def _real_windows_checks() -> None:
    """Subprocess checks that need real Windows (redirected output, console scripts)."""
    import yaml

    refused = subprocess.run(
        [sys.executable, "-c", "import nooa"],
        env=_child_env(skip=None),
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 1 and SKIP in refused.stderr, refused.stderr
    assert refused.stderr.startswith("Error: nooa does not support native Windows"), refused.stderr
    assert "Traceback" not in refused.stderr, refused.stderr
    print("refused without override: import nooa exits 1 with one clean line")

    nooa_exe = Path(sys.executable).with_name("nooa.exe")
    if not nooa_exe.exists():
        if STRICT:
            raise AssertionError(f"nooa.exe not found next to {sys.executable}")
        print("cli checks: skipped (nooa.exe not next to python)")
        return
    cli = subprocess.run(
        [str(nooa_exe), "--help"], env=_child_env(skip=None), capture_output=True, text=True
    )
    assert cli.returncode == 1, (cli.returncode, cli.stdout, cli.stderr)
    assert cli.stderr.startswith("Error: nooa does not support native Windows"), cli.stderr
    assert "Traceback" not in cli.stderr, cli.stderr
    print("refused without override: nooa --help exits 1 with one clean line")

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import nooa_cli; nooa_cli._use_utf8_output(); print('\\u2713 \\u2192 \\u21bb')",
        ],
        env=_child_env(skip="1"),
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")
    assert "\u2713" in probe.stdout.decode("utf-8"), probe.stdout
    print("redirected output: check marks encode as UTF-8")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "llm_config.yaml"
        save = subprocess.run(
            [str(nooa_exe), "connect", *CONNECT_ARGS, "--output", str(out)],
            env=_child_env(skip="1"),
            capture_output=True,
        )
        assert save.returncode == 0, save.stdout.decode(errors="replace") + save.stderr.decode(
            errors="replace"
        )
        assert "fake" in yaml.safe_load(out.read_text())["models"], out.read_text()
    print("redirected output: real `nooa connect` saved model 'fake'")


def main() -> None:
    os.environ.setdefault(SKIP, "1")  # native Windows is refused at import otherwise
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

        import importlib.util

        try:
            from click.testing import CliRunner
            from nooa_cli.commands.connect import command as connect_command
        except ImportError:
            if STRICT:
                raise
            print("connect: skipped (nooa-cli not installed)")
        else:
            if importlib.util.find_spec("msvcrt") is None:
                print("connect: save skipped (simulated Windows has no msvcrt)")
            else:
                import yaml

                with tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp) / "llm_config.yaml"
                    res = CliRunner().invoke(connect_command, [*CONNECT_ARGS, "--output", str(out)])
                    assert res.exit_code == 0, (res.exit_code, res.output)
                    assert "fake" in yaml.safe_load(out.read_text())["models"], out.read_text()
                print("connect: saved model 'fake' (filelock on Windows)")

        from nooa.storage import SQLiteStorageManager

        with SQLiteStorageManager(":memory:"):
            print("sqlite :memory: ok")

        if importlib.util.find_spec("msvcrt") is None:
            print("real-Windows checks: skipped (simulation)")
        else:
            _real_windows_checks()

    print("SMOKE OK")


if __name__ == "__main__":
    main()
