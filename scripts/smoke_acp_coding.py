# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed ACP entry point with a configured, real model.

Run with ``uv run --no-sync python scripts/smoke_acp_coding.py --model ALIAS``.
The agent receives only a small synthetic coding task in a temporary workspace.
Credentials stay in the child environment; reports contain no model configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import socket
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

from acp import PROTOCOL_VERSION, connect_to_agent, text_block
from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
from acp.schema import AgentMessageChunk, ToolCallProgress, ToolCallStart
from acp.transports import default_environment, spawn_stdio_transport

from nooa.llm_config import llm_config_chain
from nooa.secrets import load_secrets_into_env
from nooa.unifiedllm.registry import get_registry_config

_ORIGINAL = (
    "# Synthetic acceptance fixture.\n"
    "MARKER = '保留内容'\n\n"
    "def add(a: int, b: int) -> int:\n"
    "    return a - b\n"
)
_VERIFICATION = (
    "import unittest\nfrom calculator import add, MARKER\n\n"
    "class ArithmeticTests(unittest.TestCase):\n"
    "    def test_positive(self):\n        self.assertEqual(add(2, 3), 5)\n"
    "    def test_negative(self):\n        self.assertEqual(add(-2, 3), 1)\n"
    "    def test_zero(self):\n        self.assertEqual(add(0, 0), 0)\n"
    "    def test_preserved(self):\n        self.assertEqual(MARKER, '保留内容')\n\n"
    "if __name__ == '__main__':\n    unittest.main()\n"
)


class RecordingClient:
    def __init__(self) -> None:
        self.updates: list[object] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append(update)


def _redact(text: str, secrets: tuple[str, ...] = ()) -> str:
    values = set(secrets)
    values.update(
        value
        for name, value in os.environ.items()
        if any(part in name.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    )
    for value in sorted(filter(None, values), key=len, reverse=True):
        text = text.replace(value, "<REDACTED>")
    return text


def _safe_report(report: dict, secrets: tuple[str, ...] = ()) -> dict:
    return {
        key: _redact(value, secrets)[:2000] if isinstance(value, str) else value
        for key, value in report.items()
    }


class ProtocolMonitor:
    """Forward original stdout to the SDK while checking every line, including EOF."""

    def __init__(self, secrets: tuple[str, ...]) -> None:
        self.reader = asyncio.StreamReader(limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES)
        self.invalid = 0
        self.samples: list[str] = []
        self.secrets = secrets

    def _check(self, line: bytes) -> None:
        try:
            message = json.loads(line)
            valid = (
                isinstance(message, dict)
                and message.get("jsonrpc") == "2.0"
                and (
                    isinstance(message.get("method"), str)
                    or ("id" in message and (("result" in message) != ("error" in message)))
                )
            )
        except (ValueError, UnicodeError):
            valid = False
        if not valid:
            self.invalid += 1
            if len(self.samples) < 3:
                self.samples.append(
                    _redact(line.decode("utf-8", errors="replace"), self.secrets)[:300]
                )

    async def forward(self, source: asyncio.StreamReader) -> None:
        pending = b""
        try:
            while chunk := await source.read(65536):
                self.reader.feed_data(chunk)
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    self._check(line)
            if pending:
                self._check(pending)
        finally:
            self.reader.feed_eof()


def _environment(
    workspace: Path, state: Path, model: str
) -> tuple[dict[str, str], tuple[str, ...]]:
    registry_paths = llm_config_chain()
    config = get_registry_config(model)
    endpoint = urlsplit(config.get("api_base") or "")
    if endpoint.hostname in {"localhost", "127.0.0.1", "::1"}:
        port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
        try:
            with socket.create_connection((endpoint.hostname, port), timeout=3):
                pass
        except OSError:
            raise ConnectionError(
                f"Local model endpoint {endpoint.hostname}:{port} is unavailable"
            ) from None
    load_secrets_into_env()
    env = default_environment()
    credential_names = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "NVIDIA_API_KEY",
        "NVIDIA_INFERENCE_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    }
    if isinstance(config.get("api_key_env"), str):
        credential_names = {config["api_key_env"]}
    for name in credential_names:
        if value := os.environ.get(name):
            env[name] = value
    env.update(
        {
            "HOME": str(state),
            "USERPROFILE": str(state),
            "NEMO_OO_USER_DIR": str(state / "config"),
            "NEMO_OO_PROJECT_DIR": str(workspace / ".nooa"),
            "NOOA_SESSIONS_DIR": str(state / "sessions"),
            "NEMO_OO_LLM_CONFIG": ",".join(str(path) for path in registry_paths),
            "PYTHONUTF8": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env, tuple(env[name] for name in credential_names if name in env)


async def _verify(workspace: Path) -> tuple[int, str]:
    environment = default_environment()
    state = workspace.parent / "state"
    environment.update(
        {
            "HOME": str(state),
            "USERPROFILE": str(state),
            "NEMO_OO_USER_DIR": str(state / "config"),
            "NEMO_OO_PROJECT_DIR": str(workspace / ".nooa"),
        }
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-X",
        "utf8",
        "-B",
        "-c",
        "import sys, types, unittest\n"
        "from pathlib import Path\n"
        "source = Path(sys.argv[1]) / 'calculator.py'\n"
        "module = types.ModuleType('calculator')\n"
        "module.__file__ = str(source)\n"
        "sys.modules['calculator'] = module\n"
        "exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)\n"
        "sys.argv = ['verify.py']\n"
        f"exec(compile({_VERIFICATION!r}, 'verify.py', 'exec'))\n",
        str(workspace),
        cwd=workspace,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    assert process.returncode is not None
    return process.returncode, stdout.decode("utf-8", errors="replace")


def _check_workspace(workspace: Path) -> None:
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if relative.parts[0] == ".nooa":  # Framework-owned session databases and libraries.
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        if relative.as_posix() not in {"calculator.py", "verify.py"} or path.is_symlink():
            raise RuntimeError(f"The agent created an unexpected file: {relative}")
    if (workspace / "verify.py").read_bytes() != _VERIFICATION.encode("utf-8"):
        raise RuntimeError("The agent changed the acceptance test instead of only the source")
    if "# Synthetic acceptance fixture." not in (workspace / "calculator.py").read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("The agent removed the source comment it was asked to preserve")


async def _finish_transport(connection, writer, process, output_task, stderr_task, report):
    writer.close()
    with suppress(Exception):
        await asyncio.wait_for(writer.wait_closed(), timeout=5)
    try:
        await asyncio.wait_for(process.wait(), timeout=15)
    except TimeoutError:
        report["forced_shutdown"] = True
        with suppress(ProcessLookupError):
            process.kill()
        await asyncio.wait_for(process.wait(), timeout=5)
    finally:
        report["server_exit_code"] = process.returncode
        try:
            await asyncio.wait_for(output_task, timeout=5)
            stderr = await asyncio.wait_for(stderr_task, timeout=5)
            report["server_stderr"] = stderr.decode("utf-8", errors="replace")
        finally:
            try:
                if connection is not None:
                    await asyncio.wait_for(connection.close(), timeout=5)
            finally:
                for task in (output_task, stderr_task):
                    task.cancel()
                await asyncio.gather(output_task, stderr_task, return_exceptions=True)
    if report["forced_shutdown"] or process.returncode != 0:
        raise RuntimeError(f"ACP did not exit cleanly after EOF: {process.returncode}")


async def run_smoke(model: str, entrypoint: str, timeout: float) -> dict[str, object]:
    executable = shutil.which(entrypoint)
    if executable is None:
        raise RuntimeError(f"{entrypoint} is not installed; install the nooa-acp workspace package")
    arguments = ["acp"] if entrypoint == "nooa" else []
    arguments.extend(["--model", model])
    started = time.monotonic()
    report: dict[str, object] = {
        "passed": False,
        "model": model,
        "entrypoint": entrypoint,
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "forced_shutdown": False,
        "invalid_protocol_messages": 0,
    }
    secrets: tuple[str, ...] = ()
    turn_error: BaseException | None = None
    temporary = tempfile.TemporaryDirectory(prefix="nooa-acp-live-")
    try:
        root = Path(temporary.name).resolve()
        workspace = root / "中文 coding workspace"
        state = root / "state"
        workspace.mkdir()
        state.mkdir()
        source = workspace / "calculator.py"
        verification = workspace / "verify.py"
        source.write_text(_ORIGINAL, encoding="utf-8")
        verification.write_text(_VERIFICATION, encoding="utf-8", newline="\n")
        baseline_code, _ = await _verify(workspace)
        report["baseline_exit_code"] = baseline_code
        if baseline_code == 0:
            raise RuntimeError("The synthetic fixture must fail before the agent edits it")

        quote = "'" + sys.executable.replace("'", "''") + "'"
        test_command = (
            f"& {quote} -X utf8 -B verify.py"
            if sys.platform == "win32"
            else f"{shlex.quote(sys.executable)} -X utf8 -B verify.py"
        )
        prompt = (
            "修复当前仓库 calculator.py 中 add 的实现，使两个整数正确相加。"
            "先运行测试确认失败，再用 self.repo.symbols 或 refs 定位，"
            "用 self.shell.replace 修改 calculator.py，最后运行测试确认通过。"
            "只允许修改 calculator.py；保留 MARKER 和注释，不修改 verify.py。"
            "所有文件都是合成验收数据，不读取目录外文件、环境变量或访问网络。"
            f"测试命令是：{test_command}。完成后用中文说明修复和测试结果。"
        )
        client = RecordingClient()
        environment, secrets = _environment(workspace, state, model)
        monitor = ProtocolMonitor(secrets)
        async with spawn_stdio_transport(
            executable, *arguments, cwd=workspace, env=environment, shutdown_timeout=15
        ) as (reader, writer, process):
            assert process.stderr is not None
            output_task = asyncio.create_task(monitor.forward(reader))
            stderr_task = asyncio.create_task(process.stderr.read())
            connection = None
            try:
                connection = connect_to_agent(
                    client,  # type: ignore[arg-type]
                    writer,
                    monitor.reader,
                    use_unstable_protocol=True,
                )
                async with asyncio.timeout(timeout):
                    await connection.initialize(PROTOCOL_VERSION)
                    session = await connection.new_session(str(workspace))
                    response = await connection.prompt(session.session_id, [text_block(prompt)])
                    report["stop_reason"] = response.stop_reason
                    await connection.close_session(session.session_id)
            except BaseException as error:
                turn_error = error
                raise
            finally:
                try:
                    await _finish_transport(
                        connection, writer, process, output_task, stderr_task, report
                    )
                except Exception as cleanup_error:
                    if turn_error is None:
                        raise
                    turn_error.add_note(f"ACP shutdown: {cleanup_error}")
                finally:
                    report["invalid_protocol_messages"] = monitor.invalid
                    if monitor.samples:
                        report["invalid_protocol_samples"] = "\n".join(monitor.samples)

        if monitor.invalid:
            raise RuntimeError("ACP stdout contained invalid JSON-RPC messages")
        _check_workspace(workspace)
        verification_code, verification_output = await _verify(workspace)
        terminal_codes = [
            update.raw_output["exit_code"]
            for update in client.updates
            if isinstance(update, ToolCallProgress)
            and isinstance(update.raw_output, dict)
            and "exit_code" in update.raw_output
        ]
        edit_count = sum(
            isinstance(update, ToolCallStart) and update.kind == "edit" for update in client.updates
        )
        message_count = sum(isinstance(update, AgentMessageChunk) for update in client.updates)
        passed = (
            response.stop_reason == "end_turn"
            and source.read_text(encoding="utf-8") != _ORIGINAL
            and verification_code == 0
            and 0 in terminal_codes
            and edit_count > 0
            and message_count > 0
        )
        report.update(
            passed=passed,
            verification_exit_code=verification_code,
            verification_output=verification_output,
            terminal_exit_codes=terminal_codes,
            file_edit_events=edit_count,
            agent_message_events=message_count,
        )
    except Exception as error:
        if turn_error is not None and error is not turn_error:
            turn_error.add_note(f"ACP transport cleanup: {error}")
            error = turn_error
        report.update(error_type=type(error).__name__, error=str(error))
        if notes := getattr(error, "__notes__", None):
            report["error_notes"] = "\n".join(notes)
    finally:
        try:
            # Windows may release terminated children's directory handles shortly
            # after process exit. Only a successfully removed directory is clean.
            for attempt in range(5):
                try:
                    temporary.cleanup()
                    break
                except OSError as cleanup_error:
                    if (
                        getattr(cleanup_error, "winerror", None) not in {32, 33, 145}
                        or attempt == 4
                    ):
                        raise
                    await asyncio.sleep(0.05 * 2**attempt)
        except OSError as cleanup_error:
            report["passed"] = False
            report["cleanup_error"] = str(cleanup_error)
            report["retained_workspace"] = temporary.name
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return _safe_report(report, secrets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="An existing model alias or provider model")
    parser.add_argument("--entrypoint", choices=["nooa-acp", "nooa"], default="nooa-acp")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--report", type=Path, help="Optional JSON result file")
    args = parser.parse_args()
    try:
        report = asyncio.run(run_smoke(args.model, args.entrypoint, args.timeout))
    except Exception as error:
        report = {
            "passed": False,
            "model": args.model,
            "entrypoint": args.entrypoint,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    report = _safe_report(report)
    serialized = json.dumps(report, ensure_ascii=True, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
