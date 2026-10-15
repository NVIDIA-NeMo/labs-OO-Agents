# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed ACP entry points talking only to a scripted loopback model server."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from acp import connect_to_agent

_MODEL = "nooa-native-test"
_ALIAS = "native-acp-loopback"
_TEST_KEY = "local-test-no-remote-access"
type ScriptedResponse = str | dict[str, Any] | tuple[int, dict[str, Any]]


def code_response(code: str, *, response_id: str = "native") -> dict[str, Any]:
    """One Chat Completions response asking the real CodeAct loop to execute code."""
    return {
        "id": f"chatcmpl-{response_id}",
        "object": "chat.completion",
        "created": 0,
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{response_id}",
                            "type": "function",
                            "function": {
                                "name": "execute_python",
                                "arguments": json.dumps({"code": code}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


class ScriptedModelServer:
    """Consume code strings or OpenAI wire responses in request order.

    ``requests`` records the actual model-client requests. Unexpected routes,
    exhausted responses, or attempted non-loopback HTTP requests are errors.
    An explicit ``(status_code, response_body)`` scripts a planned HTTP error.
    The server also serves as a rejecting proxy for the child process, keeping
    optional discovery/telemetry from reaching the network.
    """

    def __init__(self, responses: Sequence[ScriptedResponse]) -> None:
        self._responses = deque(responses)
        self._lock = Lock()
        self.requests: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.protocol_errors: list[str] = []
        self.stderr = ""
        self.forced_shutdown = False
        self._response_prefix = uuid4().hex
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def _reply(self, status: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)

            def _reject(self, message: str) -> None:
                with owner._lock:
                    owner.errors.append(message)
                self._reply(400, {"error": {"message": message, "type": "invalid_request_error"}})

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/v1/chat/completions":
                    self._reject(f"Unexpected model route: POST {self.path}")
                    return
                if self.headers.get("Authorization") != f"Bearer {_TEST_KEY}":
                    self._reject("The model client did not use the isolated test credential")
                    return
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                with owner._lock:
                    owner.requests.append(request)
                    index = len(owner.requests)
                    response = owner._responses.popleft() if owner._responses else None
                if response is None:
                    self._reject(f"No scripted model response for request {index}")
                    return
                if isinstance(response, str):
                    response = code_response(
                        response, response_id=f"{owner._response_prefix}-{index}"
                    )
                status, payload = response if isinstance(response, tuple) else (200, response)
                self._reply(status, payload)

            def do_GET(self) -> None:  # noqa: N802
                self._reject(f"Unexpected outbound request: GET {self.path}")

            def do_CONNECT(self) -> None:  # noqa: N802
                self._reject(f"Blocked non-loopback connection: {self.path}")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05})

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._responses)

    def __enter__(self) -> ScriptedModelServer:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _ProtocolReader(asyncio.StreamReader):
    """Observe every stdout line without hiding invalid bytes from the ACP client."""

    def __init__(self, reader: asyncio.StreamReader, model: ScriptedModelServer) -> None:
        super().__init__()
        self._reader = reader
        self._model = model
        self._chunks: list[bytes] = []

    def _record(self, line: bytes) -> None:
        if line:
            try:
                message = json.loads(line)
            except (ValueError, UnicodeError):
                self._model.protocol_errors.append(repr(line))
            else:
                if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                    self._model.protocol_errors.append(repr(line))

    async def readuntil(self, separator: bytes = b"\n") -> bytes:
        try:
            line = await self._reader.readuntil(separator)
        except asyncio.IncompleteReadError as error:
            self._record(b"".join(self._chunks) + error.partial)
            self._chunks.clear()
            raise
        self._record(b"".join(self._chunks) + line)
        self._chunks.clear()
        return line

    async def readexactly(self, n: int) -> bytes:
        chunk = await self._reader.readexactly(n)
        self._chunks.append(chunk)
        return chunk


def _child_environment(
    workspace: Path, user_dir: Path, url: str, shell_executable: str | Path | None
):
    """Build an isolated child environment; never mutate the test runner's PATH."""
    user_dir.mkdir(parents=True, exist_ok=True)
    isolated_home = user_dir / "home"
    isolated_home.mkdir(exist_ok=True)
    config = user_dir / "llm_config.yaml"
    config.write_text(
        json.dumps(
            {
                "models": {
                    _ALIAS: {
                        "model_name": f"openai/{_MODEL}",
                        "client_type": "completion",
                        "api_base": f"{url}/v1",
                        "api_key_env": "NOOA_NATIVE_TEST_KEY",
                        "max_tokens": 1024,
                        "context_window": 128_000,
                        "retry_config": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    for name in ("settings.yaml", "secrets.yaml"):
        (user_dir / name).write_text("{}\n", encoding="utf-8")

    root = Path(__file__).resolve().parents[3]
    sources = [root / "src"] + [
        root / "packages" / name / "src"
        for name in ("nooa-cli", "nooa-acp", "nooa-memory", "nooa-bench")
    ]
    environment = {
        name: os.environ[name]
        for name in ("SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "TEMP", "TMP", "PATHEXT")
        if name in os.environ
    }
    # Each test selects an installed native shell using only the child's PATH.
    # In particular, the 5.1 case cannot accidentally discover pwsh elsewhere.
    selected = shell_executable or shutil.which("pwsh") or shutil.which("powershell")
    if selected is None:
        raise AssertionError("Native ACP tests require PowerShell")
    shell_path = Path(selected).resolve()
    system_root = Path(os.environ["SYSTEMROOT"])
    path_entries = [
        Path(sys.executable).parent,
        shell_path.parent,
        system_root / "System32",
        system_root,
        system_root / "System32" / "Wbem",
    ]
    environment.update(
        {
            "PATH": os.pathsep.join(dict.fromkeys(str(path) for path in path_entries)),
            "HOME": str(isolated_home),
            "USERPROFILE": str(isolated_home),
            "APPDATA": str(user_dir / "appdata"),
            "LOCALAPPDATA": str(user_dir / "localappdata"),
            "PYTHONPATH": os.pathsep.join(str(path) for path in sources),
            "PYTHONUTF8": "0",
            "NEMO_OO_USER_DIR": str(user_dir),
            "NEMO_OO_PROJECT_DIR": str(workspace / ".nooa"),
            "NEMO_OO_LLM_CONFIG": str(config),
            "NEMO_OO_SETTINGS": str(user_dir / "settings.yaml"),
            "NEMO_OO_SECRETS": str(user_dir / "secrets.yaml"),
            "NOOA_SESSIONS_DIR": str(workspace / ".nooa" / "sessions"),
            "NOOA_NATIVE_TEST_KEY": _TEST_KEY,
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "DO_NOT_TRACK": "1",
            "HTTP_PROXY": url,
            "HTTPS_PROXY": url,
            "ALL_PROXY": url,
            "NO_PROXY": "127.0.0.1,localhost",
        }
    )
    return environment


@asynccontextmanager
async def launch_native_acp(
    client: Any,
    workspace: Path,
    responses: Sequence[ScriptedResponse],
    *,
    entrypoint: str = "nooa-acp",
    shell_executable: str | Path | None = None,
    user_dir: Path | None = None,
) -> AsyncIterator[tuple[Any, asyncio.subprocess.Process, ScriptedModelServer]]:
    """Launch an installed CLI and yield ``(connection, process, model_server)``.

    ``entrypoint`` is ``nooa-acp`` or ``nooa acp`` (``nooa`` is an alias).
    Reuse ``workspace`` and optionally ``user_dir`` when testing durable reloads.
    Cleanup sends stdin EOF and requires a graceful zero exit. Forced teardown
    is only failure cleanup, is exposed by ``forced_shutdown``, and fails the test.
    """
    if sys.platform != "win32":
        raise RuntimeError("This harness exercises installed native Windows ACP entry points")
    if entrypoint not in {"nooa-acp", "nooa", "nooa acp"}:
        raise ValueError(f"Unknown ACP entry point: {entrypoint}")
    command_name = "nooa-acp" if entrypoint == "nooa-acp" else "nooa"
    executable = shutil.which(command_name)
    if executable is None:
        raise AssertionError(f"The {command_name} console script is not installed")
    args = [] if command_name == "nooa-acp" else ["acp"]
    args.extend(["--model", _ALIAS, "--client-type", "completion"])
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    user_dir = (user_dir or workspace.parent / f"acp-user-{workspace.name}").resolve()

    with ScriptedModelServer(responses) as model:
        environment = _child_environment(workspace, user_dir, model.url, shell_executable)
        process = await asyncio.create_subprocess_exec(
            executable,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
            cwd=workspace,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        assert (
            process.stdin is not None and process.stdout is not None and process.stderr is not None
        )
        stderr_task = asyncio.create_task(process.stderr.read())
        connection = None
        active_error: BaseException | None = None
        try:
            connection = connect_to_agent(
                client,
                process.stdin,
                _ProtocolReader(process.stdout, model),
                use_unstable_protocol=True,
            )
            yield connection, process, model
        except BaseException as error:
            active_error = error
            raise
        finally:
            process.stdin.close()
            with suppress(Exception):
                await asyncio.wait_for(process.stdin.wait_closed(), timeout=5)
            try:
                await asyncio.wait_for(process.wait(), timeout=15)
            except TimeoutError:
                model.forced_shutdown = True
                with suppress(ProcessLookupError):
                    process.kill()
                await asyncio.wait_for(process.wait(), timeout=5)
            finally:
                if connection is not None:
                    with suppress(Exception):
                        await asyncio.wait_for(connection.close(), timeout=5)
                try:
                    stderr = await asyncio.wait_for(stderr_task, timeout=5)
                    model.stderr = stderr.decode("utf-8", errors="replace")
                except TimeoutError:
                    model.stderr = "ACP stderr did not reach EOF after shutdown"
                    model.forced_shutdown = True

            problems = list(model.errors)
            if model.protocol_errors:
                problems.append(
                    "Invalid JSON-RPC on ACP stdout:\n" + "\n".join(model.protocol_errors)
                )
            if model.forced_shutdown:
                problems.append("ACP did not shut down gracefully after stdin EOF")
            if process.returncode != 0:
                problems.append(f"ACP process exited with {process.returncode}")
            diagnostic = "\n".join([*problems, model.stderr]).strip()
            if active_error is not None:
                if model.requests:
                    failed_cells = [
                        str(message.get("content", ""))
                        for message in model.requests[-1].get("messages", [])
                        if message.get("role") != "system"
                        and "Traceback (most recent call last)" in str(message.get("content", ""))
                    ]
                    if failed_cells:
                        active_error.add_note("CodeAct output:\n" + "\n".join(failed_cells[-3:]))
                if diagnostic:
                    active_error.add_note(diagnostic)
            elif problems:
                raise AssertionError(diagnostic)
