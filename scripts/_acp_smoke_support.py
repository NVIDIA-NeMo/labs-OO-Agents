# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared, standard-library-only model responder for native ACP acceptance."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import deque
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

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

    The requests list records the actual model-client requests. Unexpected routes,
    exhausted responses, or attempted non-loopback HTTP requests are errors.
    An explicit (status_code, response_body) scripts a planned HTTP error.
    The server also serves as a rejecting proxy for the child process, keeping
    optional discovery/telemetry from reaching the network.
    An optional responder callback can select responses for interactive editor
    prompts; requests and callback failures are still recorded.
    """

    def __init__(
        self,
        responses: Sequence[ScriptedResponse] = (),
        *,
        responder: Callable[[dict[str, Any]], ScriptedResponse] | None = None,
    ) -> None:
        self._responses = deque(responses)
        self._responder = responder
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
                if owner._responder is not None:
                    try:
                        response = owner._responder(request)
                    except Exception as error:
                        self._reject(f"Scripted model responder failed: {error}")
                        return
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


def child_environment(
    workspace: Path,
    user_dir: Path,
    url: str,
    shell_executable: str | Path | None = None,
    *,
    tokenizer_cache: Path | None = None,
) -> dict[str, str]:
    """Isolate configuration, credentials, network proxies, shell, and import paths.

    This environment contains no source checkout paths. Source-based tests may
    add their own PYTHONPATH after this function returns; wheel checks must not.
    """
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
    # Windows runners prewarm this machine-level module cache. Keep its path
    # independently of the isolated home to avoid a cold scan in every ACP
    # process, which can exhaust PowerShell 5.1's startup deadline.
    environment = {
        name: os.environ[name]
        for name in (
            "SYSTEMROOT",
            "WINDIR",
            "SYSTEMDRIVE",
            "COMSPEC",
            "TEMP",
            "TMP",
            "PATHEXT",
            "PSModuleAnalysisCachePath",
        )
        if name in os.environ
    }
    # Selecting one native shell prevents a 5.1 run from discovering pwsh.
    selected = shell_executable or shutil.which("pwsh") or shutil.which("powershell")
    if selected is None:
        raise RuntimeError("Native ACP acceptance requires PowerShell")
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
            "PYTHONUTF8": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
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
    if tokenizer_cache is not None:
        # LiteLLM overwrites TIKTOKEN_CACHE_DIR unless its CUSTOM override is set.
        environment["TIKTOKEN_CACHE_DIR"] = str(tokenizer_cache.resolve())
        environment["CUSTOM_TIKTOKEN_CACHE_DIR"] = str(tokenizer_cache.resolve())
    return environment
