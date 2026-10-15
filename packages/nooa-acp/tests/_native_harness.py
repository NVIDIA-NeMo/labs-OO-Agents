# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed ACP entry points talking only to a scripted loopback model server."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from acp import connect_to_agent

# Load only the standalone support file; never add checkout sources to sys.path.
_support_path = Path(__file__).resolve().parents[3] / "scripts" / "_acp_smoke_support.py"
_support_spec = importlib.util.spec_from_file_location("acp_native_support", _support_path)
assert _support_spec is not None and _support_spec.loader is not None
_support = importlib.util.module_from_spec(_support_spec)
_support_spec.loader.exec_module(_support)
ScriptedModelServer = _support.ScriptedModelServer
ScriptedResponse = _support.ScriptedResponse
code_response = _support.code_response
_ALIAS = _support._ALIAS


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
    """Source tests opt in to checkout imports; standalone wheel smoke never does."""
    environment = _support.child_environment(workspace, user_dir, url, shell_executable)
    root = Path(__file__).resolve().parents[3]
    sources = [root / "src"] + [
        root / "packages" / name / "src"
        for name in ("nooa-cli", "nooa-acp", "nooa-memory", "nooa-bench")
    ]
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in sources)
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
