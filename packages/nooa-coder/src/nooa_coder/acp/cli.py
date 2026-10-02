# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``nooa coder`` command, a plugin of the ``nooa`` command (``nooa_cli.commands``)."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

_acp_stdio: tuple[int, int] | None = None


def reserve_stdio_for_acp() -> tuple[int, int]:
    """Keep standard input and output for ACP frames only; return ``(input_fd, output_fd)``.

    The server's own stdio is its protocol channel, but libraries print
    (``nooa.tracing`` prints "OTel tracing enabled ..." when it finds an
    endpoint), and one stray line corrupts the stream; a subprocess a cell
    starts could likewise read the client's frames. The real descriptors
    are duplicated for the ACP transport; descriptor 1 (so ``print`` and
    anything else writing to it, C code included) is pointed at standard
    error and descriptor 0 at ``/dev/null``. Idempotent.
    """
    global _acp_stdio
    if _acp_stdio is None:
        sys.stdout.flush()
        output_fd = os.dup(1)
        os.dup2(2, 1)
        input_fd = os.dup(0)
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0)
        os.close(null)
        _acp_stdio = (input_fd, output_fd)
    return _acp_stdio


def _resolve_agent_spec(_ctx: click.Context, _param: click.Parameter, value: str | None) -> Any:
    """Make a relative ``file.py:Class`` spec absolute against the directory the command runs in.

    Sessions resolve relative file specs against their workspace, which is
    not where the person typed the path.
    """
    if not value:
        return value
    from nooa_coder.session.loader import _is_file_spec

    module, _, class_path = value.rpartition(":")
    if not module or not _is_file_spec(module):
        return value
    path = Path(module).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return f"{path}:{class_path}"


@click.command()
@click.option(
    "--model",
    envvar="NOOA_MODEL",
    required=True,
    help="Model alias or LiteLLM model name new sessions start with. Or set NOOA_MODEL.",
)
@click.option(
    "--client-type",
    type=click.Choice(("completion", "responses")),
    default=None,
    help="Override the configured NOOA LLM client type.",
)
@click.option(
    "--agent",
    "agent_spec",
    callback=_resolve_agent_spec,
    help=(
        "Agent class for new sessions (module:Class or file.py:Class). Default: the "
        "workspace's coding.agent_spec setting, else the coding agent."
    ),
)
@click.option("--legacy-agent", is_flag=True, help="Use the standard coding agent class.")
@click.option(
    "--sessions-dir",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="NOOA_SESSIONS_DIR",
    default=None,
    help="Where sessions are stored (default: the user directory's sessions folder).",
)
@click.option(
    "--tee",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Append every ACP frame, both directions, to this JSON Lines file (mode 0600).",
)
@click.option("--worker", default=None, hidden=True, help="Reserved for the worker process.")
def command(
    model: str,
    client_type: str | None,
    agent_spec: str | None,
    legacy_agent: bool,
    sessions_dir: Path | None,
    tee: Path | None,
    worker: str | None,
) -> None:
    """Serve the NOOA coding agent over ACP on standard input/output."""
    reserve_stdio_for_acp()
    from nooa.secrets import load_secrets_into_env

    if agent_spec and legacy_agent:
        raise click.UsageError("--agent and --legacy-agent cannot be used together.")
    if worker is not None:
        raise click.UsageError("--worker is not available yet.")
    if legacy_agent:
        from nooa_coder.coding.identity import CODING_AGENT

        agent_spec = CODING_AGENT
    load_secrets_into_env()
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")

    def llm_factory(alias: str | None, workspace: Path) -> Any:
        # Called per session with the alias it asks for; None is the default.
        # workspace is the seam for resolving aliases per workspace later.
        del workspace
        from nooa.unifiedllm import get_llm_client

        name = alias or model
        overrides = (
            {"api_key": nvidia_api_key} if nvidia_api_key and name.startswith("nvidia_nim/") else {}
        )
        return get_llm_client(name, client_type=client_type, **overrides)

    run(
        llm_factory=llm_factory,
        model=model,
        agent_spec=agent_spec,
        sessions_dir=sessions_dir,
        tee=tee,
    )


def run(
    *,
    llm_factory: Callable[[str | None, Path], Any],
    model: str | None = None,
    agent_spec: str | None = None,
    sessions_dir: Path | None = None,
    tee: Path | None = None,
    agent_factory: Any = None,
) -> None:
    """Serve ACP on stdio until the client leaves (the test fixture's entry point too)."""
    acp_stdin, acp_stdout = reserve_stdio_for_acp()
    asyncio.run(
        _serve(
            llm_factory=llm_factory,
            model=model,
            agent_spec=agent_spec,
            sessions_dir=sessions_dir,
            tee=tee,
            agent_factory=agent_factory,
            acp_stdin=acp_stdin,
            acp_stdout=acp_stdout,
        )
    )


async def _serve(
    *,
    llm_factory: Callable[[str | None, Path], Any],
    model: str | None,
    agent_spec: str | None,
    sessions_dir: Path | None,
    tee: Path | None,
    agent_factory: Any,
    acp_stdin: int,
    acp_stdout: int,
) -> None:
    from nooa_coder.acp._mcp_trace import MCPHandoffTrace
    from nooa_coder.acp.server import serve
    from nooa_coder.acp.tee import FrameLog
    from nooa_coder.session.registry import SessionRegistry
    from nooa_coder.session.store import SessionStore

    registry = SessionRegistry(
        SessionStore(sessions_dir), agent_factory=agent_factory, llm_factory=llm_factory
    )
    observers: list[Any] = []
    trace = MCPHandoffTrace.from_env()
    if trace is not None:
        observers.append(trace)
    frame_log = FrameLog(tee) if tee is not None else None
    if frame_log is not None:
        observers.append(frame_log)
    try:
        await serve(
            registry,
            agent_spec=agent_spec,
            model=model,
            observers=observers,
            input_fd=acp_stdin,
            output_fd=acp_stdout,
        )
    finally:
        try:
            if frame_log is not None:
                frame_log.close()
        finally:
            if trace is not None:
                trace.close()


if __name__ == "__main__":
    command()
