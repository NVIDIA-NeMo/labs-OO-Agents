# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``nooa atom`` command, a plugin of the ``nooa`` command (``nooa_cli.commands``).

The server runs in one of four roles:

- Router (the default): the client speaks ACP to this process on standard
  input and output; each root session runs in its own worker process.
- ``--http``: clients connect over WebSocket at ``ws://HOST:PORT/acp``
  (``nooa_atom.acp.websocket``); each connection gets its own router and
  workers, as on standard input and output.
- ``--single-process``: every session runs in this process.
- ``--worker-fd N --id-base B`` (hidden): a worker, started by the router on
  one end of a socket pair. Not for direct use.

The router starts its workers by re-running the command that started it
(``sys.orig_argv``) with the worker options added, so a worker is the same
installation (or test script) with the same options.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

TOKEN_ENV = "NOOA_ATOM_TOKEN"
"""The environment variable ``--http`` reads its token from (and removes, so workers and cells never see it)."""

DEFAULT_PORT = 8765
"""The port ``--http`` listens on by default."""

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
    from nooa_atom.session.loader import _is_file_spec

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
        "workspace's coding.agent_spec setting, else the Atom agent."
    ),
)
@click.option(
    "--sessions-dir",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="NOOA_SESSIONS_DIR",
    default=None,
    help=(
        "One shared directory for the sessions of all workspaces. Or set NOOA_SESSIONS_DIR. "
        "Default: <workspace>/.nooa/sessions."
    ),
)
@click.option(
    "--tee",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Append every ACP frame, both directions, to this JSON Lines file (mode 0600).",
)
@click.option(
    "--single-process",
    is_flag=True,
    help="Run every session in this process instead of one worker process per root session.",
)
@click.option(
    "--http",
    is_flag=True,
    help=(
        "Serve ACP over WebSocket at ws://HOST:PORT/acp instead of standard input/output. "
        f"Clients send the token in {TOKEN_ENV} as 'Authorization: Bearer <token>' or ?token=."
    ),
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Address --http listens on. Use an SSH tunnel or a TLS proxy to reach it from elsewhere.",
)
@click.option("--port", type=int, default=DEFAULT_PORT, show_default=True, help="Port for --http.")
@click.option(
    "--allowed-origin",
    "allowed_origins",
    multiple=True,
    help="With --http, also accept this browser Origin (exact, e.g. https://app.example). Repeatable.",
)
@click.option(
    "--no-auth",
    is_flag=True,
    help=f"With --http on a loopback address, accept connections without the {TOKEN_ENV} token.",
)
@click.option("--worker-fd", type=int, default=None, hidden=True)
@click.option("--id-base", type=int, default=None, hidden=True)
@click.pass_context
def command(
    ctx: click.Context,
    model: str,
    client_type: str | None,
    agent_spec: str | None,
    sessions_dir: Path | None,
    tee: Path | None,
    single_process: bool,
    http: bool,
    host: str,
    port: int,
    allowed_origins: tuple[str, ...],
    no_auth: bool,
    worker_fd: int | None,
    id_base: int | None,
) -> None:
    """Serve NOOA Atom over ACP on standard input/output, or over WebSocket.

    \b
    Roles:
      router (default)  The client talks to a router; each root session and
                        its subagents run in their own worker process.
      --http            Clients connect at ws://HOST:PORT/acp; each
                        connection gets its own router and workers.
      --single-process  Every session runs in this process.
      worker            Started by the router with --worker-fd and --id-base
                        on one end of a socket pair; not for direct use.
    """
    reserve_stdio_for_acp()
    from nooa.secrets import load_secrets_into_env

    if (worker_fd is None) != (id_base is None):
        raise click.UsageError("--worker-fd and --id-base go together.")
    if worker_fd is not None and single_process:
        raise click.UsageError("--single-process and --worker-fd cannot be used together.")
    # A worker started by an --http server runs the same command line, so
    # it sees --http too; it is a worker all the same.
    token = os.environ.pop(TOKEN_ENV, None) or None
    if http and worker_fd is None:
        if single_process:
            raise click.UsageError("--http runs a router per connection; drop --single-process.")
        if tee is not None:
            raise click.UsageError(
                "--tee records standard input/output; it does not work with --http."
            )
        from nooa_atom.acp.websocket import is_loopback_host

        if no_auth and not is_loopback_host(host):
            raise click.UsageError(
                f"--no-auth is only allowed on a loopback address, not on {host}."
            )
        if token is None and not no_auth:
            raise click.UsageError(
                f"--http needs a token: set {TOKEN_ENV} (for example to the output of "
                "'openssl rand -hex 32'), or pass --no-auth on a loopback address."
            )
    load_secrets_into_env()
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")

    def llm_factory(alias: str | None, workspace: Path) -> Any:
        # Called per session with the alias it asks for; None is the default.
        # The alias resolves against the session workspace's configuration.
        from nooa_atom.workspace import models

        name = alias or model
        overrides = (
            {"api_key": nvidia_api_key} if nvidia_api_key and name.startswith("nvidia_nim/") else {}
        )
        return models.workspace_llm_client(name, workspace, client_type=client_type, **overrides)

    # A test script can run this command with its own model factory.
    obj = ctx.obj if isinstance(ctx.obj, dict) else {}
    run(
        llm_factory=obj.get("llm_factory") or llm_factory,
        model=model,
        agent_spec=agent_spec,
        sessions_dir=sessions_dir,
        tee=tee,
        single_process=single_process,
        worker_fd=worker_fd,
        id_base=id_base,
        http=HttpOptions(
            host=host,
            port=port,
            token=None if no_auth else token,
            allowed_origins=allowed_origins,
        )
        if http
        else None,
    )


@dataclass(frozen=True)
class HttpOptions:
    """Where and how ``--http`` listens; ``token`` ``None`` accepts any client."""

    host: str
    port: int
    token: str | None
    allowed_origins: tuple[str, ...] = ()


def run(
    *,
    llm_factory: Callable[[str | None, Path], Any],
    model: str | None = None,
    agent_spec: str | None = None,
    sessions_dir: Path | None = None,
    tee: Path | None = None,
    agent_factory: Any = None,  # None: create_session_agent
    single_process: bool | None = None,
    worker_fd: int | None = None,
    id_base: int | None = None,
    http: HttpOptions | None = None,
) -> None:
    """Serve ACP until the client leaves, in the role the options select.

    Also the test fixture's entry point: a script calling
    ``run(llm_factory=...)`` is started as the server command, and the
    router re-runs that script for each worker with ``--worker-fd`` and
    ``--id-base`` added. So when ``single_process``, ``worker_fd`` and
    ``id_base`` are not passed, they are read from ``sys.argv``.
    """
    acp_stdin, acp_stdout = reserve_stdio_for_acp()
    if single_process is None and worker_fd is None and id_base is None:
        single_process, worker_fd, id_base = _role_from_argv(sys.argv[1:])
    if worker_fd is not None:
        if id_base is None:
            raise SystemExit("--worker-fd needs --id-base")
        _run_worker(
            fd=worker_fd,
            id_base=id_base,
            llm_factory=llm_factory,
            model=model,
            agent_spec=agent_spec,
            sessions_dir=sessions_dir,
            agent_factory=agent_factory,
        )
    elif single_process:
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
    elif http is not None:
        _run_http(http, sessions_dir=sessions_dir)
    else:
        _run_router(sessions_dir=sessions_dir, tee=tee, acp_stdin=acp_stdin, acp_stdout=acp_stdout)


def _role_from_argv(argv: list[str]) -> tuple[bool, int | None, int | None]:
    def value(flag: str) -> int | None:
        for index, arg in enumerate(argv):
            if arg == flag and index + 1 < len(argv):
                return int(argv[index + 1])
            if arg.startswith(flag + "="):
                return int(arg.split("=", 1)[1])
        return None

    return "--single-process" in argv, value("--worker-fd"), value("--id-base")


def llm_config_summary(workspace: Path) -> str:
    """One line naming the LLM configuration files of a workspace, for the log.

    Answers "where is it loading the model configuration from?" without a
    debugger: the layered chain (bundled, user, project, the workspace's
    ``.nooa/llm_config.yaml``, then the ``NEMO_OO_LLM_CONFIG`` paths,
    highest priority last) and whether the environment variable is set.
    """
    from nooa_atom.workspace.models import llm_config_files

    paths = [str(path) for path in llm_config_files(workspace)]
    listed = ", ".join(paths) if paths else "none found"
    env = os.environ.get("NEMO_OO_LLM_CONFIG")
    source = f"NEMO_OO_LLM_CONFIG={env}" if env else "NEMO_OO_LLM_CONFIG not set"
    return f"LLM configuration for {workspace} (lowest priority first): {listed}; {source}"


def _run_router(
    *, sessions_dir: Path | None, tee: Path | None, acp_stdin: int, acp_stdout: int
) -> None:
    import logging

    from nooa_atom.acp._mcp_trace import MCPHandoffTrace
    from nooa_atom.acp.router import Router, process_spawn
    from nooa_atom.acp.tee import FrameLog

    _configure_logging("nooa-atom router")
    # Both observe the client's side, which only the router sees whole.
    observers: list[Any] = []
    if (trace := MCPHandoffTrace.from_env()) is not None:
        observers.append(trace)
    frame_log = FrameLog(tee) if tee is not None else None
    if frame_log is not None:
        observers.append(frame_log)
    router = Router(
        spawn=process_spawn(list(sys.orig_argv)),
        sessions_dir=sessions_dir,
        observers=observers,
    )
    try:
        asyncio.run(router.serve_stdio(input_fd=acp_stdin, output_fd=acp_stdout))
    finally:
        if frame_log is not None:
            frame_log.close()
        logging.shutdown()


def _run_http(options: HttpOptions, *, sessions_dir: Path | None) -> None:
    import logging
    import signal

    from nooa_atom.acp.router import process_spawn
    from nooa_atom.acp.websocket import Gate, serve_websocket

    _configure_logging("nooa-atom http")
    if options.token is None:
        logging.getLogger(__name__).warning(
            "--no-auth: any program on this machine can connect to %s:%d",
            options.host,
            options.port,
        )

    async def main() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, stop.set)
        await serve_websocket(
            host=options.host,
            port=options.port,
            spawn=process_spawn(list(sys.orig_argv)),
            sessions_dir=sessions_dir,
            gate=Gate(token=options.token, allowed_origins=options.allowed_origins),
            stop=stop,
        )

    try:
        asyncio.run(main())
    finally:
        logging.shutdown()


def _run_worker(
    *,
    fd: int,
    id_base: int,
    llm_factory: Callable[[str | None, Path], Any],
    model: str | None,
    agent_spec: str | None,
    sessions_dir: Path | None,
    agent_factory: Any,
) -> None:
    # --tee and NOOA_ACP_MCP_TRACE are ignored here: the router records the
    # client's side. The server logs each workspace's model configuration
    # files when its first session starts: the workspace is not known yet.
    from nooa_atom.acp.server import AtomACPAgent
    from nooa_atom.acp.worker import run_worker
    from nooa_atom.agent.factory import create_session_agent
    from nooa_atom.session.registry import SessionRegistry
    from nooa_atom.session.store import SessionStore

    _configure_logging(f"nooa-atom worker {id_base >> 32}")

    def new_registry(store: SessionStore) -> SessionRegistry:
        return SessionRegistry(
            store, agent_factory=agent_factory or create_session_agent, llm_factory=llm_factory
        )

    def make_agent() -> AtomACPAgent:
        return AtomACPAgent(
            new_registry, sessions_dir=sessions_dir, agent_spec=agent_spec, model=model
        )

    run_worker(
        fd,
        id_base=id_base,
        make_agent=make_agent,
    )


def _configure_logging(name: str) -> None:
    """Log the router and worker to standard error, with the role in every line."""
    import logging

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(f"%(asctime)s {name} %(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.WARNING:
        root.setLevel(logging.WARNING)
    level = os.environ.get("NOOA_ATOM_LOG_LEVEL", "INFO").upper()
    logging.getLogger("nooa_atom.acp").setLevel(level)


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
    from nooa_atom.acp._mcp_trace import MCPHandoffTrace
    from nooa_atom.acp.server import serve
    from nooa_atom.acp.tee import FrameLog
    from nooa_atom.agent.factory import create_session_agent
    from nooa_atom.session.registry import SessionRegistry
    from nooa_atom.session.store import SessionStore

    def new_registry(store: SessionStore) -> SessionRegistry:
        return SessionRegistry(
            store, agent_factory=agent_factory or create_session_agent, llm_factory=llm_factory
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
            new_registry,
            sessions_dir=sessions_dir,
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
