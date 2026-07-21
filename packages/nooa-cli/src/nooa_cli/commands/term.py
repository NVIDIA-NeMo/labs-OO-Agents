# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Web terminal command for nooa CLI.

Launches an xterm.js browser terminal connected to a real PTY running the
NVIDIA OO Agents TUI.  A rich content side-channel lets the agent push Plotly
plots, HTML, and images to a collapsible browser panel while the terminal
session runs normally.

Security: the server exposes a full interactive shell, so it binds to
127.0.0.1 by default and requires a per-session token.  At startup the
command prints the full URL (including ``?token=<...>``) — open that URL in
the browser.  To reach the terminal from outside the machine/container
(e.g. a sandbox whose loopback the host browser cannot reach), bind all
interfaces and publish the port, then open the printed token URL:

    nooa term --host 0.0.0.0 --port 8000
    # publish port 8000 to the host, then open http://<host>:8000/?token=<...>

Requests without a valid token are rejected (HTTP 403 / WebSocket close
4403).  ``--no-auth`` disables the token check entirely (not recommended).
"""

import ipaddress
import secrets
import sys

import click

_RESUME_LAST = "__last__"


@click.command()
@click.option("--model", "-m", help="LLM model to use (overrides config default)")
@click.option(
    "--agent",
    "agent_spec",
    default=None,
    metavar="MODULE:CLASS",
    help=(
        "Custom agent class instead of TUIAgent. "
        "Format: 'module.path:ClassName' or './file.py:ClassName'."
    ),
)
@click.option(
    "--working-dir",
    "-w",
    "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    default=".",
    help="Working directory for bash commands",
)
@click.option(
    "--mcp-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=str),
    default=None,
    help="MCP config file (default: .mcp.json in cwd)",
)
@click.option(
    "--skills-dir",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    help="Skills directory (can be specified multiple times)",
)
@click.option("--context-limit", type=int, help="Context limit for summarization")
@click.option("--no-trace", is_flag=True, help="Disable tracing")
@click.option("--vi", is_flag=True, help="Enable vi keybindings in the input prompt")
@click.option("--python", is_flag=True, help="Show agent Python code execution panels")
@click.option(
    "--continue",
    "-c",
    "continue_session",
    is_flag=False,
    flag_value=_RESUME_LAST,
    default=None,
    help="Resume a session: -c (last session) or -c <short-hash>",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help=(
        "Host to bind the web server to. Use 0.0.0.0 to reach the terminal "
        "from outside the machine/container (access still requires the "
        "session token printed at startup)."
    ),
)
@click.option("--port", "-p", default=8000, show_default=True, type=int, help="Port to listen on")
@click.option(
    "--no-auth",
    is_flag=True,
    help=(
        "Disable session-token authentication (NOT recommended). By default "
        "a random token is required — even on loopback binds — and the full "
        "URL including the token is printed at startup."
    ),
)
def command(
    model: str | None,
    agent_spec: str | None,
    working_dir: str,
    mcp_file: str | None,
    skills_dir: tuple[str, ...],
    context_limit: int | None,
    no_trace: bool,
    vi: bool,
    python: bool,
    continue_session: str | None,
    host: str,
    port: int,
    no_auth: bool,
):
    """Launch the NVIDIA OO Agents web terminal (xterm.js + rich side-channel).

    Opens an xterm.js terminal in your browser connected to a real PTY running
    the NVIDIA OO Agents TUI.  The agent can push Plotly plots, HTML, and images
    to a collapsible side panel using the WebPublisher tool
    (``self.web.plot(fig)``).

    The terminal is a full interactive shell, so access requires the
    per-session token embedded in the URL printed at startup — open that URL.
    The server binds to 127.0.0.1 by default; from a container or sandbox,
    use ``--host 0.0.0.0``, publish the port to the host, and open the
    printed token URL from the host browser.

    Examples:
        nooa term
        nooa term --port 8080
        nooa term --model gpt-4o
        nooa term --agent ./my_agent.py:MyAgent
        nooa term --host 0.0.0.0 --port 8000   # container/sandbox use
    """
    try:
        import uvicorn  # noqa: F401
        from uvicorn import Config, Server
    except ImportError:
        click.echo(
            'Web terminal requires uvicorn. Install with: uv add "nemo-labs-oo-agents-cli[web]"',
            err=True,
        )
        sys.exit(1)

    try:
        import ptyprocess  # noqa: F401
    except ImportError:
        click.echo(
            'Web terminal requires ptyprocess. Install with: uv add "nemo-labs-oo-agents-cli[web]"',
            err=True,
        )
        sys.exit(1)

    # Build the TUI argv to spawn inside the PTY
    tui_argv = _build_tui_argv(
        model=model,
        agent_spec=agent_spec,
        working_dir=working_dir,
        mcp_file=mcp_file,
        skills_dir=skills_dir,
        context_limit=context_limit,
        no_trace=no_trace,
        vi=vi,
        python=python,
        continue_session=continue_session,
    )

    # Per-session token protecting the shell (None disables auth entirely).
    auth_token = None if no_auth else secrets.token_urlsafe(16)

    # Rich content endpoint URL that the agent will POST to
    rich_url = f"http://127.0.0.1:{port}/rich"
    if auth_token is not None:
        rich_url += f"?token={auth_token}"
    env_extra = {"NEMO_OO_RICH_URL": rich_url}

    from nooa_cli.web.pty_server import create_pty_app

    app, kill_all_procs = create_pty_app(
        tui_argv=tui_argv, env_extra=env_extra, auth_token=auth_token
    )

    url = f"http://{host}:{port}"
    if auth_token is not None:
        url += f"/?token={auth_token}"
    click.echo(f"Starting NVIDIA OO Agents web terminal at {url}")
    click.echo(f"PTY command: {' '.join(tui_argv)}")
    if not _is_loopback(host):
        if auth_token is not None:
            warning = (
                f"Warning: --host {host} exposes the terminal to the network; "
                "it is protected only by the session token in the URL above."
            )
        else:
            warning = (
                f"Warning: --host {host} exposes the terminal to the network; "
                "authentication is disabled."
            )
        click.secho(warning, fg="yellow", err=True)
    if no_auth:
        click.secho(
            "WARNING: --no-auth disables authentication — anyone who can reach "
            f"http://{host}:{port} gets an interactive shell as your user.",
            fg="red",
            err=True,
        )

    import asyncio
    import logging
    import signal
    from contextlib import nullcontext

    _SHUTDOWN_TIMEOUT = 5  # seconds shown in countdown

    config = Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=_SHUTDOWN_TIMEOUT + 1,
    )
    server = Server(config)

    # We manage our own SIGINT/SIGTERM so we can show a countdown.
    server.capture_signals = nullcontext  # type: ignore[method-assign]

    # Suppress the "Cancel N running task(s), timeout graceful shutdown exceeded"
    # error that uvicorn logs when we force-close the long-lived PTY WebSocket.
    class _SuppressShutdownNoise(logging.Filter):
        _SUPPRESSED = ("timeout graceful shutdown", "CancelledError")

        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            exc_str = str(record.exc_info or "")
            return not any(token in msg or token in exc_str for token in self._SUPPRESSED)

    _noise_filter = _SuppressShutdownNoise()
    for _ln in ("uvicorn.error", "uvicorn", "uvicorn.access"):
        _lg = logging.getLogger(_ln)
        _lg.addFilter(_noise_filter)
        for _h in _lg.handlers:
            _h.addFilter(_noise_filter)
    # Also apply to root logger handlers in case uvicorn propagates
    for _h in logging.getLogger().handlers:
        _h.addFilter(_noise_filter)

    async def _serve() -> None:
        loop = asyncio.get_running_loop()
        _countdown_started = False
        _shutdown_task: asyncio.Task | None = None

        async def _countdown() -> None:
            for remaining in range(_SHUTDOWN_TIMEOUT, 0, -1):
                click.echo(f"\rShutting down... {remaining}s ", err=True, nl=False)
                await asyncio.sleep(1)
            click.echo("\rShutting down...           ", err=True, nl=False)
            server.force_exit = True

        def _on_signal() -> None:
            nonlocal _countdown_started, _shutdown_task
            if _countdown_started:
                return
            _countdown_started = True
            server.should_exit = True
            # Kill PTY processes immediately so WebSockets drain before uvicorn's
            # graceful-shutdown timer expires — this avoids the CancelledError that
            # occurs when uvicorn force-closes long-lived connections.
            kill_all_procs()
            click.echo("\nShutting down... ", err=True, nl=False)
            _shutdown_task = asyncio.ensure_future(_countdown())

        try:
            loop.add_signal_handler(signal.SIGINT, _on_signal)
            loop.add_signal_handler(signal.SIGTERM, _on_signal)
        except NotImplementedError:
            pass  # Windows

        await server.serve()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass
    click.echo("", err=True)  # newline after countdown
    sys.exit(0)


def _is_loopback(host: str) -> bool:
    """Return True if ``host`` refers to a loopback interface."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _build_tui_argv(
    *,
    model: str | None,
    agent_spec: str | None,
    working_dir: str,
    mcp_file: str | None,
    skills_dir: tuple[str, ...],
    context_limit: int | None,
    no_trace: bool,
    vi: bool,
    python: bool,
    continue_session: str | None,
) -> list[str]:
    """Build the argv list for spawning ``nooa tui`` in the PTY."""
    import shutil

    # Find the nooa executable in the current environment
    nooa_exe = shutil.which("nooa")
    if nooa_exe is None:
        # Fallback: run as a module using the same Python interpreter
        nooa_argv = [sys.executable, "-m", "nooa_cli"]
    else:
        nooa_argv = [nooa_exe]

    argv = nooa_argv + ["tui"]

    if model:
        argv += ["--model", model]
    if agent_spec:
        argv += ["--agent", agent_spec]
    if working_dir and working_dir != ".":
        argv += ["--working-dir", working_dir]
    if mcp_file:
        argv += ["--mcp-file", mcp_file]
    for sd in skills_dir:
        argv += ["--skills-dir", sd]
    if context_limit is not None:
        argv += ["--context-limit", str(context_limit)]
    if no_trace:
        argv.append("--no-trace")
    if vi:
        argv.append("--vi")
    if python:
        argv.append("--python")
    if continue_session:
        if continue_session == _RESUME_LAST:
            argv.append("--continue")
        else:
            argv += ["--continue", continue_session]

    return argv
