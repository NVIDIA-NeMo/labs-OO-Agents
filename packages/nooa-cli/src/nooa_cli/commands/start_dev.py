# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start the development viewer.

Usage:
    nooa start-dev                  # Start viewer on :5001
    nooa start-dev --port 5002      # Custom port
"""

import logging

import click

NAME = "start-dev"

# Routes that are too noisy to log on every request.
_SUPPRESSED_PATHS = ("/v1/traces", "/api/trace", "/api/refresh")


class _AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in _SUPPRESSED_PATHS)


def _find_pid_on_port(port: int) -> str | None:
    """Return the PID of the process listening on *port*, or None."""
    import shutil
    import subprocess

    # Try lsof first (macOS + most Linux)
    if shutil.which("lsof"):
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f":{port}"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                # May return multiple PIDs; take the first.
                return out.splitlines()[0]
        except subprocess.CalledProcessError:
            pass

    # Fallback: ss (Linux)
    if shutil.which("ss"):
        try:
            out = subprocess.check_output(
                ["ss", "-tlnp", f"sport = :{port}"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            import re

            m = re.search(r"pid=(\d+)", out)
            if m:
                return m.group(1)
        except subprocess.CalledProcessError:
            pass

    return None


@click.command()
@click.option("--port", "-p", type=int, default=5001, help="Port number (default: 5001).")
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0).")
@click.option(
    "--db",
    "db_path_opt",
    type=click.Path(dir_okay=False),
    default=None,
    help="SQLite trace store path. Defaults to ~/.config/nooa/traces.db "
    "(or $NOOA_TRACE_DB if set). Pass an explicit path to run a second viewer "
    "side-by-side with the default one.",
)
def command(port: int, host: str, db_path_opt: str | None):
    """Start the unified trace + evaluation viewer."""
    import os
    from pathlib import Path

    from nooa.paths import get_user_dir

    # Resolve the DB path with --db winning, then $NOOA_TRACE_DB, then the
    # legacy $NEMO_OO_TRACE_DB, then the user-dir default. Set both env vars
    # unconditionally so the viewer module picks it up at import time.
    if db_path_opt:
        db_path = Path(db_path_opt).expanduser().resolve()
    elif "NOOA_TRACE_DB" in os.environ:
        db_path = Path(os.environ["NOOA_TRACE_DB"]).expanduser().resolve()
    elif "NEMO_OO_TRACE_DB" in os.environ:
        db_path = Path(os.environ["NEMO_OO_TRACE_DB"]).expanduser().resolve()
    else:
        db_path = get_user_dir("traces.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["NOOA_TRACE_DB"] = str(db_path)
    os.environ["NEMO_OO_TRACE_DB"] = str(db_path)

    try:
        from nooa.viewer.main import app
    except ImportError:
        click.secho(
            "Error: viewer dependencies are not installed.\n"
            'Install them with:  uv add "nooa[viewer]"',
            fg="red",
            err=True,
        )
        raise SystemExit(1) from None

    import copy

    import uvicorn

    logging.getLogger("uvicorn.access").addFilter(_AccessLogFilter())

    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["loggers"]["nooa.viewer"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }

    click.echo()
    click.secho("  NVIDIA OO Agents Viewer", fg="cyan", bold=True)
    click.echo(f"  URL:  http://localhost:{port}")
    click.echo(f"  DB:   {db_path}")
    click.echo()

    try:
        uvicorn.run(app, host=host, port=port, log_config=log_config)
    except SystemExit:
        raise
    except OSError as exc:
        import errno as _errno

        if exc.errno == _errno.EADDRINUSE:
            pid = _find_pid_on_port(port)
            pid_suffix = f" by process {pid}" if pid else ""
            click.secho(
                f"[Errno 48] error while attempting to bind on address "
                f"('{host}', {port}): address already in use{pid_suffix}",
                fg="red",
                err=True,
            )
            raise SystemExit(1) from None
        raise
