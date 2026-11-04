# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP terminal authentication that runs the ``nooa connect`` wizard.

A client that advertises ``clientCapabilities.auth.terminal`` (Pool, for a
local agent) is offered one ``terminal`` auth method. When a request fails
with ``-32000`` ("Authentication required"), the client relaunches the
agent command with ``--connect`` appended in an interactive terminal, waits
for it to exit, reconnects and retries.

``nooa atom --connect`` runs the CLI wizard (:func:`run_terminal_connect`).
On a save it writes a one-shot result file in the user's NOOA directory, so
the next ``/connect`` (possibly in a new agent process: the client
reconnects) can say what was saved instead of asking for authentication
again. The file holds the alias, the model name, the server (without
credentials, query or fragment) and the time; never a key.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from acp.schema import ClientCapabilities, TerminalAuthMethod

CONNECT_METHOD_ID = "nooa-connect"
CONNECT_ARG = "--connect"
RESULT_FILE = "acp_connect_result.json"
RESULT_MAX_AGE = 15 * 60
"""Seconds a saved result stays valid for the next ``/connect``."""


def connect_auth_method() -> TerminalAuthMethod:
    """The terminal method that relaunches the agent with ``--connect``."""
    return TerminalAuthMethod(
        type="terminal",
        id=CONNECT_METHOD_ID,
        name="Connect a model",
        description="Set up a model provider in a terminal",
        args=[CONNECT_ARG],
    )


def supports_terminal_auth(capabilities: ClientCapabilities | None) -> bool:
    """Whether the client said it runs ``terminal`` auth methods."""
    auth = capabilities.auth if capabilities is not None else None
    return bool(auth is not None and auth.terminal)


# ---- the one-shot result -------------------------------------------------


def result_path() -> Path:
    from nooa import paths

    return paths.get_user_dir(RESULT_FILE)


def _server_label(url: Any) -> str:
    """``url`` without user info, query or fragment (a key must never reach the file)."""
    if not isinstance(url, str) or not url:
        return ""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return ""


def clear_result() -> None:
    """Remove any saved result: a cancelled or failed run must not count as a save."""
    try:
        result_path().unlink()
    except FileNotFoundError:
        pass


def write_result(alias: str, model: str, server: str, *, now: float | None = None) -> Path:
    """Record a save for the next ``/connect`` (mode 0600, written atomically)."""
    path = result_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "alias": alias,
        "model": model,
        "server": _server_label(server),
        "saved_at": time.time() if now is None else now,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    os.replace(temporary, path)
    return path


def consume_result(*, max_age: float = RESULT_MAX_AGE, now: float | None = None) -> dict | None:
    """Take the saved result if it is fresh; it is removed either way.

    The file is renamed away before it is read, so two ``/connect`` prompts
    cannot both take one save.
    """
    path = result_path()
    taken = path.with_name(f".{path.name}.taken.{os.getpid()}")
    try:
        os.replace(path, taken)
    except FileNotFoundError:
        return None
    try:
        data = json.loads(taken.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    finally:
        taken.unlink(missing_ok=True)
    if not isinstance(data, dict):
        return None
    alias, saved_at = data.get("alias"), data.get("saved_at")
    if not isinstance(alias, str) or not alias or not isinstance(saved_at, (int, float)):
        return None
    age = (time.time() if now is None else now) - saved_at
    if not 0 <= age <= max_age:
        return None
    return {
        "alias": alias,
        "model": data.get("model") if isinstance(data.get("model"), str) else "",
        "server": data.get("server") if isinstance(data.get("server"), str) else "",
    }


def describe_result(result: dict) -> str:
    """The ``/connect`` reply for a consumed result."""
    text = f"Saved model alias {result['alias']}"
    if result.get("model"):
        text += f": {result['model']}"
    if result.get("server"):
        text += f" at {result['server']}"
    return text + ". Pick it in the model menu."


# ---- the wizard ------------------------------------------------------------


def _saved_entry(alias: str) -> dict:
    """The entry the wizard wrote for ``alias`` in the user ``llm_config.yaml``."""
    import yaml

    from nooa import paths

    try:
        data = yaml.safe_load(paths.get_user_dir("llm_config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    models = data.get("models") if isinstance(data, dict) else None
    entry = models.get(alias) if isinstance(models, dict) else None
    return entry if isinstance(entry, dict) else {}


def run_terminal_connect() -> int:
    """Run ``nooa connect`` (its defaults, no arguments) here; 0 only when it saved an entry.

    The wizard's own options come from the ``nooa connect`` command itself, so
    the defaults cannot drift. Errors and Ctrl-C propagate (click turns them
    into a non-zero exit).
    """
    import click
    from nooa_cli.commands import connect as connect_command
    from nooa_cli.commands._connect_wizard import run_wizard

    clear_result()
    with connect_command.command.make_context("connect", []) as ctx:
        options = dict(ctx.params)
    options.pop("stage", None)
    options.pop("input_file", None)
    alias = run_wizard(**options)
    if not alias:
        click.echo("Nothing was saved.", err=True)
        return 1
    entry = _saved_entry(alias)
    write_result(alias, str(entry.get("model_name") or ""), str(entry.get("api_base") or ""))
    click.echo("Return to the client; it reconnects with the new model.")
    return 0


__all__ = [
    "CONNECT_ARG",
    "CONNECT_METHOD_ID",
    "RESULT_MAX_AGE",
    "clear_result",
    "connect_auth_method",
    "consume_result",
    "describe_result",
    "run_terminal_connect",
    "supports_terminal_auth",
    "write_result",
]
