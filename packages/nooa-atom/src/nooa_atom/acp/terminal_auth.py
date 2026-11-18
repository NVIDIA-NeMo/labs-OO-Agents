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
again. Whichever request of a terminal-auth client first finds a fresh
result (``/connect``, ``authenticate``, a prompt, ``session/load`` or
``session/new``) takes it and switches the session to the saved alias. The file holds the alias, the model name, the server (without
credentials, query or fragment) and the time; never a key.

An installed model source (``nooa_atom.workspace.model_sources``) adds one
more method: ``nooa atom --sign-in SOURCE_ID`` runs that source's sign-in
(:func:`run_terminal_sign_in`) and leaves the same kind of result.
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
SIGN_IN_ARG = "--sign-in"
SOURCE_METHOD_PREFIX = "nooa-source-"
RESULT_FILE = "acp_connect_result.json"
RESULT_MAX_AGE = 15 * 60
"""Seconds a saved result stays valid for the next ``/connect``."""


def connect_auth_method(*, beside_sources: bool = False) -> TerminalAuthMethod:
    """The terminal method that relaunches the agent with ``--connect``.

    ``beside_sources``: model sources are offered too, so this one is named
    for the models they do not cover.
    """
    return TerminalAuthMethod(
        type="terminal",
        id=CONNECT_METHOD_ID,
        name="Other models (nooa connect)" if beside_sources else "Connect a model",
        description="Set up a model provider in a terminal",
        args=[CONNECT_ARG],
    )


def source_method_id(source_id: str) -> str:
    """The auth method id of the model source ``source_id``."""
    return f"{SOURCE_METHOD_PREFIX}{source_id}"


def auth_methods() -> list[TerminalAuthMethod]:
    """The terminal methods to offer: ``nooa connect``, then one per installed model source."""
    from nooa_atom.workspace.model_sources import model_sources

    sources = model_sources()
    methods = [
        TerminalAuthMethod(
            type="terminal",
            id=source_method_id(source.id),
            name=source.name,
            description=source.description,
            args=[SIGN_IN_ARG, source.id],
        )
        for source in sources
    ]
    return [connect_auth_method(beside_sources=bool(sources)), *methods]


def auth_method_ids() -> list[str]:
    """The ids of :func:`auth_methods`."""
    return [method.id for method in auth_methods()]


def is_auth_method(method_id: Any) -> bool:
    """Whether ``method_id`` is one of the terminal methods this agent offers."""
    return isinstance(method_id, str) and method_id in auth_method_ids()


def supports_terminal_auth(capabilities: ClientCapabilities | None) -> bool:
    """Whether the client said it runs ``terminal`` auth methods."""
    auth = capabilities.auth if capabilities is not None else None
    # The schema's default for a missing ``auth`` is the plain dict
    # ``{"terminal": False}`` (pydantic does not validate defaults), so a
    # client that leaves ``auth`` out gives a dict here, not AuthCapabilities.
    if isinstance(auth, dict):
        return bool(auth.get("terminal"))
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
    return text + "."


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

    The ``nooa connect`` command itself runs (with no arguments) and returns
    the alias the wizard saved, so its option-to-wizard mapping and defaults
    cannot drift. Errors and Ctrl-C propagate (click turns them into a
    non-zero exit).
    """
    import click
    from nooa_cli.commands import connect as connect_command

    clear_result()
    alias = connect_command.command.main(args=[], prog_name="nooa connect", standalone_mode=False)
    if not alias:
        click.echo("Nothing was saved.", err=True)
        return 1
    entry = _saved_entry(alias)
    write_result(alias, str(entry.get("model_name") or ""), str(entry.get("api_base") or ""))
    click.echo("Return to the client; it reconnects with the new model.")
    return 0


def ask_secret(label: str) -> str:
    """Ask for a key in this terminal, showing one ``*`` per character typed or pasted.

    Uses the same masked field as ``nooa connect``; without a terminal,
    Click's hidden line input. Returns the stripped answer (``""`` when
    nothing was entered). Ctrl-C propagates as ``click.Abort``.
    """
    from nooa_cli.commands import _connect_prompts as prompts

    return prompts.prompt(label, hide_input=True, default="", show_default=False).strip()


def choose_model(source_name: str, models: dict[str, Any], default: str) -> str:
    """Ask which of a source's ``models`` to use; Enter keeps ``default``.

    A searchable list in a terminal (type part of a name, Tab to select),
    Click line input otherwise. Only a listed alias is accepted. Ctrl-C
    propagates as ``click.Abort``.
    """
    import click
    from nooa_cli.commands import _connect_prompts as prompts

    names = list(models)
    if default not in models:
        default = names[0]
    click.echo(
        f"{len(names)} {source_name} models. Type part of a name and press Tab to "
        f"select one, or press Enter for {default}."
    )
    while True:
        choice = prompts.prompt("Model", default=default, choices=names, searchable=True).strip()
        if choice in models:
            return choice
        click.echo(f"{choice!r} is not one of the {source_name} models.", err=True)


def run_terminal_sign_in(source_id: str) -> int:
    """Run the sign-in of model source ``source_id`` here; 0 only when it set a model up.

    Exits 2 for an unknown source and 1 when the sign-in set nothing up or
    failed. Ctrl-C propagates.
    """
    import click

    from nooa_atom.workspace.model_sources import find_source

    clear_result()
    source = find_source(source_id)
    if source is None:
        click.echo(f"No model source {source_id!r} is installed.", err=True)
        return 2
    try:
        alias = source.sign_in()
    except Exception as exc:
        click.echo(f"{source.name} sign-in failed: {exc}", err=True)
        return 1
    if not alias or not isinstance(alias, str):
        click.echo("Nothing was set up.", err=True)
        return 1
    listing = getattr(source, "choices", None) or source.models
    try:
        models = listing(Path.cwd())
    except Exception:
        models = {}
    models = models if isinstance(models, dict) else {}
    if len(models) > 1:
        alias = choose_model(source.name, models, alias)
    remember = getattr(source, "select", None)
    if callable(remember):
        try:
            remember(alias)
        except Exception as exc:
            click.echo(f"Could not remember {alias}: {exc}", err=True)
            return 1
    entry = models.get(alias)
    entry = entry if isinstance(entry, dict) else {}
    write_result(alias, str(entry.get("model_name") or ""), str(entry.get("api_base") or ""))
    click.echo("Return to the client; it reconnects with the new model.")
    return 0


__all__ = [
    "CONNECT_ARG",
    "CONNECT_METHOD_ID",
    "RESULT_MAX_AGE",
    "SIGN_IN_ARG",
    "auth_method_ids",
    "ask_secret",
    "auth_methods",
    "choose_model",
    "clear_result",
    "connect_auth_method",
    "consume_result",
    "describe_result",
    "is_auth_method",
    "run_terminal_connect",
    "run_terminal_sign_in",
    "source_method_id",
    "supports_terminal_auth",
    "write_result",
]
