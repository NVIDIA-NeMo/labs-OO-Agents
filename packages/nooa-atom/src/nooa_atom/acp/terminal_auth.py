# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP terminal authentication that runs the ``nooa connect`` wizard.

A client that advertises ``clientCapabilities.auth.terminal`` (Pool, for a
local agent) is offered one ``terminal`` auth method. When a request fails
with ``-32000`` ("Authentication required"), the client relaunches the
agent command with ``--connect`` appended in an interactive terminal, waits
for it to exit, reconnects and retries.

The agent and the wizard are separate processes, and the client restarts
the agent after the wizard, so they hand over through two kinds of small
files in ``acp_connect/`` in the user's NOOA directory:

- A *request*, written by the agent when it answers ``-32000``: which
  session asked (``/connect`` or a prompt without a model), or, for a
  ``session/new`` without any model, which workspace.
- A *result*, written by ``nooa atom --connect`` (:func:`run_terminal_connect`)
  after a save. The wizard claims the newest fresh request and binds its
  result to that session (or workspace), so another window cannot take it.

The session that asked picks its result up when it is loaded again (the
model is switched before queued work starts; the result is kept, marked
``applied``), on its next prompt, or on the retried ``/connect``, which
says what was saved. A retried ``session/new`` takes a result bound to its
workspace. A result holds the alias, the model name, the server (scheme,
host and port only), a fingerprint of the saved entry and the time; never a
key. Files are mode 0600, written atomically, and expire after
``RESULT_MAX_AGE``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit

from acp.schema import ClientCapabilities, TerminalAuthMethod

CONNECT_METHOD_ID = "nooa-connect"
CONNECT_ARG = "--connect"
HANDOFF_DIR = "acp_connect"
RESULT_MAX_AGE = 15 * 60
"""Seconds a request or a saved result stays valid."""
ANY = "any"
"""The key of a result saved when no request was pending (``nooa atom --connect`` run by hand)."""


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
    # The schema's default for a missing ``auth`` is the plain dict
    # ``{"terminal": False}`` (pydantic does not validate defaults), so a
    # client that leaves ``auth`` out gives a dict here, not AuthCapabilities.
    if isinstance(auth, dict):
        return bool(auth.get("terminal"))
    return bool(auth is not None and auth.terminal)


# ---- keys and files --------------------------------------------------------------


def session_key(session_id: str) -> str:
    """The handoff key of one session."""
    return "s-" + hashlib.sha256(session_id.encode()).hexdigest()[:24]


def workspace_key(workspace: Path | str) -> str:
    """The handoff key of a ``session/new`` in ``workspace`` (no session yet)."""
    return "w-" + hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:24]


def handoff_dir() -> Path:
    from nooa import paths

    return paths.get_user_dir(HANDOFF_DIR)


def _path(kind: str, key: str) -> Path:
    return handoff_dir() / f"{kind}-{key}.json"


def _write(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` atomically, mode 0600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read(path: Path) -> dict | None:
    """``path``'s JSON object, or ``None`` when it is missing or not an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _fresh(stamp: Any, *, now: float | None = None, max_age: float = RESULT_MAX_AGE) -> bool:
    if not isinstance(stamp, (int, float)):
        return False
    age = (time.time() if now is None else now) - stamp
    return 0 <= age <= max_age


def _server_label(url: Any) -> str:
    """``url``'s scheme, host and port only: a key or a path token must never be shown."""
    if not isinstance(url, str) or not url:
        return ""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parts.port is not None:
            host = f"{host}:{parts.port}"
    except ValueError:
        return ""
    if not parts.scheme or not host:
        return ""
    return urlunsplit((parts.scheme, host, "", "", ""))


def entry_fingerprint(entry: Any) -> str:
    """A stable digest of a model entry, to tell whether the workspace uses the saved one."""
    text = json.dumps(entry, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


# ---- requests (written by the agent) ----------------------------------------------


def record_request(
    *, session_id: str | None, workspace: Path | str, now: float | None = None
) -> None:
    """Note that ``session_id`` (or a new session in ``workspace``) asked for the terminal Connect."""
    key = session_key(session_id) if session_id else workspace_key(workspace)
    _write(
        _path("request", key),
        {
            "key": key,
            "session_id": session_id,
            "workspace": str(workspace),
            "requested_at": time.time() if now is None else now,
        },
    )


def claim_request(*, now: float | None = None) -> str:
    """Take the newest fresh request and return its key (``ANY`` without one); drop stale ones.

    Reserves the file with ``os.replace`` before returning its key, the way
    :func:`consume_result` reserves a result, so two terminal processes can
    never both claim the same request.
    """
    try:
        paths = sorted(handoff_dir().glob("request-*.json"))
    except OSError:
        return ANY
    candidates: list[tuple[float, str, Path]] = []
    for path in paths:
        data = _read(path)
        stamp = data.get("requested_at") if data else None
        key = data.get("key") if data else None
        if not isinstance(key, str) or not _fresh(stamp, now=now):
            path.unlink(missing_ok=True)
            continue
        candidates.append((stamp, key, path))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    for _, key, path in candidates:
        claimed = path.with_name(f".{path.name}.claimed.{os.getpid()}.{time.monotonic_ns()}")
        try:
            os.replace(path, claimed)
        except FileNotFoundError:
            continue  # another process already claimed it
        claimed.unlink(missing_ok=True)
        return key
    return ANY


# ---- results (written by the wizard) -----------------------------------------------


def write_result(
    key: str,
    alias: str,
    model: str,
    server: str,
    *,
    fingerprint: str = "",
    now: float | None = None,
) -> Path:
    """Record a save for the session (or workspace) ``key``."""
    path = _path("result", key)
    _write(
        path,
        {
            "alias": alias,
            "model": model,
            "server": _server_label(server),
            "fingerprint": fingerprint,
            "saved_at": time.time() if now is None else now,
            "applied": False,
        },
    )
    return path


def discard_result(key: str) -> None:
    _path("result", key).unlink(missing_ok=True)


def _valid(data: dict | None, *, now: float | None, max_age: float) -> dict | None:
    if data is None:
        return None
    alias = data.get("alias")
    if (
        not isinstance(alias, str)
        or not alias
        or not _fresh(data.get("saved_at"), now=now, max_age=max_age)
    ):
        return None
    return {
        "alias": alias,
        "model": data.get("model") if isinstance(data.get("model"), str) else "",
        "server": data.get("server") if isinstance(data.get("server"), str) else "",
        "fingerprint": data.get("fingerprint") if isinstance(data.get("fingerprint"), str) else "",
        "applied": data.get("applied") is True,
    }


def peek_result(
    key: str, *, max_age: float = RESULT_MAX_AGE, now: float | None = None
) -> dict | None:
    """The fresh result for ``key``, left in place; a stale or broken one is removed.

    Raises ``OSError`` when the file exists but cannot be read.
    """
    path = _path("result", key)
    result = _valid(_read(path), now=now, max_age=max_age)
    if result is None:
        path.unlink(missing_ok=True)
    return result


def mark_applied(key: str, result: dict) -> None:
    """Keep ``result`` for the retried ``/connect``, noting the session already switched."""
    path = _path("result", key)
    data = _read(path)
    if data is None or data.get("alias") != result["alias"]:
        return  # taken, or replaced by a newer save
    _write(path, {**data, "applied": True})


def consume_result(
    key: str, *, max_age: float = RESULT_MAX_AGE, now: float | None = None
) -> dict | None:
    """Take the result for ``key`` if it is fresh; it is removed either way.

    The file is renamed away before it is read, so two requests cannot both
    take one save. Raises ``OSError`` when the file cannot be moved or read.
    """
    path = _path("result", key)
    taken = path.with_name(f".{path.name}.taken.{os.getpid()}.{time.monotonic_ns()}")
    try:
        os.replace(path, taken)
    except FileNotFoundError:
        return None
    try:
        return _valid(_read(taken), now=now, max_age=max_age)
    finally:
        taken.unlink(missing_ok=True)


def describe_result(result: dict) -> str:
    """The text that says what was saved."""
    text = f"Saved model alias {result['alias']}"
    if result.get("model"):
        text += f": {result['model']}"
    if result.get("server"):
        text += f" at {result['server']}"
    return text + "."


# ---- the wizard ------------------------------------------------------------


class SavedEntry(NamedTuple):
    model: str
    server: str
    fingerprint: str


def _saved_entry(alias: str, path: Path) -> SavedEntry:
    """What the wizard wrote for ``alias`` in ``path``."""
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return SavedEntry("", "", "")
    models = data.get("models") if isinstance(data, dict) else None
    entry = models.get(alias) if isinstance(models, dict) else None
    if not isinstance(entry, dict):
        return SavedEntry("", "", "")
    server = entry.get("api_base") or entry.get("endpoint") or ""
    return SavedEntry(str(entry.get("model_name") or ""), str(server), entry_fingerprint(entry))


def run_terminal_connect() -> int:
    """Run ``nooa connect`` (its defaults, no arguments) here; 0 only when it saved an entry.

    The ``nooa connect`` command itself runs (with no arguments) and returns
    what the wizard saved, so its option-to-wizard mapping and defaults
    cannot drift. Errors and Ctrl-C propagate (click turns them into a
    non-zero exit).
    """
    import click
    from nooa_cli.commands import connect as connect_command

    key = claim_request()
    discard_result(key)  # an older save for the same session must not count
    saved = connect_command.command.main(args=[], prog_name="nooa connect", standalone_mode=False)
    if not saved:
        click.echo("Nothing was saved.", err=True)
        return 1
    entry = _saved_entry(saved.alias, saved.path)
    try:
        write_result(key, saved.alias, entry.model, entry.server, fingerprint=entry.fingerprint)
    except OSError as exc:
        # The model is saved; only the note for the client is missing. The
        # client can still use the model (pick it in the model menu).
        click.echo(
            f"Saved model alias {saved.alias}, but could not tell the client ({exc}). "
            "Choose it in the client's model menu after it reconnects.",
            err=True,
        )
        return 0
    click.echo("Return to the client; it reconnects with the new model.")
    return 0


__all__ = [
    "ANY",
    "CONNECT_ARG",
    "CONNECT_METHOD_ID",
    "RESULT_MAX_AGE",
    "claim_request",
    "connect_auth_method",
    "consume_result",
    "describe_result",
    "discard_result",
    "entry_fingerprint",
    "mark_applied",
    "peek_result",
    "record_request",
    "run_terminal_connect",
    "session_key",
    "supports_terminal_auth",
    "workspace_key",
    "write_result",
]
