# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The built-in OpenRouter model source: every OpenRouter text model with one key.

Once an OpenRouter key is set up (``OPENROUTER_API_KEY`` in the environment
or in NOOA's ``secrets.yaml``), Atom lists every model of OpenRouter's public
text-model catalogue as ``openrouter/<OpenRouter model ID>``, for example
``openrouter/anthropic/claude-sonnet-4.5``. Each entry is exactly what
``nooa connect --provider openrouter`` saves for that model, without its test
calls (see ``nooa.unifiedllm.connect.openrouter_entry``).

The catalogue is cached in ``openrouter_models.json`` in the user NOOA
directory and fetched again when it is older than 24 hours. ``models()``
sends a request only when the cache is missing or stale, waits at most 10
seconds, and then falls back to the stale cache (or no models); it never
raises.

``sign_in()`` runs in a terminal (``nooa atom --sign-in openrouter``): it asks
for the key, checks it with one GET of OpenRouter's ``/key`` endpoint, saves
it the way ``nooa connect`` does and fetches the catalogue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KEY_ENV = "OPENROUTER_API_KEY"
"""The variable the entries name as ``api_key_env``, as Connect's preset does."""
ALIAS_PREFIX = "openrouter/"
CACHE_FILE = "openrouter_models.json"
"""In the user NOOA directory (``nooa.paths.get_user_dir``)."""
MAX_AGE = 24 * 60 * 60
"""Seconds before the cached catalogue is fetched again."""
FETCH_TIMEOUT = 10.0
"""Seconds a catalogue fetch may take on the ``models()`` path."""
RETRY_AFTER = 10 * 60
"""Seconds before ``models()`` tries again after a failed fetch."""
KEY_URL = "https://openrouter.ai/api/v1/key"
PREFERRED = (
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5",
    "google/gemini-2.5-pro",
)
"""Model IDs ``sign_in`` returns, in this order, when the catalogue lists them."""

_lock = threading.Lock()
_memo: tuple[tuple[str, int, int], float, dict[str, dict[str, Any]]] | None = None
"""The cache file's version (path, mtime, size), fetch time and entries."""
_failed_at: float | None = None
"""When the last fetch failed (``time.monotonic``), so it is not retried at once."""
_warned = False


def cache_path() -> Path:
    """Where the catalogue cache lives."""
    from nooa import paths

    return paths.get_user_dir(CACHE_FILE)


def api_key() -> str | None:
    """The OpenRouter key: the environment first, then NOOA's saved secrets.

    Saved secrets are loaded into the environment as at NOOA's start (a
    variable already set is kept), so a key saved by a sign-in in another
    process is found, and used by the model client, without a restart.
    """
    value = os.environ.get(KEY_ENV, "").strip()
    if value:
        return value
    try:
        from nooa.secrets import load_secrets_into_env

        load_secrets_into_env()
    except Exception:
        logger.debug("Could not read the NOOA secrets", exc_info=True)
    return os.environ.get(KEY_ENV, "").strip() or None


def _fetch(timeout: float = FETCH_TIMEOUT) -> list[dict]:
    """The public text-model catalogue, read with Connect's ``catalogue()``.

    Runs on its own thread, so it also works when called from a running
    event loop. Raises when the catalogue cannot be read.
    """
    from nooa.unifiedllm import connect

    with ThreadPoolExecutor(1, thread_name_prefix="openrouter-catalogue") as pool:
        return pool.submit(
            lambda: asyncio.run(connect.catalogue(timeout=timeout, outputs="text"))
        ).result()


def _read_cache(path: Path) -> tuple[float, list[dict]] | None:
    """The cache's fetch time and records, or ``None`` when it is missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("Ignoring the unreadable OpenRouter model cache %s", path)
        return None
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("fetched_at"), (int, float))
        or not isinstance(data.get("models"), list)
    ):
        logger.warning("Ignoring the OpenRouter model cache %s: unknown format", path)
        return None
    return float(data["fetched_at"]), [m for m in data["models"] if isinstance(m, dict)]


def write_cache(records: list[dict], path: Path | None = None) -> Path:
    """Save the catalogue records, replacing the cache file atomically."""
    from nooa.unifiedllm import connect

    path = path or cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"url": connect.CATALOGUE_URL, "fetched_at": time.time(), "models": records}
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as temporary:
        json.dump(payload, temporary)
    os.replace(temporary.name, path)
    return path


def entries(records: list[dict]) -> dict[str, dict[str, Any]]:
    """The model entries for catalogue records, by alias, sorted by model ID.

    A record Connect cannot turn into an entry is skipped.
    """
    from nooa.unifiedllm import connect

    result: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda r: str(r.get("id"))):
        try:
            entry = connect.openrouter_entry(record)
        except Exception as exc:
            logger.debug("Skipped OpenRouter model %r: %s", record.get("id"), exc)
            continue
        result[ALIAS_PREFIX + record["id"]] = entry
    return result


def _load(path: Path) -> tuple[float, dict[str, dict[str, Any]]] | None:
    """The cache's fetch time and entries, built once per version of the file."""
    global _memo
    try:
        status = path.stat()
    except OSError:
        return None
    key = (str(path), status.st_mtime_ns, status.st_size)
    if _memo is not None and _memo[0] == key:
        return _memo[1], _memo[2]
    cached = _read_cache(path)
    if cached is None:
        return None
    _memo = (key, cached[0], entries(cached[1]))
    return _memo[1], _memo[2]


def _refresh(path: Path) -> bool:
    """Fetch the catalogue into the cache; False (logged once) when that fails."""
    global _failed_at, _warned
    try:
        records = _fetch()
        if not records:
            raise ValueError("the catalogue lists no models")
        write_cache(records, path)
    except Exception as exc:
        _failed_at = time.monotonic()
        level = logging.DEBUG if _warned else logging.WARNING
        _warned = True
        logger.log(
            level, "Could not fetch the OpenRouter model list: %s", exc or type(exc).__name__
        )
        return False
    _failed_at = None
    return True


def reset() -> None:
    """Forget the in-process memo and fetch failures (for tests)."""
    global _memo, _failed_at, _warned
    _memo = None
    _failed_at = None
    _warned = False


class OpenRouterSource:
    """Every OpenRouter text model, configured as ``nooa connect`` would save it."""

    id = "openrouter"
    name = "OpenRouter"
    description = "All OpenRouter models with one key"

    def models(self, workspace: Path) -> dict[str, dict[str, Any]]:
        """The OpenRouter entries by alias; ``{}`` without a key or a catalogue. Never raises."""
        try:
            if api_key() is None:
                return {}
            path = cache_path()
            with _lock:
                loaded = _load(path)
                stale = loaded is None or time.time() - loaded[0] > MAX_AGE
                retry = _failed_at is None or time.monotonic() - _failed_at >= RETRY_AFTER
                if stale and retry and _refresh(path):
                    loaded = _load(path)
                return dict(loaded[1]) if loaded else {}
        except Exception:
            logger.warning("Could not list the OpenRouter models", exc_info=True)
            return {}

    def sign_in(self) -> str | None:
        """Ask for the key in this terminal, check it, save it and fetch the models.

        Returns the alias to switch to, or ``None`` when nothing was set up.
        Ctrl-C propagates. The key is never printed.
        """
        import click

        from nooa import paths

        from_environment = os.environ.get(KEY_ENV, "").strip()
        existing = api_key()
        key = None
        if existing and click.confirm(
            "An OpenRouter key is already set up. Keep it?", default=True
        ):
            key = existing
        if key is None:
            click.echo("Create a key at https://openrouter.ai/settings/keys.")
            key = click.prompt(
                "OpenRouter API key", hide_input=True, default="", show_default=False
            )
            key = key.strip()
            if not key:
                click.echo("No key entered; nothing was set up.", err=True)
                return None
        problem = check_key(key)
        if problem is not None:
            click.echo(f"{problem} Nothing was saved.", err=True)
            return None
        if key != existing:
            from nooa.secrets import write_secret_env

            secrets = paths.get_user_dir("secrets.yaml")
            write_secret_env(secrets, KEY_ENV, key)
            click.echo(
                f"Saved the key as {KEY_ENV} in {secrets} (plain text, readable only by you)."
            )
            if from_environment not in ("", key):
                click.echo(
                    f"Your environment sets a different {KEY_ENV}, which still wins; "
                    "unset or update it before starting NOOA again.",
                    err=True,
                )
            os.environ[KEY_ENV] = key
        try:
            records = _fetch()
            if not records:
                raise ValueError("the catalogue lists no models")
            write_cache(records)
        except Exception as exc:
            click.echo(
                f"Could not fetch the OpenRouter model list: {exc or type(exc).__name__}", err=True
            )
            return None
        reset()
        models = entries(records)
        if not models:
            click.echo("OpenRouter lists no usable models.", err=True)
            return None
        click.echo(f"{len(models)} OpenRouter models are now available (cached in {cache_path()}).")
        return default_alias(models)


def default_alias(models: dict[str, dict[str, Any]]) -> str:
    """The first preferred model that is listed, else the first model."""
    return next(
        (ALIAS_PREFIX + model for model in PREFERRED if ALIAS_PREFIX + model in models),
        next(iter(models)),
    )


def check_key(key: str) -> str | None:
    """Check ``key`` with one GET of OpenRouter's ``/key``; why it failed, or ``None``.

    Sends no completion request.
    """
    import httpx

    try:
        response = httpx.get(
            KEY_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=FETCH_TIMEOUT,
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        return f"Could not reach OpenRouter to check the key ({type(exc).__name__})."
    if response.status_code in (401, 403):
        return "OpenRouter did not accept this key."
    if not response.is_success:
        return f"OpenRouter answered the key check with HTTP {response.status_code}."
    return None


source = OpenRouterSource()

__all__ = [
    "ALIAS_PREFIX",
    "CACHE_FILE",
    "KEY_ENV",
    "MAX_AGE",
    "OpenRouterSource",
    "api_key",
    "cache_path",
    "check_key",
    "default_alias",
    "entries",
    "reset",
    "source",
    "write_cache",
]
