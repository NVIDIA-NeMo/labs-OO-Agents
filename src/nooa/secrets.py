# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secrets loading: ``secrets.yaml`` → ``os.environ`` (non-clobbering).

Part of the project's "one config story": secrets live in ``secrets.yaml``
next to ``llm_config.yaml`` and ``settings.yaml``, in the same directories,
discovered through the same
:func:`nooa.layered_config.load_layered_yaml` helper.

Schema — a single ``env:`` mapping of env-var name → value::

    # ~/.config/nooa/secrets.yaml
    env:
      NVIDIA_INFERENCE_API_KEY: sk-...
      ANTHROPIC_API_KEY: sk-ant-...

This matches the existing ``api_key_env: NVIDIA_INFERENCE_API_KEY`` pattern
in unifiedllm — YAML names the env var, the env var holds the secret. One
mental model.

:func:`load_secrets_into_env` pushes those names into ``os.environ``
**non-clobbering**: an already-set process env var always wins over a file
value, so an explicit ``export`` in the shell still takes precedence. The
call is idempotent — running it twice is a no-op for keys already present.
"""

from __future__ import annotations

import logging
import os

from nooa.layered_config import load_layered_yaml

logger = logging.getLogger(__name__)

_SECRETS_FILENAME = "secrets.yaml"
_SECRETS_ENV_VAR = "NEMO_OO_SECRETS"


def load_secrets_into_env() -> list[str]:
    """Load layered ``secrets.yaml`` and push its ``env:`` map into ``os.environ``.

    Non-clobbering: a name already present in ``os.environ`` is left
    untouched (the process / shell value wins). Returns the list of env
    var names actually set by this call (i.e. those that were missing),
    for diagnostics. Safe to call multiple times.
    """
    merged = load_layered_yaml(_SECRETS_FILENAME, _SECRETS_ENV_VAR)
    env_map = merged.get("env")
    if env_map is None:
        return []
    if not isinstance(env_map, dict):
        logger.warning(
            "secrets.yaml `env:` is not a mapping (%s); ignoring", type(env_map).__name__
        )
        return []

    applied: list[str] = []
    for name, value in env_map.items():
        if value is None:
            continue
        key = str(name)
        if key in os.environ:
            # Process / shell value wins — never clobber.
            continue
        os.environ[key] = str(value)
        applied.append(key)
    return applied


def write_secret_env(path, name: str, value: str) -> None:
    """Atomically save one credential with owner-only permissions.

    This explicit write does not change the process environment. Frontends own
    consent and must explain that an exported shell value still takes priority.
    Secret-bearing YAML parser excerpts never escape this helper.
    """
    import re
    import tempfile
    from pathlib import Path

    import yaml

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("Secret variable must be a valid environment variable name")
    if not isinstance(value, str) or not value:
        raise ValueError("Secret value cannot be empty")
    path = Path(path)
    original = path.read_text() if path.exists() else None
    try:
        with path.open() as source:
            data = yaml.safe_load(source) or {}
    except FileNotFoundError:
        data = {}
    except yaml.YAMLError:
        raise ValueError(f"Secrets file {path} contains invalid YAML; no changes made") from None
    if not isinstance(data, dict) or not isinstance(data.get("env", {}), dict):
        raise ValueError(f"Secrets file {path} must contain an env mapping")
    data.setdefault("env", {})[name] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            yaml.safe_dump(data, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        if (path.read_text() if path.exists() else None) != original:
            raise ValueError(f"Secrets file {path} changed during write; retry after reloading")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = ["load_secrets_into_env", "write_secret_env"]
