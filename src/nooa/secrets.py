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


__all__ = ["load_secrets_into_env"]
