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
from contextlib import ExitStack, contextmanager

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


def _write_secret_env(path, name: str, value: str, *, create_only=False) -> bytes:
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
    path = Path(path).resolve()
    original = path.read_text() if path.exists() else None
    try:
        with path.open() as source:
            data = yaml.safe_load(source) or {}
    except FileNotFoundError:
        data = {}
    except yaml.YAMLError:
        raise ValueError(f"Secrets file {path} contains invalid YAML; no changes made") from None
    if isinstance(data, dict) and data.get("env") is None:
        data["env"] = {}
    if not isinstance(data, dict) or not isinstance(data.get("env", {}), dict):
        raise ValueError(f"Secrets file {path} must contain an env mapping")
    if create_only and name in data.get("env", {}):
        raise ValueError(
            "Secret variable already exists; choose another name or review it separately"
        )
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
        owned = temporary.read_bytes()
        if (path.read_text() if path.exists() else None) != original:
            raise ValueError(f"Secrets file {path} changed during write; retry after reloading")
        os.replace(temporary, path)
        return owned
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = ["load_secrets_into_env", "write_secret_env", "new_secret_env"]


def write_secret_env(path, name: str, value: str) -> None:
    """Explicit unconditional library write, serialized on a stable sidecar.

    Frontends needing create-only ownership and rollback use new_secret_env.
    This does not modify the process environment or remove the lock inode.
    """
    with _secret_lock(path) as resolved:
        _write_secret_env(resolved, name, value)


@contextmanager
def _secret_lock(path):
    import fcntl
    from pathlib import Path

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f".{path.name}.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield path


@contextmanager
def new_secret_env(path, name: str, value: str, *, model_may_be_saved=None, rollback_guard=None):
    """Create a new variable and roll it back if the enclosed model write fails.

    Hold the secret sidecar through the model commit (lock order: secret, model).
    Never overwrite an existing variable, including null. Other variables survive.
    Two files are NOT atomic: exceptions before model commit trigger byte-owned
    rollback. A non-cooperating change prevents rollback and reports partial state.
    Callers must put only the model commit in this block, not post-save display.
    model_may_be_saved checks an ambiguous post-rename error: if committed or
    unknowable, keep the key and report partial state rather than breaking the
    possibly saved profile. It must inspect credential dependencies in the
    selected file only, never globals. Frontends supply rollback_guard to hold
    the model sidecar through inspection AND rollback, blocking cooperative
    profile changes while deciding whether the new credential is still needed.
    """
    import tempfile
    from pathlib import Path

    with _secret_lock(path) as path:
        original = path.read_bytes() if path.exists() else None
        mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
        owned = _write_secret_env(path, name, value, create_only=True)
        try:
            yield
            if path.read_bytes() != owned:
                raise ValueError("Secret changed during model commit")
        except BaseException:
            with ExitStack() as stack:
                if rollback_guard is not None:
                    try:
                        stack.enter_context(rollback_guard())
                    except Exception:
                        raise ValueError(
                            "Model save failed; partial state: rollback could not acquire its guard. "
                            "The new key may remain; review both files before retrying. No success claimed."
                        ) from None
                if model_may_be_saved is not None:
                    try:
                        may_be_saved = model_may_be_saved()
                    except Exception:
                        may_be_saved = True  # Cannot safely rule out a committed profile.
                    if may_be_saved:
                        raise ValueError(
                            "Model save did not finish cleanly; partial state: the model entry may be saved "
                            "and the new key remains. Review both files before retrying. No success claimed."
                        ) from None
                temporary = None
                try:
                    if path.read_bytes() != owned:
                        raise ValueError("Secret changed after creation")
                    if original is None:
                        path.unlink()
                    else:
                        with tempfile.NamedTemporaryFile(
                            dir=path.parent, prefix=f".{path.name}.", delete=False
                        ) as stream:
                            temporary = Path(stream.name)
                            os.chmod(temporary, mode)
                            stream.write(original)
                            stream.flush()
                            os.fsync(stream.fileno())
                        if path.read_bytes() != owned:
                            raise ValueError("Secret changed during rollback")
                        os.replace(temporary, path)
                except Exception:
                    raise ValueError(
                        "Model save failed; partial state: the new key may remain in the secrets file. "
                        "Rollback was unsafe or failed; review the files before retrying. No success claimed."
                    ) from None
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                raise ValueError(
                    "Model save failed; the new key was rolled back. "
                    "Reload the model file and re-review consent before retrying."
                ) from None
