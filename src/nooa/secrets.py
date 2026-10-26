# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load API keys from ``secrets.yaml`` into ``os.environ``, and save new ones.

Secrets live in ``secrets.yaml`` next to ``llm_config.yaml`` and
``settings.yaml``, in the same directories, and are found through the same
:func:`nooa.layered_config.load_layered_yaml` helper.

Schema — a single ``env:`` mapping of env-var name → value::

    # ~/.config/nooa/secrets.yaml
    env:
      NVIDIA_INFERENCE_API_KEY: sk-...
      ANTHROPIC_API_KEY: sk-ant-...

This matches how model entries in ``llm_config.yaml`` name their key
(``api_key_env: NVIDIA_INFERENCE_API_KEY``): the model entry names the
environment variable, and the variable holds the secret.

:func:`load_secrets_into_env` copies those values into ``os.environ`` without
overwriting: a variable that is already set in the process (for example by
``export`` in the shell) keeps its value. Calling it again changes nothing for
variables that are already set.

:func:`write_secret_env` and :func:`new_secret_env` save a variable to a
secrets file. Writes replace the file atomically, keep it readable by its
owner only (mode 600), and are serialized by an exclusive lock on a separate
lock file next to it (``.secrets.yaml.lock``).
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

    A variable already present in ``os.environ`` is left unchanged, so the
    process or shell value wins. Entries with a null value are skipped, and a
    non-mapping ``env:`` is logged and ignored.

    Returns:
        The names of the variables this call set (those that were missing),
        for diagnostics. Safe to call several times.
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
            # A value already set in the process or shell takes priority.
            continue
        os.environ[key] = str(value)
        applied.append(key)
    return applied


def _write_secret_env(path, name: str, value: str, *, create_only=False) -> bytes:
    """Save one variable into the ``env:`` mapping of a secrets file, atomically.

    The new content is written to a temporary file with mode 600 in the same
    directory and then renamed over ``path``. If the file changed while this
    was being prepared, nothing is replaced. Other variables are kept. The
    process environment is not changed; callers that ask the user for consent
    should also explain that an exported shell value still takes priority.
    YAML parse errors are reported without the parser's message, because that
    message can quote parts of the file, including secret values.

    Args:
        path: The secrets file; it is created if missing.
        name: Environment variable name.
        value: The secret value; must be a non-empty string.
        create_only: If true, fail when ``name`` already exists in the file,
            even with a null value.

    Returns:
        The bytes written, so callers can later check that the file still
        holds exactly what they wrote.

    Raises:
        ValueError: If the name or value is invalid, the file is not valid
            YAML or has no ``env`` mapping, the variable exists and
            ``create_only`` is set, or the file changed during the write.
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
    """Save ``name=value`` in a secrets file, replacing any existing value.

    The write holds the file's lock (see the module docstring). It does not
    change the process environment, and the lock file is left in place.
    Callers that must not overwrite an existing variable, or that need to
    undo the write when a related save fails, use :func:`new_secret_env`.
    """
    with _secret_lock(path) as resolved:
        _write_secret_env(resolved, name, value)


@contextmanager
def _secret_lock(path):
    """Hold an exclusive lock for a secrets file and yield its resolved path.

    The lock is taken on a separate ``.<name>.lock`` file in the same
    directory, not on the secrets file itself, because writes replace the
    secrets file with a new one.
    """
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
    """Add a new variable to a secrets file, and remove it again if the block fails.

    Used when saving a model entry together with its new API key: the key is
    written first, then the caller saves the model entry inside the ``with``
    block. Saving two files cannot be atomic, so if the block raises, this
    function tries to restore the secrets file to its previous content.

    The secrets file lock is held for the whole block. Callers that also lock
    the model file must take that lock inside the block (secrets file first,
    then model file) to avoid deadlocks. Put only the model save inside the
    block; anything that runs after a successful save (such as printing a
    summary) belongs after it, so its errors do not undo the key.

    Args:
        path: The secrets file.
        name: New environment variable name. It must not exist in the file
            yet, not even with a null value. Other variables are kept.
        value: The secret value.
        model_may_be_saved: Optional callable, called after a failure, that
            returns whether the model entry may have been saved anyway (for
            example when an error happened after the file was renamed into
            place). If it returns true, or raises, the key is kept so that a
            saved model entry still finds its credential. It should look only
            at the model file being saved.
        rollback_guard: Optional callable returning a context manager (for
            example the model file's lock) that is held while
            ``model_may_be_saved`` runs and while the key is removed, so that
            other cooperating writers cannot change the model file meanwhile.

    Raises:
        ValueError: Always, if the block raises (the original exception is not
            chained). The message says whether the key was removed or may
            remain ("partial state"), in which case the user should review both
            files. The key is not removed if the secrets file was changed by
            someone else after this function wrote it. Also raised if the
            variable already exists or the secrets file changed during the
            block.
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
                        may_be_saved = True  # Unknown: assume the model entry was saved.
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
