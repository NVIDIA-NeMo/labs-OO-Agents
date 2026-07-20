# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The currently-active, resolved file config as typed objects.

``resolved_config()`` is the programmatic equivalent of ``nooa config
show``: it reads the three layered config files (``settings.yaml``,
``secrets.yaml``, ``llm_config.yaml``) through the shared loader and returns
a typed :class:`ResolvedConfig` rather than raw dicts.

    from nooa.config import resolved_config
    print(resolved_config())

Registry / loader imports are kept lazy (inside the functions) to avoid an
import cycle and to keep this module's load cheap.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from nooa.config.model_config import ModelConfig

# (filename, env-var) for each layered config file.
_SETTINGS = ("settings.yaml", "NEMO_OO_SETTINGS")
_SECRETS = ("secrets.yaml", "NEMO_OO_SECRETS")
_LLM = ("llm_config.yaml", "NEMO_OO_LLM_CONFIG")


class Secrets(BaseModel):
    """Loaded view of ``secrets.yaml`` — the ``env:`` name→value map.

    Values are :class:`pydantic.SecretStr`, so ``repr`` / ``str`` /
    ``model_dump`` mask them as ``'**********'`` — printing a
    :class:`ResolvedConfig` never leaks a key. Call ``.get_secret_value()``
    on a value to read the plaintext.

    This is the *loaded representation*; the values are still pushed into
    ``os.environ`` by :func:`nooa.secrets.load_secrets_into_env`,
    which is what providers/litellm actually read. The map is intentionally
    open-ended (any env-var name), so it stays a ``dict`` rather than typed
    per-key fields.
    """

    model_config = ConfigDict(frozen=True)
    env: dict[str, SecretStr] = Field(default_factory=dict)


class ResolvedConfig(BaseModel):
    """Everything the three config files resolve to, as typed objects.

    - ``models`` — alias → :class:`ModelConfig` (from ``llm_config.yaml`` +
      bundled defaults).
    - ``secrets`` — :class:`Secrets`; values are ``SecretStr`` (masked on
      print/dump; ``.get_secret_value()`` for plaintext).
    - ``settings`` — merged ``settings.yaml`` mapping. Left as a dict because
      the rich settings object (``Config``) lives in the CLI package, which
      core can't import.
    - ``sources`` — the resolved file paths that contributed to each, lowest
      priority first (the "which file is winning?" answer).
    """

    model_config = ConfigDict(frozen=True)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    secrets: Secrets = Field(default_factory=Secrets)
    settings: dict[str, Any] = Field(default_factory=dict)
    sources: dict[str, list[str]] = Field(default_factory=dict)


def get_model_config(name: str) -> ModelConfig | None:
    """Return the typed :class:`ModelConfig` for alias *name*, or ``None``.

    Reads the unifiedllm registry (auto-loading it if needed). This is the
    typed accessor over the raw ``MODELS`` dict.
    """
    from nooa.unifiedllm import MODELS
    from nooa.unifiedllm.registry import _registry_lock, ensure_loaded

    ensure_loaded()
    with _registry_lock:
        raw = MODELS.get(name)
        raw = dict(raw) if raw is not None else None
    if raw is None:
        return None
    return ModelConfig.from_registry(name, raw)


def resolved_config() -> ResolvedConfig:
    """Return the currently active, resolved config across all layers.

    Secret *values* are wrapped in :class:`pydantic.SecretStr`, so they're
    masked on print / ``model_dump`` — ``print(resolved_config())`` never
    leaks a key. Read a plaintext value with
    ``resolved_config().secrets.env["KEY"].get_secret_value()``.
    """
    from nooa.layered_config import layered_paths, load_layered_yaml
    from nooa.llm_config import bundled_config_paths, llm_config_chain

    settings = load_layered_yaml(*_SETTINGS)

    secrets_raw = load_layered_yaml(*_SECRETS)
    env_raw = secrets_raw.get("env")
    env = (
        {str(k): SecretStr(str(v)) for k, v in env_raw.items()} if isinstance(env_raw, dict) else {}
    )

    llm_merged = load_layered_yaml(*_LLM, prepend=bundled_config_paths())
    models_raw = llm_merged.get("models")
    models: dict[str, ModelConfig] = {}
    if isinstance(models_raw, dict):
        for name, cfg in models_raw.items():
            if isinstance(cfg, dict):
                models[name] = ModelConfig.from_registry(name, cfg)

    return ResolvedConfig(
        models=models,
        secrets=Secrets(env=env),
        settings=settings,
        sources={
            "settings": [str(p) for p in layered_paths(*_SETTINGS)],
            "secrets": [str(p) for p in layered_paths(*_SECRETS)],
            "llm_config": [str(p) for p in llm_config_chain()],
        },
    )


__all__ = ["Secrets", "ResolvedConfig", "resolved_config", "get_model_config"]
