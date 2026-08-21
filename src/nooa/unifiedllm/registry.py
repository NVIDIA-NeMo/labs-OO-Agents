# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YAML-based model registry for UnifiedLLM.

The registry is a *thin* layer on top of litellm. Most users do not need it:
litellm already knows how to route every common public model (OpenAI, Anthropic,
Google, Azure, etc.) via its built-in model database. Pass the model name
directly to ``get_llm_client()`` and litellm handles the rest.

The registry is useful for *custom* models — models behind a proxy/gateway,
models with non-standard API keys, or convenience aliases for a team.
Define them in YAML and point the framework at them.

Discovery is *not* done here. ``unifiedllm`` is intentionally ignorant of
project filesystem conventions; the framework helper
:func:`nooa.llm_config.llm_config_chain` returns the YAML files
to load — including any bundled defaults registered by external
packages via the ``nooa.bundled_configs`` entry-point group
(install ``nemo-oo-agents-nvidia`` for the NVIDIA-gateway aliases).
The registry exposes:

- :func:`reload_registry` — load the registry from explicit paths.
- :func:`ensure_loaded` — idempotently auto-load via the framework helper.
  Called automatically on the first :func:`get_llm_client` call.
- :func:`resolve_api_key_from_config` — public helper for resolving
  ``api_key_env`` → env var value, with a WARN when the named env var
  is unset. Used by callers outside the registry (NAT plugin, viewer).

YAML schema::

    models:
      my-alias:
        model_name: openai/my-org/my-model   # exact litellm routing string
        api_base: https://my-gateway.example.com/v1
        api_key_env: MY_API_KEY
        context_window: 128000               # optional
        temperature: 0.0                     # optional
        top_p: 1.0                           # optional
        max_tokens: 4096                     # optional
        drop_params: true                    # optional, defaults to true

Set a model to ``null`` in a later layer to remove it.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from nooa.unifiedllm.requirements import LLMRequirements

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM

logger = logging.getLogger(__name__)

# Serializes the (_loaded, MODELS) check-and-mutate so concurrent
# ensure_loaded() / reload_registry() calls — possible when async agent
# tool calls fire in parallel before bootstrap finishes — cannot
# observe MODELS in a half-cleared state or both run the discovery.
# Reentrant because reload_registry() is called from inside
# ensure_loaded() while the lock is held.
_registry_lock = threading.RLock()

# NVIDIA_INTERNAL_API_KEY was renamed to NVIDIA_INFERENCE_API_KEY (the
# inference.nvidia.com endpoint is public, not internal). Config files may
# declare either name; whichever is declared, the other is accepted as a
# fallback so existing environments and older YAMLs keep working.
_NVIDIA_KEY_SYNONYMS = {
    "NVIDIA_INFERENCE_API_KEY": "NVIDIA_INTERNAL_API_KEY",
    "NVIDIA_INTERNAL_API_KEY": "NVIDIA_INFERENCE_API_KEY",
}
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPENAI_GPT5_RE = re.compile(r"^gpt-5(?:[.\-]|$)")
_RAW_PROVIDER_PREFIXES = (
    "chatgpt-",
    "claude-",
    "gemini-",
    "gpt-",
    "mistral-",
    "o1",
    "o3",
    "o4",
)


def resolve_api_key_from_config(
    model_name: str,
    config: Mapping[str, Any],
    *,
    allowed_env_vars: Collection[str] | None = None,
) -> str | None:
    """Read the api_key_env-named env var declared by a registry config.

    Returns the env var's value on success, or ``None`` if ``api_key_env``
    isn't set in ``config``. **Logs a WARN if ``api_key_env`` is set but
    the named env var is unset or empty** — callers should propagate the
    ``None`` so litellm can fall back to its own defaults, but the user
    sees a visible signal that their config promised an env var that
    doesn't exist.

    Shared helper for callers outside the registry (NAT plugin, viewer
    routes) that need the same resolution + diagnostic semantics.

    ``allowed_env_vars`` is a defense-in-depth boundary for callers handling
    partially untrusted config. When supplied, only those exact env-var names
    may be resolved.
    """
    api_key_env = config.get("api_key_env")
    if not api_key_env:
        return None
    # Defend against malformed YAML where api_key_env was written as a
    # number, list, etc. — os.getenv would raise TypeError.
    if not isinstance(api_key_env, str):
        logger.warning(
            "Model %r has invalid api_key_env %r: expected a string env var name, got %s.",
            model_name,
            api_key_env,
            type(api_key_env).__name__,
        )
        return None
    if not _ENV_VAR_NAME_RE.fullmatch(api_key_env):
        logger.warning(
            "Model %r has invalid api_key_env %r: expected a shell-style env var name.",
            model_name,
            api_key_env,
        )
        return None
    if allowed_env_vars is not None and api_key_env not in allowed_env_vars:
        logger.warning(
            "Model %r requested api_key_env %r, which is not in the caller's allowed env-var set.",
            model_name,
            api_key_env,
        )
        return None
    api_key = os.getenv(api_key_env)
    if api_key:
        return api_key
    synonym = _NVIDIA_KEY_SYNONYMS.get(api_key_env)
    if synonym:
        api_key = os.getenv(synonym)
        if api_key:
            return api_key
    logger.warning(
        "Model %r is configured to read its API key from env var %r, but "
        "that variable is unset or empty. Falling back to litellm defaults "
        "(which may use OPENAI_API_KEY).",
        model_name,
        api_key_env,
    )
    return None


def _load_models_from_yaml(path: Path) -> dict[str, dict[str, Any] | None]:
    """Load the 'models' section from a YAML config file."""
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:
        logger.exception("Failed to load config file: %s", path)
        return {}

    if not isinstance(data, dict):
        return {}

    models = data.get("models")
    if models is None:
        return {}
    if not isinstance(models, dict):
        logger.warning("'models' key is not a mapping in: %s", path)
        return {}

    return models


# MODELS is the merged view of all YAML config layers configured by the
# framework (see :func:`reload_registry` / :func:`ensure_loaded`).
# Consumers:
#   - get_llm_client() — looks up config when you pass a registered alias
#   - TUI /model and /models commands — list and autocomplete aliases
#   - External tools (eval_pipeline, nat plugin) — resolve alias → endpoint/key
#
# Starts empty. The first get_llm_client() call triggers ensure_loaded(),
# which auto-discovers via nooa.llm_config.llm_config_chain. The
# framework (TUI bootstrap) can call reload_registry() explicitly during
# startup. reload_registry() updates this dict in-place so existing
# references stay live.
# NOTE: entries are raw dicts so litellm can receive arbitrary passthrough
# kwargs and external readers (nat plugin, eval_pipeline, viewer) keep working.
# The typed boundary is ``nooa.config.ModelConfig`` /
# ``get_model_config()``. Long term this should become
# ``dict[str, ModelConfig]`` once those readers are migrated.
MODELS: dict[str, dict[str, Any]] = {}

# Set to True once reload_registry has run (with or without paths) so
# subsequent ensure_loaded() calls are no-ops.
_loaded = False


def reload_registry(*paths: Path) -> dict[str, dict[str, Any]]:
    """Reload ``MODELS`` from YAML files, last-wins.

    With explicit ``paths``: load only those files. This is the
    framework-driven path (e.g. TUI bootstrap does
    ``reload_registry(*llm_config_chain())``).

    With no arguments: **re-discover via**
    :func:`nooa.llm_config.llm_config_chain` and load.
    Matches the pre-refactor behavior — callers can edit
    ``llm_config.yaml`` or change ``NEMO_OO_LLM_CONFIG`` and call
    ``reload_registry()`` to pick the changes up. To truly clear the
    registry, use ``MODELS.clear()`` directly or pass paths that
    don't exist.

    Marks the registry as explicitly loaded — subsequent
    :func:`get_llm_client` calls will *not* trigger auto-discovery.
    ``MODELS`` is updated in place so existing references stay live.

    Thread-safe: the ``MODELS.clear()`` / ``MODELS.update()`` mutation
    is serialized with :func:`ensure_loaded` so concurrent callers
    cannot observe an empty registry between the two steps.
    """
    global _loaded

    if not paths:
        # Imported lazily so the unifiedllm package doesn't pull in
        # nooa.llm_config / paths at module-load time.
        from nooa.llm_config import llm_config_chain

        paths = tuple(llm_config_chain())

    fresh: dict[str, dict[str, Any]] = {}
    for path in paths:
        logger.debug("Loading LLM registry config from: %s", path)
        for name, cfg in _load_models_from_yaml(path).items():
            if cfg is None:
                fresh.pop(name, None)
            elif isinstance(cfg, dict):
                fresh[name] = cfg
            else:
                logger.warning(
                    "Ignoring model %r in %s: expected mapping or null, got %s",
                    name,
                    path,
                    type(cfg).__name__,
                )
                fresh.pop(name, None)

    with _registry_lock:
        MODELS.clear()
        MODELS.update(fresh)
        _loaded = True
    return MODELS


def ensure_loaded() -> None:
    """Idempotently populate ``MODELS`` via the framework's config chain.

    Called automatically from :func:`get_llm_client` so standalone
    scripts and benchmark runners (no TUI bootstrap) populate the
    registry without ceremony. Call explicitly from other entry points
    that read ``MODELS`` directly before any :func:`get_llm_client`
    call (e.g. the ``nat`` plugin).

    Once :func:`reload_registry` has been called (with or without
    paths), this is a no-op — explicit reloads win over
    auto-discovery. To force a refresh use ``reload_registry()``
    directly.

    Thread-safe: the check-and-load is serialized so two concurrent
    callers cannot both trigger discovery (and clobber each other's
    in-place mutation of ``MODELS``).
    """
    # The check + load must be atomic w.r.t. other callers; otherwise a
    # second thread can pass the ``if _loaded`` guard while the first is
    # still inside reload_registry(). RLock allows reload_registry() to
    # re-acquire the lock from within the same thread.
    with _registry_lock:
        if _loaded:
            return
        reload_registry()


def get_registry_config(name: str) -> dict[str, Any]:
    """Return a defensive snapshot of the raw registry config for *name*.

    Public accessor over the raw ``MODELS`` dict for external readers (the
    ``nat`` plugin, ``eval_pipeline``, viewer) that need the untyped
    passthrough dict without importing the module-private ``MODELS`` /
    ``_registry_lock``. Prefer :func:`nooa.config.get_model_config`
    when a typed :class:`~nooa.config.ModelConfig` is enough.

    Triggers :func:`ensure_loaded` first so standalone processes (no TUI
    bootstrap) still see registered aliases, then copies the entry under
    the registry lock so a concurrent :func:`reload_registry` — which does
    ``MODELS.clear()`` then ``update()`` — can never expose a half-cleared
    dict. Returns an empty dict when *name* is not registered.
    """
    ensure_loaded()
    with _registry_lock:
        return dict(MODELS.get(name, {}))


def _requires_tool_transport(requirements: LLMRequirements | None) -> bool:
    return bool(
        requirements is not None and (requirements.function_tools or requirements.multi_turn_tools)
    )


def _canonical_openai_model(model: str) -> str | None:
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    if "/" not in model and model.startswith(("chatgpt-", "gpt-", "o1", "o3", "o4")):
        return model
    return None


def _is_openai_gpt5_model(model: str) -> bool:
    canonical = _canonical_openai_model(model)
    return bool(canonical and _OPENAI_GPT5_RE.match(canonical))


def _is_ambiguous_raw_model(model: str) -> bool:
    if "/" in model:
        return False
    return not model.startswith(_RAW_PROVIDER_PREFIXES)


def _transport_error(model: str) -> ValueError:
    return ValueError(
        f"Cannot choose an LLM transport for model {model!r} with tool-calling "
        "requirements. Register the model in llm_config.yaml with an explicit "
        "client_type, pass client_type='completion' or client_type='responses' to "
        "get_llm_client(), or use a provider-qualified model name."
    )


def _select_client_type(
    *,
    model: str,
    configured_client_type: str,
    client_type_source: str,
    requirements: LLMRequirements | None,
) -> str:
    if configured_client_type != "completion":
        return configured_client_type
    if client_type_source in {"explicit", "registry"}:
        return configured_client_type
    if not _requires_tool_transport(requirements):
        return configured_client_type
    if _is_openai_gpt5_model(model):
        return "responses"
    if _is_ambiguous_raw_model(model):
        raise _transport_error(model)
    return configured_client_type


def _record_client_selection(
    client: UnifiedLLM,
    *,
    registry_name: str,
    config: dict[str, Any],
    client_type: str,
    client_type_source: str,
    requirements: LLMRequirements | None,
) -> None:
    client._registry_config = config  # For context_window lookup
    client._registry_name = registry_name
    client._client_type = client_type
    client._client_type_source = client_type_source
    client._llm_requirements = requirements


def resolve_llm_client_for_requirements(
    client: UnifiedLLM, requirements: LLMRequirements | None
) -> UnifiedLLM:
    """Return a client compatible with strategy requirements when auto-selection can.

    Directly constructed clients and explicit registry/client_type choices are
    treated as user-authored request plans and are left untouched. Only clients
    built by :func:`get_llm_client` through the default completion fallback are
    eligible for automatic transport upgrades.
    """
    if not _requires_tool_transport(requirements):
        return client

    from nooa.unifiedllm import CompletionClient, ResponsesClient

    if isinstance(client, ResponsesClient):
        return client
    if not isinstance(client, CompletionClient):
        return client

    client_type_source = getattr(client, "_client_type_source", "direct")
    if client_type_source == "direct":
        return client
    current_client_type = getattr(client, "_client_type", "completion")
    selected = _select_client_type(
        model=client.model,
        configured_client_type=current_client_type,
        client_type_source=client_type_source,
        requirements=requirements,
    )
    if selected == current_client_type:
        return client

    cache_key = (selected, requirements)
    cache = getattr(client, "_requirements_client_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        client._requirements_client_cache = cache
    if cache_key in cache:
        return cache[cache_key]

    replacement = ResponsesClient(
        model=client.model,
        retry_config=getattr(client, "retry_config", None),
        http_config=getattr(client, "_http_config", None),
        cache_control_injection_points=list(getattr(client, "cache_control_injection_points", [])),
        **dict(getattr(client, "config", {})),
    )
    _record_client_selection(
        replacement,
        registry_name=getattr(client, "_registry_name", ""),
        config=dict(getattr(client, "_registry_config", None) or {}),
        client_type=selected,
        client_type_source="auto",
        requirements=requirements,
    )
    cache[cache_key] = replacement
    return replacement


def get_llm_client(
    name: str,
    *,
    client_type: str | None = None,
    **overrides: Any,
) -> UnifiedLLM:
    """Create an LLM client, optionally using registry config.

    If ``name`` is a registry key, its config (model_name, endpoint, API key,
    defaults) is applied. Otherwise ``name`` is passed directly to litellm,
    which handles routing for every common public provider.

    Triggers :func:`ensure_loaded` on the first call so the registry
    auto-populates from
    :func:`nooa.llm_config.llm_config_chain` if the framework
    hasn't already called :func:`reload_registry`.

    Client class selection (first wins):
    1. Explicit ``client_type`` parameter
    2. ``client_type`` field in registry YAML
    3. Strategy ``llm_requirements`` automatic transport selection
    4. Default: ``"completion"``

    Args:
        name: Registry key or a litellm-supported model string.
        client_type: ``"completion"`` or ``"responses"``. Overrides YAML config.
        llm_requirements: Optional strategy requirements used for automatic
            transport selection. Tool-using OpenAI GPT-5 models use the
            Responses API unless an explicit ``client_type`` overrides it.
        **overrides: Override any parameter (max_tokens, temperature, etc.)

    Returns:
        Configured UnifiedLLM instance (CompletionClient or ResponsesClient).

    Example::

        # Pass-through to litellm (no registry entry needed)
        llm = get_llm_client("gpt-4o-mini")
        llm = get_llm_client("claude-sonnet-4-5-20250514")

        # Custom alias from llm_config.yaml
        llm = get_llm_client("my-internal-model", max_tokens=1000)

        # Responses API model (client_type: responses in YAML)
        llm = get_llm_client("gpt-5.3-codex")

        # Responses API without YAML — pass client_type directly
        llm = get_llm_client("openai/gpt-5.3-codex", client_type="responses")
    """
    from nooa.unifiedllm import CompletionClient, ResponsesClient, RetryConfig

    ensure_loaded()

    # Snapshot the alias's config under the lock so a concurrent
    # reload_registry() — which does MODELS.clear() then update() —
    # can't make us observe an empty dict mid-mutation. The lock is
    # held only for the lookup; CompletionClient construction
    # happens after release.
    with _registry_lock:
        config = dict(MODELS.get(name, {}))

    if config:
        model = config.get("model_name", name)
        logger.info(
            "LLM registry hit for %r → model=%r, api_base=%r", name, model, config.get("api_base")
        )
    else:
        model = name
        logger.debug("LLM registry miss for %r — passing through to litellm", name)

    params: dict[str, Any] = {
        "model": model,
        "drop_params": config.get("drop_params", True),
    }

    if "api_base" not in overrides and (api_base := config.get("api_base")):
        params["api_base"] = api_base

    # Only resolve the registry's api_key_env when the caller hasn't
    # supplied an explicit api_key — otherwise resolve_api_key_from_config
    # would emit a misleading "env var unset" warning for credentials
    # the user is already passing in directly.
    if (
        "api_key" not in overrides
        and (api_key := resolve_api_key_from_config(name, config)) is not None
    ):
        params["api_key"] = api_key

    # Copy model-specific defaults from config (overrides win).  Keep LiteLLM
    # pass-through controls here too: OpenAI-compatible gateways drop unknown
    # params unless aliases explicitly whitelist them.
    for key in (
        "temperature",
        "top_p",
        "max_tokens",
        "reasoning",
        "reasoning_effort",
        "allowed_openai_params",
        "additional_drop_params",
        "extra_body",
    ):
        if key in config and key not in overrides:
            params[key] = config[key]

    # Registry aliases can centrally tune or disable the clients' default endpoint
    # retry behavior. ``retry_config: false`` means a single attempt for every
    # endpoint error; a mapping is passed to RetryConfig(...). Explicit call-site
    # overrides still win.
    if "retry_config" in config and "retry_config" not in overrides:
        retry_config = config["retry_config"]
        if retry_config is False or retry_config is None:
            params["retry_config"] = RetryConfig(max_retries=0, rate_limit_extra_retries=0)
        elif isinstance(retry_config, dict):
            params["retry_config"] = RetryConfig(**retry_config)
        else:
            logger.warning(
                "Ignoring model %r retry_config: expected mapping, false, or null; got %s.",
                name,
                type(retry_config).__name__,
            )

    llm_requirements = overrides.pop("llm_requirements", None)
    if llm_requirements is not None and not isinstance(llm_requirements, LLMRequirements):
        raise TypeError("llm_requirements must be an LLMRequirements instance or None")

    params.update(overrides)

    # Select client class: explicit param > YAML config > requirements > default.
    client_type_source = (
        "explicit" if client_type is not None else "registry" if "client_type" in config else "auto"
    )
    configured_client_type = client_type or config.get("client_type", "completion")
    selected_client_type = _select_client_type(
        model=model,
        configured_client_type=configured_client_type,
        client_type_source=client_type_source,
        requirements=llm_requirements,
    )
    client = (
        ResponsesClient(**params)
        if selected_client_type == "responses"
        else CompletionClient(**params)
    )
    _record_client_selection(
        client,
        registry_name=name if config else "",
        config=config,
        client_type=selected_client_type,
        client_type_source=client_type_source,
        requirements=llm_requirements,
    )
    return client
