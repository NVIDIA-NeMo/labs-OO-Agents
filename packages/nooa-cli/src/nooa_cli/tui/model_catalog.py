# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible model-catalog discovery and local registry updates."""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

_MAX_CATALOG_BYTES = 5 * 1024 * 1024
_MAX_MODELS = 5_000
_MAX_TOKEN_LIMIT = 100_000_000
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_TOKEN_LIMIT = re.compile(r"^(\d+(?:\.\d+)?)([km]?)$", re.IGNORECASE)


@dataclass(frozen=True)
class CatalogModel:
    """One model advertised by an OpenAI-compatible ``/models`` endpoint."""

    id: str
    context_window: int | None = None
    max_tokens: int | None = None


class ModelCatalogError(ValueError):
    """A catalog or registry operation could not be completed safely."""


def _coerce_catalog_limit(value: Any) -> int | None:
    """Return a plausible positive token count from endpoint metadata."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        result = int(value.strip())
    else:
        return None
    return result if 0 < result <= _MAX_TOKEN_LIMIT else None


def parse_optional_token_limit(value: str, label: str) -> int | None:
    """Parse an optional human token count such as ``131072`` or ``128k``."""
    raw = value.strip().replace(",", "").replace("_", "")
    if not raw:
        return None
    match = _TOKEN_LIMIT.fullmatch(raw)
    if not match:
        raise ModelCatalogError(f"{label} must be a positive token count such as 131072 or 128k.")
    number = float(match.group(1))
    suffix = match.group(2).lower()
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[suffix]
    result = int(number * multiplier)
    if result <= 0 or result > _MAX_TOKEN_LIMIT:
        raise ModelCatalogError(f"{label} must be between 1 and {_MAX_TOKEN_LIMIT:,} tokens.")
    return result


def normalize_catalog_endpoint(value: str) -> tuple[str, str]:
    """Return ``(api_base, models_url)`` for an HTTP(S) API URL."""
    raw = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ModelCatalogError("Server URL must be an absolute http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ModelCatalogError("Put credentials in an environment variable, not in the URL.")
    if parsed.query or parsed.fragment:
        raise ModelCatalogError("Server URL cannot contain a query string or fragment.")

    path = parsed.path.rstrip("/")
    if path.endswith("/models"):
        api_path = path[: -len("/models")]
        models_path = path
    else:
        api_path = path
        models_path = f"{path}/models"
    api_base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, api_path, "", ""))
    models_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, models_path, "", ""))
    return api_base.rstrip("/"), models_url


def suggested_api_key_env(api_base: str) -> str:
    """Suggest a conventional API-key variable without reading its value."""
    hostname = (urllib.parse.urlsplit(api_base).hostname or "").lower()
    if hostname == "nvidia.com" or hostname.endswith(".nvidia.com"):
        return "NVIDIA_INFERENCE_API_KEY"
    if hostname == "api.openai.com" or hostname.endswith(".openai.com"):
        return "OPENAI_API_KEY"
    return ""


def validate_api_key_env(value: str) -> str:
    """Validate and normalize an optional shell environment-variable name."""
    name = value.strip()
    if name and not _ENV_NAME.fullmatch(name):
        raise ModelCatalogError(
            "API key environment variable must be a shell name such as OPENAI_API_KEY."
        )
    return name


def validate_alias(value: str) -> str:
    """Validate a concise YAML registry alias."""
    alias = value.strip()
    if not _ALIAS.fullmatch(alias):
        raise ModelCatalogError(
            "Alias must start with a letter or number and contain only letters, numbers, ., _, /, or -."
        )
    return alias


def default_alias(model_id: str) -> str:
    """Derive a valid, concise alias from a provider model ID."""
    candidate = model_id.strip().removeprefix("openai/").strip("/")
    candidate = re.sub(r"[^A-Za-z0-9._/-]+", "-", candidate).strip("-./_")
    return candidate[:128] or "custom-model"


def fetch_model_catalog(
    server_url: str,
    *,
    api_key: str | None = None,
    timeout: float = 15.0,
) -> tuple[str, list[CatalogModel]]:
    """Fetch a bounded OpenAI-compatible ``GET /models`` catalog."""
    import httpx

    api_base, models_url = normalize_catalog_endpoint(server_url)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", models_url, headers=headers) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > _MAX_CATALOG_BYTES:
                        raise ModelCatalogError("Model catalog is larger than 5 MiB.")
                    chunks.append(chunk)
    except ModelCatalogError:
        raise
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in {401, 403}:
            raise ModelCatalogError(
                f"Model catalog rejected authentication (HTTP {status}). Check the API key environment variable."
            ) from exc
        raise ModelCatalogError(f"Model catalog request failed with HTTP {status}.") from exc
    except httpx.HTTPError as exc:
        raise ModelCatalogError(f"Could not reach model catalog: {exc}") from exc

    try:
        payload = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCatalogError("Model catalog did not return valid JSON.") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ModelCatalogError("Model catalog must use the OpenAI shape: {'data': [{'id': ...}]}.")

    models: dict[str, CatalogModel] = {}
    for item in data[: _MAX_MODELS + 1]:
        model_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if not model_id or len(model_id) > 500 or any(ord(c) < 32 for c in model_id):
            continue
        context_window = next(
            (
                limit
                for field in (
                    "context_window",
                    "max_model_len",
                    "context_length",
                    "max_input_tokens",
                )
                if (limit := _coerce_catalog_limit(item.get(field))) is not None
            ),
            None,
        )
        max_tokens = next(
            (
                limit
                for field in ("max_output_tokens", "max_completion_tokens")
                if (limit := _coerce_catalog_limit(item.get(field))) is not None
            ),
            None,
        )
        previous = models.get(model_id)
        models[model_id] = CatalogModel(
            id=model_id,
            context_window=(previous.context_window if previous else None) or context_window,
            max_tokens=(previous.max_tokens if previous else None) or max_tokens,
        )
    if len(data) > _MAX_MODELS:
        raise ModelCatalogError(f"Model catalog contains more than {_MAX_MODELS:,} entries.")
    if not models:
        raise ModelCatalogError("Model catalog did not contain any usable model IDs.")
    return api_base, sorted(models.values(), key=lambda model: model.id.casefold())


def lookup_model_token_limits(model_id: str) -> tuple[int | None, int | None]:
    """Look up context and output limits in LiteLLM's local model database."""
    try:
        import litellm
    except ImportError:
        return None, None

    parts = model_id.split("/")
    candidates = [model_id]
    if not model_id.startswith("openai/"):
        candidates.append(f"openai/{model_id}")
    if len(parts) > 1:
        candidates.extend(("/".join(parts[1:]), parts[-1]))

    context_window: int | None = None
    max_tokens: int | None = None
    for candidate in dict.fromkeys(candidates):
        try:
            # LiteLLM prints provider help for unknown names; model setup owns
            # the UI and should remain quiet when a metadata lookup misses.
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                info = litellm.get_model_info(candidate)
        except Exception:
            continue
        if not isinstance(info, dict):
            continue
        context_window = context_window or _coerce_catalog_limit(
            info.get("max_input_tokens") or info.get("context_window")
        )
        max_tokens = max_tokens or _coerce_catalog_limit(info.get("max_output_tokens"))
        if context_window is not None and max_tokens is not None:
            break
    return context_window, max_tokens


def registry_entry(
    model_id: str,
    api_base: str,
    api_key_env: str = "",
    *,
    context_window: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build the minimal registry entry for an OpenAI-compatible server."""
    model_name = model_id if model_id.startswith("openai/") else f"openai/{model_id}"
    entry: dict[str, Any] = {"model_name": model_name, "api_base": api_base}
    if api_key_env:
        entry["api_key_env"] = validate_api_key_env(api_key_env)
    if context_window is not None:
        if context_window <= 0 or context_window > _MAX_TOKEN_LIMIT:
            raise ModelCatalogError("Context window is outside the supported range.")
        entry["context_window"] = context_window
    if max_tokens is not None:
        if max_tokens <= 0 or max_tokens > _MAX_TOKEN_LIMIT:
            raise ModelCatalogError("Maximum output tokens is outside the supported range.")
        if context_window is not None and max_tokens > context_window:
            raise ModelCatalogError("Maximum output tokens cannot exceed the context window.")
        entry["max_tokens"] = max_tokens
    return entry


def write_model_alias(path: Path, alias: str, entry: dict[str, Any]) -> Path:
    """Add one alias while preserving comments and existing registry entries."""
    import yaml

    alias = validate_alias(alias)
    path = Path(path)
    source = path.read_text() if path.exists() else ""
    try:
        loaded = yaml.safe_load(source) if source.strip() else {}
    except yaml.YAMLError as exc:
        raise ModelCatalogError(f"Existing registry is not valid YAML: {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ModelCatalogError("Existing registry root must be a YAML mapping.")
    models = loaded.get("models")
    if models is not None and not isinstance(models, dict):
        raise ModelCatalogError("Existing registry 'models' value must be a mapping.")
    if isinstance(models, dict) and alias in models:
        raise ModelCatalogError(f"Model alias '{alias}' already exists in {path}.")

    dumped = yaml.safe_dump({alias: entry}, sort_keys=False, default_flow_style=False).rstrip()
    indented = "\n".join(f"  {line}" for line in dumped.splitlines()) + "\n"
    if not source.strip():
        updated = f"models:\n{indented}"
    else:
        document = yaml.compose(source)
        models_node = None
        if document is not None and hasattr(document, "value"):
            for key_node, value_node in document.value:
                if getattr(key_node, "value", None) == "models":
                    models_node = value_node
                    break
        if models_node is None:
            separator = "" if source.endswith("\n") else "\n"
            updated = f"{source}{separator}models:\n{indented}"
        elif getattr(models_node, "flow_style", False):
            # Flow mappings cannot accept an indented child. Re-render only
            # that mapping value; leave every other top-level key untouched.
            flow_models = dict(models or {})
            flow_models[alias] = entry
            flow_dump = yaml.safe_dump(
                flow_models, sort_keys=False, default_flow_style=False
            ).rstrip()
            replacement = "\n" + "\n".join(f"  {line}" for line in flow_dump.splitlines()) + "\n"
            start = models_node.start_mark.index
            end = models_node.end_mark.index
            updated = f"{source[:start]}{replacement}{source[end:]}"
        else:
            insertion = models_node.end_mark.index
            prefix = "" if insertion == 0 or source[insertion - 1] == "\n" else "\n"
            updated = f"{source[:insertion]}{prefix}{indented}{source[insertion:]}"

    try:
        reparsed = yaml.safe_load(updated)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive invariant
        raise ModelCatalogError(f"Could not construct a valid registry update: {exc}") from exc
    if not isinstance(reparsed, dict) or reparsed.get("models", {}).get(alias) != entry:
        raise ModelCatalogError("Could not verify the updated registry.")

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            stream.write(updated)
            temp_name = stream.name
        os.replace(temp_name, path)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
    return path


__all__ = [
    "CatalogModel",
    "ModelCatalogError",
    "default_alias",
    "fetch_model_catalog",
    "lookup_model_token_limits",
    "normalize_catalog_endpoint",
    "parse_optional_token_limit",
    "registry_entry",
    "suggested_api_key_env",
    "validate_alias",
    "validate_api_key_env",
    "write_model_alias",
]
