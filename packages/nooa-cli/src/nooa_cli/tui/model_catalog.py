# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible model-catalog discovery and local registry updates."""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

_MAX_CATALOG_BYTES = 5 * 1024 * 1024
_MAX_MODELS = 5_000
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")


class ModelCatalogError(ValueError):
    """A catalog or registry operation could not be completed safely."""


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
) -> tuple[str, list[str]]:
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

    model_ids: list[str] = []
    seen: set[str] = set()
    for item in data[: _MAX_MODELS + 1]:
        model_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if not model_id or len(model_id) > 500 or any(ord(c) < 32 for c in model_id):
            continue
        if model_id not in seen:
            seen.add(model_id)
            model_ids.append(model_id)
    if len(data) > _MAX_MODELS:
        raise ModelCatalogError(f"Model catalog contains more than {_MAX_MODELS:,} entries.")
    if not model_ids:
        raise ModelCatalogError("Model catalog did not contain any usable model IDs.")
    return api_base, sorted(model_ids, key=str.casefold)


def registry_entry(model_id: str, api_base: str, api_key_env: str = "") -> dict[str, Any]:
    """Build the minimal registry entry for an OpenAI-compatible server."""
    model_name = model_id if model_id.startswith("openai/") else f"openai/{model_id}"
    entry: dict[str, Any] = {"model_name": model_name, "api_base": api_base}
    if api_key_env:
        entry["api_key_env"] = validate_api_key_env(api_key_env)
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
    "ModelCatalogError",
    "default_alias",
    "fetch_model_catalog",
    "normalize_catalog_endpoint",
    "registry_entry",
    "suggested_api_key_env",
    "validate_alias",
    "validate_api_key_env",
    "write_model_alias",
]
