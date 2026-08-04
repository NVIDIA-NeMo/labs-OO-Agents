# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve local or URL-hosted LLM registry YAML files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import yaml

_MANIFEST_NAME = "nooa-registry.yaml"
_DEFAULT_CONFIG_NAME = "llm_config.yaml"
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_CONFIG_BYTES = 5 * 1024 * 1024
_HTTP_TIMEOUT_SECONDS = 30


class RegistrySourceError(ValueError):
    """A registry source could not be fetched or did not satisfy the contract."""


def _cache_root() -> Path:
    override = os.environ.get("NEMO_OO_REGISTRY_CACHE")
    if override:
        root = Path(override).expanduser()
    else:
        from nooa.paths import get_user_dir

        root = get_user_dir("registry-cache")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    return root.resolve()


def _validate_http_url(source: str) -> None:
    parsed = urlsplit(source)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RegistrySourceError("Registry URLs must use HTTP or HTTPS.")
    if parsed.username or parsed.password:
        raise RegistrySourceError(
            "Registry URLs must not contain credentials. Use an authenticated raw-file URL."
        )
    if parsed.fragment:
        raise RegistrySourceError(
            "Registry URLs must not contain fragments; put the branch or revision in the URL."
        )


def _validate_registry_yaml(content: bytes) -> None:
    try:
        data = yaml.safe_load(content)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RegistrySourceError("The registry URL did not return valid YAML.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise RegistrySourceError(
            "The registry URL must return a YAML mapping containing a 'models' mapping."
        )


def _download_registry(source: str) -> bytes:
    _validate_http_url(source)
    request = Request(  # noqa: S310 - URL scheme and credentials are validated above
        source,
        headers={
            "Accept": "application/yaml, text/yaml, text/plain",
            "User-Agent": "nooa-cli/registry-fetch",
        },
    )
    try:
        with urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310
            final_url = response.geturl()
            _validate_http_url(final_url)
            if source.startswith("https://") and not final_url.startswith("https://"):
                raise RegistrySourceError("Registry URL redirected from HTTPS to an insecure URL.")

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    if int(content_length) > _MAX_CONFIG_BYTES:
                        raise RegistrySourceError("LLM registry config is too large.")
                except ValueError:
                    pass

            content = response.read(_MAX_CONFIG_BYTES + 1)
    except RegistrySourceError:
        raise
    except HTTPError as exc:
        raise RegistrySourceError(
            "Could not fetch the registry URL. Check access to the raw YAML file."
        ) from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise RegistrySourceError(
            "Could not fetch the registry URL. Check the URL and network connection."
        ) from exc

    if len(content) > _MAX_CONFIG_BYTES:
        raise RegistrySourceError("LLM registry config is too large.")
    _validate_registry_yaml(content)
    return content


def _materialize_url_config(source: str) -> Path:
    content = _download_registry(source)
    key = hashlib.sha256(source.encode()).hexdigest()[:24]
    destination = _cache_root() / f"url-{key}.yaml"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    temporary.replace(destination)
    return destination.resolve()


def _manifest_config_path(manifest_bytes: bytes) -> Path:
    try:
        data = yaml.safe_load(manifest_bytes)
    except yaml.YAMLError as exc:
        raise RegistrySourceError(f"Invalid {_MANIFEST_NAME}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistrySourceError(f"{_MANIFEST_NAME} must be a YAML mapping.")
    version = data.get("version", 1)
    if version != 1:
        raise RegistrySourceError(f"Unsupported registry manifest version: {version!r}.")
    value = data.get("llm_config")
    if not isinstance(value, str) or not value.strip():
        raise RegistrySourceError(
            f"{_MANIFEST_NAME} must define a non-empty string 'llm_config' path."
        )
    return Path(value)


def _config_path_in_directory(directory: Path) -> Path:
    manifest = directory / _MANIFEST_NAME
    if manifest.is_file():
        if manifest.stat().st_size > _MAX_MANIFEST_BYTES:
            raise RegistrySourceError(f"{_MANIFEST_NAME} is too large.")
        relative = _manifest_config_path(manifest.read_bytes())
        if relative.is_absolute() or ".." in relative.parts:
            raise RegistrySourceError(
                "Registry manifest llm_config path must stay inside the directory."
            )
        config = (directory / relative).resolve()
        try:
            config.relative_to(directory.resolve())
        except ValueError as exc:
            raise RegistrySourceError("Registry manifest path escapes its directory.") from exc
    else:
        config = (directory / _DEFAULT_CONFIG_NAME).resolve()
    if not config.is_file():
        raise RegistrySourceError(
            f"Registry source must contain {_DEFAULT_CONFIG_NAME} or {_MANIFEST_NAME}."
        )
    if config.stat().st_size > _MAX_CONFIG_BYTES:
        raise RegistrySourceError("LLM registry config is too large.")
    return config


def resolve_registry_source(source: str | Path) -> Path:
    """Resolve one local YAML/directory or raw HTTP(S) YAML URL to a path."""
    raw = str(source).strip()
    if not raw:
        raise RegistrySourceError("Registry source cannot be empty.")

    local = Path(raw).expanduser()
    if local.is_file():
        if local.stat().st_size > _MAX_CONFIG_BYTES:
            raise RegistrySourceError("LLM registry config is too large.")
        return local.resolve()
    if local.is_dir():
        return _config_path_in_directory(local.resolve())
    if urlsplit(raw).scheme in {"http", "https"}:
        return _materialize_url_config(raw)
    raise RegistrySourceError(f"Registry source does not exist: {raw}")


def resolve_registry_sources(sources: tuple[str, ...] | list[str]) -> list[Path]:
    """Resolve registry sources in argument order (later paths override earlier ones)."""
    return [resolve_registry_source(source) for source in sources]


__all__ = ["RegistrySourceError", "resolve_registry_source", "resolve_registry_sources"]
