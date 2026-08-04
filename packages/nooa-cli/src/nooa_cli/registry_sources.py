# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve trusted local or Git-hosted LLM registry sources.

A registry Git repository contains either ``llm_config.yaml`` at its root or a
``nooa-registry.yaml`` manifest with a repository-relative ``llm_config`` path.
Only YAML is read; source packages from the repository are never installed or
executed.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import yaml

_MANIFEST_NAME = "nooa-registry.yaml"
_DEFAULT_CONFIG_NAME = "llm_config.yaml"
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_CONFIG_BYTES = 5 * 1024 * 1024
_GIT_TIMEOUT_SECONDS = 120
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


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


def _split_git_ref(source: str) -> tuple[str, str | None]:
    marker = "#ref="
    if marker not in source:
        return source, None
    url, ref = source.rsplit(marker, 1)
    if not url or not _SAFE_REF.fullmatch(ref) or ".." in ref or "@{" in ref:
        raise RegistrySourceError("Invalid Git ref in registry source.")
    return url, ref


def _looks_like_git_source(source: str) -> bool:
    if source.startswith("git@") and ":" in source:
        return True
    return urlsplit(source).scheme in {"file", "git", "http", "https", "ssh"}


def _validate_remote_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.password:
        raise RegistrySourceError(
            "Registry URLs must not contain credentials. Configure a Git credential helper instead."
        )


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    return env


def _run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(  # noqa: S603 - fixed executable and argument vector, no shell
            ["git", *args],
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_git_env(),
        )
    except FileNotFoundError as exc:
        raise RegistrySourceError("Git is required to use a remote registry source.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RegistrySourceError("Timed out while fetching the registry repository.") from exc
    if check and result.returncode != 0:
        raise RegistrySourceError(
            "Git could not fetch the registry repository. Verify the URL, your Git credentials, "
            "and repository access."
        )
    return result


def _cached_git_repository(source: str) -> tuple[Path, str]:
    url, ref = _split_git_ref(source)
    _validate_remote_url(url)
    key = hashlib.sha256(source.encode()).hexdigest()[:24]
    root = _cache_root()
    checkout = root / key

    if (checkout / ".git").is_dir():
        fetch_ref = ref or "HEAD"
        _run_git(
            [
                "-C",
                str(checkout),
                "fetch",
                "--depth",
                "1",
                "--no-tags",
                "origin",
                fetch_ref,
            ]
        )
        return checkout, "FETCH_HEAD"

    if checkout.exists():
        raise RegistrySourceError(
            f"Registry cache entry is not a Git checkout: {checkout}. Remove it and retry."
        )

    with tempfile.TemporaryDirectory(prefix=f"{key}-", dir=root) as temp_dir:
        staged = Path(temp_dir) / "repo"
        args = ["clone", "--depth", "1", "--no-tags", "--no-checkout"]
        if ref:
            args.extend(["--branch", ref])
        args.extend([url, str(staged)])
        _run_git(args)
        try:
            staged.replace(checkout)
        except OSError as exc:
            # Another process won the initial clone race. Its checkout is the
            # canonical cache entry; fetch it below on the next invocation.
            if not (checkout / ".git").is_dir():
                raise RegistrySourceError("Concurrent registry cache creation failed.") from exc
    return checkout, "HEAD"


def _safe_repository_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistrySourceError(
            f"{_MANIFEST_NAME} must define a non-empty string 'llm_config' path."
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise RegistrySourceError("Registry manifest llm_config path must stay inside the repository.")
    return path.as_posix()


def _git_blob(repo: Path, revision: str, path: str, *, limit: int) -> bytes | None:
    spec = f"{revision}:{path}"
    size_result = _run_git(["-C", str(repo), "cat-file", "-s", spec], check=False)
    if size_result.returncode != 0:
        return None
    try:
        size = int(size_result.stdout.strip())
    except ValueError as exc:
        raise RegistrySourceError("Git returned an invalid registry blob size.") from exc
    if size > limit:
        raise RegistrySourceError(f"Registry file {path!r} exceeds the {limit}-byte limit.")
    content = _run_git(["-C", str(repo), "show", spec])
    if len(content.stdout) > limit:
        raise RegistrySourceError(f"Registry file {path!r} exceeds the {limit}-byte limit.")
    return content.stdout


def _manifest_config_path(manifest_bytes: bytes) -> str:
    try:
        data = yaml.safe_load(manifest_bytes)
    except yaml.YAMLError as exc:
        raise RegistrySourceError(f"Invalid {_MANIFEST_NAME}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistrySourceError(f"{_MANIFEST_NAME} must be a YAML mapping.")
    version = data.get("version", 1)
    if version != 1:
        raise RegistrySourceError(f"Unsupported registry manifest version: {version!r}.")
    return _safe_repository_path(data.get("llm_config"))


def _config_path_in_directory(directory: Path) -> Path:
    manifest = directory / _MANIFEST_NAME
    if manifest.is_file():
        if manifest.stat().st_size > _MAX_MANIFEST_BYTES:
            raise RegistrySourceError(f"{_MANIFEST_NAME} is too large.")
        relative = _manifest_config_path(manifest.read_bytes())
        config = (directory / relative).resolve()
        try:
            config.relative_to(directory.resolve())
        except ValueError as exc:
            raise RegistrySourceError("Registry manifest path escapes its repository.") from exc
    else:
        config = (directory / _DEFAULT_CONFIG_NAME).resolve()
    if not config.is_file():
        raise RegistrySourceError(
            f"Registry source must contain {_DEFAULT_CONFIG_NAME} or {_MANIFEST_NAME}."
        )
    if config.stat().st_size > _MAX_CONFIG_BYTES:
        raise RegistrySourceError("LLM registry config is too large.")
    return config


def _materialize_git_config(source: str) -> Path:
    repo, revision = _cached_git_repository(source)
    manifest = _git_blob(
        repo,
        revision,
        _MANIFEST_NAME,
        limit=_MAX_MANIFEST_BYTES,
    )
    config_path = (
        _manifest_config_path(manifest) if manifest is not None else _DEFAULT_CONFIG_NAME
    )
    config = _git_blob(repo, revision, config_path, limit=_MAX_CONFIG_BYTES)
    if config is None:
        raise RegistrySourceError(
            f"Registry repository must contain {_DEFAULT_CONFIG_NAME} or a valid "
            f"{_MANIFEST_NAME}."
        )

    destination = repo / "resolved-llm-config.yaml"
    temporary = repo / f".resolved-llm-config.{os.getpid()}.tmp"
    temporary.write_bytes(config)
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    temporary.replace(destination)
    return destination.resolve()


def resolve_registry_source(source: str | Path) -> Path:
    """Resolve one local YAML/directory or cached Git repository to a YAML path."""
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
    if _looks_like_git_source(raw):
        return _materialize_git_config(raw)
    raise RegistrySourceError(f"Registry source does not exist: {raw}")


def resolve_registry_sources(sources: tuple[str, ...] | list[str]) -> list[Path]:
    """Resolve registry sources in argument order (later paths override earlier ones)."""
    return [resolve_registry_source(source) for source in sources]


__all__ = ["RegistrySourceError", "resolve_registry_source", "resolve_registry_sources"]
