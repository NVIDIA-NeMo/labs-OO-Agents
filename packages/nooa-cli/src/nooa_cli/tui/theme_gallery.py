# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-demand, cached access to the canonical Tinted Theming scheme catalog."""

from __future__ import annotations

import io
import os
import tarfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from .theme_catalog import ThemeRecord, parse_theme

CATALOG_URL = "https://github.com/tinted-theming/schemes/archive/refs/heads/spec-0.11.tar.gz"
_ALLOWED_HOSTS = {"github.com", "codeload.github.com"}
_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
_MAX_MEMBER_BYTES = 64 * 1024
_MAX_TOTAL_BYTES = 16 * 1024 * 1024
_MAX_SCHEMES = 1024
_MAX_ARCHIVE_MEMBERS = 2048


@dataclass(frozen=True, slots=True)
class GalleryTheme:
    """One validated remote scheme plus its original YAML document."""

    id: str
    system: str
    record: ThemeRecord
    document: dict[str, object]


@dataclass(frozen=True, slots=True)
class GalleryInstall:
    """Filesystem receipt used to roll back an interrupted install."""

    path: Path
    previous: bytes | None


@dataclass(frozen=True, slots=True)
class GalleryCatalog:
    """Validated schemes and isolated diagnostics from one archive."""

    themes: dict[str, GalleryTheme]
    diagnostics: tuple[str, ...] = ()


_loaded_catalog: GalleryCatalog | None = None


def gallery_loaded() -> bool:
    """Return whether this process has loaded the remote catalog on demand."""
    return _loaded_catalog is not None


def _cache_path() -> Path:
    from nooa.paths import get_user_dir

    return get_user_dir("theme-gallery", "schemes-spec-0.11.tar.gz")


def _validate_catalog_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in _ALLOWED_HOSTS:
        raise ValueError(f"Unexpected theme catalog URL: {url}")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects away from the pinned catalog hosts before following them."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_catalog_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_archive(url: str = CATALOG_URL) -> bytes:
    """Download one bounded archive from the pinned canonical origin."""
    _validate_catalog_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "NOOA theme gallery"})
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    with opener.open(request, timeout=30) as response:
        _validate_catalog_url(response.geturl())
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > _MAX_ARCHIVE_BYTES:
            raise ValueError("Theme catalog archive exceeds 4 MiB")
        data = response.read(_MAX_ARCHIVE_BYTES + 1)
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise ValueError("Theme catalog archive exceeds 4 MiB")
    return data


def _flatten_document(document: dict[str, object]) -> dict[str, object]:
    """Flatten the current Tinted schema's nested palette for NOOA parsing."""
    palette = document.get("palette")
    if not isinstance(palette, dict):
        raise ValueError("scheme palette must be a mapping")
    return {**document, **palette}


def parse_gallery_archive(data: bytes) -> GalleryCatalog:
    """Validate all Base16/Base24 YAML schemes from a bounded tar archive."""
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise ValueError("Theme catalog archive exceeds 4 MiB")
    themes: dict[str, GalleryTheme] = {}
    diagnostics: list[str] = []
    total_bytes = 0
    seen = 0
    member_count = 0
    try:
        archive = tarfile.open(fileobj=io.BytesIO(data), mode="r|gz")
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"Invalid theme catalog archive: {exc}") from exc
    with archive:
        for member in archive:
            member_count += 1
            if member_count > _MAX_ARCHIVE_MEMBERS:
                raise ValueError("Theme catalog contains too many archive members")
            if member.size < 0 or member.size > _MAX_TOTAL_BYTES - total_bytes:
                raise ValueError("Theme catalog expands beyond 16 MiB")
            total_bytes += member.size
            parts = Path(member.name).parts
            if (
                not member.isfile()
                or len(parts) != 3
                or parts[1] not in {"base16", "base24"}
                or Path(parts[2]).suffix.lower() not in {".yaml", ".yml"}
            ):
                continue
            seen += 1
            if seen > _MAX_SCHEMES:
                raise ValueError("Theme catalog contains too many schemes")
            if member.size > _MAX_MEMBER_BYTES:
                diagnostics.append(f"Skipped {member.name}: file exceeds 64 KiB")
                continue
            try:
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("scheme file is unreadable")
                raw = extracted.read(_MAX_MEMBER_BYTES + 1)
                if len(raw) > _MAX_MEMBER_BYTES:
                    raise ValueError("scheme file exceeds 64 KiB")
                document = yaml.safe_load(raw.decode("utf-8"))
                if not isinstance(document, dict):
                    raise ValueError("scheme document must be a mapping")
                system = parts[1]
                slug = Path(parts[2]).stem.lower()
                theme_id = f"{system}-{slug}"
                normalized = _flatten_document(document)
                normalized["id"] = theme_id
                record = parse_theme(
                    normalized,
                    fallback_id=theme_id,
                    source=f"gallery:{system}",
                )
                themes[theme_id] = GalleryTheme(theme_id, system, record, document)
            except Exception as exc:
                diagnostics.append(f"Skipped {member.name}: {exc}")
    return GalleryCatalog(dict(sorted(themes.items())), tuple(diagnostics))


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def update_gallery_catalog(
    fetch: Callable[[], bytes] | None = None,
) -> GalleryCatalog:
    """Fetch, validate, cache, and activate the canonical catalog."""
    global _loaded_catalog
    data = (fetch or _fetch_archive)()
    catalog = parse_gallery_archive(data)
    if not catalog.themes:
        raise ValueError("Theme catalog contains no valid schemes")
    _atomic_write(_cache_path(), data)
    _loaded_catalog = catalog
    return catalog


def ensure_gallery_catalog(
    fetch: Callable[[], bytes] | None = None,
) -> GalleryCatalog:
    """Load the catalog lazily from memory/cache, fetching only when absent."""
    global _loaded_catalog
    if _loaded_catalog is not None:
        return _loaded_catalog
    cache = _cache_path()
    if cache.is_file():
        try:
            if cache.stat().st_size > _MAX_ARCHIVE_BYTES:
                raise ValueError("Cached theme catalog archive exceeds 4 MiB")
            with cache.open("rb") as stream:
                data = stream.read(_MAX_ARCHIVE_BYTES + 1)
            if len(data) > _MAX_ARCHIVE_BYTES:
                raise ValueError("Cached theme catalog archive exceeds 4 MiB")
            _loaded_catalog = parse_gallery_archive(data)
            return _loaded_catalog
        except Exception:
            pass
    return update_gallery_catalog(fetch)


def install_gallery_theme(entry: GalleryTheme) -> GalleryInstall:
    """Atomically install one validated gallery scheme in the user theme directory."""
    from nooa.paths import get_user_dir

    target = get_user_dir("themes", f"{entry.id}.yaml")
    previous = target.read_bytes() if target.is_file() else None
    document = dict(entry.document)
    document["id"] = entry.id
    payload = yaml.safe_dump(document, sort_keys=False, allow_unicode=True).encode("utf-8")
    if len(payload) > _MAX_MEMBER_BYTES:
        raise ValueError("Installed theme exceeds 64 KiB")
    _atomic_write(target, payload)
    return GalleryInstall(target, previous)


def rollback_gallery_install(install: GalleryInstall) -> None:
    """Restore the file state captured before an unsuccessful install."""
    if install.previous is None:
        install.path.unlink(missing_ok=True)
    else:
        _atomic_write(install.path, install.previous)
