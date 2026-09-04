# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the on-demand Tinted theme gallery catalog."""

from __future__ import annotations

import io
import tarfile

import pytest
import yaml
from nooa_cli.tui import theme_gallery
from nooa_cli.tui.theme_gallery import (
    install_gallery_theme,
    parse_gallery_archive,
)


def _scheme(system: str = "base16", name: str = "Ocean") -> dict[str, object]:
    palette = {
        "base00": "181818",
        "base01": "282828",
        "base02": "383838",
        "base03": "585858",
        "base04": "b8b8b8",
        "base05": "d8d8d8",
        "base06": "e8e8e8",
        "base07": "f8f8f8",
        "base08": "ab4642",
        "base09": "dc9656",
        "base0A": "f7ca88",
        "base0B": "a1b56c",
        "base0C": "86c1b9",
        "base0D": "7cafc2",
        "base0E": "ba8baf",
        "base0F": "a16946",
    }
    if system == "base24":
        palette.update({f"base{index:02X}": "ffffff" for index in range(16, 24)})
    return {"system": system, "name": name, "variant": "dark", "palette": palette}


def _archive(files: dict[str, object]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for relative, document in files.items():
            payload = yaml.safe_dump(document).encode("utf-8")
            info = tarfile.TarInfo(f"schemes-spec-0.11/{relative}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _raw_archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for relative, payload in files.items():
            info = tarfile.TarInfo(f"schemes-spec-0.11/{relative}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def test_gallery_catalog_is_lazy_on_import(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path))
    monkeypatch.setattr(theme_gallery, "_loaded_catalog", None)

    assert theme_gallery.gallery_loaded() is False
    assert not theme_gallery._cache_path().exists()


def test_parse_gallery_archive_accepts_nested_base16_and_base24() -> None:
    catalog = parse_gallery_archive(
        _archive(
            {
                "base16/ocean.yaml": _scheme(),
                "base24/ayu.yaml": _scheme("base24", "Ayu"),
            }
        )
    )

    assert list(catalog.themes) == ["base16-ocean", "base24-ayu"]
    assert catalog.themes["base16-ocean"].record.name == "Ocean"
    assert catalog.themes["base24-ayu"].system == "base24"
    assert catalog.diagnostics == ()


def test_parse_gallery_archive_skips_malformed_and_unsafe_members() -> None:
    data = _archive(
        {
            "base16/good.yaml": _scheme(),
            "base16/bad.yaml": {"name": "Bad", "palette": {"base00": "000000"}},
            "../base16/escape.yaml": _scheme(name="Escape"),
        }
    )

    catalog = parse_gallery_archive(data)

    assert list(catalog.themes) == ["base16-good"]
    assert len(catalog.diagnostics) == 1
    assert "bad.yaml" in catalog.diagnostics[0]


def test_parse_gallery_archive_bounds_all_members(monkeypatch) -> None:
    monkeypatch.setattr(theme_gallery, "_MAX_ARCHIVE_MEMBERS", 1)
    data = _raw_archive({"notes/one.txt": b"one", "notes/two.txt": b"two"})

    with pytest.raises(ValueError, match="too many archive members"):
        parse_gallery_archive(data)


def test_parse_gallery_archive_bounds_ignored_expanded_data(monkeypatch) -> None:
    monkeypatch.setattr(theme_gallery, "_MAX_TOTAL_BYTES", 10)
    data = _raw_archive({"notes/readme.txt": b"x" * 11})

    with pytest.raises(ValueError, match="expands beyond"):
        parse_gallery_archive(data)


def test_oversized_disk_cache_is_replaced_without_parsing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path))
    monkeypatch.setattr(theme_gallery, "_loaded_catalog", None)
    cache = theme_gallery._cache_path()
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"x" * (theme_gallery._MAX_ARCHIVE_BYTES + 1))
    payload = _archive({"base16/ocean.yaml": _scheme()})

    catalog = theme_gallery.ensure_gallery_catalog(lambda: payload)

    assert list(catalog.themes) == ["base16-ocean"]
    assert cache.read_bytes() == payload


def test_update_fetches_once_and_gallery_reuses_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path))
    monkeypatch.setattr(theme_gallery, "_loaded_catalog", None)
    payload = _archive({"base16/ocean.yaml": _scheme()})
    calls = 0

    def fetch() -> bytes:
        nonlocal calls
        calls += 1
        return payload

    updated = theme_gallery.update_gallery_catalog(fetch)
    cached = theme_gallery.ensure_gallery_catalog(lambda: (_ for _ in ()).throw(AssertionError()))

    assert calls == 1
    assert cached is updated
    assert theme_gallery._cache_path().read_bytes() == payload


def test_gallery_uses_disk_cache_after_process_restart(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path))
    payload = _archive({"base16/ocean.yaml": _scheme()})
    path = theme_gallery._cache_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    monkeypatch.setattr(theme_gallery, "_loaded_catalog", None)

    catalog = theme_gallery.ensure_gallery_catalog(
        lambda: (_ for _ in ()).throw(AssertionError("network should not be used"))
    )

    assert list(catalog.themes) == ["base16-ocean"]


def test_install_gallery_theme_writes_source_document(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path))
    catalog = parse_gallery_archive(_archive({"base16/ocean.yaml": _scheme()}))

    install = install_gallery_theme(catalog.themes["base16-ocean"])

    assert install.path == tmp_path / "themes" / "base16-ocean.yaml"
    assert install.previous is None
    document = yaml.safe_load(install.path.read_text(encoding="utf-8"))
    assert document["name"] == "Ocean"
    assert document["id"] == "base16-ocean"


def test_install_gallery_theme_can_replace_and_roll_back(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path))
    catalog = parse_gallery_archive(_archive({"base16/ocean.yaml": _scheme()}))
    target = tmp_path / "themes" / "base16-ocean.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("existing", encoding="utf-8")

    install = install_gallery_theme(catalog.themes["base16-ocean"])
    theme_gallery.rollback_gallery_install(install)

    assert install.previous == b"existing"
    assert target.read_text(encoding="utf-8") == "existing"
