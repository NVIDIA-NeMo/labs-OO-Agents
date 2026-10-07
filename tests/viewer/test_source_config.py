# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only explicitly configured, unambiguous source plugins are loaded."""

import json
from types import SimpleNamespace

import pytest

from nooa.viewer import sources


@pytest.fixture
def source_config(tmp_path, monkeypatch):
    sources.configured_sources.cache_clear()
    path = tmp_path / "sources.json"
    monkeypatch.setenv("NOOA_VIEWER_SOURCES_CONFIG", str(path))
    yield path
    sources.configured_sources.cache_clear()


def test_plugin_discovery_is_explicit_and_cached(source_config, monkeypatch):
    loaded = []

    def factory(**kwargs):
        loaded.append(kwargs)
        return SimpleNamespace(version="1")

    def forbidden():
        raise AssertionError("Unconfigured plugins must not load")

    entries = [
        SimpleNamespace(name="selected", load=lambda: factory),
        SimpleNamespace(name="other", load=forbidden),
    ]
    monkeypatch.setattr(sources, "entry_points", lambda **kwargs: entries)
    source_config.write_text(
        json.dumps(
            {
                "sources": [
                    {"name": "example", "plugin": "selected", "options": {"setting": "value"}}
                ]
            }
        )
    )
    first = sources.configured_sources()
    assert sources.configured_sources() is first
    assert loaded == [{"name": "example", "options": {"setting": "value"}}]


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_ambiguous_plugin_fails(source_config, monkeypatch, count):
    source_config.write_text(json.dumps({"sources": [{"name": "example", "plugin": "selected"}]}))
    entries = [SimpleNamespace(name="selected") for _ in range(count)]
    monkeypatch.setattr(sources, "entry_points", lambda **kwargs: entries)
    with pytest.raises(ValueError, match="one installed"):
        sources.configured_sources()


@pytest.mark.parametrize("names", [["invalid/name"], ["example", "example"]])
def test_invalid_or_duplicate_source_names_fail(source_config, monkeypatch, names):
    source_config.write_text(
        json.dumps({"sources": [{"name": name, "plugin": "selected"} for name in names]})
    )
    monkeypatch.setattr(
        sources,
        "entry_points",
        lambda **kwargs: [
            SimpleNamespace(
                name="selected", load=lambda: lambda **kwargs: SimpleNamespace(version="1")
            )
        ],
    )
    with pytest.raises(ValueError, match="Source names"):
        sources.configured_sources()


def test_native_defaults_do_not_load_plugins(source_config, monkeypatch):
    monkeypatch.delenv("NOOA_VIEWER_SOURCES_CONFIG")

    def forbidden(**kwargs):
        raise AssertionError("No plugin discovery without configuration")

    monkeypatch.setattr(sources, "entry_points", forbidden)
    monkeypatch.setattr(sources.otlp_store, "list_experiments", lambda: ["Native"])
    assert sources.list_experiments() == ["Native"]
