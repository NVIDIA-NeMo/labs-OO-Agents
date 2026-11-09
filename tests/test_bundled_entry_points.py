# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the bundled-default entry-point discovery.

Covers :func:`nooa.layered_config.entry_point_paths`, its two group
wrappers :func:`nooa.layered_config.bundled_settings_paths` and
:func:`nooa.llm_config.bundled_config_paths`, and the bundled layer's
place (lowest priority) in the settings and LLM-config chains.

The entry points are fakes: ``importlib.metadata.entry_points`` is patched
on the ``metadata`` module ``nooa.layered_config`` calls it through, so no
installed package is consulted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from nooa import layered_config
from nooa.layered_config import (
    BUNDLED_SETTINGS_ENTRY_POINT_GROUP,
    bundled_settings_paths,
    entry_point_paths,
    layered_paths,
)
from nooa.llm_config import bundled_config_paths, llm_config_chain

_GROUP = "nooa.test_group"


class _FakeEntryPoint:
    """Stand-in for :class:`importlib.metadata.EntryPoint` (``name`` + ``load()``)."""

    def __init__(self, name: str, value: object = None, *, load_error: Exception | None = None):
        self.name = name
        self._value = value
        self._load_error = load_error

    def load(self) -> object:
        if self._load_error is not None:
            raise self._load_error
        return self._value


def _patch_entry_points(monkeypatch, groups: dict[str, list[_FakeEntryPoint]]) -> list[str]:
    """Make ``metadata.entry_points(group=...)`` return *groups*; return the groups asked for."""
    asked: list[str] = []

    def fake_entry_points(*, group: str):
        asked.append(group)
        return list(groups.get(group, []))

    monkeypatch.setattr(layered_config.metadata, "entry_points", fake_entry_points)
    return asked


def _raise(exc: Exception):
    def fn():
        raise exc

    return fn


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    d = tmp_path / "user"
    d.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(d))
    return d


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    d = tmp_path / "project"
    d.mkdir()
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(d))
    return d


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in ("NEMO_OO_SETTINGS", "NEMO_OO_LLM_CONFIG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


class TestEntryPointPaths:
    def test_sorted_by_entry_point_name(self, monkeypatch, tmp_path):
        eps = [
            _FakeEntryPoint("zeta", lambda: tmp_path / "z.yaml"),
            _FakeEntryPoint("alpha", lambda: tmp_path / "a.yaml"),
            _FakeEntryPoint("mid", lambda: tmp_path / "m.yaml"),
        ]
        _patch_entry_points(monkeypatch, {_GROUP: eps})
        assert entry_point_paths(_GROUP) == [
            tmp_path / "a.yaml",
            tmp_path / "m.yaml",
            tmp_path / "z.yaml",
        ]

    def test_callable_and_plain_values_path_or_str(self, monkeypatch, tmp_path):
        eps = [
            _FakeEntryPoint("a_call_path", lambda: tmp_path / "a.yaml"),
            _FakeEntryPoint("b_call_str", lambda: str(tmp_path / "b.yaml")),
            _FakeEntryPoint("c_plain_path", tmp_path / "c.yaml"),
            _FakeEntryPoint("d_plain_str", str(tmp_path / "d.yaml")),
        ]
        _patch_entry_points(monkeypatch, {_GROUP: eps})
        result = entry_point_paths(_GROUP)
        assert result == [tmp_path / f"{c}.yaml" for c in "abcd"]
        assert all(isinstance(p, Path) for p in result)

    def test_none_is_skipped_silently(self, monkeypatch, tmp_path, caplog):
        eps = [
            _FakeEntryPoint("a", lambda: None),
            _FakeEntryPoint("b", None),
            _FakeEntryPoint("c", lambda: tmp_path / "c.yaml"),
        ]
        _patch_entry_points(monkeypatch, {_GROUP: eps})
        with caplog.at_level(logging.WARNING, logger="nooa.layered_config"):
            assert entry_point_paths(_GROUP) == [tmp_path / "c.yaml"]
        assert caplog.records == []

    def test_non_path_return_logged_and_skipped(self, monkeypatch, tmp_path, caplog):
        eps = [
            _FakeEntryPoint("a_bad", lambda: 42),
            _FakeEntryPoint("b_good", lambda: tmp_path / "b.yaml"),
        ]
        _patch_entry_points(monkeypatch, {_GROUP: eps})
        with caplog.at_level(logging.WARNING, logger="nooa.layered_config"):
            assert entry_point_paths(_GROUP) == [tmp_path / "b.yaml"]
        msgs = [r.getMessage() for r in caplog.records]
        assert any("'a_bad'" in m and "42" in m and "not a path" in m for m in msgs), msgs

    @pytest.mark.parametrize(
        "broken",
        [
            _FakeEntryPoint("b_broken", load_error=ImportError("no module")),
            _FakeEntryPoint("b_broken", _raise(RuntimeError("boom"))),
        ],
        ids=["load-raises", "call-raises"],
    )
    def test_raising_entry_point_logged_others_still_load(
        self, monkeypatch, tmp_path, caplog, broken
    ):
        eps = [
            _FakeEntryPoint("a_good", lambda: tmp_path / "a.yaml"),
            broken,
            _FakeEntryPoint("c_good", str(tmp_path / "c.yaml")),
        ]
        _patch_entry_points(monkeypatch, {_GROUP: eps})
        with caplog.at_level(logging.WARNING, logger="nooa.layered_config"):
            assert entry_point_paths(_GROUP) == [tmp_path / "a.yaml", tmp_path / "c.yaml"]
        bad = [r for r in caplog.records if "'b_broken'" in r.getMessage()]
        assert len(bad) == 1
        assert bad[0].levelno == logging.WARNING
        assert bad[0].exc_info is not None

    def test_broken_metadata_returns_empty_with_warning(self, monkeypatch, caplog):
        def broken_entry_points(*, group: str):
            raise RuntimeError("corrupt dist-info")

        monkeypatch.setattr(layered_config.metadata, "entry_points", broken_entry_points)
        with caplog.at_level(logging.WARNING, logger="nooa.layered_config"):
            assert entry_point_paths(_GROUP) == []
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(_GROUP in m for m in msgs), msgs

    def test_empty_group(self, monkeypatch):
        _patch_entry_points(monkeypatch, {})
        assert entry_point_paths(_GROUP) == []


class TestGroupWrappers:
    def test_bundled_settings_paths_uses_settings_group(self, monkeypatch, tmp_path):
        asked = _patch_entry_points(
            monkeypatch,
            {
                "nooa.bundled_settings": [_FakeEntryPoint("s", lambda: tmp_path / "s.yaml")],
                "nooa.bundled_configs": [_FakeEntryPoint("c", lambda: tmp_path / "c.yaml")],
            },
        )
        assert BUNDLED_SETTINGS_ENTRY_POINT_GROUP == "nooa.bundled_settings"
        assert bundled_settings_paths() == [tmp_path / "s.yaml"]
        assert asked == ["nooa.bundled_settings"]

    def test_bundled_config_paths_uses_configs_group(self, monkeypatch, tmp_path):
        asked = _patch_entry_points(
            monkeypatch,
            {
                "nooa.bundled_settings": [_FakeEntryPoint("s", lambda: tmp_path / "s.yaml")],
                "nooa.bundled_configs": [_FakeEntryPoint("c", lambda: tmp_path / "c.yaml")],
            },
        )
        assert bundled_config_paths() == [tmp_path / "c.yaml"]
        assert asked == ["nooa.bundled_configs"]


def _write(path: Path, text: str = "a: 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class TestBundledLayerLowestPriority:
    def test_settings_chain_bundled_first(self, monkeypatch, tmp_path, user_dir, project_dir):
        bundled = _write(tmp_path / "bundled" / "settings.yaml")
        user = _write(user_dir / "settings.yaml")
        project = _write(project_dir / "settings.yaml")
        env = _write(tmp_path / "env" / "settings.yaml")
        monkeypatch.setenv("NEMO_OO_SETTINGS", str(env))
        _patch_entry_points(
            monkeypatch,
            {"nooa.bundled_settings": [_FakeEntryPoint("pkg", lambda: bundled)]},
        )
        chain = layered_paths("settings.yaml", "NEMO_OO_SETTINGS", prepend=bundled_settings_paths())
        assert chain == [p.resolve() for p in (bundled, user, project, env)]

    def test_resolved_config_settings_sources_bundled_first(
        self, monkeypatch, tmp_path, user_dir, project_dir
    ):
        from nooa.config.resolved import resolved_config

        bundled = _write(tmp_path / "bundled" / "settings.yaml", "k: bundled\nonly_bundled: 1\n")
        user = _write(user_dir / "settings.yaml", "k: user\n")
        _patch_entry_points(
            monkeypatch,
            {"nooa.bundled_settings": [_FakeEntryPoint("pkg", lambda: bundled)]},
        )
        cfg = resolved_config()
        assert cfg.sources["settings"] == [str(bundled.resolve()), str(user.resolve())]
        assert cfg.settings["k"] == "user"
        assert cfg.settings["only_bundled"] == 1

    def test_llm_config_chain_bundled_first(self, monkeypatch, tmp_path, user_dir, project_dir):
        bundled = _write(tmp_path / "bundled" / "llm_config.yaml", "x: {}\n")
        user = _write(user_dir / "llm_config.yaml", "y: {}\n")
        _patch_entry_points(
            monkeypatch,
            {"nooa.bundled_configs": [_FakeEntryPoint("pkg", lambda: bundled)]},
        )
        assert llm_config_chain() == [bundled.resolve(), user.resolve()]
