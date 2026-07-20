# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`nooa.llm_config.llm_config_chain`.

The chain layers, lowest priority first:

1. ``get_user_dir("llm_config.yaml")``
2. ``NEMO_OO_LLM_CONFIG`` env var (comma-separated)
3. ``get_project_dir("llm_config.yaml")``

These tests use the ``NEMO_OO_USER_DIR`` / ``NEMO_OO_PROJECT_DIR``
env-var overrides exposed by :mod:`nooa.paths` so they
never touch the real user config directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from nooa.llm_config import llm_config_chain


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    """Isolated user-config directory."""
    d = tmp_path / "user"
    d.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(d))
    return d


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    """Isolated project-config directory."""
    d = tmp_path / "project"
    d.mkdir()
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(d))
    return d


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Ensure NEMO_OO_LLM_CONFIG is unset, CWD is a clean temp dir,
    and bundled-default entry-points are stubbed empty.

    Tests that need to exercise the bundled layer explicitly
    re-patch :func:`bundled_config_paths`.
    """
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    # Stub out the entry-point lookup so tests don't pick up the
    # workspace-installed nemo-oo-agents-nvidia bundled YAML.
    monkeypatch.setattr("nooa.llm_config.bundled_config_paths", lambda: [])
    monkeypatch.chdir(tmp_path)


def _write_yaml(path: Path, alias: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"models:\n  {alias}:\n    model_name: m\n")


class TestEmpty:
    def test_no_files_no_env_returns_empty(self, user_dir, project_dir):
        assert llm_config_chain() == []


class TestSingleLayer:
    def test_user_only(self, user_dir, project_dir):
        path = user_dir / "llm_config.yaml"
        _write_yaml(path)
        chain = llm_config_chain()
        assert chain == [path.resolve()]

    def test_project_only(self, user_dir, project_dir):
        path = project_dir / "llm_config.yaml"
        _write_yaml(path)
        chain = llm_config_chain()
        assert chain == [path.resolve()]

    def test_env_only(self, user_dir, project_dir, tmp_path, monkeypatch):
        path = tmp_path / "from_env.yaml"
        _write_yaml(path)
        monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(path))
        chain = llm_config_chain()
        assert chain == [path.resolve()]


class TestLayering:
    def test_user_and_project_order(self, user_dir, project_dir):
        u = user_dir / "llm_config.yaml"
        p = project_dir / "llm_config.yaml"
        _write_yaml(u)
        _write_yaml(p)
        chain = llm_config_chain()
        # Project is higher priority — must be last.
        assert chain == [u.resolve(), p.resolve()]

    def test_env_var_is_highest_priority(self, user_dir, project_dir, tmp_path, monkeypatch):
        """`NEMO_OO_LLM_CONFIG` is the global override — last in the chain,
        winning over both user and project."""
        u = user_dir / "llm_config.yaml"
        p = project_dir / "llm_config.yaml"
        e1 = tmp_path / "e1.yaml"
        e2 = tmp_path / "e2.yaml"
        for path in (u, p, e1, e2):
            _write_yaml(path)
        monkeypatch.setenv("NEMO_OO_LLM_CONFIG", f"{e1},{e2}")
        chain = llm_config_chain()
        # Order: user → project → env entries (highest at the end).
        assert chain == [u.resolve(), p.resolve(), e1.resolve(), e2.resolve()]

    def test_whitespace_entries_ignored(self, user_dir, project_dir, tmp_path, monkeypatch):
        e = tmp_path / "e.yaml"
        _write_yaml(e)
        monkeypatch.setenv("NEMO_OO_LLM_CONFIG", f" , {e} , ,")
        chain = llm_config_chain()
        assert chain == [e.resolve()]


class TestDedup:
    def test_user_path_repeated_in_env_keeps_higher(self, user_dir, project_dir, monkeypatch):
        u = user_dir / "llm_config.yaml"
        _write_yaml(u)
        monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(u))
        chain = llm_config_chain()
        # Same path appears in user (low) and env-var (highest).
        # Lower-priority occurrence is dropped; the env-var slot wins.
        assert chain == [u.resolve()]

    def test_env_var_duplicate_entries_collapse_to_last(
        self, user_dir, project_dir, tmp_path, monkeypatch
    ):
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        _write_yaml(a)
        _write_yaml(b)
        monkeypatch.setenv("NEMO_OO_LLM_CONFIG", f"{a},{b},{a}")
        chain = llm_config_chain()
        # 'a' appears twice; only its last occurrence is kept.
        assert chain == [b.resolve(), a.resolve()]

    def test_symlink_dedup(self, user_dir, project_dir, tmp_path, monkeypatch):
        target = tmp_path / "target.yaml"
        _write_yaml(target)
        link = user_dir / "llm_config.yaml"
        link.symlink_to(target)
        monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(target))
        chain = llm_config_chain()
        # Symlinked user-dir entry resolves to the same real path as
        # the env-var entry. Lower-priority occurrence is dropped.
        assert chain == [target.resolve()]


class TestMissing:
    def test_env_var_missing_path_warns(self, user_dir, project_dir, monkeypatch, caplog):
        monkeypatch.setenv("NEMO_OO_LLM_CONFIG", "/nope/nope.yaml")
        with caplog.at_level(logging.WARNING, logger="nooa.llm_config"):
            chain = llm_config_chain()
        assert chain == []
        assert "does not exist" in caplog.text

    def test_user_missing_silent(self, user_dir, project_dir, caplog):
        # User-dir file simply not present — no warning expected.
        with caplog.at_level(logging.WARNING, logger="nooa.llm_config"):
            chain = llm_config_chain()
        assert chain == []
        assert "does not exist" not in caplog.text


class TestBundledDefaults:
    """Bundled defaults are the lowest-priority layer.

    External packages register their YAML under the
    ``nooa.bundled_configs`` entry-point group; tests
    monkeypatch :func:`bundled_config_paths` to stub providers in/out.
    """

    def test_bundled_included_in_chain(self, user_dir, project_dir, tmp_path, monkeypatch):
        # Stub the entry-point lookup to return one synthetic bundled YAML.
        synthetic = tmp_path / "synthetic.yaml"
        _write_yaml(synthetic)
        monkeypatch.setattr(
            "nooa.llm_config.bundled_config_paths",
            lambda: [synthetic],
        )
        chain = llm_config_chain()
        assert chain[0] == synthetic.resolve()

    def test_bundled_absent_chain_empty(self, user_dir, project_dir):
        # The autouse fixture already stubs bundled to [].
        assert llm_config_chain() == []

    def test_bundled_then_project(self, user_dir, project_dir, tmp_path, monkeypatch):
        synthetic = tmp_path / "synthetic.yaml"
        _write_yaml(synthetic)
        monkeypatch.setattr(
            "nooa.llm_config.bundled_config_paths",
            lambda: [synthetic],
        )
        project_yaml = project_dir / "llm_config.yaml"
        _write_yaml(project_yaml)
        chain = llm_config_chain()
        # Bundled is first (lowest), project is last (highest).
        assert chain[0] == synthetic.resolve()
        assert chain[-1] == project_yaml.resolve()

    def test_multiple_bundled_providers_all_included(
        self, user_dir, project_dir, tmp_path, monkeypatch
    ):
        """Two providers registered → both load, in the order returned."""
        first = tmp_path / "first.yaml"
        second = tmp_path / "second.yaml"
        _write_yaml(first)
        _write_yaml(second)
        monkeypatch.setattr(
            "nooa.llm_config.bundled_config_paths",
            lambda: [first, second],
        )
        chain = llm_config_chain()
        assert chain == [first.resolve(), second.resolve()]
