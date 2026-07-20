# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`nooa.layered_config`.

Two entry points:

- :func:`layered_paths` — priority-ordered, deduplicated list of
  existing YAML paths (lowest priority first).
- :func:`load_layered_yaml` — deep-merge of every layer into one dict
  (last wins; ``null`` deletes).

Like ``test_llm_config_chain``, these use the ``NEMO_OO_USER_DIR`` /
``NEMO_OO_PROJECT_DIR`` overrides so they never touch the real config
directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from nooa.layered_config import layered_paths, load_layered_yaml

_ENV_VAR = "NEMO_OO_SETTINGS"
_FILENAME = "settings.yaml"


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
    monkeypatch.delenv(_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


# ── layered_paths ──────────────────────────────────────────────────────────


class TestLayeredPaths:
    def test_empty(self, user_dir, project_dir):
        assert layered_paths(_FILENAME, _ENV_VAR) == []

    def test_user_only(self, user_dir, project_dir):
        p = user_dir / _FILENAME
        _write(p, "a: 1\n")
        assert layered_paths(_FILENAME, _ENV_VAR) == [p.resolve()]

    def test_user_then_project_order(self, user_dir, project_dir):
        u = user_dir / _FILENAME
        pr = project_dir / _FILENAME
        _write(u, "a: 1\n")
        _write(pr, "a: 2\n")
        assert layered_paths(_FILENAME, _ENV_VAR) == [u.resolve(), pr.resolve()]

    def test_env_highest(self, user_dir, project_dir, tmp_path, monkeypatch):
        u = user_dir / _FILENAME
        pr = project_dir / _FILENAME
        e = tmp_path / "e.yaml"
        for x in (u, pr, e):
            _write(x, "a: 1\n")
        monkeypatch.setenv(_ENV_VAR, str(e))
        assert layered_paths(_FILENAME, _ENV_VAR) == [u.resolve(), pr.resolve(), e.resolve()]

    def test_prepend_is_lowest(self, user_dir, project_dir, tmp_path):
        bundled = tmp_path / "bundled.yaml"
        u = user_dir / _FILENAME
        _write(bundled, "a: 1\n")
        _write(u, "a: 2\n")
        chain = layered_paths(_FILENAME, _ENV_VAR, prepend=[bundled])
        assert chain == [bundled.resolve(), u.resolve()]

    def test_no_env_var_disables_layer(self, user_dir, project_dir, tmp_path, monkeypatch):
        e = tmp_path / "e.yaml"
        _write(e, "a: 1\n")
        monkeypatch.setenv(_ENV_VAR, str(e))
        # env_var=None → env layer ignored entirely
        assert layered_paths(_FILENAME, None) == []

    def test_resolved_path_dedup_keeps_higher(self, user_dir, project_dir, monkeypatch):
        u = user_dir / _FILENAME
        _write(u, "a: 1\n")
        monkeypatch.setenv(_ENV_VAR, str(u))
        # Same path in user (low) + env (high) → collapses to the env slot.
        assert layered_paths(_FILENAME, _ENV_VAR) == [u.resolve()]

    def test_env_missing_path_warns(self, user_dir, project_dir, monkeypatch, caplog):
        monkeypatch.setenv(_ENV_VAR, "/nope/nope.yaml")
        with caplog.at_level(logging.WARNING, logger="nooa.layered_config"):
            assert layered_paths(_FILENAME, _ENV_VAR) == []
        assert "does not exist" in caplog.text


# ── load_layered_yaml ──────────────────────────────────────────────────────


class TestLoadLayeredYaml:
    def test_empty_returns_empty_dict(self, user_dir, project_dir):
        assert load_layered_yaml(_FILENAME, _ENV_VAR) == {}

    def test_single_layer(self, user_dir, project_dir):
        _write(user_dir / _FILENAME, "model: foo\nvi: true\n")
        assert load_layered_yaml(_FILENAME, _ENV_VAR) == {"model": "foo", "vi": True}

    def test_last_wins_scalar(self, user_dir, project_dir):
        _write(user_dir / _FILENAME, "model: foo\n")
        _write(project_dir / _FILENAME, "model: bar\n")
        assert load_layered_yaml(_FILENAME, _ENV_VAR)["model"] == "bar"

    def test_deep_merge_nested(self, user_dir, project_dir):
        _write(
            user_dir / _FILENAME, "summarization:\n  policy: token_budget\n  preserve_recent: 50\n"
        )
        _write(project_dir / _FILENAME, "summarization:\n  preserve_recent: 99\n")
        merged = load_layered_yaml(_FILENAME, _ENV_VAR)
        assert merged["summarization"] == {"policy": "token_budget", "preserve_recent": 99}

    def test_null_deletes_key(self, user_dir, project_dir):
        _write(user_dir / _FILENAME, "model: foo\nvi: true\n")
        _write(project_dir / _FILENAME, "vi: null\n")
        merged = load_layered_yaml(_FILENAME, _ENV_VAR)
        assert merged == {"model": "foo"}

    def test_non_mapping_skipped(self, user_dir, project_dir, caplog):
        _write(user_dir / _FILENAME, "model: foo\n")
        _write(project_dir / _FILENAME, "- just\n- a\n- list\n")
        with caplog.at_level(logging.WARNING, logger="nooa.layered_config"):
            merged = load_layered_yaml(_FILENAME, _ENV_VAR)
        assert merged == {"model": "foo"}
        assert "not a YAML mapping" in caplog.text

    def test_empty_file_skipped(self, user_dir, project_dir):
        _write(user_dir / _FILENAME, "model: foo\n")
        _write(project_dir / _FILENAME, "")
        assert load_layered_yaml(_FILENAME, _ENV_VAR) == {"model": "foo"}
