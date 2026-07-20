# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`nooa.secrets.load_secrets_into_env`.

Secrets live in layered ``secrets.yaml`` (user → project →
``NEMO_OO_SECRETS``) with an ``env:`` mapping pushed into ``os.environ``
non-clobbering.
"""

from __future__ import annotations

import logging
import os

import pytest

from nooa.secrets import load_secrets_into_env

_KEY = "NEMO_TEST_SECRET_KEY"
_KEY2 = "NEMO_TEST_SECRET_KEY2"


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
def _clean_env(monkeypatch):
    monkeypatch.delenv("NEMO_OO_SECRETS", raising=False)
    monkeypatch.delenv(_KEY, raising=False)
    monkeypatch.delenv(_KEY2, raising=False)


def _write(d, body):
    (d / "secrets.yaml").write_text(body)


def test_no_file_is_noop(user_dir, project_dir):
    assert load_secrets_into_env() == []


def test_loads_env_map(user_dir, project_dir):
    _write(user_dir, f"env:\n  {_KEY}: sk-123\n")
    applied = load_secrets_into_env()
    assert applied == [_KEY]
    assert os.environ[_KEY] == "sk-123"


def test_non_clobber_existing_env_wins(user_dir, project_dir, monkeypatch):
    monkeypatch.setenv(_KEY, "from-shell")
    _write(user_dir, f"env:\n  {_KEY}: from-file\n")
    applied = load_secrets_into_env()
    assert applied == []
    assert os.environ[_KEY] == "from-shell"


def test_project_overrides_user(user_dir, project_dir):
    _write(user_dir, f"env:\n  {_KEY}: user-val\n")
    _write(project_dir, f"env:\n  {_KEY}: project-val\n")
    load_secrets_into_env()
    assert os.environ[_KEY] == "project-val"


def test_idempotent(user_dir, project_dir):
    _write(user_dir, f"env:\n  {_KEY}: sk-123\n")
    assert load_secrets_into_env() == [_KEY]
    # Second call: already present → nothing applied.
    assert load_secrets_into_env() == []
    assert os.environ[_KEY] == "sk-123"


def test_value_coerced_to_str(user_dir, project_dir):
    _write(user_dir, f"env:\n  {_KEY}: 12345\n")
    load_secrets_into_env()
    assert os.environ[_KEY] == "12345"


def test_null_value_skipped(user_dir, project_dir):
    _write(user_dir, f"env:\n  {_KEY}: sk-123\n  {_KEY2}: null\n")
    applied = load_secrets_into_env()
    assert applied == [_KEY]
    assert _KEY2 not in os.environ


def test_env_var_override_layer(user_dir, project_dir, tmp_path, monkeypatch):
    _write(user_dir, f"env:\n  {_KEY}: user-val\n")
    override = tmp_path / "override.yaml"
    override.write_text(f"env:\n  {_KEY}: override-val\n")
    monkeypatch.setenv("NEMO_OO_SECRETS", str(override))
    load_secrets_into_env()
    assert os.environ[_KEY] == "override-val"


def test_non_mapping_env_warns(user_dir, project_dir, caplog):
    _write(user_dir, "env:\n  - just\n  - a list\n")
    with caplog.at_level(logging.WARNING, logger="nooa.secrets"):
        assert load_secrets_into_env() == []
    assert "not a mapping" in caplog.text
