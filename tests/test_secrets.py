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

from nooa.secrets import load_secrets_into_env, write_secret_env

_KEY = "NEMO_TEST_SECRET_KEY"
_KEY2 = "NEMO_TEST_SECRET_KEY2"


def test_write_secret_preserves_values_and_does_not_export(tmp_path, monkeypatch):
    path = tmp_path / "secrets.yaml"
    path.write_text("env:\n  OTHER: old-value\nmetadata: keep\n")
    monkeypatch.delenv(_KEY, raising=False)
    write_secret_env(path, _KEY, "new-value")
    import yaml

    assert yaml.safe_load(path.read_text()) == {
        "env": {"OTHER": "old-value", _KEY: "new-value"},
        "metadata": "keep",
    }
    assert path.stat().st_mode & 0o777 == 0o600
    assert _KEY not in os.environ


def test_invalid_secret_yaml_is_not_disclosed_or_replaced(tmp_path):
    path = tmp_path / "secrets.yaml"
    original = "env: [PRIVATE-OLD-KEY"
    path.write_text(original)
    with pytest.raises(ValueError) as caught:
        write_secret_env(path, _KEY, "PRIVATE-NEW-KEY")
    assert "PRIVATE" not in str(caught.value)
    assert path.read_text() == original


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


@pytest.fixture
def refresh_state(monkeypatch):
    import nooa.secrets as secrets

    monkeypatch.setattr(secrets, "_file_env", {})
    return secrets.reload_secret_env


def test_reload_added_and_rotated_file_key(refresh_state, user_dir, project_dir):
    assert refresh_state(_KEY) is None
    _write(project_dir, f"env:\n  {_KEY}: first-key\n")
    assert refresh_state(_KEY) == "first-key"
    _write(project_dir, f"env:\n  {_KEY}: second-key\n")
    assert refresh_state(_KEY) == "second-key"
    assert os.environ[_KEY] == "second-key"


def test_reload_recognizes_keys_exported_by_startup_loader(refresh_state, user_dir, project_dir):
    _write(project_dir, f"env:\n  {_KEY}: startup-key\n  {_KEY2}: unrelated\n")
    load_secrets_into_env()
    _write(project_dir, f"env:\n  {_KEY}: rotated-key\n  {_KEY2}: changed\n")
    assert refresh_state(_KEY) == "rotated-key"
    assert os.environ[_KEY2] == "unrelated"  # Only the selected key is refreshed.


@pytest.mark.parametrize("export", ["shell-key", ""])
def test_reload_preserves_explicit_exports(
    refresh_state, user_dir, project_dir, monkeypatch, export
):
    monkeypatch.setenv(_KEY, export)
    _write(project_dir, f"env:\n  {_KEY}: file-key\n")
    load_secrets_into_env()
    assert refresh_state(_KEY) == export


def test_reload_preserves_later_process_override(refresh_state, user_dir, project_dir, monkeypatch):
    _write(project_dir, f"env:\n  {_KEY}: file-key\n")
    load_secrets_into_env()
    monkeypatch.setenv(_KEY, "process-override")
    _write(project_dir, f"env:\n  {_KEY}: rotated-key\n")
    assert refresh_state(_KEY) == "process-override"


@pytest.mark.parametrize("replacement", [None, "env: {}\n", f"env:\n  {_KEY}: null\n"])
def test_reload_removes_deleted_file_value(refresh_state, user_dir, project_dir, replacement):
    _write(project_dir, f"env:\n  {_KEY}: removed-key\n")
    load_secrets_into_env()
    if replacement is None:
        (project_dir / "secrets.yaml").unlink()
    else:
        _write(project_dir, replacement)
    assert refresh_state(_KEY) is None
    assert _KEY not in os.environ


def test_reload_uses_session_directory_and_override_layers(
    refresh_state, user_dir, project_dir, tmp_path, monkeypatch
):
    session = tmp_path / "other-session"
    session.mkdir()
    _write(user_dir, f"env:\n  {_KEY}: user-key\n")
    _write(project_dir, f"env:\n  {_KEY}: launch-key\n")
    _write(session, f"env:\n  {_KEY}: session-key\n")
    load_secrets_into_env()
    assert refresh_state(_KEY, project_dir=session) == "session-key"
    override = tmp_path / "override.yaml"
    override.write_text(f"env:\n  {_KEY}: override-key\n")
    monkeypatch.setenv("NEMO_OO_SECRETS", str(override))
    assert refresh_state(_KEY, project_dir=session) == "override-key"
    assert refresh_state(_KEY, project_dir=project_dir) == "override-key"


@pytest.mark.parametrize(
    "bad_yaml",
    [
        "env: [PRIVATE-SENTINEL",
        "[PRIVATE-SENTINEL]",
        "env: PRIVATE-SENTINEL",
        f"env:\n  {_KEY}: [PRIVATE-SENTINEL]\n",
    ],
)
def test_reload_invalid_file_never_exposes_key_or_applies_partial_data(
    refresh_state, user_dir, project_dir, caplog, bad_yaml
):
    _write(project_dir, f"env:\n  {_KEY}: previous-key\n")
    load_secrets_into_env()
    _write(project_dir, bad_yaml)
    with pytest.raises(ValueError) as caught:
        refresh_state(_KEY)
    assert "PRIVATE-SENTINEL" not in str(caught.value) + caplog.text
    assert os.environ[_KEY] == "previous-key"


def test_fresh_secret_names_do_not_export_values(refresh_state, user_dir, project_dir):
    from nooa.secrets import secret_env_names

    _write(project_dir, f"env:\n  {_KEY}: PRIVATE-SENTINEL\n  {_KEY2}: null\n")
    assert secret_env_names() == [_KEY]
    assert _KEY not in os.environ
