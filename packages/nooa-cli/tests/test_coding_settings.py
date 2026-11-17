# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for workspace-aware coding host settings."""

import pytest
from nooa_cli.coding import load_coding_skills_dirs


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point ``Path.home()`` at an empty directory.

    ``load_coding_skills_dirs`` reads conventional user roots such as
    ``~/.agents/skills`` straight from the real home — correct in production,
    since those are third-party conventions rather than NOOA config, and so
    unaffected by NEMO_OO_USER_DIR. Without this, every assertion here depends
    on whether the developer happens to have those directories, so the suite
    passes on a clean CI runner and fails on a working machine.
    """
    home = tmp_path / "isolated-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def test_legacy_tui_section_and_config_toml_are_ignored(tmp_path, monkeypatch):
    """Only ``coding.additional_skills_dirs`` names skill roots; old TUI settings do not."""
    workspace = tmp_path / "workspace"
    skills = workspace / "nemo-oo-skills"
    user_config = tmp_path / "user-config"
    skills.mkdir(parents=True)
    user_config.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(f"tui:\n  additional_skills_dirs:\n    - {skills}\n")
    (config_dir / "config.toml").write_text('[tui]\nlibs_dirs = ["nemo-oo-skills"]\n')

    assert load_coding_skills_dirs(workspace) == []


def test_shared_coding_skill_dirs_and_workspace_conventions_are_loaded(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    configured = workspace / "shared-skills"
    conventional = workspace / ".agents" / "skills"
    user_config = tmp_path / "user-config"
    configured.mkdir(parents=True)
    conventional.mkdir(parents=True)
    user_config.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        "coding:\n  additional_skills_dirs:\n    - shared-skills\n"
    )

    assert load_coding_skills_dirs(workspace) == [
        configured.resolve(),
        conventional.resolve(),
    ]


def test_missing_configured_skill_dirs_are_ignored(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    user_config = tmp_path / "user-config"
    workspace.mkdir()
    user_config.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text("coding:\n  additional_skills_dirs:\n    - absent\n")

    assert load_coding_skills_dirs(workspace) == []


def test_environment_settings_override_user_and_workspace_layers(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    user_config = tmp_path / "user-config"
    user_skills = tmp_path / "user-skills"
    workspace_skills = tmp_path / "workspace-skills"
    override_skills = tmp_path / "override-skills"
    workspace.mkdir()
    user_config.mkdir()
    user_skills.mkdir()
    workspace_skills.mkdir()
    override_skills.mkdir()
    workspace_config = workspace / ".nooa"
    workspace_config.mkdir()
    override = tmp_path / "override.yaml"
    (user_config / "settings.yaml").write_text(
        f"coding:\n  additional_skills_dirs:\n    - {user_skills}\n"
    )
    (workspace_config / "settings.yaml").write_text(
        f"coding:\n  additional_skills_dirs:\n    - {workspace_skills}\n"
    )
    override.write_text(f"coding:\n  additional_skills_dirs:\n    - {override_skills}\n")
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.setenv("NEMO_OO_SETTINGS", str(override))

    assert load_coding_skills_dirs(workspace) == [override_skills.resolve()]


def test_an_explicit_empty_workspace_list_overrides_the_user_list(tmp_path, monkeypatch):
    """``additional_skills_dirs: []`` in the workspace means "none"."""
    workspace = tmp_path / "workspace"
    user_skills = tmp_path / "user-skills"
    user_config = tmp_path / "user-config"
    workspace.mkdir()
    user_skills.mkdir()
    user_config.mkdir()
    (user_config / "settings.yaml").write_text(
        f"coding:\n  additional_skills_dirs:\n    - {user_skills}\n"
    )
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)

    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text("coding:\n  additional_skills_dirs: []\n")

    assert load_coding_skills_dirs(workspace) == []


def test_a_non_utf8_settings_file_does_not_abort_discovery(tmp_path, monkeypatch):
    """read_text raises UnicodeError, which neither handler caught.

    A malformed workspace settings file then aborted skill discovery entirely
    instead of logging and falling back.
    """
    workspace = tmp_path / "workspace"
    conventional = workspace / ".agents" / "skills"
    user_config = tmp_path / "user-config"
    conventional.mkdir(parents=True)
    user_config.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)

    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_bytes(b"\xff\xfe coding:\n")

    assert load_coding_skills_dirs(workspace) == [conventional.resolve()]
