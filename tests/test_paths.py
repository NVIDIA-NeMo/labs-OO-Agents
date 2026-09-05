# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nooa.paths import find_project_root, get_project_dir, get_user_dir


def test_xdg_user_dir_uses_nooa_name(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_OO_USER_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert get_user_dir() == tmp_path / "nooa"


@pytest.mark.parametrize("nested", [False, True])
def test_project_root_starts_at_working_directory(tmp_path, monkeypatch, nested):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").touch()
    cwd = project / "src" if nested else project
    cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("NEMO_OO_PROJECT_DIR", raising=False)

    assert find_project_root() == project
    assert get_project_dir("sessions") == project / ".nooa" / "sessions"


def test_project_root_uses_nearest_project(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    child = tmp_path / "child"
    child.mkdir()
    (child / "pyproject.toml").touch()
    monkeypatch.chdir(child)
    assert find_project_root() == child


def test_project_root_without_marker_falls_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_project_root() == tmp_path


def test_project_dir_honors_override(tmp_path, monkeypatch):
    override = tmp_path / "custom-nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(override))
    assert get_project_dir("settings.yaml") == override / "settings.yaml"
