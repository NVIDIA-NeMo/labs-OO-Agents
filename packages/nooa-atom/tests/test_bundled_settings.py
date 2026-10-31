# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bundled-defaults settings layer: ``nooa.bundled_settings`` entry points."""

from pathlib import Path

import pytest
from nooa_atom.workspace import settings as atom_settings
from nooa_atom.workspace.options import AtomOptions
from nooa_atom.workspace.settings import load_settings_layers, load_skills_dirs

from nooa import layered_config


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """An empty home and user dir, no NEMO_OO_SETTINGS."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    return user


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A package's bundled settings.yaml with a skills directory beside it."""
    package = tmp_path / "site-packages" / "nooa_atom_nvidia"
    (package / "skills").mkdir(parents=True)
    path = package / "settings.yaml"
    path.write_text(
        "atom:\n"
        "  default_model: bundled-model\n"
        "  verbosity: concise\n"
        "  additional_skills_dirs: [skills]\n"
        "  active_skills: [nvidia.search, nvidia.docs]\n"
    )
    monkeypatch.setattr(atom_settings, "bundled_settings_paths", lambda: [path])
    return path


class _EntryPoint:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self._value = value

    def load(self) -> object:
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


def test_entry_points_name_a_path_or_a_callable_returning_one(tmp_path, monkeypatch):
    a, b, c = (tmp_path / f"{name}.yaml" for name in "abc")
    eps = [
        _EntryPoint("c_path", str(c)),
        _EntryPoint("a_callable", lambda: a),
        _EntryPoint("b_none", lambda: None),
        _EntryPoint("broken", ImportError("no such module")),
        _EntryPoint("b_callable", lambda: b),
    ]
    seen = []

    def entry_points(*, group):
        seen.append(group)
        return eps

    monkeypatch.setattr(layered_config.metadata, "entry_points", entry_points)
    assert layered_config.bundled_settings_paths() == [a, b, c]
    assert seen == ["nooa.bundled_settings"]


def test_bundled_settings_are_the_lowest_layer(tmp_path, isolated, bundle, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / ".nooa").mkdir(parents=True)
    assert load_settings_layers(workspace / ".nooa")["atom"]["default_model"] == "bundled-model"

    (isolated / "settings.yaml").write_text("atom:\n  default_model: user-model\n")
    data = load_settings_layers(workspace / ".nooa")
    assert data["atom"]["default_model"] == "user-model"
    assert data["atom"]["verbosity"] == "concise"  # untouched bundled keys remain

    (workspace / ".nooa" / "settings.yaml").write_text("atom:\n  default_model: ws-model\n")
    assert load_settings_layers(workspace / ".nooa")["atom"]["default_model"] == "ws-model"

    env = tmp_path / "env.yaml"
    env.write_text("atom:\n  default_model: env-model\n  verbosity: null\n")
    monkeypatch.setenv("NEMO_OO_SETTINGS", str(env))
    data = load_settings_layers(workspace / ".nooa")
    assert data["atom"]["default_model"] == "env-model"
    assert "verbosity" not in data["atom"]  # null deletes a bundled key too


def test_bundled_skill_dirs_are_relative_to_the_bundled_file(tmp_path, bundle):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert load_skills_dirs(workspace) == [(bundle.parent / "skills").resolve()]


def test_user_skill_dirs_extend_the_bundled_ones(tmp_path, isolated, bundle):
    workspace = tmp_path / "workspace"
    (workspace / "mine").mkdir(parents=True)
    (isolated / "settings.yaml").write_text("atom:\n  additional_skills_dirs: [mine]\n")
    assert load_skills_dirs(workspace) == [
        (bundle.parent / "skills").resolve(),
        (workspace / "mine").resolve(),
    ]

    (isolated / "settings.yaml").write_text("atom:\n  additional_skills_dirs: null\n")
    assert load_skills_dirs(workspace) == []


def test_user_active_skills_extend_the_bundled_ones(tmp_path, isolated, bundle):
    """The documented rule: a user list adds to the bundled list; inactive_skills opts out."""
    workspace = tmp_path / "workspace"
    (workspace / ".nooa").mkdir(parents=True)
    assert AtomOptions.load(workspace).active_skills == ["nvidia.search", "nvidia.docs"]

    (isolated / "settings.yaml").write_text("atom:\n  active_skills: [web, nvidia.docs]\n")
    (workspace / ".nooa" / "settings.yaml").write_text(
        "atom:\n  active_skills: [repo]\n  inactive_skills: [nvidia.search]\n"
    )
    options = AtomOptions.load(workspace)
    # Among the user, project and env layers a list still replaces (project wins
    # over user); the result is then added after the bundled list.
    assert options.active_skills == ["nvidia.search", "nvidia.docs", "repo"]
    assert options.inactive_skills == ["nvidia.search"]

    (workspace / ".nooa" / "settings.yaml").write_text("atom:\n  active_skills: null\n")
    assert AtomOptions.load(workspace).active_skills == []


def test_bundled_files_are_never_rewritten(tmp_path, monkeypatch):
    """The coding -> atom copy only touches the user's files, not a package's."""
    path = tmp_path / "bundled.yaml"
    path.write_text("coding:\n  default_model: m\n")
    monkeypatch.setattr(atom_settings, "bundled_settings_paths", lambda: [path])
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = atom_settings.load_settings_data(workspace)
    assert data["coding"] == {"default_model": "m"}
    assert path.read_text() == "coding:\n  default_model: m\n"


def test_an_invalid_bundled_file_is_named(tmp_path, bundle, caplog):
    bundle.write_text("atom:\n  active_skills: one\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert atom_settings.load_behavior_settings(workspace) == {}
    assert str(bundle.resolve()) in caplog.text


def test_resolved_config_lists_the_bundled_settings(tmp_path, bundle, monkeypatch):
    from nooa.config.resolved import resolved_config

    monkeypatch.setattr(layered_config, "bundled_settings_paths", lambda: [bundle])
    monkeypatch.setattr("nooa.llm_config.bundled_config_paths", lambda: [])
    config = resolved_config()
    assert config.sources["settings"][0] == str(bundle.resolve())
    assert config.settings["atom"]["default_model"] == "bundled-model"


def test_no_bundled_settings_change_nothing(tmp_path, isolated, monkeypatch):
    monkeypatch.setattr(atom_settings, "bundled_settings_paths", lambda: [])
    (isolated / "settings.yaml").write_text("atom:\n  active_skills: [web]\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert AtomOptions.load(workspace).active_skills == ["web"]


def test_path_type_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(
        layered_config.metadata,
        "entry_points",
        lambda *, group: [_EntryPoint("p", lambda: tmp_path), _EntryPoint("bad", lambda: 3)],
    )
    assert layered_config.entry_point_paths("g") == [Path(tmp_path)]
