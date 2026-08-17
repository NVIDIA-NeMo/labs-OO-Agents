# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`nooa_cli.tui.settings`.

Settings live in layered ``settings.yaml`` files (user → project →
``NEMO_OO_SETTINGS``), discovered through the shared
:func:`nooa.layered_config.load_layered_yaml` helper, and
round-trip with :func:`dump_settings`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from nooa_cli.tui.config import Config
from nooa_cli.tui.settings import (
    dump_settings,
    load_settings,
    settings_present,
    settings_to_dict,
)


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
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)


class TestRoundTrip:
    def test_defaults_round_trip(self, user_dir, project_dir):
        """dump → load yields the same persistable fields as the defaults."""
        original = Config()
        (user_dir / "settings.yaml").write_text(dump_settings(original))
        loaded = load_settings(Config())
        assert settings_to_dict(loaded) == settings_to_dict(original)

    def test_modified_round_trip(self, user_dir, project_dir):
        original = Config()
        original.tui.default_model = "my-model"
        original.tui.vi_mode = True
        original.tui.toolbar_items = ["model", "cwd", "session"]
        original.tui.active_skills = ["nvzurich.agent_mesh"]
        original.agent.working_dir = "/tmp"
        original.agent.summarization.preserve_recent = 123
        (user_dir / "settings.yaml").write_text(dump_settings(original))
        loaded = load_settings(Config())
        assert loaded.tui.default_model == "my-model"
        assert loaded.tui.vi_mode is True
        assert loaded.tui.toolbar_items == ["model", "cwd", "session"]
        assert loaded.tui.active_skills == ["nvzurich.agent_mesh"]
        assert loaded.agent.working_dir == "/tmp"
        assert loaded.agent.summarization.preserve_recent == 123

    def test_dump_omits_computed_skills_dirs(self):
        data = settings_to_dict(Config())
        assert "skills_dirs" not in data["tui"]
        assert "no_splash" not in data
        assert "no_trace" not in data


class TestLayering:
    def test_project_overrides_user(self, user_dir, project_dir):
        (user_dir / "settings.yaml").write_text("tui:\n  default_model: user-model\n")
        (project_dir / "settings.yaml").write_text("tui:\n  default_model: project-model\n")
        loaded = load_settings(Config())
        assert loaded.tui.default_model == "project-model"

    def test_env_overrides_all(self, user_dir, project_dir, tmp_path, monkeypatch):
        (user_dir / "settings.yaml").write_text("tui:\n  default_model: user-model\n")
        (project_dir / "settings.yaml").write_text("tui:\n  default_model: project-model\n")
        env_file = tmp_path / "env.yaml"
        env_file.write_text("tui:\n  default_model: env-model\n")
        monkeypatch.setenv("NEMO_OO_SETTINGS", str(env_file))
        loaded = load_settings(Config())
        assert loaded.tui.default_model == "env-model"

    def test_null_deletes_lower_layer_key(self, user_dir, project_dir):
        (user_dir / "settings.yaml").write_text("tui:\n  agent_spec: foo\n")
        (project_dir / "settings.yaml").write_text("tui:\n  agent_spec: null\n")
        loaded = load_settings(Config())
        # null removed the key from the merged dict → field keeps its default.
        assert loaded.tui.agent_spec is None

    def test_path_coercion(self, user_dir, project_dir):
        (user_dir / "settings.yaml").write_text(
            "tui:\n  mcp_file: custom.mcp.json\n  additional_skills_dirs:\n    - shared/skills\n"
        )
        loaded = load_settings(Config())
        assert loaded.tui.mcp_file == Path("custom.mcp.json")
        assert loaded.tui.additional_skills_dirs == [Path("shared/skills")]


class TestPresence:
    def test_absent(self, user_dir, project_dir):
        assert settings_present() is False

    def test_present_when_user_file_exists(self, user_dir, project_dir):
        (user_dir / "settings.yaml").write_text("tui:\n  vi_mode: true\n")
        assert settings_present() is True

    def test_dump_is_valid_yaml(self):
        text = dump_settings(Config())
        parsed = yaml.safe_load(text)
        assert "tui" in parsed and "agent" in parsed

    def test_toolbar_items_round_trip(self, user_dir, project_dir):
        original = Config()
        original.tui.toolbar_items = ["cwd", "model"]
        (user_dir / "settings.yaml").write_text(dump_settings(original))
        loaded = load_settings(Config())
        assert loaded.tui.toolbar_items == ["cwd", "model"]

    def test_keep_going_round_trip(self, user_dir, project_dir):
        original = Config()
        original.tui.keep_going = True
        original.tui.keep_going_model = "audit-model"
        (user_dir / "settings.yaml").write_text(dump_settings(original))
        loaded = load_settings(Config())
        assert loaded.tui.keep_going is True
        assert loaded.tui.keep_going_model == "audit-model"
