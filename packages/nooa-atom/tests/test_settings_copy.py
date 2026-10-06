# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom copies the ``coding`` settings it used to read into its own ``atom`` section.

The TUI and ``nooa-acp`` keep reading ``coding``. The first time Atom reads
a settings file that has ``coding`` and no ``atom``, it adds an ``atom``
copy; from then on the two sections change independently.
"""

import logging
import os
from pathlib import Path

import pytest
import yaml
from nooa_atom.workspace.options import AtomOptions
from nooa_atom.workspace.settings import copy_coding_settings, write_settings_updates

PROJECT_SETTINGS = """\
# My workspace settings.
coding:
  default_model: ws-model  # the team model
  active_skills: [review]
tui:
  theme: dark
"""


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    workspace = tmp_path / "workspace"
    (workspace / ".nooa").mkdir(parents=True)
    return workspace


def _settings(workspace):
    return workspace / ".nooa" / "settings.yaml"


def test_loading_copies_coding_to_atom_and_keeps_the_rest_of_the_file(workspace):
    _settings(workspace).write_text(PROJECT_SETTINGS)

    options = AtomOptions.load(workspace)

    assert (options.default_model, options.active_skills) == ("ws-model", ["review"])
    text = _settings(workspace).read_text()
    assert text.startswith(PROJECT_SETTINGS)  # comments and layout untouched
    data = yaml.safe_load(text)
    assert (
        data["atom"]
        == data["coding"]
        == {
            "default_model": "ws-model",
            "active_skills": ["review"],
        }
    )
    assert data["tui"] == {"theme": "dark"}


def test_the_copy_happens_once_and_the_sections_then_change_independently(workspace):
    _settings(workspace).write_text(PROJECT_SETTINGS)
    AtomOptions.load(workspace)

    # The TUI changes its section afterwards; Atom keeps its own.
    data = yaml.safe_load(_settings(workspace).read_text())
    data["coding"]["default_model"] = "tui-model"
    _settings(workspace).write_text(yaml.safe_dump(data, sort_keys=False))
    before = _settings(workspace).read_text()

    assert AtomOptions.load(workspace).default_model == "ws-model"
    assert _settings(workspace).read_text() == before


def test_a_file_that_already_has_atom_is_left_alone(workspace):
    text = "coding:\n  default_model: old\natom:\n  default_model: new\n"
    _settings(workspace).write_text(text)

    assert AtomOptions.load(workspace).default_model == "new"
    assert _settings(workspace).read_text() == text


@pytest.mark.parametrize(
    "text",
    ["tui:\n  theme: dark\n", "coding:\n", "coding: not-a-mapping\n", "- a list\n", ""],
)
def test_a_file_without_a_coding_mapping_is_left_alone(workspace, text):
    _settings(workspace).write_text(text)

    copy_coding_settings(workspace / ".nooa")

    assert _settings(workspace).read_text() == text


def test_the_user_layer_is_copied_too(workspace):
    user_settings = Path(os.environ["NEMO_OO_USER_DIR"]) / "settings.yaml"
    user_settings.parent.mkdir(parents=True, exist_ok=True)
    user_settings.write_text("coding:\n  default_model: user-model\n")

    assert AtomOptions.load(workspace).default_model == "user-model"
    assert yaml.safe_load(user_settings.read_text())["atom"] == {"default_model": "user-model"}


def test_a_file_without_a_final_newline_gets_a_valid_copy(workspace):
    _settings(workspace).write_text("coding:\n  default_model: m")

    copy_coding_settings(workspace / ".nooa")

    data = yaml.safe_load(_settings(workspace).read_text())
    assert data == {"coding": {"default_model": "m"}, "atom": {"default_model": "m"}}


def test_a_flow_style_file_is_rewritten_with_the_copy(workspace):
    _settings(workspace).write_text("{coding: {default_model: m}}\n")

    copy_coding_settings(workspace / ".nooa")

    data = yaml.safe_load(_settings(workspace).read_text())
    assert data == {"coding": {"default_model": "m"}, "atom": {"default_model": "m"}}


def test_a_first_write_keeps_the_copied_coding_settings(workspace):
    _settings(workspace).write_text(PROJECT_SETTINGS)

    write_settings_updates({("atom", "default_model"): "picked"}, workspace=workspace)

    data = yaml.safe_load(_settings(workspace).read_text())
    assert data["atom"] == {"default_model": "picked", "active_skills": ["review"]}
    assert data["coding"]["default_model"] == "ws-model"


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions")
def test_a_file_that_cannot_be_written_is_reported_and_left_alone(workspace, caplog):
    _settings(workspace).write_text(PROJECT_SETTINGS)
    _settings(workspace).chmod(0o444)
    try:
        with caplog.at_level(logging.WARNING):
            options = AtomOptions.load(workspace)
    finally:
        _settings(workspace).chmod(0o644)

    assert "Could not copy the coding settings" in caplog.text
    assert _settings(workspace).read_text() == PROJECT_SETTINGS
    assert options.default_model != "ws-model"
