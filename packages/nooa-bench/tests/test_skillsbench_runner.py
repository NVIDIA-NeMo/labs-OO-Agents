# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the local NOOA SkillsBench one-task runner."""

from __future__ import annotations

import json
import shutil

import pytest
from nooa_bench.skillsbench_runner import (
    ConditionResult,
    _build_nooa_runner_args,
    _condition_settings,
    _copy_nooa_source,
    _credentials,
    _install_nooa_command,
    _load_env_file,
    _load_existing_results,
    _read_activated_skills,
    _run_manifest,
    _skill_dirs,
    _task_agent_timeout,
    _translate_task_library_skills,
    _write_summary,
)


def test_load_env_file_reads_api_values_without_shell_eval(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# local credentials",
                "API_KEY='secret-key'",
                'API_URL="https://example.test/v1"',
                "IGNORED",
            ]
        )
    )

    assert _load_env_file(env_file) == {
        "API_KEY": "secret-key",
        "API_URL": "https://example.test/v1",
    }


def test_credentials_map_api_env_names(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("API_KEY=file-key\nAPI_URL=https://file.test/v1\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_URL", raising=False)

    assert _credentials(env_file) == {
        "OPENAI_API_KEY": "file-key",
        "OPENAI_BASE_URL": "https://file.test/v1",
    }


def test_copy_nooa_source_excludes_local_secrets(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (src / ".env").write_text("API_KEY=secret\n")
    (src / ".env.local").write_text("API_KEY=secret-local\n")
    (src / "client.pem").write_text("private key\n")
    (src / "module.py").write_text("print('ok')\n")

    copied = _copy_nooa_source(src)
    try:
        assert (copied / "pyproject.toml").is_file()
        assert (copied / "module.py").is_file()
        assert not (copied / ".env").exists()
        assert not (copied / ".env.local").exists()
        assert not (copied / "client.pem").exists()
    finally:
        shutil.rmtree(copied.parent, ignore_errors=True)


def test_condition_settings_keep_oracle_out_of_no_skill(tmp_path):
    task_dir = tmp_path / "citation-check"
    task_dir.mkdir()

    settings = _condition_settings(task_dir, "no_skill")

    assert settings.rollout_skill_mode == "no-skill"
    assert settings.rollout_skills_dir is None
    assert settings.runner_skill_mode == "no_skill"
    assert settings.runner_skills_dir is None


def test_condition_settings_text_skill_uses_environment_skills(tmp_path):
    task_dir = tmp_path / "citation-check"
    skills_dir = task_dir / "environment" / "skills"
    skills_dir.mkdir(parents=True)
    (task_dir / "oracle").mkdir()

    settings = _condition_settings(task_dir, "text_skill")

    assert settings.rollout_skill_mode == "with-skill"
    assert settings.rollout_skills_dir == skills_dir
    assert settings.runner_skill_mode == "text_skill"
    assert settings.runner_skills_dir == "/skills"
    assert "oracle" not in str(settings.rollout_skills_dir)


def test_condition_settings_library_skill_uses_environment_skills(tmp_path):
    task_dir = tmp_path / "citation-check"
    skills_dir = task_dir / "environment" / "skills"
    skills_dir.mkdir(parents=True)
    (task_dir / "oracle").mkdir()

    settings = _condition_settings(task_dir, "library_skill")

    assert settings.rollout_skill_mode == "with-skill"
    assert settings.rollout_skills_dir == skills_dir
    assert settings.runner_skill_mode == "library_skill"
    assert settings.runner_skills_dir == "/skills"
    assert "oracle" not in str(settings.rollout_skills_dir)


def test_condition_settings_text_skill_requires_skills_dir(tmp_path):
    task_dir = tmp_path / "citation-check"
    task_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="requires skills dir"):
        _condition_settings(task_dir, "text_skill")


def test_condition_settings_library_skill_requires_skills_dir(tmp_path):
    task_dir = tmp_path / "citation-check"
    task_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="requires skills dir"):
        _condition_settings(task_dir, "library_skill")


def test_translate_task_library_skills_writes_valid_packages(tmp_path):
    task_dir = tmp_path / "citation-check"
    skill_dir = task_dir / "environment" / "skills" / "citation-helper"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: citation-helper",
                "description: Citation helper",
                "---",
                "",
                "Use citation helper guidance.",
                "",
            ]
        )
    )

    output_dir = tmp_path / "translated"
    summaries = _translate_task_library_skills(task_dir, output_dir)

    assert [path.name for path in _skill_dirs(task_dir / "environment" / "skills")] == [
        "citation-helper"
    ]
    assert len(summaries) == 1
    assert summaries[0]["translator"] == "SlimTextSkillTranslator"
    assert summaries[0]["registry_name"] == "local.citation-helper"
    assert summaries[0]["validation"]["ok"] is True
    assert (output_dir / "citation-helper" / "pyproject.toml").is_file()
    assert (output_dir / "translation_summary.json").is_file()


def test_task_agent_timeout_reads_task_document(tmp_path):
    task_dir = tmp_path / "example-task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text(
        "\n".join(
            [
                "---",
                "schema_version: '1.3'",
                "agent:",
                "  timeout_sec: 1800.0",
                "verifier:",
                "  type: test-script",
                "  timeout_sec: 300.0",
                "sandbox:",
                "  network_mode: public",
                "---",
                "",
                "Do the task.",
            ]
        )
    )

    assert _task_agent_timeout(task_dir) == 1800


def test_build_nooa_runner_args_adds_skills_and_api_base(tmp_path):
    task_dir = tmp_path / "citation-check"
    skills_dir = task_dir / "environment" / "skills"
    skills_dir.mkdir(parents=True)
    settings = _condition_settings(task_dir, "text_skill")

    args = _build_nooa_runner_args(
        instruction="do the task",
        model="openai/openai/openai/gpt-5.2",
        settings=settings,
        agent_env={"OPENAI_BASE_URL": "https://api.test/v1"},
    )

    assert args[:3] == ["/opt/nooa-bench-venv/bin/python", "-m", "nooa_bench.runner"]
    assert args[args.index("--instruction") + 1] == "do the task"
    assert args[args.index("--model") + 1] == "openai/openai/openai/gpt-5.2"
    assert args[args.index("--skill-mode") + 1] == "text_skill"
    assert args[args.index("--skills-dir") + 1] == "/skills"
    assert args[args.index("--api-base") + 1] == "https://api.test/v1"


def test_install_nooa_command_bootstraps_missing_curl_and_uv():
    command = _install_nooa_command("/tmp/nooa src")

    assert command.startswith("set -eu;")
    assert "command -v uv" in command
    assert "command -v curl" in command
    assert "apt-get update && apt-get install -y curl ca-certificates" in command
    assert "https://astral.sh/uv/0.9.7/install.sh" in command
    assert "cd '/tmp/nooa src';" in command
    assert "UV_PROJECT_ENVIRONMENT=/opt/nooa-bench-venv uv sync --package nooa-bench" in command


def test_read_activated_skills_from_agent_result(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "result.json").write_text(
        json.dumps({"success": True, "activated_skills": ["cmd.xlsx"]})
    )

    assert _read_activated_skills(agent_dir) == ["cmd.xlsx"]


def test_run_manifest_records_non_secret_eval_context(tmp_path):
    manifest = _run_manifest(
        task="citation-check",
        model="openai/openai/openai/gpt-5.2",
        sandbox="docker",
        conditions=("no_skill", "text_skill"),
        skillsbench_dir=tmp_path / "skillsbench",
        jobs_dir=tmp_path / "jobs",
        repo_src=tmp_path,
    )

    assert manifest["task"] == "citation-check"
    assert manifest["conditions"] == ["no_skill", "text_skill"]
    assert "API_KEY" not in json.dumps(manifest)


def test_write_summary_records_pass_fail_reward_rollout_dirs_and_manifest(tmp_path):
    results = [
        ConditionResult(
            condition="no_skill",
            rollout_dir="/tmp/no",
            passed=False,
            reward=0.0,
            error="failed",
            verifier_error=None,
            agent_return_code=1,
            activated_skills=[],
        ),
        ConditionResult(
            condition="text_skill",
            rollout_dir="/tmp/text",
            passed=True,
            reward=1.0,
            error=None,
            verifier_error=None,
            agent_return_code=0,
            activated_skills=["cmd.xlsx"],
        ),
    ]
    manifest = {
        "task": "citation-check",
        "model": "model",
        "sandbox": "docker",
        "repo_commit": "abc123",
    }

    _write_summary(tmp_path, "job", results, manifest=manifest)

    payload = json.loads((tmp_path / "job" / "summary.json").read_text())
    summary_md = (tmp_path / "job" / "summary.md").read_text()

    assert payload["manifest"]["repo_commit"] == "abc123"
    assert payload["results"][0]["rollout_dir"] == "/tmp/no"
    assert payload["results"][1]["reward"] == 1.0
    assert payload["results"][1]["activated_skills"] == ["cmd.xlsx"]
    assert "- repo_commit: abc123" in summary_md
    assert "## no_skill" in summary_md
    assert "- passed: True" in summary_md


def test_load_existing_results_for_resume(tmp_path):
    results = [
        ConditionResult(
            condition="text_skill",
            rollout_dir="/tmp/text",
            passed=True,
            reward=1.0,
            error=None,
            verifier_error=None,
            agent_return_code=0,
            activated_skills=["cmd.xlsx"],
        )
    ]
    _write_summary(tmp_path, "job", results)

    loaded = _load_existing_results(tmp_path, "job")

    assert loaded["text_skill"].passed is True
    assert loaded["text_skill"].activated_skills == ["cmd.xlsx"]
