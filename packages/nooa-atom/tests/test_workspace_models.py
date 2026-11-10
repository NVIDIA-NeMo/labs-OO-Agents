# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A session reads the model configuration of its own workspace."""

import os
import time
from pathlib import Path

import pytest
from atom_test_agents import EchoAgent
from nooa_atom.agent.factory import default_llm_factory
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from nooa_atom.workspace.models import llm_config_files, workspace_llm_client, workspace_models


def write_models(path: Path, **aliases: str) -> Path:
    """Write an ``llm_config.yaml`` mapping each alias to a ``model_name``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["models:"]
    for alias, model_name in aliases.items():
        lines += [f"  {alias}:", f"    model_name: {model_name}"]
    path.write_text("\n".join(lines) + "\n")
    # A same-size rewrite within one clock tick must still be seen.
    stamp = time.time_ns() + 2_000_000_000
    os.utime(path, ns=(stamp, stamp))
    return path


def workspace_file(workspace: Path) -> Path:
    return workspace / ".nooa" / "llm_config.yaml"


@pytest.fixture(autouse=True)
def _no_outside_config(tmp_path, monkeypatch):
    """No env file and an empty package-project directory unless a test adds one."""
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "package-project"))


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


def test_a_workspace_only_alias_is_offered_and_resolves(workspace):
    write_models(workspace_file(workspace), mine="openai/mine-model")
    assert workspace_models(workspace)["mine"]["model_name"] == "openai/mine-model"
    assert workspace_llm_client("mine", workspace).model == "openai/mine-model"


def test_an_alias_added_later_is_seen_without_a_restart(workspace):
    write_models(workspace_file(workspace), first="openai/first")
    assert "second" not in workspace_models(workspace)
    write_models(workspace_file(workspace), first="openai/first", second="openai/second")
    assert workspace_models(workspace)["second"]["model_name"] == "openai/second"


def test_a_workspace_file_created_later_is_seen(workspace):
    assert "mine" not in workspace_models(workspace)
    write_models(workspace_file(workspace), mine="openai/mine-model")
    assert "mine" in workspace_models(workspace)


def test_unchanged_files_are_not_read_again(workspace, monkeypatch):
    import nooa_atom.workspace.models as models

    write_models(workspace_file(workspace), mine="openai/mine-model")
    workspace_models(workspace)
    reads: list[Path] = []
    real = models._read_models
    monkeypatch.setattr(models, "_read_models", lambda path: reads.append(path) or real(path))
    workspace_models(workspace)
    assert reads == []


def test_the_env_file_wins_then_the_workspace_then_the_user(
    workspace, tmp_path, _user_dir, monkeypatch
):
    user = write_models(_user_dir / "llm_config.yaml", shared="openai/user", only_user="u")
    project = write_models(tmp_path / "package-project" / "llm_config.yaml", shared="openai/pkg")
    own = write_models(workspace_file(workspace), shared="openai/workspace")
    assert workspace_models(workspace)["shared"]["model_name"] == "openai/workspace"
    assert workspace_models(workspace)["only_user"]["model_name"] == "u"
    assert llm_config_files(workspace) == [user.resolve(), project.resolve(), own.resolve()]

    env = write_models(tmp_path / "env.yaml", shared="openai/env")
    monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(env))
    assert workspace_models(workspace)["shared"]["model_name"] == "openai/env"
    assert llm_config_files(workspace)[-2:] == [own.resolve(), env.resolve()]


def test_an_alias_of_workspace_a_does_not_resolve_in_workspace_b(tmp_path):
    one, two = tmp_path / "a", tmp_path / "b"
    write_models(workspace_file(one), a_only="openai/a-model")
    two.mkdir()
    assert "a_only" in workspace_models(one)
    assert "a_only" not in workspace_models(two)
    assert workspace_llm_client("a_only", one).model == "openai/a-model"
    # Not a registered alias in B: passed through to litellm as a model name.
    assert workspace_llm_client("a_only", two).model == "a_only"


def test_the_default_llm_factory_resolves_workspace_aliases(tmp_path):
    one, two = tmp_path / "a", tmp_path / "b"
    write_models(workspace_file(one), a_only="openai/a-model")
    two.mkdir()
    make = default_llm_factory()
    assert make("a_only", one).model == "openai/a-model"
    assert make("a_only", two).model == "a_only"


async def test_a_headless_registry_builds_a_workspace_only_alias(workspace, tmp_path):
    """The registry a headless host builds with default_llm_factory() (run_task does).

    The client carries the alias's reasoning levels, which the ACP reasoning option shows.
    """
    own = workspace_file(workspace)
    own.parent.mkdir()
    own.write_text(
        "models:\n  mine:\n    model_name: openai/mine-model\n"
        "    reasoning_levels:\n"
        "      low: {reasoning: {effort: low}}\n"
        "      high: {reasoning: {effort: high}}\n"
    )
    registry = SessionRegistry(
        SessionStore(tmp_path / "sessions"),
        agent_factory=lambda options, storage: EchoAgent(storage=storage, llm=options.llm),
        llm_factory=default_llm_factory(),
    )
    try:
        root = await registry.create(
            SessionOptions(workspace=workspace, agent_spec="echo", model="mine")
        )
        assert root.info.model == "mine"
        assert root._agent.llm.model == "openai/mine-model"
        assert root.model_info().reasoning_levels == ["low", "high"]
    finally:
        await registry.close_all()
