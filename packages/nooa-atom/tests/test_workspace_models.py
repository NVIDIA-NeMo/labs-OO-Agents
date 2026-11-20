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
    from nooa.runtime.llm_call import model_label

    a, b = make("a_only", one), make("a_only", two)
    try:
        assert model_label(a) == "openai/a-model"
        assert model_label(b) == "a_only"
    finally:
        a.close()
        b.close()


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
        from nooa.runtime.llm_call import model_label

        assert model_label(root._agent.llm) == "openai/mine-model"
        assert root.model_info().reasoning_levels == ["low", "high"]
    finally:
        await registry.close_all()


@pytest.fixture
async def sixth_workspace_native_models(monkeypatch):
    from nooa.unifiedllm import registry
    from tests.runtime.public_llm_fixtures import NativeModels

    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})
    native = NativeModels()
    try:
        yield native
    finally:
        await native.close()


async def test_sixth_contracted_workspace_helper_native_requirements_and_isolation(
    workspace, sixth_workspace_native_models
):
    import copy

    from nooa_atom.workspace.models import workspace_llm, workspace_load_llm

    from nooa.unifiedllm import Need, Requirements, registry
    from nooa.unifiedllm.errors import LLMFailure
    from nooa.unifiedllm.request import LLMRequest

    native = sixth_workspace_native_models
    native.make()
    own = workspace_file(workspace)
    own.parent.mkdir()
    own.write_text(
        "models:\n  native:\n    adapter: openai-chat\n    model: workspace-native\n"
        "    context_window: 12800\n    tools_supported: true\n    max_tokens: 101\n"
        "    base_url: https://fixture.invalid/v1\n    api_key: fixture-key\n"
    )
    registry.MODELS["native"] = {"adapter": "invalid-global", "context_window": 9}
    before = copy.deepcopy(registry.MODELS)
    entry = copy.deepcopy(workspace_models(workspace)["native"])
    required = Requirements(Need.TOOLS, min_context=12000)
    # Injecting a borrowed SDK is a route override: restate its route facts explicitly.
    strict = workspace_load_llm(
        "native",
        workspace,
        require=required,
        client=native.sdks[-1],
        context_window=12800,
        tools_supported=True,
    )
    compatibility = workspace_llm(
        "native", workspace, client_type="completion", client=native.sdks[-1], max_tokens=102
    )
    native.views.extend([strict, compatibility])
    for view in (strict, compatibility):
        response = await view.acall(LLMRequest(messages=[{"role": "user", "content": "native"}]))
        assert response.usage.input_tokens > 0
    assert native.seen[-2][1]["model"] == native.seen[-1][1]["model"] == "workspace-native"
    assert native.seen[-2][1]["max_completion_tokens"] == 101
    assert native.seen[-1][1]["max_completion_tokens"] == 102
    with pytest.raises(LLMFailure):
        workspace_load_llm(
            "native",
            workspace,
            require=Requirements(min_context=20000),
            client=native.sdks[-1],
            context_window=12800,
        )
    with pytest.raises(TypeError):
        workspace_llm("native", workspace, client_type="decision", client=native.sdks[-1])
    assert registry.MODELS == before
    assert workspace_models(workspace)["native"] == entry
    assert native.sdks[-1].max_retries == 9
    assert not native.http[-1].is_closed
