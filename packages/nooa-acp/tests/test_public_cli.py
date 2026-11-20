# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The reachable CLI factory produces a strict view used by an actual actor."""

import click.testing
from nooa_acp.cli import command

from nooa import Agent
from nooa.unifiedllm import load_llm
from tests.runtime.public_llm_fixtures import native_models as native_models
from tests.runtime.test_public_llm_factory_calls import generate


async def test_cli_factory_native_wire_uses_pinned_client_type(native_models, monkeypatch):
    root = native_models.make()
    captured = {}

    def serve(factory, *, model_alias):
        captured["model_alias"] = model_alias
        captured["factory"] = factory
        return "not-a-coroutine"

    def loader(alias, **options):
        captured["load"] = alias, options
        return load_llm(root)

    monkeypatch.setattr("nooa.secrets.load_secrets_into_env", lambda: None)
    monkeypatch.setattr("nooa_acp.server.serve", serve)
    monkeypatch.setattr("nooa_acp.cli.asyncio.run", lambda value: value)
    monkeypatch.setattr("nooa.unifiedllm.load_llm", loader)
    result = click.testing.CliRunner().invoke(
        command, ["--model", "configured", "--client-type", "completion"]
    )
    assert result.exit_code == 0, result.output
    view = captured["factory"]()
    agent = Agent(llm=view)
    try:
        await generate(agent, max_tokens=250)
        assert captured["load"] == ("configured", {"client_type": "completion"})
        assert captured["model_alias"] == "configured"
        assert native_models.seen[0][1]["max_completion_tokens"] == 250
    finally:
        await agent.aclose()
        await view.aclose()
    assert not root._closed


import pytest  # noqa: E402 (additive integration test section)


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
async def test_d2_cli_real_yaml_credentials_and_public_native_loader(
    family, native_models, monkeypatch, tmp_path
):
    """Only the ACP stdio loop is replaced; config and factory loading are real."""
    from nooa.unifiedllm import registry

    borrowed = native_models.make(family)
    config = tmp_path / "llm_config.yaml"
    config.write_text(
        "models:\n  configured-native:\n"
        f"    adapter: {family}\n"
        "    model_name: fixture-model\n"
        "    api_base: https://fixture.invalid/v1\n"
        "    api_key_env: D2_CLI_FIXTURE_KEY\n"
        "    context_window: 12800\n"
        "    max_tokens: 100\n"
        "    retry_config: false\n"
    )
    monkeypatch.setenv("D2_CLI_FIXTURE_KEY", "fixture-key")
    registry.reload_registry(config)
    # A real, borrowed core SDK is the offline transport seam. No loader,
    # acall, normalized response, or model binding is replaced.
    registry.MODELS["configured-native"]["client"] = native_models.sdks[-1]
    captured = {}

    def serve(factory, *, model_alias):
        captured.update(factory=factory, model_alias=model_alias)
        return "offline-stdio-boundary"

    monkeypatch.setattr("nooa.secrets.load_secrets_into_env", lambda: None)
    monkeypatch.setattr("nooa_acp.server.serve", serve)
    monkeypatch.setattr("nooa_acp.cli.asyncio.run", lambda operation: operation)
    result = click.testing.CliRunner().invoke(command, ["--model", "configured-native"])
    assert result.exit_code == 0, result.output
    assert captured["model_alias"] == "configured-native"
    assert not native_models.seen
    view = captured["factory"]()
    native_models.views.append(view)
    agent = Agent(llm=view)
    try:
        await generate(agent, max_tokens=250)
        family_seen, request = native_models.seen[-1]
        assert family_seen == family
        assert request["model"] == "fixture-model"
        assert (
            request[
                "max_completion_tokens"
                if family == "openai-chat"
                else "max_output_tokens"
                if family == "openai-responses"
                else "max_tokens"
            ]
            == 250
        )
        assert registry.get_registry_config("configured-native")["api_key_env"] == (
            "D2_CLI_FIXTURE_KEY"
        )
    finally:
        await agent.aclose()
        await view.aclose()
    assert not borrowed._closed
    assert not native_models.http[-1].is_closed
    assert native_models.sdks[-1].max_retries == 9
