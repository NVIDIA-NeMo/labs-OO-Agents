# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model setup crosses the shared command/library boundary without real API calls."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
import yaml
from nooa_cli.interactive.connect import ConnectControl

from nooa.unifiedllm import connect


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("CONNECT_TEST_KEY", "private-connect-key")
    return ConnectControl(SimpleNamespace(cwd=tmp_path), SimpleNamespace(), workspace=tmp_path)


@pytest.fixture
def requests(monkeypatch):
    sent = []

    def respond(request):
        sent.append(request)
        assert request.headers["authorization"] == "Bearer private-connect-key"
        if request.method == "GET":
            return httpx.Response(
                200, json={"data": [{"id": "vendor/model", "context_window": 32000}]}
            )
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "vendor/model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "323"},
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
            },
        )

    original = httpx.AsyncClient.__init__

    def init(client, **kwargs):
        original(client, **{**kwargs, "transport": httpx.MockTransport(respond)})

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    return sent


async def preview(setup):
    result = await setup.invoke("https://gateway.example/v1 --api-key-env CONNECT_TEST_KEY")
    assert result.success, str(result)
    result = await setup.invoke("model vendor/model --as work --max-tokens 256")
    assert result.success, str(result)
    return result


async def test_discover_preview_check_save_are_separate(setup, requests):
    shown = await preview(setup)
    assert [r.method for r in requests] == ["GET"]
    assert not setup.registry_path.exists()
    assert "private-connect-key" not in str(shown)
    assert "CONNECT_TEST_KEY" in str(shown)
    assert setup.proposal.entry["context_window"] == 32000
    assert setup.proposal.entry["transport"] == "direct"
    result = await setup.invoke("check minimal")
    assert result.success, str(result)
    assert [r.method for r in requests] == ["GET", "POST"]
    body = json.loads(requests[-1].content)
    assert body["model"] == "vendor/model"
    assert body["max_tokens"] == 256
    assert not setup.registry_path.exists()
    result = await setup.invoke("save")
    assert result.success, str(result)
    saved = yaml.safe_load(setup.registry_path.read_text())["models"]["work"]
    assert saved["provenance"]["probes"]["routing"]["outcome"] == "accepted"
    assert saved["api_key_env"] == "CONNECT_TEST_KEY"
    assert "private-connect-key" not in setup.registry_path.read_text()
    assert setup.proposal is None
    assert len(requests) == 2


async def test_save_without_checks_reports_unconfirmed_and_preserves_neighbors(setup, requests):
    setup.registry_path.parent.mkdir()
    setup.registry_path.write_text("# keep this\nmodels:\n  other:\n    model_name: openai/other\n")
    await preview(setup)
    result = await setup.invoke("save")
    assert result.success
    assert "unconfirmed" in str(result)
    text = setup.registry_path.read_text()
    assert text.startswith("# keep this\n")
    assert yaml.safe_load(text)["models"]["other"] == {"model_name": "openai/other"}
    assert [r.method for r in requests] == ["GET"]


async def test_alias_created_after_preview_requires_replace(setup, requests):
    await preview(setup)
    setup.registry_path.parent.mkdir()
    original = "models:\n  work:\n    model_name: openai/existing\n"
    setup.registry_path.write_text(original)
    result = await setup.invoke("save")
    assert not result.success
    assert setup.registry_path.read_text() == original
    assert setup.proposal is not None
    result = await setup.invoke("save --replace")
    assert result.success
    assert (
        yaml.safe_load(setup.registry_path.read_text())["models"]["work"]["model_name"]
        == "openai/vendor/model"
    )


async def test_settings_use_library_reasoning_and_reply_plans(setup, requests):
    await preview(setup)
    result = await setup.invoke(
        "model vendor/model --as reason --max-tokens 512 --context-window 24000 --reasoning-template effort --levels low,high --reasoning-default high --budget-tokens 8192"
    )
    assert result.success, str(result)
    p = setup.proposal
    assert p.entry["reasoning_levels"] == {
        "low": {"reasoning_effort": "low"},
        "high": {"reasoning_effort": "high"},
    }
    assert p.entry["reasoning_default"] == "high"
    assert p.entry["context_window"] == 24000
    assert p.budget_tokens == 8192
    assert all(probe.body["max_tokens"] == 512 for probe in p.probes)
    assert len(requests) == 1


async def test_cancelled_check_cancels_library_and_does_not_save(setup, requests, monkeypatch):
    await preview(setup)
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(connect, "run", blocked)
    task = asyncio.create_task(setup.invoke("check all"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert not setup.registry_path.exists()
    assert not setup.proposal.entry["provenance"]["probes"]
    assert (await setup.invoke("cancel")).success
    assert setup.proposal is None and setup.connection is None and setup._api_key is None


async def test_sessions_do_not_share_drafts_or_workspace_destinations(setup, requests, tmp_path):
    await preview(setup)
    other = ConnectControl(
        SimpleNamespace(cwd=tmp_path / "other"), SimpleNamespace(), workspace=tmp_path / "other"
    )
    assert not (await other.invoke("save")).success
    assert other.registry_path == tmp_path / "other" / ".nooa" / "llm_config.yaml"
    assert setup.proposal is not None


@pytest.mark.parametrize(
    "args",
    [
        "check",
        "check none",
        "save --yes",
        "model id --max-tokens -1",
        "https://gateway.example --api-key private-connect-key",
        "https://gateway.example --api-key-env invalid-name",
    ],
)
async def test_invalid_commands_make_no_requests(setup, requests, args):
    result = await setup.invoke(args)
    assert not result.success
    assert "private-connect-key" not in str(result)
    assert not requests
    assert not setup.registry_path.exists()


async def test_masked_host_key_is_not_serialized_or_echoed(setup, requests, monkeypatch):
    monkeypatch.delenv("CONNECT_TEST_KEY")
    result = await setup.start(
        "https://gateway.example/v1", api_key_env="CONNECT_TEST_KEY", api_key="private-connect-key"
    )
    assert result.success
    result = setup.select_model("vendor/model", max_tokens=256)
    assert "private-connect-key" not in str(result)
    assert (await setup.invoke("check minimal")).success
    assert (await setup.invoke("save")).success
    assert "private-connect-key" not in setup.registry_path.read_text()
    assert setup._api_key is None


async def test_missing_named_key_cannot_fall_back_to_another_provider(setup, requests, monkeypatch):
    monkeypatch.delenv("CONNECT_TEST_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-secret")
    result = await setup.invoke("https://gateway.example/v1 --api-key-env CONNECT_TEST_KEY")
    assert not result.success
    assert "Set CONNECT_TEST_KEY" in str(result)
    assert not requests


async def test_provider_exception_is_not_echoed(setup, requests, monkeypatch):
    await preview(setup)

    async def fail(*args, **kwargs):
        raise RuntimeError("private-connect-key from provider body")

    monkeypatch.setattr(connect, "run", fail)
    result = await setup.invoke("check minimal")
    assert not result.success
    assert "private-connect-key" not in str(result)
    assert not setup.registry_path.exists()


async def test_abbreviated_key_flag_is_rejected_without_echo(setup, caplog):
    result = await setup.invoke("https://gateway.example/v1 --api-key PASTEDKEY")
    assert not result.success
    assert "PASTEDKEY" not in str(result) and "PASTEDKEY" not in caplog.text


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_no_auth_discovery_and_checks_do_not_borrow_credentials(setup, monkeypatch, style):
    from tests.unifiedllm.connect.connect_http import mock_http, response_body

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-anthropic-token")
    sent = []

    def respond(request):
        sent.append(request)
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers
        payload = {"data": [{"id": "model"}]} if request.method == "GET" else response_body(style)
        return httpx.Response(200, json=payload)

    mock_http(monkeypatch, respond)
    assert (await setup.invoke(f"https://custom.example/v1 --api-style {style}")).success
    assert (await setup.invoke("model model --as local --max-tokens 128")).success
    result = await setup.invoke("check minimal")
    assert result.success and "Checks passed: routing" in str(result)
    assert [r.method for r in sent] == ["GET", "POST"]


async def test_missing_file_key_can_be_added_then_discovery_retried(
    setup, requests, tmp_path, monkeypatch
):
    import nooa.secrets as secrets

    monkeypatch.setattr(secrets, "_file_env", {})
    monkeypatch.delenv("CONNECT_TEST_KEY")
    monkeypatch.delenv("NEMO_OO_SECRETS", raising=False)
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    result = await setup.invoke("https://gateway.example/v1 --api-key-env CONNECT_TEST_KEY")
    assert not result.success and not requests
    assert str(tmp_path / ".nooa/secrets.yaml") in str(result)
    assert "/connect retry" in str(result)
    (tmp_path / ".nooa").mkdir()
    (tmp_path / ".nooa/secrets.yaml").write_text("env:\n  CONNECT_TEST_KEY: private-connect-key\n")
    result = await setup.invoke("retry")
    assert result.success, str(result)
    assert [r.method for r in requests] == ["GET"]
    assert "private-connect-key" not in str(result)
    assert not setup.registry_path.exists()


async def test_rotated_key_rechecks_instead_of_reusing_accepted_evidence(
    setup, monkeypatch, tmp_path
):
    import nooa.secrets as secrets

    monkeypatch.setattr(secrets, "_file_env", {})
    monkeypatch.delenv("CONNECT_TEST_KEY")
    monkeypatch.delenv("NEMO_OO_SECRETS", raising=False)
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    directory = setup.registry_path.parent
    directory.mkdir()
    path = directory / "secrets.yaml"
    path.write_text("env:\n  CONNECT_TEST_KEY: first-key\n")
    sent = []

    def respond(request):
        sent.append((request.method, request.headers.get("authorization")))
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "vendor/model"}]})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "vendor/model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "323"},
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
            },
        )

    original = httpx.AsyncClient.__init__
    monkeypatch.setattr(
        httpx.AsyncClient,
        "__init__",
        lambda client, **kwargs: original(
            client, **{**kwargs, "transport": httpx.MockTransport(respond)}
        ),
    )
    await preview(setup)
    assert (await setup.invoke("check minimal")).success
    assert sent == [("GET", "Bearer first-key"), ("POST", "Bearer first-key")]
    # An unchanged key can reuse accepted evidence.
    assert (await setup.invoke("check minimal")).success
    assert len(sent) == 2
    path.write_text("env:\n  CONNECT_TEST_KEY: replacement-key\n")
    assert (await setup.invoke("check minimal")).success
    assert sent[-1] == ("POST", "Bearer replacement-key") and len(sent) == 3
    path.unlink()
    result = await setup.invoke("check minimal")
    assert not result.success and len(sent) == 3
    assert "replacement-key" not in str(result)
