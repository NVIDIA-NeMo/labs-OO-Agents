# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete remaining review failures, through library and CLI boundaries."""

import httpx
import pytest
import yaml

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.mark.parametrize(
    "fragment",
    [
        {"api_key": "test-secret"},
        {"extra_body": {"api_key": "test-secret"}},
        {"reasoning_levels": {"high": {"headers": {"Authorization": "Bearer test-secret"}}}},
    ],
)
def test_library_refuses_credentials_at_any_depth(fragment, tmp_path):
    with pytest.raises(ValueError, match="credential") as error:
        connect.write(
            {"model_name": "openai/model", **fragment}, tmp_path / "config.yaml", alias="model"
        )
    assert "test-secret" not in str(error.value)
    assert not (tmp_path / "config.yaml").exists()


async def test_existing_include_without_evidence_does_not_lose_paid_result(monkeypatch):
    mock_http(monkeypatch, lambda r: httpx.Response(200, json=response_body("responses")))
    plan = connect.plan("model", "model", "responses", "https://api.test/v1", "")
    plan.entry["provenance"].pop("encrypted_reasoning")
    result = await connect.run(plan, approved="minimal", api_key="test-key")
    assert result.entry["provenance"]["encrypted_reasoning"]["outcome"] == "accepted"


@pytest.mark.parametrize("include", [[], (), set()])
def test_empty_include_sequences_are_explicit_opt_outs(include):
    from nooa.unifiedllm.replay_state import add_encrypted_reasoning_include

    params = {"include": include, "api_base": "https://api.openai.com/v1"}
    add_encrypted_reasoning_include(params, "responses:openai:https://api.openai.com/v1")
    assert "include" not in params


def test_failed_fsync_cleans_staging_file(tmp_path, monkeypatch):
    path = tmp_path / "registry.yaml"
    path.write_text("models: {}\n")

    def fail(fd):
        raise OSError("test disk full")

    monkeypatch.setattr(connect.os, "fsync", fail)
    with pytest.raises(OSError):
        connect.write({"model_name": "openai/model"}, path, alias="model")
    assert not list(tmp_path.glob("registry.yaml.*"))
    assert path.read_text() == "models: {}\n"


@pytest.mark.parametrize(
    "source",
    [
        "defaults: &d {model_name: openai/model}\nmodels:\n  model: *d\n",
        "defaults: &d {model: {model_name: openai/model}}\nmodels:\n  <<: *d\n",
    ],
)
def test_anchors_are_refused_with_actionable_error_without_changing_file(tmp_path, source):
    path = tmp_path / "registry.yaml"
    path.write_text(source)
    with pytest.raises(ValueError, match="anchors|merge keys"):
        connect.write({"model_name": "openai/replacement"}, path, alias="model")
    assert path.read_text() == source


async def test_422_is_rejected_not_unchecked(monkeypatch):
    mock_http(monkeypatch, lambda r: httpx.Response(422))
    plan = connect.plan("model", "model", "chat", "https://api.test/v1", "")
    result = await connect.run(plan, approved="minimal", api_key="test-key")
    assert result.entry["provenance"]["probes"]["routing"]["outcome"] == "rejected"


def test_null_secrets_env_is_populated(tmp_path):
    from nooa.secrets import write_secret_env

    path = tmp_path / "secrets.yaml"
    path.write_text("env:\n")
    write_secret_env(path, "TEST_KEY", "test-only")
    assert yaml.safe_load(path.read_text())["env"] == {"TEST_KEY": "test-only"}


def _write_from_process(path, alias, first_inside, release_first, second_started, second_inside):
    original_write = connect._write_entry

    def held_write(entry, path, *, alias):
        if alias == "first":
            first_inside.set()
            assert release_first.wait(15)
        else:
            second_inside.set()
        original_write(entry, path, alias=alias)

    connect._write_entry = held_write
    if alias == "second":
        second_started.set()
    connect.write({"model_name": "openai/model"}, path, alias=alias)


def test_concurrent_registry_writers_serialize_the_entire_update(tmp_path):
    import multiprocessing

    # Pytest may import this directory as `connect`; spawn needs its stable path.
    from tests.unifiedllm.connect.test_connect_severin_review import _write_from_process as writer

    ctx = multiprocessing.get_context("spawn")
    signals = [ctx.Event() for _ in range(4)]
    first_inside, release_first, second_started, second_inside = signals
    path = tmp_path / "registry.yaml"
    first, second = [
        ctx.Process(target=writer, args=(path, name, *signals)) for name in ("first", "second")
    ]
    try:
        first.start()
        assert first_inside.wait(15)
        second.start()
        assert second_started.wait(15)
        assert not second_inside.wait(0.2)
    finally:
        release_first.set()
        first.join(5)
        if second.pid:
            second.join(5)
        for child in (first, second):
            if child.is_alive():
                child.terminate()
                child.join(5)
    assert first.exitcode == second.exitcode == 0
    assert set(yaml.safe_load(path.read_text())["models"]) == {"first", "second"}


async def test_unobserved_unauthenticated_request_is_not_called_dropped_settings(monkeypatch):
    from unittest.mock import AsyncMock

    from nooa.llm_types import LLMResponse
    from nooa.unifiedllm import registry

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    real_factory = registry.client_from_config

    def accepted_without_owned_pool(*args, **kwargs):
        client = real_factory(*args, **kwargs)
        assert client._http.async_client is None
        # Simulate a successful legacy fallback; no request traverses our owned pool.
        client.acall = AsyncMock(return_value=LLMResponse(content="323"))
        return client

    monkeypatch.setattr(registry, "client_from_config", accepted_without_owned_pool)
    plan = connect.plan("model", "model", "chat", "http://localhost:8000/v1", "")
    result = await connect.run(plan, approved="minimal")
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["outcome"] == "not_confirmed"
    assert record["settings_sent"] is None
    assert "Could not observe" in record["reason"]
