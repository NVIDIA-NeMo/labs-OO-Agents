# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive model-catalog discovery and registry updates."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import yaml
from nooa_cli.tui.commands import ModelCommand
from nooa_cli.tui.config import TUIConfig
from nooa_cli.tui.model_catalog import (
    CatalogModel,
    ModelCatalogError,
    fetch_model_catalog,
    normalize_catalog_endpoint,
    parse_optional_token_limit,
    registry_entry,
    write_model_alias,
)


def test_normalize_catalog_endpoint_accepts_api_base_or_models_url() -> None:
    expected = (
        "https://inference-api.nvidia.com/v1",
        "https://inference-api.nvidia.com/v1/models",
    )
    assert normalize_catalog_endpoint(expected[0]) == expected
    assert normalize_catalog_endpoint(expected[1]) == expected


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "gitlab.example/models", "https://user:secret@example.test/v1"],
)
def test_normalize_catalog_endpoint_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(ModelCatalogError):
        normalize_catalog_endpoint(url)


def test_fetch_model_catalog_uses_bearer_and_openai_shape(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "z/model",
                        "context_window": "262144",
                        "max_output_tokens": 16384,
                    },
                    {"id": "a/model", "max_model_len": 131072},
                    {"id": "a/model", "max_completion_tokens": 8192},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    api_base, models = fetch_model_catalog("https://models.example.test/v1", api_key="secret")

    assert api_base == "https://models.example.test/v1"
    assert models == [
        CatalogModel(id="a/model", context_window=131072, max_tokens=8192),
        CatalogModel(id="z/model", context_window=262144, max_tokens=16384),
    ]
    assert seen == {
        "url": "https://models.example.test/v1/models",
        "authorization": "Bearer secret",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", None), ("131,072", 131072), ("128k", 128000), ("1.5m", 1500000)],
)
def test_parse_optional_token_limit(value: str, expected: int | None) -> None:
    assert parse_optional_token_limit(value, "Context window") == expected


def test_registry_entry_includes_token_limits() -> None:
    assert registry_entry(
        "org/model",
        "https://models.example.test/v1",
        context_window=131072,
        max_tokens=8192,
    ) == {
        "model_name": "openai/org/model",
        "api_base": "https://models.example.test/v1",
        "context_window": 131072,
        "max_tokens": 8192,
    }


def test_write_model_alias_preserves_comments_and_siblings(tmp_path) -> None:
    path = tmp_path / "llm_config.yaml"
    path.write_text(
        "# team registry\nmodels:\n  existing:\n    model_name: openai/existing\n"
        "# keep this comment\nother_setting: true\n"
    )
    entry = registry_entry("org/new-model", "https://models.example.test/v1", "MODEL_KEY")

    write_model_alias(path, "new-model", entry)

    text = path.read_text()
    loaded = yaml.safe_load(text)
    assert "# team registry" in text
    assert "# keep this comment" in text
    assert loaded["other_setting"] is True
    assert loaded["models"]["existing"]["model_name"] == "openai/existing"
    assert loaded["models"]["new-model"] == entry


def test_write_model_alias_preserves_entries_in_flow_mapping(tmp_path) -> None:
    path = tmp_path / "llm_config.yaml"
    path.write_text("models: {existing: {model_name: openai/existing}}\n")

    write_model_alias(
        path,
        "new-model",
        registry_entry("new-model", "http://localhost:8000/v1"),
    )

    assert set(yaml.safe_load(path.read_text())["models"]) == {"existing", "new-model"}


class _WorkflowFrontend:
    def __init__(self, scope: str = "This project (.nooa/llm_config.yaml)") -> None:
        self.text_answers = ["", "my-model", "131072", "8192"]
        self.choice_answers = ["org/model", scope, "Add only"]
        self.text_prompts = []

    async def prompt_text(self, *_args):
        self.text_prompts.append(_args)
        return self.text_answers.pop(0)

    async def prompt_choice(self, *_args):
        return self.choice_answers.pop(0)


@pytest.mark.asyncio
async def test_model_add_to_registry_workflow_writes_local_file(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.fetch_model_catalog",
        lambda *_args, **_kwargs: (
            "http://localhost:8000/v1",
            [CatalogModel(id="org/model")],
        ),
    )
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.lookup_model_token_limits",
        lambda *_args: (None, None),
    )
    frontend = _WorkflowFrontend()
    command = ModelCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute(["add-to-registry", "http://localhost:8000/v1"])

    assert result.success is True
    command._reload_model_registry.assert_called_once_with()
    saved = yaml.safe_load((project_dir / "llm_config.yaml").read_text())
    assert saved["models"]["my-model"] == {
        "model_name": "openai/org/model",
        "api_base": "http://localhost:8000/v1",
        "context_window": 131072,
        "max_tokens": 8192,
    }
    assert any("/model my-model" in output.content for output in result.outputs)


@pytest.mark.asyncio
async def test_model_add_to_registry_defaults_to_user_config(tmp_path, monkeypatch) -> None:
    user_dir = tmp_path / "user-config"
    project_dir = tmp_path / "project" / ".nooa"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_dir))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.fetch_model_catalog",
        lambda *_args, **_kwargs: (
            "http://localhost:8000/v1",
            [CatalogModel(id="org/model", context_window=262144, max_tokens=16384)],
        ),
    )
    frontend = _WorkflowFrontend("All projects (~/.config/nooa/llm_config.yaml)")
    frontend.text_answers = ["", "my-model", "262144", "16384"]
    command = ModelCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute(["add-to-registry", "http://localhost:8000/v1"])

    assert result.success is True
    assert (user_dir / "llm_config.yaml").exists()
    assert not (project_dir / "llm_config.yaml").exists()
    assert frontend.text_prompts[2][2] == "262144"
    assert frontend.text_prompts[3][2] == "16384"
