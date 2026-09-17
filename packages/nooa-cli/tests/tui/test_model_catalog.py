# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive model-catalog discovery and registry updates."""

from __future__ import annotations

import httpx
import pytest
import yaml
from nooa_cli.tui.model_catalog import (
    CatalogModel,
    ModelCatalogError,
    fetch_model_catalog,
    fetch_native_provider_models,
    model_alias_exists,
    native_provider_registry_entry,
    normalize_catalog_endpoint,
    normalize_native_provider,
    parse_optional_token_limit,
    registry_entry,
    write_model_alias,
    write_secret_env,
)


@pytest.fixture(autouse=True)
def _stub_ollama_probe(monkeypatch):
    """Prevent the connect flow from hitting the network to probe /api/tags."""
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.probe_ollama_backend",
        lambda *_args, **_kwargs: False,
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


def test_fetch_model_catalog_retries_v1_when_root_models_404(monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if str(request.url) == "http://localhost:11434/models":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "qwen3:1.7b"}]})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    api_base, models = fetch_model_catalog("http://localhost:11434")

    assert api_base == "http://localhost:11434/v1"
    assert models == [CatalogModel(id="qwen3:1.7b")]
    assert seen == [
        "http://localhost:11434/models",
        "http://localhost:11434/v1/models",
    ]


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


def test_write_model_alias_replaces_existing_when_requested(tmp_path) -> None:
    path = tmp_path / "llm_config.yaml"
    path.write_text(
        "# team registry\n"
        "models:\n"
        "  qwen3-1.7b:\n"
        "    model_name: openai/qwen3:1.7b\n"
        "    api_base: http://localhost:11434/v1\n"
        "  sibling:\n"
        "    model_name: openai/sibling\n"
    )
    entry = registry_entry("qwen3:1.7b", "http://localhost:11434/v1")

    assert model_alias_exists(path, "qwen3-1.7b") is True
    write_model_alias(path, "qwen3-1.7b", entry, replace=True)

    text = path.read_text()
    loaded = yaml.safe_load(text)
    assert "# team registry" in text
    assert loaded["models"]["qwen3-1.7b"] == {
        "model_name": "openai/qwen3:1.7b",
        "api_base": "http://localhost:11434/v1",
    }
    assert loaded["models"]["sibling"]["model_name"] == "openai/sibling"


def test_write_model_alias_replace_last_entry_keeps_trailing_newline(tmp_path) -> None:
    path = tmp_path / "llm_config.yaml"
    path.write_text(
        "# team registry\n"
        "models:\n"
        "  existing:\n"
        "    model_name: openai/existing\n"
        "    api_base: http://localhost:11434/v1\n",  # file ends with newline
    )
    entry = registry_entry("existing", "http://localhost:11434/v1")

    assert model_alias_exists(path, "existing") is True
    write_model_alias(path, "existing", entry, replace=True)

    text = path.read_text()
    # Replacing the only/last entry must not strip the file's trailing newline.
    assert text.endswith("\n")
    loaded = yaml.safe_load(text)
    assert loaded["models"]["existing"] == {
        "model_name": "openai/existing",
        "api_base": "http://localhost:11434/v1",
    }


def test_registry_entry_uses_openai_prefix_for_openai_compatible_server() -> None:
    assert registry_entry("qwen3:1.7b", "http://localhost:11434/v1") == {
        "model_name": "openai/qwen3:1.7b",
        "api_base": "http://localhost:11434/v1",
    }


def test_native_provider_registry_entry_uses_provider_and_key_env() -> None:
    assert normalize_native_provider("claude") == "anthropic"
    assert native_provider_registry_entry(
        "anthropic",
        "claude-sonnet-4-5",
        "ANTHROPIC_API_KEY",
        context_window=200000,
        max_tokens=64000,
    ) == {
        "model_name": "anthropic/claude-sonnet-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "context_window": 200000,
        "max_tokens": 64000,
    }


def test_fetch_native_provider_models_uses_anthropic_models_api(monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                str(request.url),
                request.headers.get("x-api-key"),
                request.headers.get("anthropic-version"),
            )
        )
        if "after_id=model-a" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "model-b"}],
                    "has_more": False,
                    "last_id": "model-b",
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [{"id": "model-a"}],
                "has_more": True,
                "last_id": "model-a",
            },
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    models = fetch_native_provider_models("anthropic", "secret-value")

    assert models == [CatalogModel(id="model-a"), CatalogModel(id="model-b")]
    assert seen == [
        (
            "https://api.anthropic.com/v1/models?limit=100",
            "secret-value",
            "2023-06-01",
        ),
        (
            "https://api.anthropic.com/v1/models?limit=100&after_id=model-a",
            "secret-value",
            "2023-06-01",
        ),
    ]


def test_write_secret_env_persists_and_sets_process_env(tmp_path, monkeypatch) -> None:
    path = tmp_path / "secrets.yaml"
    monkeypatch.delenv("MY_MODEL_KEY", raising=False)

    write_secret_env(path, "MY_MODEL_KEY", "secret-value")

    assert yaml.safe_load(path.read_text()) == {"env": {"MY_MODEL_KEY": "secret-value"}}
    assert __import__("os").environ["MY_MODEL_KEY"] == "secret-value"
