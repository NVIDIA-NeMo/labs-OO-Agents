# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive model-catalog discovery and registry updates."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import yaml
from nooa_cli.tui.commands import ConnectCommand
from nooa_cli.tui.config import TUIConfig
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


class _WorkflowFrontend:
    def __init__(self, scope: str = "This project (.nooa/llm_config.yaml)") -> None:
        self.text_answers = ["my-model", "131072", "8192"]
        self.choice_answers = ["org/model", scope, "Add only"]
        self.text_prompts = []
        self.sensitive_answers: list[str] = []
        self.choice_prompts = []

    async def prompt_text(self, *_args):
        self.text_prompts.append(_args)
        return self.text_answers.pop(0)

    async def prompt_choice(self, *_args):
        self.choice_prompts.append(_args)
        return self.choice_answers.pop(0)

    async def prompt_sensitive(self, *_args, **_kwargs):
        return self.sensitive_answers.pop(0)


def _stub_successful_model_switch(monkeypatch) -> None:
    class Healthy:
        ok = True
        error_message = None
        fix_hint = None

    async def fake_probe(_candidate):
        return Healthy()

    monkeypatch.setattr(
        "nooa_cli.tui.config.get_llm_for_model",
        lambda selected, *_args: f"llm:{selected}",
    )
    monkeypatch.setattr("nooa_cli.tui.health_check.probe_llm", fake_probe)
    monkeypatch.setattr("nooa.interactive.apply_model_limits", lambda _agent: None)


@pytest.mark.asyncio
async def test_connect_workflow_writes_local_registry(tmp_path, monkeypatch) -> None:
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
    _stub_successful_model_switch(monkeypatch)
    startup_info = SimpleNamespace(
        model="not connected",
        short_model="No LLM",
        llm_ready=False,
        llm_status="not_connected",
    )
    registry = SimpleNamespace(blocking_llm_health=object(), startup_info=startup_info)
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        registry=registry,
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute(["http://localhost:8000/v1"])

    assert result.success is True
    command._reload_model_registry.assert_called_once_with()
    saved = yaml.safe_load((project_dir / "llm_config.yaml").read_text())
    assert saved["models"]["org/model"] == {
        "model_name": "openai/org/model",
        "api_base": "http://localhost:8000/v1",
    }
    assert any("Switched to model: org/model" in output.content for output in result.outputs)
    assert registry.blocking_llm_health is None
    assert startup_info.model == "org/model"
    assert startup_info.short_model == "model"
    assert startup_info.llm_ready is True
    assert startup_info.llm_status == "ready"
    assert startup_info not in result.outputs


@pytest.mark.asyncio
async def test_connect_workflow_defaults_to_project_config(tmp_path, monkeypatch) -> None:
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
    _stub_successful_model_switch(monkeypatch)
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute(["http://localhost:8000/v1"])

    assert result.success is True
    assert not (user_dir / "llm_config.yaml").exists()
    assert (project_dir / "llm_config.yaml").exists()
    assert frontend.text_prompts == []


@pytest.mark.asyncio
async def test_connect_workflow_reuses_model_catalog_setup(tmp_path, monkeypatch) -> None:
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
    _stub_successful_model_switch(monkeypatch)
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute(["http://localhost:8000/v1"])

    assert result.success is True
    command._reload_model_registry.assert_called_once_with()
    saved = yaml.safe_load((project_dir / "llm_config.yaml").read_text())
    assert saved["models"]["org/model"]["api_base"] == "http://localhost:8000/v1"
    assert frontend.text_prompts == []


@pytest.mark.asyncio
async def test_connect_without_url_prompts_for_endpoint(tmp_path, monkeypatch) -> None:
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
    frontend.text_answers = ["http://localhost:8000/v1"]
    frontend.choice_answers = ["Custom OpenAI-compatible endpoint...", "org/model"]
    _stub_successful_model_switch(monkeypatch)
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute([])

    assert result.success is True
    assert frontend.choice_prompts[0][0] == "Connect model backend"
    assert frontend.choice_prompts[0][2] == [
        "OpenAI",
        "Anthropic",
        "Ollama local",
        "Custom OpenAI-compatible endpoint...",
    ]
    assert frontend.text_prompts[0][0] == "Custom model endpoint"
    assert any("Switched to model: org/model" in output.content for output in result.outputs)


@pytest.mark.asyncio
async def test_connect_without_url_can_use_standard_endpoint_preset(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    calls = []

    def fake_fetch(server_url, *_args, **_kwargs):
        calls.append(server_url)
        return ("http://localhost:11434/v1", [CatalogModel(id="qwen3:1.7b")])

    monkeypatch.setattr("nooa_cli.tui.model_catalog.fetch_model_catalog", fake_fetch)
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.lookup_model_token_limits",
        lambda *_args: (None, None),
    )
    frontend = _WorkflowFrontend()
    frontend.choice_answers = ["Ollama local", "qwen3:1.7b"]
    _stub_successful_model_switch(monkeypatch)
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute([])

    assert result.success is True
    assert calls == ["http://localhost:11434"]
    assert frontend.text_prompts == []
    assert any("Switched to model: qwen3-1.7b" in output.content for output in result.outputs)


@pytest.mark.asyncio
async def test_connect_anthropic_writes_native_provider_alias(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.fetch_native_provider_models",
        lambda *_args, **_kwargs: [
            CatalogModel(id="claude-sonnet-4-5"),
            CatalogModel(id="claude-opus-4-8"),
        ],
    )
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.lookup_model_token_limits",
        lambda *_args: (200000, 64000),
    )
    frontend = _WorkflowFrontend()
    frontend.choice_answers = ["claude-sonnet-4-5"]
    frontend.sensitive_answers = ["anthropic-secret"]
    _stub_successful_model_switch(monkeypatch)
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute(["https://api.anthropic.com"])

    assert result.success is True
    command._reload_model_registry.assert_called_once_with()
    assert yaml.safe_load((project_dir / "secrets.yaml").read_text()) == {
        "env": {"ANTHROPIC_API_KEY": "anthropic-secret"}
    }
    saved = yaml.safe_load((project_dir / "llm_config.yaml").read_text())
    assert saved["models"]["claude-sonnet-4-5"] == {
        "model_name": "anthropic/claude-sonnet-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "context_window": 200000,
        "max_tokens": 64000,
    }
    assert frontend.choice_prompts[0][1] == "Found 2 models. Type to filter, then choose one."
    assert frontend.choice_prompts[0][2] == [
        "claude-sonnet-4-5",
        "claude-opus-4-8",
        "Custom model...",
    ]
    assert frontend.text_prompts == []
    assert any(
        "Switched to model: claude-sonnet-4-5" in output.content for output in result.outputs
    )


@pytest.mark.asyncio
async def test_connect_anthropic_reprompts_for_rejected_saved_secret(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-value")
    calls = []

    def fake_fetch(_provider, api_key, **_kwargs):
        calls.append(api_key)
        if api_key == "stale-value":
            raise ModelCatalogError("Provider model catalog rejected authentication (HTTP 401).")
        return [CatalogModel(id="claude-sonnet-4-5")]

    monkeypatch.setattr("nooa_cli.tui.model_catalog.fetch_native_provider_models", fake_fetch)
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.lookup_model_token_limits",
        lambda *_args: (200000, 64000),
    )
    frontend = _WorkflowFrontend()
    frontend.choice_answers = ["claude-sonnet-4-5"]
    frontend.sensitive_answers = ["fresh-value"]
    _stub_successful_model_switch(monkeypatch)
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute(["https://api.anthropic.com"])

    assert result.success is True
    assert calls == ["stale-value", "fresh-value"]
    assert yaml.safe_load((project_dir / "secrets.yaml").read_text()) == {
        "env": {"ANTHROPIC_API_KEY": "fresh-value"}
    }
    assert __import__("os").environ["ANTHROPIC_API_KEY"] == "fresh-value"


@pytest.mark.asyncio
async def test_connect_registers_openai_compatible_backend_when_probe_fails(
    tmp_path, monkeypatch
) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    project_dir.mkdir(parents=True)
    (project_dir / "llm_config.yaml").write_text(
        "models:\n"
        "  qwen3-1.7b:\n"
        "    model_name: openai/qwen3:1.7b\n"
        "    api_base: http://localhost:8000/v1\n"
    )
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.fetch_model_catalog",
        lambda *_args, **_kwargs: (
            "http://localhost:8000/v1",
            [CatalogModel(id="qwen3:1.7b")],
        ),
    )
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.lookup_model_token_limits",
        lambda *_args: (None, None),
    )
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.probe_ollama_backend",
        lambda *_args, **_kwargs: False,
    )
    frontend = _WorkflowFrontend()
    frontend.choice_answers = ["qwen3:1.7b", "Replace only"]
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute(["http://localhost:8000"])

    assert result.success is True
    saved = yaml.safe_load((project_dir / "llm_config.yaml").read_text())
    assert saved["models"]["qwen3-1.7b"] == {
        "model_name": "openai/qwen3:1.7b",
        "api_base": "http://localhost:8000/v1",
    }
    assert frontend.choice_prompts[-1][2] == ["Replace and use now", "Replace only", "Cancel"]


@pytest.mark.asyncio
async def test_connect_routes_ollama_backend_via_probe(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.fetch_model_catalog",
        lambda *_args, **_kwargs: (
            "http://localhost:11434/v1",
            [CatalogModel(id="qwen3:1.7b")],
        ),
    )
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.lookup_model_token_limits",
        lambda *_args: (None, None),
    )
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.probe_ollama_backend",
        lambda *_args, **_kwargs: True,
    )
    frontend = _WorkflowFrontend()
    frontend.choice_answers = ["qwen3:1.7b"]
    _stub_successful_model_switch(monkeypatch)
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute(["http://localhost:11434"])

    assert result.success is True
    saved = yaml.safe_load((project_dir / "llm_config.yaml").read_text())
    assert saved["models"]["qwen3-1.7b"] == {
        "model_name": "ollama_chat/qwen3:1.7b",
        "api_base": "http://localhost:11434",
    }


@pytest.mark.asyncio
async def test_connect_prompts_for_missing_secret_and_saves_it(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_INTERNAL_API_KEY", raising=False)
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.fetch_model_catalog",
        lambda *_args, **_kwargs: (
            "https://inference-api.nvidia.com/v1",
            [CatalogModel(id="org/model")],
        ),
    )
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.lookup_model_token_limits",
        lambda *_args: (None, None),
    )
    frontend = _WorkflowFrontend()
    frontend.sensitive_answers = ["secret-value"]
    _stub_successful_model_switch(monkeypatch)
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute(["https://inference-api.nvidia.com/v1"])

    assert result.success is True
    assert yaml.safe_load((project_dir / "secrets.yaml").read_text()) == {
        "env": {"NVIDIA_INFERENCE_API_KEY": "secret-value"}
    }
    assert __import__("os").environ["NVIDIA_INFERENCE_API_KEY"] == "secret-value"
    saved = yaml.safe_load((project_dir / "llm_config.yaml").read_text())
    assert saved["models"]["org/model"]["api_key_env"] == "NVIDIA_INFERENCE_API_KEY"
    assert frontend.text_prompts == []


@pytest.mark.asyncio
async def test_connect_cancel_after_secret_prompt_does_not_save_secret(
    tmp_path, monkeypatch
) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_INTERNAL_API_KEY", raising=False)
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.fetch_model_catalog",
        lambda *_args, **_kwargs: (
            "https://inference-api.nvidia.com/v1",
            [CatalogModel(id="org/model")],
        ),
    )
    frontend = _WorkflowFrontend()
    frontend.sensitive_answers = ["secret-value"]
    frontend.choice_answers = [""]
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )

    result = await command.execute(["https://inference-api.nvidia.com/v1"])

    assert result.success is True
    assert any("Model setup cancelled." in output.content for output in result.outputs)
    assert not (project_dir / "secrets.yaml").exists()
    assert "NVIDIA_INFERENCE_API_KEY" not in __import__("os").environ


@pytest.mark.asyncio
async def test_connect_cancel_alias_replacement_does_not_save_secret(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_INTERNAL_API_KEY", raising=False)
    project_dir.mkdir(parents=True)
    (project_dir / "llm_config.yaml").write_text(
        "models:\n"
        "  org/model:\n"
        "    model_name: openai/org/model\n"
        "    api_base: https://old.example/v1\n"
    )
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.fetch_model_catalog",
        lambda *_args, **_kwargs: (
            "https://inference-api.nvidia.com/v1",
            [CatalogModel(id="org/model")],
        ),
    )
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.lookup_model_token_limits",
        lambda *_args: (None, None),
    )
    frontend = _WorkflowFrontend()
    frontend.sensitive_answers = ["secret-value"]
    frontend.choice_answers = ["org/model", "Cancel"]
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )

    result = await command.execute(["https://inference-api.nvidia.com/v1"])

    assert result.success is True
    assert any("Model setup cancelled." in output.content for output in result.outputs)
    assert not (project_dir / "secrets.yaml").exists()
    assert "NVIDIA_INFERENCE_API_KEY" not in __import__("os").environ
    saved = yaml.safe_load((project_dir / "llm_config.yaml").read_text())
    assert saved["models"]["org/model"]["api_base"] == "https://old.example/v1"


@pytest.mark.asyncio
async def test_connect_reprompts_for_rejected_saved_secret(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "stale-value")
    calls = []

    def fake_fetch(_server_url, *, api_key=None):
        calls.append(api_key)
        if api_key == "stale-value":
            raise ModelCatalogError("Model catalog rejected authentication (HTTP 401).")
        return ("https://inference-api.nvidia.com/v1", [CatalogModel(id="org/model")])

    monkeypatch.setattr("nooa_cli.tui.model_catalog.fetch_model_catalog", fake_fetch)
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.lookup_model_token_limits",
        lambda *_args: (None, None),
    )
    frontend = _WorkflowFrontend()
    frontend.sensitive_answers = ["fresh-value"]
    _stub_successful_model_switch(monkeypatch)
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        MagicMock(),
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    result = await command.execute(["https://inference-api.nvidia.com/v1"])

    assert result.success is True
    assert calls == ["stale-value", "fresh-value"]
    assert yaml.safe_load((project_dir / "secrets.yaml").read_text()) == {
        "env": {"NVIDIA_INFERENCE_API_KEY": "fresh-value"}
    }
    assert frontend.text_prompts == []


@pytest.mark.asyncio
async def test_connect_add_and_use_switches_alias_not_endpoint(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.fetch_model_catalog",
        lambda *_args, **_kwargs: (
            "https://inference-api.nvidia.com/v1",
            [CatalogModel(id="azure/moonshotai/kimi-k2.6")],
        ),
    )
    monkeypatch.setattr(
        "nooa_cli.tui.model_catalog.lookup_model_token_limits",
        lambda *_args: (None, None),
    )
    switched = {}

    class Healthy:
        ok = True
        error_message = None
        fix_hint = None

    async def fake_probe(_candidate):
        return Healthy()

    monkeypatch.setattr(
        "nooa_cli.tui.config.get_llm_for_model",
        lambda selected, *_args: f"llm:{selected}",
    )
    monkeypatch.setattr("nooa_cli.tui.health_check.probe_llm", fake_probe)
    monkeypatch.setattr("nooa.interactive.apply_model_limits", lambda _agent: None)

    frontend = _WorkflowFrontend()
    frontend.choice_answers = ["azure/moonshotai/kimi-k2.6"]
    frontend.sensitive_answers = ["secret-value"]
    agent = MagicMock()
    agent.set_llm.side_effect = lambda llm: switched.setdefault("llm", llm)
    command = ConnectCommand(
        frontend,
        TUIConfig(),
        agent,
        root_config=SimpleNamespace(llm_config_paths=[]),
    )
    command._reload_model_registry = MagicMock()

    async def run_async(fn):
        return fn()

    command.agent_run_async = run_async

    result = await command.execute(["https://inference-api.nvidia.com/v1"])

    assert result.success is True
    assert switched == {"llm": "llm:azure/moonshotai/kimi-k2.6"}
    assert any(
        "Switched to model: azure/moonshotai/kimi-k2.6" in getattr(output, "content", "")
        for output in result.outputs
    )
