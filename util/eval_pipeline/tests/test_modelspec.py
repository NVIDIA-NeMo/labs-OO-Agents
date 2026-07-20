# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ModelSpec inline model configuration."""

import pytest

from eval_pipeline.config import load_config
from eval_pipeline.eval_types import ModelSpec


class TestModelSpec:
    """Tests for ModelSpec dataclass."""

    def test_create_modelspec(self):
        """ModelSpec can be created with all fields."""
        spec = ModelSpec(
            id="gpt-4",
            model_name="openai/gpt-4",
            endpoint="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
        )
        assert spec.id == "gpt-4"
        assert spec.model_name == "openai/gpt-4"
        assert spec.endpoint == "https://api.openai.com/v1"
        assert spec.api_key_env == "OPENAI_API_KEY"
        assert spec.max_tokens == 4096

    def test_modelspec_default_max_tokens(self):
        """ModelSpec defaults max_tokens to None (use provider default)."""
        spec = ModelSpec(
            id="gpt-4",
            model_name="openai/gpt-4",
            endpoint="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        )
        assert spec.max_tokens is None


class TestLoadConfigWithDictModels:
    """Tests for loading config with dict-format models."""

    def test_loads_dict_modelspec(self, tmp_path):
        """Dict-format models are parsed into ModelSpec objects."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
name: test_eval
models:
  gpt-4:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    max_tokens: 8192

test_suite: []
""")
        config = load_config(config_file)

        assert len(config.models) == 1
        assert "gpt-4" in config.models
        model = config.models["gpt-4"]
        assert isinstance(model, ModelSpec)
        assert model.id == "gpt-4"
        assert model.model_name == "openai/gpt-4"
        assert model.endpoint == "https://api.openai.com/v1"
        assert model.api_key_env == "OPENAI_API_KEY"
        assert model.max_tokens == 8192

    def test_loads_multiple_dict_modelspecs(self, tmp_path):
        """Multiple dict models are all loaded."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
name: test_eval
models:
  gpt-4:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    max_tokens: 8192
  claude-3:
    model_name: anthropic/claude-3
    endpoint: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    max_tokens: 4096
  qwen:
    model_name: nvidia/qwen3-80b
    endpoint: https://integrate.api.nvidia.com/v1
    api_key_env: NVIDIA_API_KEY

test_suite: []
""")
        config = load_config(config_file)

        assert len(config.models) == 3
        assert all(isinstance(m, ModelSpec) for m in config.models.values())
        assert set(config.models.keys()) == {"gpt-4", "claude-3", "qwen"}
        # qwen has no max_tokens specified, so it should be None
        assert config.models["qwen"].max_tokens is None

    def test_agent_models_list(self, tmp_path):
        """agent_models specifies which models to use for agents."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
name: test_eval
models:
  gpt-4:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
  claude-3:
    model_name: anthropic/claude-3
    endpoint: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
  judge-model:
    model_name: anthropic/claude-3-haiku
    endpoint: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    max_tokens: 1024

agent_models:
  - gpt-4
  - claude-3

test_suite: []
""")
        config = load_config(config_file)

        # All 3 models available
        assert len(config.models) == 3
        # Only 2 specified for agent evaluation
        assert config.agent_models == ["gpt-4", "claude-3"]

    def test_agent_models_defaults_to_all(self, tmp_path):
        """Without agent_models, all models are used."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
name: test_eval
models:
  gpt-4:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
  claude-3:
    model_name: anthropic/claude-3
    endpoint: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY

test_suite: []
""")
        config = load_config(config_file)

        # Without agent_models, defaults to all model keys
        assert set(config.agent_models) == {"gpt-4", "claude-3"}

    def test_config_without_models(self, tmp_path):
        """Config without models has empty dict."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
name: test_eval
test_suite: []
""")
        config = load_config(config_file)

        assert config.models == {}
        assert config.agent_models == []

    def test_dict_spec_preserves_all_fields(self, tmp_path):
        """All ModelSpec fields are preserved from dict YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
name: test_eval
models:
  my-model:
    model_name: provider/path/to/model
    endpoint: https://custom-endpoint.com/v1/chat
    api_key_env: MY_CUSTOM_API_KEY
    max_tokens: 16384

test_suite: []
""")
        config = load_config(config_file)

        model = config.models["my-model"]
        assert model.id == "my-model"
        assert model.model_name == "provider/path/to/model"
        assert model.endpoint == "https://custom-endpoint.com/v1/chat"
        assert model.api_key_env == "MY_CUSTOM_API_KEY"
        assert model.max_tokens == 16384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
