# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the typed file-config API under ``nooa.config``.

``resolved_config()`` / ``ModelConfig`` / ``Secrets`` are the typed boundary
over the layered YAML files (the programmatic ``nooa config show``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nooa.config import (
    ModelConfig,
    ResolvedConfig,
    Secrets,
    get_model_config,
    resolved_config,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    user = tmp_path / "user"
    user.mkdir()
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(proj))
    for v in ("NEMO_OO_SETTINGS", "NEMO_OO_SECRETS", "NEMO_OO_LLM_CONFIG"):
        monkeypatch.delenv(v, raising=False)
    # Stub bundled providers empty so the registry view is deterministic.
    monkeypatch.setattr("nooa.llm_config.bundled_config_paths", lambda: [])
    return user


class TestModelConfig:
    def test_from_registry_defaults_model_name_to_alias(self):
        mc = ModelConfig.from_registry(
            "my-alias",
            {"api_base": "https://example.test/v1", "api_key_env": "K"},
        )
        assert mc.model_name == "my-alias"
        assert mc.api_base == "https://example.test/v1"
        assert mc.api_key_env == "K"

    def test_extra_passthrough_preserved(self):
        mc = ModelConfig.from_registry("a", {"model_name": "m", "num_retries": 7})
        # Unknown litellm passthrough keys are kept (extra="allow").
        assert mc.num_retries == 7  # type: ignore[attr-defined]

    def test_frozen(self):
        mc = ModelConfig.from_registry("a", {"model_name": "m"})
        with pytest.raises(ValidationError):
            mc.model_name = "other"  # type: ignore[misc]


class TestResolvedConfig:
    def test_returns_typed_object(self, _isolate):
        rc = resolved_config()
        assert isinstance(rc, ResolvedConfig)
        assert isinstance(rc.secrets, Secrets)

    def test_secret_values_masked_on_print_and_dump(self, _isolate):
        (_isolate / "secrets.yaml").write_text("env:\n  NVIDIA_INTERNAL_API_KEY: sk-supersecret\n")
        rc = resolved_config()
        # SecretStr masks on str/repr/model_dump(_json) — no plaintext leak.
        assert "sk-supersecret" not in str(rc)
        assert "sk-supersecret" not in str(rc.model_dump())
        assert "sk-supersecret" not in rc.model_dump_json()

    def test_plaintext_via_get_secret_value(self, _isolate):
        (_isolate / "secrets.yaml").write_text("env:\n  K: realvalue\n")
        rc = resolved_config()
        assert rc.secrets.env["K"].get_secret_value() == "realvalue"

    def test_settings_carried_as_dict(self, _isolate):
        (_isolate / "settings.yaml").write_text("tui:\n  default_model: foo\n")
        rc = resolved_config()
        assert rc.settings == {"tui": {"default_model": "foo"}}

    def test_models_are_typed(self, _isolate):
        (_isolate / "llm_config.yaml").write_text(
            "models:\n"
            "  m1:\n"
            "    model_name: openai/x\n"
            "    api_base: https://api.openai.com/v1\n"
            "    api_key_env: OPENAI_API_KEY\n"
        )
        rc = resolved_config()
        assert isinstance(rc.models["m1"], ModelConfig)
        assert rc.models["m1"].api_base == "https://api.openai.com/v1"
        assert rc.models["m1"].api_key_env == "OPENAI_API_KEY"

    def test_sources_lists_winning_files(self, _isolate, monkeypatch):
        s = _isolate / "settings.yaml"
        s.write_text("tui: {}\n")
        rc = resolved_config()
        assert str(s.resolve()) in rc.sources["settings"]


class TestGetModelConfig:
    def test_known_alias(self, _isolate):
        (_isolate / "llm_config.yaml").write_text("models:\n  a:\n    model_name: openai/m\n")
        from nooa.llm_config import llm_config_chain
        from nooa.unifiedllm import reload_registry

        reload_registry(*llm_config_chain())
        mc = get_model_config("a")
        assert isinstance(mc, ModelConfig)
        assert mc.model_name == "openai/m"

    def test_unknown_alias_returns_none(self, _isolate):
        from nooa.unifiedllm import reload_registry

        reload_registry()  # empty (bundled stubbed, no files)
        assert get_model_config("does-not-exist") is None
