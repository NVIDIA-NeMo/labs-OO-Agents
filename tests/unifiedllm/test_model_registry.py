# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for YAML-based model registry."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from nooa.unifiedllm import (
    MODELS,
    CompletionClient,
    RetryConfig,
    ensure_loaded,
    get_llm_client,
    get_registry_config,
    reload_registry,
)


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """Reset registry + redirect user/project dirs to a clean temp dir.

    Without this, tests inherit whatever the real user has in
    ``~/.config/nooa/llm_config.yaml`` (or in the project's
    ``.nooa/`` directory) and become flaky.

    Bundled-default entry-points are stubbed empty so the test suite
    is insensitive to whether ``nemo-oo-agents-nvidia`` (or other
    bundled-config providers) is installed alongside.
    """
    user = tmp_path / "user"
    project = tmp_path / "project"
    user.mkdir()
    project.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project))
    monkeypatch.setattr("nooa.llm_config.bundled_config_paths", lambda: [])
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    # Force a known starting state.
    reload_registry()
    yield
    reload_registry()


def _write_project_config(project_dir: Path, body: str) -> Path:
    path = project_dir / "llm_config.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def _write_user_config(user_dir: Path, body: str) -> Path:
    path = user_dir / "llm_config.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def _project_dir(tmp_path: Path) -> Path:
    return tmp_path / "project"


def _user_dir(tmp_path: Path) -> Path:
    return tmp_path / "user"


class TestEmptyDefaultRegistry:
    """The registry ships empty; public models rely on litellm's built-in routing."""

    def test_registry_empty_after_clear(self):
        """``reload_registry()`` with no args clears MODELS."""
        # Populate first so the post-clear assertion is actually
        # checking that reload_registry() *clears*, not just the
        # fixture's starting state.
        MODELS["pre-existing"] = {"model_name": "x"}
        reload_registry()
        assert MODELS == {}

    def test_registry_empty_when_no_files(self, tmp_path):
        """No user/project/env files → registry stays empty after ensure_loaded()."""
        from nooa.unifiedllm import registry as _registry

        # Reset _loaded so ensure_loaded actually runs.
        _registry._loaded = False
        ensure_loaded()
        assert MODELS == {}


class TestGetRegistryConfig:
    """Public snapshot accessor over the raw MODELS dict."""

    def test_returns_snapshot_copy_for_known_alias(self, tmp_path):
        path = _write_project_config(
            _project_dir(tmp_path),
            """
            models:
              my-alias:
                model_name: openai/my-org/my-model
                api_base: https://gw.example.com/v1
            """,
        )
        reload_registry(path)

        cfg = get_registry_config("my-alias")
        assert cfg["model_name"] == "openai/my-org/my-model"
        assert cfg["api_base"] == "https://gw.example.com/v1"

        # Mutating the returned dict must not corrupt the live registry.
        cfg["api_base"] = "tampered"
        assert MODELS["my-alias"]["api_base"] == "https://gw.example.com/v1"

    def test_returns_empty_dict_for_unknown_alias(self):
        assert get_registry_config("does-not-exist") == {}

    def test_triggers_auto_load(self, tmp_path):
        from nooa.unifiedllm import registry as _registry

        _write_project_config(
            _project_dir(tmp_path),
            """
            models:
              lazy-alias:
                model_name: openai/lazy
            """,
        )
        # Simulate an un-bootstrapped process: registry not yet loaded.
        _registry._loaded = False
        MODELS.clear()

        cfg = get_registry_config("lazy-alias")
        assert cfg.get("model_name") == "openai/lazy"


class TestGetLlmClient:
    """Tests for get_llm_client() function."""

    def test_returns_completion_client(self):
        llm = get_llm_client("gpt-4o-mini")
        assert isinstance(llm, CompletionClient)

    def test_unknown_model_passes_through(self):
        """Unknown model names should pass through to CompletionClient directly."""
        llm = get_llm_client("some-unknown-model-xyz")
        assert llm.model == "some-unknown-model-xyz"

    def test_public_provider_model_strings_pass_through(self):
        """Common public providers should work without registry aliases."""
        openai_llm = get_llm_client("gpt-4o-mini")
        anthropic_llm = get_llm_client("claude-sonnet-4-5-20250514")

        assert openai_llm.model == "gpt-4o-mini"
        assert anthropic_llm.model == "claude-sonnet-4-5-20250514"
        assert openai_llm.config.get("drop_params") is True
        assert anthropic_llm.config.get("drop_params") is True

    def test_registry_model_uses_model_name(self, tmp_path):
        """Registry model should use the model_name from config."""
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              my-alias:
                model_name: openai/my-org/my-model
            """,
        )
        reload_registry(path)
        llm = get_llm_client("my-alias")
        assert llm.model == "openai/my-org/my-model"

    def test_registry_retry_config_mapping(self, tmp_path):
        """Registry aliases can tune default endpoint retries centrally."""
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              my-alias:
                model_name: openai/my-org/my-model
                retry_config:
                  max_retries: 1
                  base_delay: 0.01
                  rate_limit_extra_retries: 0
            """,
        )
        reload_registry(path)

        llm = get_llm_client("my-alias")

        assert isinstance(llm.retry_config, RetryConfig)
        assert llm.retry_config.max_retries == 1
        assert llm.retry_config.base_delay == 0.01
        assert llm.retry_config.rate_limit_extra_retries == 0
        assert "retry_config" not in llm.config

    def test_registry_retry_config_false_disables_retries(self, tmp_path):
        """Registry aliases can opt out of default endpoint retries centrally."""
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              my-alias:
                model_name: openai/my-org/my-model
                retry_config: false
            """,
        )
        reload_registry(path)

        llm = get_llm_client("my-alias")

        assert llm.retry_config.max_retries == 0
        assert llm.retry_config.rate_limit_extra_retries == 0
        assert "retry_config" not in llm.config

    def test_retry_config_override_beats_registry(self, tmp_path):
        """Explicit call-site retry_config overrides YAML defaults."""
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              my-alias:
                model_name: openai/my-org/my-model
                retry_config: false
            """,
        )
        reload_registry(path)
        override = RetryConfig(max_retries=5)

        llm = get_llm_client("my-alias", retry_config=override)

        assert llm.retry_config is override

    def test_overrides_take_precedence(self):
        """User overrides should take precedence over registry defaults."""
        llm = get_llm_client("gpt-4o-mini", temperature=0.9, max_tokens=100)
        assert llm.config.get("temperature") == 0.9
        assert llm.config.get("max_tokens") == 100

    def test_registry_preserves_litellm_pass_through_controls(self, tmp_path):
        """Reasoning aliases can whitelist gateway params for LiteLLM."""
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              reasoning-alias:
                model_name: openai/aws/anthropic/bedrock-claude-opus-4-6
                reasoning_effort: high
                allowed_openai_params:
                  - reasoning_effort
                extra_body:
                  trace: true
            """,
        )
        reload_registry(path)

        llm = get_llm_client("reasoning-alias")

        assert llm.model == "openai/aws/anthropic/bedrock-claude-opus-4-6"
        assert llm.config["reasoning_effort"] == "high"
        assert llm.config["allowed_openai_params"] == ["reasoning_effort"]
        assert llm.config["extra_body"] == {"trace": True}

    def test_drop_params_default_true(self):
        llm = get_llm_client("gpt-4o-mini")
        assert llm.config.get("drop_params") is True

    def test_registry_hit_logs_info(self, tmp_path, caplog):
        """Registry hits should be logged at INFO level for user visibility."""
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              my-alias:
                model_name: openai/my-org/my-model
                api_base: https://example.com/v1
            """,
        )
        reload_registry(path)

        import logging

        with caplog.at_level(logging.INFO, logger="nooa.unifiedllm.registry"):
            get_llm_client("my-alias")
        assert "registry hit" in caplog.text.lower()

    def test_first_call_triggers_auto_load(self, tmp_path):
        """When ``reload_registry`` has not been called, the first
        ``get_llm_client`` triggers auto-discovery via the standard chain.
        """
        from nooa.unifiedllm import registry as _registry

        _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              auto-loaded:
                model_name: auto/loaded
            """,
        )
        # Pretend the registry was never explicitly loaded.
        MODELS.clear()
        _registry._loaded = False

        llm = get_llm_client("auto-loaded")
        assert llm.model == "auto/loaded"
        assert "auto-loaded" in MODELS


class TestApiKeyHandling:
    """Tests for API key environment variable handling."""

    def test_api_key_from_env(self, tmp_path, monkeypatch):
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              test-keyed-model:
                model_name: test-model
                api_key_env: MY_TEST_KEY
            """,
        )
        monkeypatch.setenv("MY_TEST_KEY", "test-key-abc")
        reload_registry(path)
        llm = get_llm_client("test-keyed-model")
        assert llm.config.get("api_key") == "test-key-abc"

    def test_missing_api_key_handled_gracefully(self, tmp_path, monkeypatch):
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              test-keyed-model:
                model_name: test-model
                api_key_env: NONEXISTENT_KEY
            """,
        )
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        reload_registry(path)
        llm = get_llm_client("test-keyed-model")
        assert llm.config.get("api_key") is None

    def test_unknown_model_no_api_key_env(self):
        llm = get_llm_client("totally-made-up-model")
        assert llm.config.get("api_key") is None

    def test_nvidia_key_synonym_old_config_new_env(self, tmp_path, monkeypatch):
        """A config declaring the legacy NVIDIA_INTERNAL_API_KEY resolves from
        the renamed NVIDIA_INFERENCE_API_KEY."""
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              test-nvidia-model:
                model_name: test-model
                api_key_env: NVIDIA_INTERNAL_API_KEY
            """,
        )
        monkeypatch.delenv("NVIDIA_INTERNAL_API_KEY", raising=False)
        monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "new-name-key")
        reload_registry(path)
        llm = get_llm_client("test-nvidia-model")
        assert llm.config.get("api_key") == "new-name-key"

    def test_nvidia_key_synonym_new_config_old_env(self, tmp_path, monkeypatch):
        """A config declaring NVIDIA_INFERENCE_API_KEY resolves from the legacy
        NVIDIA_INTERNAL_API_KEY still set in older environments."""
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              test-nvidia-model:
                model_name: test-model
                api_key_env: NVIDIA_INFERENCE_API_KEY
            """,
        )
        monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY", raising=False)
        monkeypatch.setenv("NVIDIA_INTERNAL_API_KEY", "old-name-key")
        reload_registry(path)
        llm = get_llm_client("test-nvidia-model")
        assert llm.config.get("api_key") == "old-name-key"

    def test_declared_env_var_wins_over_synonym(self, tmp_path, monkeypatch):
        """When both names are set, the one the config declares wins."""
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              test-nvidia-model:
                model_name: test-model
                api_key_env: NVIDIA_INFERENCE_API_KEY
            """,
        )
        monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "declared-key")
        monkeypatch.setenv("NVIDIA_INTERNAL_API_KEY", "synonym-key")
        reload_registry(path)
        llm = get_llm_client("test-nvidia-model")
        assert llm.config.get("api_key") == "declared-key"

    def test_non_string_api_key_env_logged_and_dropped(self, tmp_path, caplog):
        """A malformed ``api_key_env`` (e.g. a number) must not crash ``os.getenv``.

        ``resolve_api_key_from_config`` logs a warning naming the
        offending type and returns ``None``; ``get_llm_client`` then
        proceeds without an API key.
        """
        path = tmp_path / "bad.yaml"
        path.write_text(
            textwrap.dedent("""\
                models:
                  numeric-env:
                    model_name: m
                    api_key_env: 12345
            """)
        )
        reload_registry(path)

        import logging

        with caplog.at_level(logging.WARNING, logger="nooa.unifiedllm.registry"):
            llm = get_llm_client("numeric-env")

        assert llm.config.get("api_key") is None
        assert "invalid api_key_env" in caplog.text
        assert "int" in caplog.text  # type name surfaced

    def test_explicit_api_key_override_skips_env_warning(self, tmp_path, monkeypatch, caplog):
        """An explicit ``api_key=`` override must not emit a missing-env-var WARN.

        The registry's ``api_key_env`` is irrelevant when the caller is
        already passing credentials directly — surfacing a warning here
        would look like a real misconfiguration even though the request
        succeeds.
        """
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              test-keyed-model:
                model_name: test-model
                api_key_env: ABSENT_ON_PURPOSE
            """,
        )
        monkeypatch.delenv("ABSENT_ON_PURPOSE", raising=False)
        reload_registry(path)

        import logging

        with caplog.at_level(logging.WARNING, logger="nooa.unifiedllm.registry"):
            llm = get_llm_client("test-keyed-model", api_key="explicit-key")

        assert llm.config.get("api_key") == "explicit-key"
        assert "ABSENT_ON_PURPOSE" not in caplog.text


class TestApiBaseHandling:
    """Tests for explicit local/self-hosted endpoint URL handling."""

    def test_api_base_static_config(self, tmp_path):
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              local-model:
                model_name: hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct
                api_base: http://127.0.0.1:8000/v1
            """,
        )
        reload_registry(path)

        llm = get_llm_client("local-model")

        assert llm.model == "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct"
        assert llm.config["api_base"] == "http://127.0.0.1:8000/v1"

    def test_explicit_api_base_override_beats_registry(self, tmp_path):
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              local-vllm:
                model_name: hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct
                api_base: http://127.0.0.1:8000/v1
            """,
        )
        reload_registry(path)

        llm = get_llm_client("local-vllm", api_base="http://127.0.0.1:9000/v1")

        assert llm.model == "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct"
        assert llm.config.get("api_base") == "http://127.0.0.1:9000/v1"


class TestConfigLayering:
    """Tests for YAML config file layering via the chain helper."""

    def test_project_config_loaded(self, tmp_path):
        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              my-custom-model:
                model_name: openai/my-custom-model
                api_base: https://my-endpoint.example.com/v1
                api_key_env: MY_API_KEY
                context_window: 128000
            """,
        )
        registry = reload_registry(path)
        assert "my-custom-model" in registry
        assert registry["my-custom-model"]["api_base"] == "https://my-endpoint.example.com/v1"

    def test_null_removes_model(self, tmp_path):
        """Setting a model to null in a later layer should remove it."""
        env_config = tmp_path / "env.yaml"
        env_config.write_text(
            textwrap.dedent("""\
                models:
                  removable:
                    model_name: removable
            """)
        )
        project_config = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              removable: null
            """,
        )
        # Pass paths in the order the chain helper would: env first, project last.
        registry = reload_registry(env_config, project_config)
        assert "removable" not in registry

    def test_chain_user_project_env_priority(self, tmp_path, monkeypatch):
        """End-to-end: user < project < NEMO_OO_LLM_CONFIG, last wins.

        The env var is the **global override** — anything set in a
        shell session beats files on disk.
        """
        _write_user_config(
            _user_dir(tmp_path),
            """\
            models:
              who-wins:
                model_name: from-user
            """,
        )
        _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              who-wins:
                model_name: from-project
            """,
        )
        env_config = tmp_path / "env_config.yaml"
        env_config.write_text(
            textwrap.dedent("""\
                models:
                  who-wins:
                    model_name: from-env
            """)
        )
        monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(env_config))

        from nooa.llm_config import llm_config_chain

        registry = reload_registry(*llm_config_chain())
        assert registry["who-wins"]["model_name"] == "from-env"

    def test_explicit_paths_last_wins(self, tmp_path):
        """``reload_registry(a, b, c)`` merges last-wins."""
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        c = tmp_path / "c.yaml"
        a.write_text("models:\n  k:\n    model_name: from-a\n")
        b.write_text("models:\n  k:\n    model_name: from-b\n")
        c.write_text("models:\n  k:\n    model_name: from-c\n")

        registry = reload_registry(a, b, c)
        assert registry["k"]["model_name"] == "from-c"


class TestReloadRegistry:
    """Tests for reload_registry() function."""

    def test_no_args_rediscovers_via_chain(self, tmp_path):
        """``reload_registry()`` with no args re-reads the discovered chain.

        Matches the pre-refactor behavior: edit ``llm_config.yaml`` or
        change ``NEMO_OO_LLM_CONFIG`` and call ``reload_registry()`` to
        pick the new contents up.
        """
        # Initial state: empty (no files, no env var, bundled disabled).
        reload_registry()
        assert MODELS == {}

        # Drop a file into the project layer — chain helper will find it.
        _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              one:
                model_name: one
            """,
        )
        reload_registry()
        assert "one" in MODELS

    def test_no_args_marks_loaded(self, tmp_path):
        """``reload_registry()`` sets ``_loaded`` so ensure_loaded() is a no-op afterward."""
        from nooa.unifiedllm import registry as _registry

        reload_registry()
        assert _registry._loaded is True

        # After explicit reload, ensure_loaded() is idempotent — it
        # shouldn't re-run discovery. Drop a file the chain would find
        # and verify it does NOT appear via ensure_loaded.
        _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              should-not-appear:
                model_name: x
            """,
        )
        ensure_loaded()  # no-op
        assert "should-not-appear" not in MODELS

    def test_reload_updates_in_place(self, tmp_path):
        """Callers holding a reference to MODELS should see reloaded contents."""
        original_ref = MODELS

        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              new-model:
                model_name: new-model
            """,
        )
        reload_registry(path)

        assert MODELS is original_ref
        assert "new-model" in original_ref

    def test_concurrent_get_llm_client_during_reload_is_safe(self, tmp_path):
        """A reader hitting ``get_llm_client()`` while another thread
        runs ``reload_registry()`` must never observe a half-cleared
        registry. The reader either sees the pre-reload value (registry
        miss → litellm passthrough) or the post-reload value
        (registry hit), never a transient empty state that loses the
        alias mid-mutation.
        """
        import threading
        import time

        path = _write_project_config(
            _project_dir(tmp_path),
            """\
            models:
              raceable:
                model_name: openai/raceable-model
            """,
        )
        # Seed MODELS so the alias is present before the race starts.
        reload_registry(path)
        assert "raceable" in MODELS

        stop = threading.Event()
        reader_errors: list[BaseException] = []
        bad_observations: list[str] = []

        def reader() -> None:
            try:
                while not stop.is_set():
                    llm = get_llm_client("raceable")
                    # Either we see the registry hit (model_name from
                    # YAML) or the registry miss (alias passes through
                    # to litellm as-is). The bad case the lock
                    # prevents is: ``llm.model == "raceable"`` *and*
                    # MODELS containing "raceable" — i.e. a transient
                    # empty-dict observation. We can't directly observe
                    # that, but we can record whether reads returned
                    # consistent shapes.
                    if llm.model not in {"raceable", "openai/raceable-model"}:
                        bad_observations.append(f"unexpected model: {llm.model!r}")
            except BaseException as exc:  # noqa: BLE001 — surface in test
                reader_errors.append(exc)

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        # Hammer reload_registry from the main thread while the reader
        # is running. With write-side-only locking, MODELS.clear() +
        # update() leaves a window where the reader sees an empty
        # registry; the read-side lock added in this MR closes it.
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            reload_registry(path)
        stop.set()
        t.join(timeout=2.0)

        assert not reader_errors, reader_errors
        assert not bad_observations, bad_observations
        # End state: registry holds the alias.
        assert "raceable" in MODELS
