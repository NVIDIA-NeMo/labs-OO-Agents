# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for GitLab issue #346 (eval_pipeline unwired/mis-wired params).

Covers:
  * temperature/top_p from model config actually reach the LLM client (#346-2)
  * the model_factory registry-dispatch import path resolves (#346-3)
"""

import importlib

import pytest

from eval_pipeline.config import evaluator_from_config, load_config
from eval_pipeline.eval_types import ModelSpec


class TestSamplingParamsWired:
    """temperature/top_p must survive parsing and reach the client config."""

    def test_modelspec_has_sampling_fields(self):
        spec = ModelSpec(id="m", model_name="openai/gpt-4", temperature=0.3, top_p=0.8)
        assert spec.temperature == 0.3
        assert spec.top_p == 0.8

    def test_full_spec_parses_sampling_params(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
name: test_eval
models:
  m:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    temperature: 0.7
    top_p: 0.9

test_suite: []
"""
        )
        cfg = load_config(config_file)
        assert cfg.models["m"].temperature == 0.7
        assert cfg.models["m"].top_p == 0.9

    def test_sampling_params_reach_client_config(self, tmp_path):
        """The client factory must forward temperature/top_p into the client."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
name: test_eval
models:
  m:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    max_tokens: 4096
    temperature: 0.42
    top_p: 0.88

agent_models:
  - m

test_suite: []
"""
        )
        evaluator = evaluator_from_config(config_file)
        client = evaluator._model_factories["m"]()
        # CompletionClient stores passthrough kwargs on .config
        assert client.config.get("temperature") == 0.42
        assert client.config.get("top_p") == 0.88

    def test_sampling_params_absent_when_unset(self, tmp_path):
        """No temperature/top_p configured => they are not forced into the client."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
name: test_eval
models:
  m:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    max_tokens: 4096

agent_models:
  - m

test_suite: []
"""
        )
        evaluator = evaluator_from_config(config_file)
        client = evaluator._model_factories["m"]()
        assert "temperature" not in client.config
        assert "top_p" not in client.config


class TestRegistryImportPath:
    """model_factory registry dispatch must use the real import path (#346-3)."""

    def test_correct_registry_path_importable(self):
        mod = importlib.import_module("nooa.unifiedllm.registry")
        assert hasattr(mod, "MODELS")
        assert hasattr(mod, "get_llm_client")
        assert hasattr(mod, "ensure_loaded")

    def test_old_top_level_path_does_not_exist(self):
        """The pre-fix `unifiedllm.registry` top-level package must not exist,
        confirming the old import was a permanently-swallowed ImportError."""
        with pytest.raises(ImportError):
            importlib.import_module("unifiedllm.registry")

    def test_model_factory_dispatches_after_ensure_loaded(self, monkeypatch):
        """Dispatch must call ensure_loaded() BEFORE the membership check.

        The registry starts empty, so a model only becomes visible once
        ensure_loaded() has run. This simulates that: the model is absent until
        our fake ensure_loaded populates it, proving the ordering is correct.
        """
        from eval_pipeline import model_factory
        from nooa.unifiedllm import registry

        sentinel = object()
        key = "__test_346_model__"
        registry.MODELS.pop(key, None)  # ensure absent before ensure_loaded runs

        def _fake_ensure_loaded():
            registry.MODELS[key] = {"model_name": "x"}

        monkeypatch.setattr(registry, "ensure_loaded", _fake_ensure_loaded)
        monkeypatch.setattr(registry, "_loaded", True)
        monkeypatch.setattr(registry, "get_llm_client", lambda mid, **kw: sentinel)
        try:
            result = model_factory.client(key)
        finally:
            registry.MODELS.pop(key, None)
        # If ensure_loaded ran after (or not at all) the membership check, the
        # model would be missing and dispatch would fall through -> not sentinel.
        assert result is sentinel


class _Marker:
    """Stand-in exporter that records its kind and allows attribute assignment."""

    def __init__(self, kind):
        self.kind = kind


class TestTraceFilesGating:
    """--trace-files / --no-files must actually control jsonl file exporters (#346-1)."""

    def _run_start_tracing(self, monkeypatch, tmp_path, *, use_viewer, no_files, trace_files):
        import nooa.tracing as tracing_mod
        from eval_pipeline import evaluator as evaluator_mod
        from eval_pipeline.evaluator import Evaluator

        # Fake headless backend so nothing real is started.
        class _FakeBackend:
            def start(self):
                return "http://localhost:0"

            def stop(self):
                pass

        monkeypatch.setattr(evaluator_mod, "_probe_otlp", lambda *a, **k: use_viewer)
        import eval_pipeline.headless_backend as hb

        monkeypatch.setattr(hb, "HeadlessOtlpBackend", _FakeBackend)
        monkeypatch.setattr(tracing_mod.exporters, "journal", lambda **k: _Marker("journal"))
        monkeypatch.setattr(tracing_mod.exporters, "jsonl", lambda *a, **k: _Marker("jsonl"))

        captured = {}

        def _fake_enable(exporters, experiment=None, **k):
            captured["exporters"] = exporters

        monkeypatch.setattr(tracing_mod, "enable_tracing", _fake_enable)

        ev = Evaluator()
        ev._langfuse_host = None
        ev._start_tracing(
            tmp_path,
            no_files,
            "exp",
            trace_files=trace_files,
        )
        return [m.kind for m in captured["exporters"]]

    def test_files_written_when_no_viewer(self, monkeypatch, tmp_path):
        kinds = self._run_start_tracing(
            monkeypatch, tmp_path, use_viewer=False, no_files=False, trace_files=False
        )
        assert "jsonl" in kinds

    def test_files_skipped_when_viewer_active(self, monkeypatch, tmp_path):
        kinds = self._run_start_tracing(
            monkeypatch, tmp_path, use_viewer=True, no_files=False, trace_files=False
        )
        assert "jsonl" not in kinds

    def test_trace_files_forces_files_with_viewer(self, monkeypatch, tmp_path):
        kinds = self._run_start_tracing(
            monkeypatch, tmp_path, use_viewer=True, no_files=False, trace_files=True
        )
        assert "jsonl" in kinds

    def test_no_files_overrides_trace_files(self, monkeypatch, tmp_path):
        kinds = self._run_start_tracing(
            monkeypatch, tmp_path, use_viewer=False, no_files=True, trace_files=True
        )
        assert "jsonl" not in kinds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
