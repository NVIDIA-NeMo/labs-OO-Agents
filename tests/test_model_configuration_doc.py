# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The manual setup example loads without contacting a model endpoint."""

import re
from pathlib import Path

import httpx
import yaml


def test_manual_configuration_example_loads_without_http(tmp_path, monkeypatch):
    from nooa.unifiedllm import registry

    text = (Path(__file__).parents[1] / "docs/model-configuration.md").read_text()
    config = re.search(r"```yaml\n(.*?)\n```", text, re.S).group(1)
    code = re.search(r"```python\n(.*?)\n```", text, re.S).group(1)
    (tmp_path / "llm_config.yaml").write_text(config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Manual validation must not make HTTP requests")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    exec(compile(code, "docs/model-configuration.md", "exec"), {})
    assert "my-model" in registry.MODELS
    entry = yaml.safe_load(config)["models"]["my-model"]
    assert entry["api_key_env"] == "MY_MODEL_KEY"
    assert "api_key" not in entry
