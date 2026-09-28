# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model picker re-reads the registry when a registry file changes."""

from __future__ import annotations

import os
import time
from pathlib import Path

from nooa_coder.acp import server


def _write(path: Path, *aliases: str) -> None:
    lines = ["models:"]
    for alias in aliases:
        lines += [f"  {alias}:", "    provider: openai", f"    model: {alias}-model"]
    path.write_text("\n".join(lines) + "\n")
    # A same-size rewrite within one clock tick must still be seen.
    stamp = time.time_ns() + 2_000_000_000
    os.utime(path, ns=(stamp, stamp))


def test_an_alias_connected_after_start_up_appears(tmp_path, monkeypatch):
    registry = tmp_path / "llm_config.yaml"
    _write(registry, "first")
    monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(registry))
    monkeypatch.setattr(server, "_registry_seen", None)

    assert "first" in server.model_aliases()
    assert "second" not in server.model_aliases()

    _write(registry, "first", "second")
    assert {"first", "second"} <= set(server.model_aliases())


def test_an_unchanged_registry_is_not_reloaded(tmp_path, monkeypatch):
    registry = tmp_path / "llm_config.yaml"
    _write(registry, "first")
    monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(registry))
    monkeypatch.setattr(server, "_registry_seen", None)
    server.model_aliases()

    calls: list[int] = []
    from nooa.unifiedllm import registry as registry_module

    monkeypatch.setattr(registry_module, "reload_registry", lambda *a: calls.append(1))
    server.model_aliases()
    assert calls == []
