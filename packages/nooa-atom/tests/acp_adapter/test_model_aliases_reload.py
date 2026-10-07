# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model picker re-reads the registry when a registry file changes."""

from __future__ import annotations

import os
import time
from pathlib import Path

from nooa_atom.acp import server


def _write(path: Path, *aliases: str) -> None:
    lines = ["models:"]
    for alias in aliases:
        lines += [f"  {alias}:", "    provider: openai", f"    model: {alias}-model"]
    path.write_text("\n".join(lines) + "\n")
    # A same-size rewrite within one clock tick must still be seen.
    stamp = time.time_ns() + 2_000_000_000
    os.utime(path, ns=(stamp, stamp))


def test_an_alias_connected_after_start_up_appears(tmp_path, workspace, monkeypatch):
    registry = tmp_path / "llm_config.yaml"
    _write(registry, "first")
    monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(registry))

    assert "first" in server.model_aliases(workspace)
    assert "second" not in server.model_aliases(workspace)

    _write(registry, "first", "second")
    assert {"first", "second"} <= set(server.model_aliases(workspace))


def test_an_unchanged_registry_is_not_reloaded(tmp_path, workspace, monkeypatch):
    from nooa_atom.workspace import models

    registry = tmp_path / "llm_config.yaml"
    _write(registry, "first")
    monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(registry))
    server.model_aliases(workspace)

    calls: list[int] = []
    monkeypatch.setattr(models, "_read_models", lambda *a: calls.append(1) or {})
    server.model_aliases(workspace)
    assert calls == []
