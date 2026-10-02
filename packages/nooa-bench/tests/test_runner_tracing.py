# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bench runner streams to the endpoint core resolves, including the https upgrade."""

from __future__ import annotations

from typing import Any

from nooa_bench import runner

import nooa.tracing


def _run(monkeypatch, tmp_path, resolved: str | None) -> list[Any]:
    exporters_seen: list[Any] = []
    journal_endpoints: list[str] = []
    monkeypatch.setenv("OTLP_ENDPOINT", "http://viewer:5443/v1/traces")
    monkeypatch.setattr(runner, "TRACES_DIR", tmp_path / "traces")
    monkeypatch.setattr(nooa.tracing, "resolve_otlp_endpoint", lambda endpoint: resolved)
    monkeypatch.setattr(
        nooa.tracing.exporters, "journal", lambda endpoint: journal_endpoints.append(endpoint)
    )
    monkeypatch.setattr(
        nooa.tracing,
        "enable_tracing",
        lambda exporters, extra_resource_attrs: exporters_seen.extend(exporters),
    )
    runner._setup_tracing("model", "agent")
    return journal_endpoints


def test_the_runner_streams_to_the_upgraded_https_endpoint(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, "https://viewer:5443/v1/traces") == [
        "https://viewer:5443/v1/traces"
    ]


def test_the_runner_writes_files_only_when_nothing_resolves(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, None) == []
