# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the standalone commit-to-commit capability A/B runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import capability_ab


def _write_results(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps({"_type": "result", **row}) for row in rows))
    return path


def test_configured_models_prefers_agent_model_matrix(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("agent_models:\n  - model-a\n  - model-b\nmodels:\n  scorer: scorer-model\n")

    assert capability_ab.configured_models(config) == ["model-a", "model-b"]


def test_absolute_config_is_shared_and_relative_config_is_per_tree(tmp_path):
    tree = tmp_path / "tree"
    shared = tmp_path / "shared.yaml"

    assert capability_ab._config_for_tree(shared.resolve(), tree) == shared.resolve()
    assert capability_ab._config_for_tree(Path("tests/config.yaml"), tree) == (
        tree / "tests/config.yaml"
    )


def test_parser_keeps_duplicate_case_ids_separate_by_test(tmp_path):
    path = _write_results(
        tmp_path / "results.jsonl",
        [
            {
                "model": "model-a",
                "test_name": "agent_help",
                "test_case": "shared_001",
                "tier": "stable",
                "run_id": 1,
                "passed": True,
            },
            {
                "model": "model-a",
                "test_name": "agent_no_help",
                "test_case": "shared_001",
                "tier": "stable",
                "run_id": 1,
                "passed": False,
            },
        ],
    )

    arm = capability_ab.parse_results(path, "arm")

    assert len(arm.by_task) == 2
    assert arm.counts() == (1, 2)


def test_task_clustered_bootstrap_reports_uniform_improvement(tmp_path):
    base_rows: list[dict] = []
    head_rows: list[dict] = []
    for task in range(8):
        for run in range(3):
            common = {
                "model": "model-a",
                "test_name": f"test_{task}",
                "test_case": f"case_{task}",
                "tier": "stable",
                "run_id": run + 1,
            }
            base_rows.append({**common, "passed": False})
            head_rows.append({**common, "passed": True})
    base = capability_ab.parse_results(_write_results(tmp_path / "base.jsonl", base_rows), "base")
    head = capability_ab.parse_results(_write_results(tmp_path / "head.jsonl", head_rows), "head")

    stats = capability_ab.bootstrap_comparison(base, head, samples=500, seed=7)

    assert stats["overall"]["delta"] == 1.0
    assert stats["overall"]["delta_ci95"] == [1.0, 1.0]


def test_eval_command_options_are_recorded_in_cache_signature(tmp_path, monkeypatch):
    """A changed run option must not silently reuse an incompatible result."""
    tree = tmp_path / "tree"
    tree.mkdir()
    config = tree / "config.yaml"
    config.write_text("agent_models: [model-a]\n")
    output = tmp_path / "output"
    output.mkdir()
    result_path = output / "prior.noo-eval.jsonl"
    _write_results(
        result_path,
        [
            {
                "model": "model-a",
                "test_name": "test",
                "test_case": "case",
                "tier": "stable",
                "run_id": 1,
                "passed": True,
            }
        ],
    )
    (output / "run.json").write_text("{}")
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        return type("Process", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(capability_ab, "_run", fake_run)
    capability_ab._run_arm(
        tree,
        label="arm",
        revision="abc",
        config=Path("config.yaml"),
        models=["model-a"],
        runs=3,
        parallel=40,
        output_dir=output,
        limit=None,
        timeout=900,
        test_filter=None,
        no_cache=True,
        trace_files=False,
        reuse=True,
        max_error_rate=0.5,
    )

    assert called, "mismatched signature must execute instead of reusing prior output"
