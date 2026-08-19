# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the release script's capability-diff reporting.

The expensive parts of `scripts/make_release.py` (evals, builds, `gh`) cannot
run in CI, but the part that decides whether a release is safe — parsing eval
output and classifying the diff — is pure and worth pinning down. These tests
feed synthetic `.noo-eval.jsonl` through the real parser and comparator.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# (model, test_name, test_case, tier, passed, error_type)
Row = tuple[str, str, str, str, bool, str | None]


@pytest.fixture(scope="module")
def mr():
    """Load make_release.py, which lives in scripts/ and is not importable."""
    spec = importlib.util.spec_from_file_location("_make_release", REPO / "scripts/make_release.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations via sys.modules.
    sys.modules["_make_release"] = module
    spec.loader.exec_module(module)
    return module


def write_eval(path: Path, rows: Sequence[Row]) -> Path:
    """Write a .noo-eval.jsonl from (model, test, case, tier, passed, error_type)."""
    with open(path, "w") as fh:
        fh.write(json.dumps({"_type": "metadata", "metadata": {}}) + "\n")
        for model, test, case, tier, passed, error in rows:
            fh.write(
                json.dumps(
                    {
                        "_type": "result",
                        "model": model,
                        "test_name": test,
                        "test_case": case,
                        "tier": tier,
                        "passed": passed,
                        "error_type": error,
                        "output_tokens": 900,
                        "total_tokens": 16_000,
                    }
                )
                + "\n"
            )
    return path


MODELS = ["claude-haiku", "gpt-5.4-mini"]
SUITE = [
    ("sentiment_single", "sentiment_single_001", "stable"),
    ("router_multi", "router_multi_001", "stable"),
    ("structured_nested", "structured_nested_001", "stable"),
    ("truncation_deep", "truncation_deep_001", "frontier"),
]


def all_passing(stable_only: bool = False) -> list:
    return [
        (m, t, c, tier, (tier == "stable") if stable_only else True, None)
        for m in MODELS
        for t, c, tier in SUITE
        for _ in range(3)
    ]


def test_parses_pass_rates_tiers_and_tokens(mr, tmp_path):
    arm = mr.parse_results(write_eval(tmp_path / "a.jsonl", all_passing()), "head")
    assert arm.counts() == (24, 24)
    assert arm.overall() == 1.0
    assert arm.tier_counts() == {"stable": (18, 18), "frontier": (6, 6)}
    assert arm.total_tokens == 24 * 16_000


def test_clean_diff_when_nothing_changed(mr, tmp_path):
    base = mr.parse_results(write_eval(tmp_path / "b.jsonl", all_passing()), "v0.0.8")
    head = mr.parse_results(write_eval(tmp_path / "h.jsonl", all_passing()), "head")
    diff = mr.compare(base, head, "v0.0.8", "abc123456789")
    assert diff.clean
    assert diff.floor_breach is None
    assert "Release OK" in diff.markdown


def test_flags_collapse_and_new_error_type(mr, tmp_path):
    head_rows = []
    for m in MODELS:
        for t, c, tier in SUITE:
            for _ in range(3):
                if m == "claude-haiku" and t == "router_multi":
                    head_rows.append((m, t, c, tier, False, None))
                elif m == "gpt-5.4-mini" and t == "structured_nested":
                    head_rows.append((m, t, c, tier, False, "TypeError"))
                else:
                    head_rows.append((m, t, c, tier, True, None))

    base = mr.parse_results(write_eval(tmp_path / "b.jsonl", all_passing()), "v0.0.8")
    head = mr.parse_results(write_eval(tmp_path / "h.jsonl", head_rows), "head")
    diff = mr.compare(base, head, "v0.0.8", "abc123456789")

    assert len(diff.regressions) == 2
    assert len(diff.new_errors) == 1
    assert "TypeError" in diff.new_errors[0]
    assert not diff.clean


def test_case_collapse_is_not_hidden_by_passing_sibling_case(mr, tmp_path):
    """Collapse detection must retain test-case granularity.

    Grouping only by (model, test_name) makes this look like a 50% test-family
    result and misses that one case fell from 100% to 0%.
    """
    base_rows = [
        ("claude-haiku", "truncation", case, "stable", True, None)
        for case in ("visible_suffix", "hidden_middle")
        for _ in range(3)
    ]
    head_rows = [
        ("claude-haiku", "truncation", case, "stable", case == "hidden_middle", None)
        for case in ("visible_suffix", "hidden_middle")
        for _ in range(3)
    ]
    base = mr.parse_results(write_eval(tmp_path / "b.jsonl", base_rows), "base")
    head = mr.parse_results(write_eval(tmp_path / "h.jsonl", head_rows), "head")

    diff = mr.compare(base, head, "base", "head")

    assert len(diff.regressions) == 1
    assert "visible_suffix" in diff.regressions[0]


def test_tier_regression_is_caught_when_aggregate_is_flat(mr, tmp_path):
    """Frontier gains must not mask a stable-tier drop.

    The suite has 18 stable samples and 6 frontier ones. Baseline passes all
    stable and no frontier (18/24). Head loses exactly one stable test — 6
    samples — and gains all 6 frontier ones, so the overall rate is unchanged
    at 18/24 while stable falls 100% → 66.7%. Only a per-tier check sees it.
    """
    base_rows = all_passing(stable_only=True)
    head_rows = [
        (m, t, c, tier, tier == "stable" and t != "router_multi", None)
        for m in MODELS
        for t, c, tier in SUITE
        for _ in range(3)
    ]
    head_rows = [
        (m, t, c, tier, True if tier == "frontier" else passed, err)
        for m, t, c, tier, passed, err in head_rows
    ]
    base = mr.parse_results(write_eval(tmp_path / "b.jsonl", base_rows), "v0.0.8")
    head = mr.parse_results(write_eval(tmp_path / "h.jsonl", head_rows), "head")

    assert base.counts() == (18, 24)
    assert head.counts() == (18, 24), "precondition: aggregate is flat"
    assert base.overall() == pytest.approx(head.overall())
    assert head.tier_counts()["stable"] == (12, 18)

    diff = mr.compare(base, head, "v0.0.8", "abc123456789")
    assert any("stable tier" in note for note in diff.beyond_noise)
    assert not diff.clean


def test_floor_breach_blocks(mr, tmp_path):
    head_rows = [
        (m, t, c, tier, tier != "stable", None)
        for m in MODELS
        for t, c, tier in SUITE
        for _ in range(3)
    ]
    base = mr.parse_results(write_eval(tmp_path / "b.jsonl", all_passing()), "v0.0.8")
    head = mr.parse_results(write_eval(tmp_path / "h.jsonl", head_rows), "head")
    diff = mr.compare(base, head, "v0.0.8", "abc123456789")

    assert diff.floor_breach is not None
    assert "Release BLOCKED" in diff.markdown
    assert not diff.clean


def test_floor_passes_just_above_the_line(mr, tmp_path):
    """18 stable samples, 12 passing = 66.7%, comfortably over the 60% floor."""
    head_rows = []
    for m in MODELS:
        for t, c, tier in SUITE:
            for _ in range(3):
                failed = tier == "stable" and t == "router_multi"
                head_rows.append((m, t, c, tier, not failed, None))
    base = mr.parse_results(write_eval(tmp_path / "b.jsonl", all_passing()), "v0.0.8")
    head = mr.parse_results(write_eval(tmp_path / "h.jsonl", head_rows), "head")
    diff = mr.compare(base, head, "v0.0.8", "abc123456789")

    assert head.tier_counts()["stable"] == (12, 18)
    assert diff.floor_breach is None


def test_added_tests_are_listed_but_do_not_block(mr, tmp_path):
    """A new test has no baseline to regress from, so it must not fail the run."""
    base_rows = all_passing()
    head_rows = all_passing() + [
        ("claude-haiku", "brand_new", "brand_new_001", "stable", True, None)
    ]
    base = mr.parse_results(write_eval(tmp_path / "b.jsonl", base_rows), "v0.0.8")
    head = mr.parse_results(write_eval(tmp_path / "h.jsonl", head_rows), "head")
    diff = mr.compare(base, head, "v0.0.8", "abc123456789")

    assert diff.added == ["claude-haiku/brand_new/brand_new_001"]
    assert diff.removed == []
    assert "*(new)*" in diff.markdown
    assert diff.clean


def test_removed_test_is_not_clean(mr, tmp_path):
    """Deleting or renaming a failing test must not erase the regression.

    Without this, dropping a test that fails on HEAD leaves the verdict at
    "no regressions" and `--checks-only` exiting 0, while testing less.
    """
    head_rows = [
        row for row in all_passing() if row[1] != "router_multi"
    ]  # router_multi deleted on HEAD
    base = mr.parse_results(write_eval(tmp_path / "b.jsonl", all_passing()), "v0.0.8")
    head = mr.parse_results(write_eval(tmp_path / "h.jsonl", head_rows), "head")
    diff = mr.compare(base, head, "v0.0.8", "abc123456789")

    assert sorted(diff.removed) == [
        "claude-haiku/router_multi/router_multi_001",
        "gpt-5.4-mini/router_multi/router_multi_001",
    ]
    assert not diff.clean, "a disappearing test must not report as clean"
    assert "removed test(s)" in diff.markdown


def test_model_set_change_is_not_reported_as_removed_tests(mr, tmp_path):
    """Changing GATE_MODELS must not look like the suite being deleted.

    Every test of a model that ran in only one arm would otherwise land in
    added/removed, and gating on `removed` would block every model-set change.
    """
    head_rows = [row for row in all_passing() if row[0] == "claude-haiku"]
    base = mr.parse_results(write_eval(tmp_path / "b.jsonl", all_passing()), "v0.0.8")
    head = mr.parse_results(write_eval(tmp_path / "h.jsonl", head_rows), "head")
    diff = mr.compare(base, head, "v0.0.8", "abc123456789")

    assert diff.models_changed == ["gpt-5.4-mini"]
    assert diff.removed == [], "a dropped model is not a dropped test"
    assert diff.added == []
    assert diff.clean


def test_progress_bar_encodes_gain_and_loss(mr):
    assert mr._bar(0.5, 0.5, width=10) == "🟦" * 5 + "⬜" * 5
    assert mr._bar(0.5, 0.8, width=10) == "🟦" * 5 + "🟩" * 3 + "⬜" * 2
    assert mr._bar(0.8, 0.5, width=10) == "🟦" * 5 + "🟥" * 3 + "⬜" * 2
    assert len(mr._bar(0.37, 0.94, width=20)) == 20 * len("🟦")


def test_markdown_report_has_the_expected_sections(mr, tmp_path):
    base = mr.parse_results(write_eval(tmp_path / "b.jsonl", all_passing()), "v0.0.8")
    head = mr.parse_results(write_eval(tmp_path / "h.jsonl", all_passing()), "head")
    md = mr.compare(base, head, "v0.0.8", "abc123456789", runs=3).markdown

    assert md.startswith("## 🧪 Capability Test Results")
    for section in (
        "Per-model breakdown",
        "Per-tier breakdown",
        "Per-test breakdown",
        "| Samples Passed |",
        "Task-bootstrap delta",
        "Total Tokens",
    ):
        assert section in md
    assert md.rstrip().endswith("*2 models × 3 runs* | *both arms run fresh*")


def test_release_script_reuses_shared_capability_comparator(mr):
    assert mr.compare is mr.capability_ab.compare
    assert mr.parse_results is mr.capability_ab.parse_results
