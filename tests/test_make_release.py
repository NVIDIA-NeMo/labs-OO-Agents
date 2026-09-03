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
import subprocess
import sys
from collections.abc import Sequence
from contextlib import contextmanager
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

    assert diff.added == ["claude-haiku/brand_new"]
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

    assert sorted(diff.removed) == ["claude-haiku/router_multi", "gpt-5.4-mini/router_multi"]
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
    for section in ("Per-tier breakdown", "Per-test breakdown", "| Tests Passed |", "Total Tokens"):
        assert section in md
    assert md.rstrip().endswith("*2 models × 3 runs* | *both arms run fresh*")


@pytest.mark.parametrize(
    "tag",
    ["0.0.10", "v1.2", "v01.2.3", "v1.02.3", "v1.2.03", "v1.2.3-rc1", "v1.2.3;bad"],
)
def test_release_tag_validation_fails_closed(mr, tag):
    with pytest.raises(mr.ReleaseError):
        mr.validate_tag(tag)


def test_exact_sha_validation(mr):
    sha = "a" * 40
    assert mr.validate_candidate_sha(sha) == sha
    for invalid in ("a" * 39, "A" * 40, "g" * 40, "a" * 40 + ";echo"):
        with pytest.raises(mr.ReleaseError):
            mr.validate_candidate_sha(invalid)


def test_git_returns_empty_when_an_allowed_failure_echoes_the_unresolved_ref(mr, monkeypatch):
    unresolved = "refs/tags/v0.0.10^{commit}"
    monkeypatch.setattr(
        mr,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 128, unresolved + "\n", ""),
    )

    assert mr.git("rev-parse", unresolved, check=False) == ""


def test_tool_versions_records_missing_diagnostic_tools(mr, monkeypatch):
    monkeypatch.setattr(mr.shutil, "which", lambda _executable: None)
    monkeypatch.setattr(
        mr,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing diagnostic tool was executed")
        ),
    )

    versions = mr.tool_versions("release-image@sha256:digest")

    assert versions["runner_image"] == "release-image@sha256:digest"
    for tool in ("uv", "git", "rg", "sha256sum", "base64", "curl", "gcc", "make", "gh"):
        assert versions[tool] == "unavailable"


@pytest.mark.parametrize(
    "candidate_ref",
    ["refs/heads/topic", "refs/pull/0/head", "refs/pull/163/merge", "refs/pull/1/head:evil"],
)
def test_candidate_ref_validation_is_limited_to_pull_heads(mr, candidate_ref):
    with pytest.raises(mr.ReleaseError):
        mr.validate_candidate_ref(candidate_ref)


def test_ci_candidate_remains_valid_when_main_advances(mr, monkeypatch):
    candidate = "1" * 40
    advanced_main = "2" * 40
    previous = "3" * 40

    def fake_git(*args, **_kwargs):
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "HEAD"
        if args == ("status", "--porcelain"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return candidate
        if args == ("rev-parse", "origin/main"):
            return advanced_main
        if args[0] == "describe":
            return "v0.0.9"
        if args == ("rev-parse", "v0.0.9^{commit}"):
            return previous
        raise AssertionError(args)

    commands = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mr, "git", fake_git)
    monkeypatch.setattr(mr, "run", fake_run)
    monkeypatch.setattr(mr, "validate_existing_release", lambda *_args: None)

    head, _tag, _prev, _release = mr.preflight("v0.0.10", False, candidate_sha=candidate, ci=True)

    assert head == candidate
    assert ["git", "merge-base", "--is-ancestor", candidate, advanced_main] in commands


def test_ci_unmerged_rehearsal_skips_only_main_reachability(mr, monkeypatch):
    candidate = "1" * 40
    advanced_main = "2" * 40
    previous = "3" * 40

    def fake_git(*args, **_kwargs):
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "HEAD"
        if args == ("status", "--porcelain"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return candidate
        if args == ("rev-parse", "origin/main"):
            return advanced_main
        if args[0] == "describe":
            return "v0.0.9"
        if args == ("rev-parse", "v0.0.9^{commit}"):
            return previous
        raise AssertionError(args)

    commands = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mr, "git", fake_git)
    monkeypatch.setattr(mr, "run", fake_run)
    monkeypatch.setattr(
        mr,
        "validate_existing_release",
        lambda *_args: (_ for _ in ()).throw(AssertionError("release state is production-only")),
    )

    head, _tag, _prev, existing = mr.preflight(
        "v0.0.10",
        False,
        candidate_sha=candidate,
        ci=True,
        allow_unmerged_candidate=True,
    )

    assert head == candidate
    assert existing is None
    assert not any("merge-base" in command for command in commands)


def test_existing_release_must_be_matching_unpublished_draft(mr, monkeypatch):
    sha = "a" * 40
    monkeypatch.setattr(
        mr, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", "")
    )
    monkeypatch.setattr(
        mr, "release_for_tag", lambda _tag: {"draft": True, "target_commitish": sha}
    )
    assert mr.validate_existing_release("v1.2.3", sha)["draft"]

    monkeypatch.setattr(
        mr, "release_for_tag", lambda _tag: {"draft": False, "target_commitish": sha}
    )
    with pytest.raises(mr.ReleaseError, match="already published"):
        mr.validate_existing_release("v1.2.3", sha)

    monkeypatch.setattr(
        mr, "release_for_tag", lambda _tag: {"draft": True, "target_commitish": "b" * 40}
    )
    with pytest.raises(mr.ReleaseError, match="not frozen candidate"):
        mr.validate_existing_release("v1.2.3", sha)


def test_release_lookup_uses_draft_aware_gh_command(mr, monkeypatch):
    metadata = {
        "apiUrl": f"https://api.github.com/repos/{mr.GITHUB_REPO}/releases/7",
        "isDraft": True,
        "targetCommitish": "a" * 40,
        "url": "https://draft",
    }
    commands = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, json.dumps(metadata), "")

    monkeypatch.setattr(mr, "run", fake_run)

    assert mr.release_for_tag("v1.2.3") == {
        "id": 7,
        "draft": True,
        "target_commitish": "a" * 40,
        "html_url": "https://draft",
    }
    assert commands == [
        [
            "gh",
            "release",
            "view",
            "v1.2.3",
            "--repo",
            mr.GITHUB_REPO,
            "--json",
            "apiUrl,isDraft,targetCommitish,url",
        ]
    ]


def test_release_lookup_returns_none_when_cli_reports_not_found(mr, monkeypatch):
    monkeypatch.setattr(
        mr,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 1, "", "release not found"),
    )

    assert mr.release_for_tag("v1.2.3") is None


def test_release_lookup_fails_closed_on_api_error_or_invalid_metadata(mr, monkeypatch):
    monkeypatch.setattr(
        mr,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 1, "", "HTTP 403"),
    )
    with pytest.raises(mr.ReleaseError, match="HTTP 403"):
        mr.release_for_tag("v1.2.3")

    monkeypatch.setattr(
        mr,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 0, "not-json", ""),
    )
    with pytest.raises(mr.ReleaseError, match="invalid release metadata"):
        mr.release_for_tag("v1.2.3")


def test_release_lookup_rejects_string_draft_state(mr, monkeypatch):
    metadata = {
        "apiUrl": f"https://api.github.com/repos/{mr.GITHUB_REPO}/releases/7",
        "isDraft": "false",
        "targetCommitish": "a" * 40,
        "url": "https://release",
    }
    monkeypatch.setattr(
        mr,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 0, json.dumps(metadata), ""),
    )

    with pytest.raises(mr.ReleaseError, match="invalid release metadata"):
        mr.release_for_tag("v1.2.3")


@pytest.mark.parametrize(
    "api_url",
    [
        "https://api.github.com/repos/NVIDIA-NeMo/labs-OO-Agents/releases/0",
        "https://api.github.com/repos/NVIDIA-NeMo/labs-OO-Agents/releases/not-an-id",
        "https://example.com/repos/NVIDIA-NeMo/labs-OO-Agents/releases/7",
    ],
)
def test_release_lookup_rejects_invalid_api_url(mr, monkeypatch, api_url):
    metadata = {
        "apiUrl": api_url,
        "isDraft": True,
        "targetCommitish": "a" * 40,
        "url": "https://release",
    }
    monkeypatch.setattr(
        mr,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 0, json.dumps(metadata), ""),
    )

    with pytest.raises(mr.ReleaseError, match="invalid release metadata"):
        mr.release_for_tag("v1.2.3")


def test_strict_fast_checks_sync_all_public_ci_dependencies_first(mr, monkeypatch):
    commands = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mr, "run", fake_run)

    mr.fast_checks(verify_containment=True)

    assert commands[0] == [
        "uv",
        "sync",
        "--frozen",
        "--all-extras",
        "--no-extra",
        "sandbox",
    ]
    assert commands.index(["uv", "run", "ruff", "check", "."]) < commands.index(
        ["uv", "run", "pytest", "-q", "-m", "not integration and not stress"]
    )


def test_strict_arm_uses_explicit_wheel_without_env_or_ambient_extras(mr, tmp_path, monkeypatch):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "uv.lock").write_text("locked")
    wheel = tmp_path / "nemo_oo_agents_nvidia-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    output_root = tmp_path / "evidence"
    commands = []
    eval_env = None

    def forbidden_extras():
        raise AssertionError("strict CI must not inspect the ambient environment")

    def fake_run(cmd, **kwargs):
        nonlocal eval_env
        commands.append(cmd)
        if "eval_pipeline" in cmd:
            eval_env = kwargs["env"]
            result_dir = output_root / "candidate" / "run"
            result_dir.mkdir(parents=True)
            write_eval(
                result_dir / "result.noo-eval.jsonl",
                [("model-a", "test-a", "case-a", "stable", True, None)],
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mr, "run", fake_run)
    monkeypatch.setattr(mr, "env_extras", forbidden_extras)
    monkeypatch.setenv("OTLP_ENDPOINT", "https://viewer.invalid/v1/traces")

    arm = mr.run_capability_arm(
        tree,
        "candidate",
        "candidate",
        ["model-a"],
        1,
        1,
        strict_ci=True,
        internal_wheel=wheel,
        output_root=output_root,
        sample_timeout=10,
        hang_timeout=20,
        overall_timeout=30,
    )

    assert arm.counts() == (1, 1)
    assert ["uv", "sync", "--frozen", "--all-extras", "--no-extra", "sandbox"] in commands
    assert any(str(wheel) in cmd and "pip" in cmd for cmd in commands)
    eval_cmd = next(cmd for cmd in commands if "eval_pipeline" in cmd)
    assert "--no-cache" in eval_cmd and "--trace-files" in eval_cmd
    assert "--timeout" in eval_cmd and "--hang-timeout" in eval_cmd
    assert "OTLP_ENDPOINT" not in eval_env
    assert not (tree / ".env").exists()


def test_strict_capability_installs_same_wheel_in_both_fresh_arms(mr, tmp_path, monkeypatch):
    wheel = tmp_path / "nemo_oo_agents_nvidia-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    calls = []

    @contextmanager
    def fake_worktree(ref):
        tree = tmp_path / ref.replace("/", "_")
        tree.mkdir(exist_ok=True)
        yield tree

    def fake_arm(tree, label, cache_key, models, runs, limit, **kwargs):
        calls.append((tree, kwargs["internal_wheel"], kwargs["strict_ci"]))
        arm = mr.ArmResults(label)
        arm.by_case["case"] = [True]
        arm.case_tier["case"] = "stable"
        arm.by_test[(models[0], "test")] = [True]
        arm.lock_sha256 = "f" * 64
        return arm

    monkeypatch.setattr(mr, "worktree_at", fake_worktree)
    monkeypatch.setattr(mr, "run_capability_arm", fake_arm)
    diff, _base, _head = mr.capability_diff(
        "v0.0.9",
        "a" * 40,
        ["model-a"],
        1,
        1,
        strict_ci=True,
        internal_wheel=wheel,
        artifact_dir=tmp_path / "artifacts",
    )

    assert diff.hard_gate_passed
    assert len(calls) == 2
    assert all(call[1:] == (wheel, True) for call in calls)
    assert calls[0][0] != calls[1][0]


def test_advisories_do_not_become_hard_gate_failures(mr):
    diff = mr.Diff(regressions=["collapse"], beyond_noise=["overall -6 pts"])
    assert not diff.clean
    assert diff.hard_gate_passed
    diff.floor_breach = "stable floor"
    assert not diff.hard_gate_passed


@pytest.mark.parametrize("existing", [None, {"id": 7, "draft": True}])
def test_draft_command_targets_full_sha_and_is_idempotent(mr, tmp_path, monkeypatch, existing):
    sha = "a" * 40
    notes = tmp_path / "notes.md"
    notes.write_text("safe")
    commands = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mr, "run", fake_run)
    release = {
        "id": 7,
        "draft": True,
        "target_commitish": sha,
        "html_url": "https://draft",
    }
    responses = iter([existing, release])
    monkeypatch.setattr(mr, "validate_existing_release", lambda *_args: next(responses))
    url, release_id = mr.create_or_update_draft("v1.2.3", sha, notes)

    cmd = commands[0]
    assert cmd[:3] == ["gh", "release", "edit" if existing else "create"]
    assert cmd[cmd.index("--target") + 1] == sha
    assert "--draft" in cmd
    assert not any("draft=false" in part for part in cmd)
    assert (url, release_id) == ("https://draft", 7)


def test_draft_is_rechecked_before_mutation(mr, tmp_path, monkeypatch):
    notes = tmp_path / "notes.md"
    notes.write_text("safe")
    commands = []
    monkeypatch.setattr(
        mr,
        "validate_existing_release",
        lambda *_args: (_ for _ in ()).throw(mr.ReleaseError("already published")),
    )
    monkeypatch.setattr(mr, "run", lambda cmd, **_kwargs: commands.append(cmd))
    with pytest.raises(mr.ReleaseError, match="already published"):
        mr.create_or_update_draft("v1.2.3", "a" * 40, notes)
    assert commands == []


def test_public_report_sanitization(mr):
    unsafe = (
        "api_key=supersecret\nAuthorization: Bearer-token\n"
        "https://alice:password@internal.example/path\n/home/alice/private/trace.jsonl\n"
    )
    safe = mr.sanitize_public_text(unsafe)
    assert "supersecret" not in safe
    assert "Bearer-token" not in safe
    assert "alice:password" not in safe
    assert "/home/alice" not in safe
    assert "[REDACTED]" in safe and "[private-path]" in safe


@pytest.mark.parametrize(
    ("capability_gate_ran", "diff", "expected", "unexpected"),
    [
        (False, None, "NOT RUN — the capability gate was skipped", "PASS"),
        (True, None, "PASS — all automated hard gates passed", "NOT RUN"),
        (True, "stable floor", "FAIL — stable floor", "PASS"),
    ],
)
def test_public_notes_report_the_actual_capability_gate_state(
    mr, capability_gate_ran, diff, expected, unexpected
):
    notes = mr.build_public_notes(
        tag="v1.2.3",
        sha="a" * 40,
        prev_tag="v1.2.2",
        diff=mr.Diff(floor_breach=diff, markdown="capability report"),
        change_notes="changes",
        pipeline_url="https://gitlab.example/pipelines/1",
        distributions=[],
        capability_gate_ran=capability_gate_ran,
    )

    assert expected in notes
    assert unexpected not in notes


def _ci_args(mr, tmp_path):
    wheel = tmp_path / "nemo_oo_agents_nvidia-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    return mr._parser().parse_args(
        [
            "v1.2.3",
            "--ci",
            "--candidate-sha",
            "a" * 40,
            "--controller-sha",
            "b" * 40,
            "--internal-wheel",
            str(wheel),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--pipeline-url",
            "https://gitlab.example/pipelines/1",
            "--job-url",
            "https://gitlab.example/jobs/2",
            "--image-digest",
            "registry.example/release@sha256:" + "c" * 64,
            "--create-draft",
        ]
    )


def test_unmerged_candidate_requires_reduced_scope_and_never_drafts(mr, tmp_path, monkeypatch):
    args = _ci_args(mr, tmp_path)
    args.candidate_ref = "refs/pull/163/head"
    args.create_draft = False
    monkeypatch.setenv("NVIDIA_INTERNAL_API_KEY", "disposable-test-key")
    monkeypatch.delenv("GH_TOKEN", raising=False)

    with pytest.raises(mr.ReleaseError, match="requires a reduced rehearsal"):
        mr.ci_main(args)

    args.models = "claude-haiku"
    args.runs = 1
    args.limit = 1
    args.create_draft = True
    with pytest.raises(mr.ReleaseError, match="can never create a draft"):
        mr.ci_main(args)


def test_unmerged_rehearsal_does_not_require_github_token(mr, tmp_path, monkeypatch):
    args = _ci_args(mr, tmp_path)
    args.candidate_ref = "refs/pull/163/head"
    args.create_draft = False
    args.models = "claude-haiku"
    args.runs = 1
    args.limit = 1
    monkeypatch.setenv("NVIDIA_INTERNAL_API_KEY", "disposable-test-key")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(mr, "tool_versions", lambda _image: {})
    monkeypatch.setattr(mr, "sha256", lambda _path: "e" * 64)
    monkeypatch.setattr(
        mr,
        "preflight",
        lambda *_args, **_kwargs: ("a" * 40, "v1.2.2", "d" * 40, None),
    )
    monkeypatch.setattr(mr, "fast_checks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mr, "build_and_smoke", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mr, "_copy_distributions", lambda _path: [])
    base = mr.ArmResults("base")
    head = mr.ArmResults("head")
    for arm in (base, head):
        arm.by_case["case"] = [True]
        arm.case_tier["case"] = "stable"
        arm.by_test[("model", "test")] = [True]
        arm.lock_sha256 = "f" * 64
    clean = mr.Diff(markdown="clean")
    monkeypatch.setattr(mr, "capability_diff", lambda *_args, **_kwargs: (clean, base, head))
    monkeypatch.setattr(mr, "local_change_notes", lambda *_args: "local changes")
    monkeypatch.setattr(
        mr,
        "generated_change_notes",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unmerged rehearsal called GitHub")),
    )

    assert mr.ci_main(args) == 0
    manifest = json.loads((tmp_path / "artifacts" / "release-manifest.json").read_text())
    assert manifest["status"] == "passed"
    assert manifest["unmerged_candidate"] is True


@pytest.mark.parametrize("failure_point", ["deterministic", "infrastructure", "hard-gate"])
def test_ci_never_creates_draft_after_gate_failure(mr, tmp_path, monkeypatch, failure_point):
    args = _ci_args(mr, tmp_path)
    monkeypatch.setenv("NVIDIA_INTERNAL_API_KEY", "masked")
    monkeypatch.setenv("GH_TOKEN", "masked")
    monkeypatch.setattr(mr, "tool_versions", lambda _image: {})
    monkeypatch.setattr(
        mr,
        "preflight",
        lambda *_args, **_kwargs: ("a" * 40, "v1.2.2", "d" * 40, None),
    )
    monkeypatch.setattr(mr, "sha256", lambda _path: "e" * 64)
    monkeypatch.setattr(mr, "build_and_smoke", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mr, "_copy_distributions", lambda _path: [])
    draft_calls = []
    monkeypatch.setattr(mr, "create_or_update_draft", lambda *_args: draft_calls.append(True))

    if failure_point == "deterministic":
        monkeypatch.setattr(
            mr,
            "fast_checks",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(mr.ReleaseError("lint")),
        )
    else:
        monkeypatch.setattr(mr, "fast_checks", lambda *_args, **_kwargs: None)
        if failure_point == "infrastructure":
            monkeypatch.setattr(
                mr,
                "capability_diff",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(mr.ReleaseError("endpoint")),
            )
        else:
            base = mr.ArmResults("base")
            head = mr.ArmResults("head")
            blocked = mr.Diff(floor_breach="stable floor", markdown="blocked")
            monkeypatch.setattr(
                mr, "capability_diff", lambda *_args, **_kwargs: (blocked, base, head)
            )

    with pytest.raises(mr.ReleaseError):
        mr.ci_main(args)
    assert draft_calls == []
    manifest = json.loads((tmp_path / "artifacts" / "release-manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_noninteractive_ci_drafts_when_only_advisories_exist(mr, tmp_path, monkeypatch):
    args = _ci_args(mr, tmp_path)
    monkeypatch.setenv("NVIDIA_INTERNAL_API_KEY", "masked")
    monkeypatch.setenv("GH_TOKEN", "masked")
    monkeypatch.setattr(mr, "tool_versions", lambda _image: {})
    monkeypatch.setattr(
        mr,
        "preflight",
        lambda *_args, **_kwargs: ("a" * 40, "v1.2.2", "d" * 40, None),
    )
    monkeypatch.setattr(mr, "sha256", lambda _path: "e" * 64)
    monkeypatch.setattr(mr, "fast_checks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mr, "build_and_smoke", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mr, "_copy_distributions", lambda _path: [])
    base = mr.ArmResults("base")
    head = mr.ArmResults("head")
    for arm in (base, head):
        arm.by_case["case"] = [True]
        arm.case_tier["case"] = "stable"
        arm.by_test[("model", "test")] = [True]
        arm.lock_sha256 = "f" * 64
    advisory = mr.Diff(regressions=["collapse"], markdown="advisory report")
    monkeypatch.setattr(mr, "capability_diff", lambda *_args, **_kwargs: (advisory, base, head))
    monkeypatch.setattr(mr, "generated_change_notes", lambda *_args: "changes")
    draft_calls = []

    def fake_draft(*call_args):
        draft_calls.append(call_args)
        return "https://github.example/draft", 42

    monkeypatch.setattr(mr, "create_or_update_draft", fake_draft)

    assert mr.ci_main(args) == 0
    assert len(draft_calls) == 1
    manifest = json.loads((tmp_path / "artifacts" / "release-manifest.json").read_text())
    assert manifest["status"] == "passed"
    assert manifest["capability"]["hard_gate_outcome"] == "passed"
    assert manifest["capability"]["advisory"]["collapses"] == ["collapse"]
    assert manifest["github_release_database_id"] == 42


def test_release_runner_contains_no_publish_operation(mr):
    source = (REPO / "scripts/make_release.py").read_text()
    assert "--draft=false" not in source
    assert "def publish(" not in source


def test_existing_publication_workflow_still_uses_published_release_trigger():
    workflow = (REPO / ".github/workflows/publish.yml").read_text()
    assert "release:" in workflow
    assert "types: [published]" in workflow
    assert "workflow_dispatch:" in workflow
    assert "if: github.event_name == 'release'" in workflow
