# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a fresh capability A/B comparison between two git revisions.

Both revisions execute in isolated worktrees against the same model endpoints
and credentials. A relative config path is resolved inside each revision so
test-harness changes remain part of the comparison; an absolute config path is
shared verbatim between arms when the harness must be held constant.

Examples:

    uv run python scripts/capability_ab.py v0.0.8 HEAD

    uv run python scripts/capability_ab.py main HEAD \
      --config /tmp/six-model-capability.yaml \
      --models claude-haiku,gpt-5.4-mini,nemotron3-nano-30b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path("tests/capability/config.yaml")
DEFAULT_RUNS = 3
DEFAULT_PARALLEL = 40
DEFAULT_BOOTSTRAP_SAMPLES = 100_000
DEFAULT_SEED = 20260819

# Release-policy defaults. Callers may supply a different policy without
# changing how arms are prepared or how results are parsed.
STABLE_FLOOR = 0.60
AGGREGATE_NOISE_PTS = 5.0
MAX_ERROR_RATE = 0.50
COLLAPSE_BEFORE = 0.80
COLLAPSE_AFTER = 0.20

BOLD, DIM, RED, YELLOW, GREEN, RESET = (
    ("\033[1m", "\033[2m", "\033[31m", "\033[33m", "\033[32m", "\033[0m")
    if sys.stdout.isatty()
    else ("", "", "", "", "", "")
)

CaseKey = tuple[str, str, str]
TaskKey = tuple[str, str]


class CapabilityABError(RuntimeError):
    """Raised when an A/B arm is invalid or infrastructure preparation fails."""


@dataclass(frozen=True)
class ComparisonPolicy:
    stable_floor: float = STABLE_FLOOR
    aggregate_noise_points: float = AGGREGATE_NOISE_PTS
    max_error_rate: float = MAX_ERROR_RATE
    collapse_before: float = COLLAPSE_BEFORE
    collapse_after: float = COLLAPSE_AFTER


@dataclass
class ArmResults:
    """Parsed results for one revision."""

    label: str
    result_path: Path | None = None
    by_case: dict[CaseKey, list[bool]] = field(default_factory=lambda: defaultdict(list))
    case_tier: dict[CaseKey, str] = field(default_factory=dict)
    by_task: dict[TaskKey, list[bool]] = field(default_factory=lambda: defaultdict(list))
    task_tier: dict[TaskKey, str] = field(default_factory=dict)
    by_run: dict[int, list[bool]] = field(default_factory=lambda: defaultdict(list))
    errors: dict[CaseKey, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    output_tokens: int = 0
    total_tokens: int = 0
    errored: int = 0

    def error_rate(self) -> float:
        _, total = self.counts()
        return self.errored / total if total else 0.0

    def rate(self, key: CaseKey) -> float | None:
        values = self.by_case.get(key)
        return statistics.mean(values) if values else None

    def counts(self) -> tuple[int, int]:
        flags = [flag for values in self.by_case.values() for flag in values]
        return sum(flags), len(flags)

    def overall(self) -> float:
        passed, total = self.counts()
        return passed / total if total else 0.0

    def tier_counts(self) -> dict[str, tuple[int, int]]:
        grouped: dict[str, list[bool]] = defaultdict(list)
        for key, flags in self.by_case.items():
            grouped[self.case_tier.get(key, "stable")].extend(flags)
        return {tier: (sum(flags), len(flags)) for tier, flags in grouped.items()}

    def per_model(self) -> dict[str, float]:
        grouped: dict[str, list[bool]] = defaultdict(list)
        for (model, _, _), flags in self.by_case.items():
            grouped[model].extend(flags)
        return {model: statistics.mean(flags) for model, flags in grouped.items() if flags}

    def model_names(self) -> set[str]:
        return {model for model, _, _ in self.by_case}


@dataclass
class Diff:
    regressions: list[str] = field(default_factory=list)
    new_errors: list[str] = field(default_factory=list)
    beyond_noise: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    models_changed: list[str] = field(default_factory=list)
    floor_breach: str | None = None
    markdown: str = ""
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not (
            self.regressions
            or self.new_errors
            or self.beyond_noise
            or self.removed
            or self.floor_breach
        )


@dataclass
class ABRunResult:
    base: ArmResults
    head: ArmResults
    diff: Diff
    base_sha: str
    head_sha: str
    report_path: Path
    summary_path: Path


def _run(
    command: list[str],
    *,
    cwd: Path,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if not capture:
        print(f"  {DIM}$ {' '.join(command)}{RESET}", flush=True)
    environment = os.environ.copy()
    if command and command[0] == "uv":
        # Each isolated checkout owns its own project environment. Inheriting
        # the caller's VIRTUAL_ENV makes uv warn and risks targeting the wrong
        # environment for pip-interface commands.
        environment.pop("VIRTUAL_ENV", None)
    process = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=capture,
        check=False,
    )
    if check and process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()
        raise CapabilityABError(
            f"command failed ({process.returncode}): {' '.join(command)}\n{detail}"
        )
    return process


def _git(repo: Path, *arguments: str, check: bool = True) -> str:
    return _run(["git", *arguments], cwd=repo, check=check).stdout.strip()


def parse_results(path: Path, label: str) -> ArmResults:
    """Parse an eval JSONL file, retaining model/test/case granularity."""
    arm = ArmResults(label=label, result_path=path)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("_type") != "result":
            continue

        model = str(record.get("model") or "?")
        test_name = str(record.get("test_name") or record.get("agent_class") or "?")
        test_case = str(record.get("test_case") or test_name)
        tier = str(record.get("tier") or "stable")
        passed = bool(record.get("passed"))
        case_key = (model, test_name, test_case)
        task_key = (test_name, test_case)

        arm.by_case[case_key].append(passed)
        arm.case_tier.setdefault(case_key, tier)
        arm.by_task[task_key].append(passed)
        arm.task_tier.setdefault(task_key, tier)
        try:
            run_id = int(record.get("run_id", 1))
        except (TypeError, ValueError):
            run_id = 1
        arm.by_run[run_id].append(passed)

        if record.get("error_type"):
            arm.errors[case_key][str(record["error_type"])] += 1
            arm.errored += 1
        arm.output_tokens += int(record.get("output_tokens") or 0)
        arm.total_tokens += int(record.get("total_tokens") or 0)
    return arm


def newest_eval(output_dir: Path) -> Path | None:
    found = sorted(output_dir.rglob("*.noo-eval.jsonl"), key=lambda path: path.stat().st_mtime)
    return found[-1] if found else None


def configured_models(config: Path) -> list[str]:
    """Return the config's declared agent-model matrix."""
    document = yaml.safe_load(config.read_text()) or {}
    models = document.get("agent_models")
    if isinstance(models, list) and models:
        return [str(model) for model in models]
    mapping = document.get("models")
    if isinstance(mapping, dict) and mapping:
        return [str(model) for model in mapping]
    raise CapabilityABError(f"{config} declares neither agent_models nor models")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_rates(
    before: dict[TaskKey, list[bool]],
    after: dict[TaskKey, list[bool]],
    tasks: list[TaskKey],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not tasks:
        return {}
    before_rates = [statistics.mean(before[task]) for task in tasks]
    after_rates = [statistics.mean(after[task]) for task in tasks]
    deltas = [new - old for old, new in zip(before_rates, after_rates, strict=True)]
    rng = random.Random(seed)
    boot_delta: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(tasks)) for _ in tasks]
        boot_delta.append(statistics.mean(deltas[index] for index in indices))
    return {
        "tasks": len(tasks),
        "before_rate": statistics.mean(before_rates),
        "after_rate": statistics.mean(after_rates),
        "delta": statistics.mean(deltas),
        "delta_ci95": [_percentile(boot_delta, 0.025), _percentile(boot_delta, 0.975)],
        "improved_tasks": sum(delta > 0 for delta in deltas),
        "regressed_tasks": sum(delta < 0 for delta in deltas),
        "tied_tasks": sum(delta == 0 for delta in deltas),
    }


def bootstrap_comparison(
    base: ArmResults,
    head: ArmResults,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Paired task-clustered uncertainty, retaining all model/run observations."""
    shared = sorted(set(base.by_task) & set(head.by_task))
    results: dict[str, Any] = {
        "overall": _bootstrap_rates(base.by_task, head.by_task, shared, samples=samples, seed=seed),
        "bootstrap_samples": samples,
        "seed": seed,
    }
    tiers = sorted(
        {base.task_tier.get(task, head.task_tier.get(task, "stable")) for task in shared}
    )
    for index, tier in enumerate(tiers, start=1):
        tasks = [
            task
            for task in shared
            if base.task_tier.get(task, head.task_tier.get(task, "stable")) == tier
        ]
        results[tier] = _bootstrap_rates(
            base.by_task,
            head.by_task,
            tasks,
            samples=samples,
            seed=seed + index,
        )

    per_model: dict[str, Any] = {}
    for index, model in enumerate(sorted(base.model_names() & head.model_names()), start=100):
        before = {
            (test, case): flags
            for (candidate, test, case), flags in base.by_case.items()
            if candidate == model
        }
        after = {
            (test, case): flags
            for (candidate, test, case), flags in head.by_case.items()
            if candidate == model
        }
        tasks = sorted(set(before) & set(after))
        per_model[model] = _bootstrap_rates(
            before, after, tasks, samples=samples, seed=seed + index
        )
    results["per_model"] = per_model

    def run_rates(arm: ArmResults) -> list[float]:
        return [statistics.mean(flags) for _, flags in sorted(arm.by_run.items())]

    base_runs, head_runs = run_rates(base), run_rates(head)
    results["run_rates"] = {
        "before": base_runs,
        "after": head_runs,
        "before_population_sd": statistics.pstdev(base_runs) if len(base_runs) > 1 else 0.0,
        "after_population_sd": statistics.pstdev(head_runs) if len(head_runs) > 1 else 0.0,
    }
    return results


def _mark(delta: float, *, inverse: bool = False) -> str:
    if delta == 0:
        return "➖"
    good = delta < 0 if inverse else delta > 0
    return "✅" if good else "❌"


def _bar(base_rate: float, head_rate: float, width: int = 20) -> str:
    now, was = round(head_rate * width), round(base_rate * width)
    shared = min(now, was)
    gained, lost = max(0, now - was), max(0, was - now)
    return "🟦" * shared + "🟩" * gained + "🟥" * lost + "⬜" * (width - shared - gained - lost)


def compare(
    base: ArmResults,
    head: ArmResults,
    prev_tag: str,
    sha: str,
    runs: int = DEFAULT_RUNS,
    *,
    policy: ComparisonPolicy | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> Diff:
    """Classify a capability delta and render a release-compatible report."""
    policy = policy or ComparisonPolicy()
    diff = Diff()
    base_passed, base_total = base.counts()
    head_passed, head_total = head.counts()
    base_rate, head_rate = base.overall(), head.overall()
    delta_points = (head_rate - base_rate) * 100

    shared_models = base.model_names() & head.model_names()
    diff.models_changed = sorted((base.model_names() | head.model_names()) - shared_models)

    for key in sorted(set(base.by_case) | set(head.by_case)):
        model, test_name, test_case = key
        if model not in shared_models:
            continue
        before_rate, after_rate = base.rate(key), head.rate(key)
        display = f"{model}/{test_name}/{test_case}"
        if before_rate is None:
            diff.added.append(display)
            continue
        if after_rate is None:
            diff.removed.append(display)
            continue
        if before_rate >= policy.collapse_before and after_rate <= policy.collapse_after:
            diff.regressions.append(
                f"| `{test_name}` | `{test_case}` | {model} | {before_rate:.0%} | "
                f"{after_rate:.0%} | {(after_rate - before_rate) * 100:+.1f}% ❌ |"
            )
        previous_errors = set(base.errors.get(key, {}))
        for error_type, count in head.errors.get(key, {}).items():
            if error_type not in previous_errors:
                diff.new_errors.append(
                    f"| `{test_name}` | `{test_case}` | {model} | {count}× `{error_type}` | "
                    "0 before |"
                )

    if delta_points < -policy.aggregate_noise_points:
        diff.beyond_noise.append(
            f"overall {delta_points:+.1f} pts (band ±{policy.aggregate_noise_points:g})"
        )
    base_models, head_models = base.per_model(), head.per_model()
    for model in sorted(set(base_models) & set(head_models)):
        delta = (head_models[model] - base_models[model]) * 100
        if delta < -policy.aggregate_noise_points:
            diff.beyond_noise.append(f"{model} {delta:+.1f} pts")

    base_tiers, head_tiers = base.tier_counts(), head.tier_counts()
    for tier in sorted(set(base_tiers) & set(head_tiers)):
        old_passed, old_total = base_tiers[tier]
        new_passed, new_total = head_tiers[tier]
        if not (old_total and new_total):
            continue
        delta = (new_passed / new_total - old_passed / old_total) * 100
        if delta < -policy.aggregate_noise_points:
            diff.beyond_noise.append(f"{tier} tier {delta:+.1f} pts")

    stable_passed, stable_total = head_tiers.get("stable", (0, 0))
    stable_rate = stable_passed / stable_total if stable_total else 0.0
    if stable_total and stable_rate < policy.stable_floor:
        diff.floor_breach = f"stable tier at {stable_rate:.1%}, floor is {policy.stable_floor:.0%}"

    diff.statistics = bootstrap_comparison(base, head, samples=bootstrap_samples, seed=seed)
    overall_stats = diff.statistics.get("overall", {})

    markdown: list[str] = ["## 🧪 Capability Test Results", ""]
    if diff.floor_breach:
        markdown.append(f"❌ **Release BLOCKED** — {diff.floor_breach}")
    elif diff.clean:
        markdown.append(
            f"✅ **Release OK** — Stable tier at {stable_rate:.1%} "
            f"(floor: {policy.stable_floor:.0%}), no regressions beyond noise"
        )
    else:
        markdown.append(
            f"⚠️ **Review required** — Stable tier at {stable_rate:.1%} "
            f"(floor: {policy.stable_floor:.0%}), {len(diff.regressions)} collapse(s), "
            f"{len(diff.new_errors)} new error type(s), {len(diff.removed)} removed test(s)"
        )
    markdown += [
        "",
        "---",
        "",
        f"**{head_rate:.1%}** {_bar(base_rate, head_rate)} **{delta_points:+.1f}%**",
        "",
        f"{head_passed}/{head_total} samples passing *({head_passed - base_passed:+d} from {prev_tag})*",
        "",
        f"| Metric | {prev_tag} | This revision | Change |",
        "|--------|----------|---------------|--------|",
        f"| Samples Passed | {base_passed}/{base_total} | {head_passed}/{head_total} | "
        f"{head_passed - base_passed:+d} {_mark(head_passed - base_passed)} |",
        f"| Success Rate | {base_rate:.1%} | {head_rate:.1%} | "
        f"{delta_points:+.1f}% {_mark(delta_points)} |",
        f"| Collapsed cases | — | {len(diff.regressions)} | "
        f"{len(diff.regressions)} {_mark(len(diff.regressions), inverse=True)} |",
        f"| New error types | — | {len(diff.new_errors)} | "
        f"{len(diff.new_errors)} {_mark(len(diff.new_errors), inverse=True)} |",
        f"| Output Tokens | {base.output_tokens:,} | {head.output_tokens:,} | "
        f"{head.output_tokens - base.output_tokens:+,} |",
        f"| Total Tokens | {base.total_tokens:,} | {head.total_tokens:,} | "
        f"{head.total_tokens - base.total_tokens:+,} |",
    ]
    if overall_stats:
        low, high = overall_stats["delta_ci95"]
        markdown += [
            f"| Task-bootstrap delta (95% CI) | — | {overall_stats['delta'] * 100:+.2f} pts | "
            f"{low * 100:+.2f} to {high * 100:+.2f} pts |",
        ]

    markdown += [
        "",
        "<details>",
        "<summary>📊 Per-model breakdown</summary>",
        "",
        f"| Model | {prev_tag} | This revision | Change | 95% task-bootstrap CI |",
        "|-------|----------|---------------|--------|-----------------------|",
    ]
    model_stats = diff.statistics.get("per_model", {})
    for model in sorted(set(base_models) | set(head_models)):
        if model not in base_models or model not in head_models:
            markdown.append(f"| {model} | — | — | only in one arm | — |")
            continue
        delta = (head_models[model] - base_models[model]) * 100
        stats = model_stats.get(model, {})
        ci = stats.get("delta_ci95")
        rendered_ci = f"{ci[0] * 100:+.2f} to {ci[1] * 100:+.2f}" if ci else "—"
        markdown.append(
            f"| {model} | {base_models[model]:.1%} | {head_models[model]:.1%} | "
            f"{delta:+.1f} {_mark(delta)} | {rendered_ci} |"
        )
    markdown += ["", "</details>", "", "<details>", "<summary>📊 Per-tier breakdown</summary>", ""]
    markdown += [
        f"| Tier | {prev_tag} | This revision | Change | Expected |",
        "|------|----------|---------------|--------|----------|",
    ]
    for tier in sorted(set(base_tiers) | set(head_tiers)):
        old_passed, old_total = base_tiers.get(tier, (0, 0))
        new_passed, new_total = head_tiers.get(tier, (0, 0))
        old_rate = old_passed / old_total if old_total else 0.0
        new_rate = new_passed / new_total if new_total else 0.0
        delta = (new_rate - old_rate) * 100
        expected = f"≥{policy.stable_floor:.0%}" if tier == "stable" else "—"
        markdown.append(
            f"| {tier.title()} | {old_passed}/{old_total} ({old_rate:.1%}) | "
            f"{new_passed}/{new_total} ({new_rate:.1%}) | {new_passed - old_passed:+d} / "
            f"{delta:+.1f}% {_mark(delta)} | {expected} |"
        )
    markdown += ["", "</details>", ""]

    if diff.regressions:
        markdown += [
            "<details open>",
            "<summary>❌ Collapsed cases</summary>",
            "",
            f"| Test | Case | Model | {prev_tag} | This revision | Change |",
            "|------|------|-------|----------|---------------|--------|",
            *diff.regressions,
            "",
            "</details>",
            "",
        ]
    if diff.new_errors:
        markdown += [
            "<details open>",
            "<summary>❌ New error types</summary>",
            "",
            "| Test | Case | Model | This revision | Baseline |",
            "|------|------|-------|---------------|----------|",
            *diff.new_errors,
            "",
            "</details>",
            "",
        ]

    markdown += [
        "<details>",
        "<summary>📋 Per-test breakdown</summary>",
        "",
        "| Test / case | Status |",
        "|-------------|--------|",
    ]
    for task in sorted(set(base.by_task) | set(head.by_task)):
        flags = head.by_task.get(task)
        display = "/".join(task)
        if flags is None:
            markdown.append(f"| `{display}` | ⬜ removed |")
            continue
        icon = "✅" if all(flags) else "❌"
        new = " *(new)*" if task not in base.by_task else ""
        markdown.append(f"| `{display}` | {icon} {sum(flags)}/{len(flags)}{new} |")
    markdown += [
        "",
        "</details>",
        "",
        f"*{prev_tag} → `{sha[:8]}`* | *{len(head.model_names())} models × {runs} runs* | "
        "*both arms run fresh*",
    ]
    diff.markdown = "\n".join(markdown)
    return diff


def print_comparison(base: ArmResults, head: ArmResults, diff: Diff) -> None:
    base_passed, base_total = base.counts()
    head_passed, head_total = head.counts()
    delta = (head.overall() - base.overall()) * 100
    print(f"\n{BOLD}{'═' * 72}{RESET}")
    print(f"{BOLD} CAPABILITY A/B   {base.label} → {head.label}{RESET}")
    print(f" {DIM}{len(head.model_names())} models · {len(head.by_task)} tasks{RESET}")
    print(f"{BOLD}{'═' * 72}{RESET}\n")
    print(f" {_bar(base.overall(), head.overall())}")
    print(
        f" {BOLD}{head.overall():.1%}{RESET}  {delta:+.1f} pts   "
        f"{head_passed}/{head_total} passing ({head_passed - base_passed:+d})"
    )
    print("\n PER MODEL")
    base_models, head_models = base.per_model(), head.per_model()
    for model in sorted(set(base_models) | set(head_models)):
        if model not in base_models or model not in head_models:
            print(f"   {model:<30} only in one arm")
            continue
        model_delta = (head_models[model] - base_models[model]) * 100
        print(
            f"   {model:<30} {base_models[model]:>6.1%} → "
            f"{head_models[model]:>6.1%}  {model_delta:+5.1f}"
        )
    verdict = "OK" if diff.clean else "REVIEW REQUIRED"
    if diff.floor_breach:
        verdict = "BLOCKED"
    print(f"\n{BOLD} VERDICT: {verdict}{RESET}")


def discover_env_extras(repo: Path) -> list[str]:
    """Return installed, out-of-lock packages that fresh worktrees need."""
    lock_path = repo / "uv.lock"
    locked = {
        name.lower()
        for name in re.findall(r'^name = "([^"]+)"', lock_path.read_text(), re.MULTILINE)
    }
    frozen = _run(["uv", "pip", "freeze"], cwd=repo).stdout.splitlines()
    extras: list[str] = []
    for raw in frozen:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e "):
            name, location = "", line[3:]
        elif " @ " in line:
            name, _, location = line.partition(" @ ")
        elif "==" in line:
            name, location = line.split("==", maxsplit=1)[0], ""
        else:
            continue
        resolved = location.strip().removeprefix("file://")
        if resolved.startswith(str(repo)):
            continue
        if name.strip().lower() in locked:
            continue
        extras.append(line.removeprefix("-e ").strip())
    return extras


@contextmanager
def isolated_worktree(repo: Path, revision: str, label: str) -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix=f"nooa-capability-{label}-"))
    tree = root / "tree"
    _git(repo, "worktree", "add", "--detach", str(tree), revision)
    try:
        yield tree
    finally:
        _git(repo, "worktree", "remove", "--force", str(tree), check=False)
        shutil.rmtree(root, ignore_errors=True)


def _prepare_tree(
    tree: Path,
    *,
    env_file: Path | None,
    extra_packages: list[str],
) -> None:
    if env_file and env_file.exists():
        target = tree / ".env"
        shutil.copyfile(env_file, target)
        target.chmod(0o600)
    _run(
        ["uv", "sync", "--all-extras", "--no-extra", "sandbox", "--inexact"],
        cwd=tree,
    )
    if extra_packages:
        python = tree / ".venv" / "bin" / "python"
        _run(
            ["uv", "pip", "install", "--python", str(python), *extra_packages],
            cwd=tree,
        )


def _config_for_tree(config: Path, tree: Path) -> Path:
    return config if config.is_absolute() else tree / config


def _config_digest(config: Path) -> str:
    return hashlib.sha256(config.read_bytes()).hexdigest()[:16]


def _run_arm(
    tree: Path,
    *,
    label: str,
    revision: str,
    config: Path,
    models: list[str],
    runs: int,
    parallel: int,
    output_dir: Path,
    limit: int | None,
    timeout: int | None,
    test_filter: str | None,
    no_cache: bool,
    trace_files: bool,
    reuse: bool,
    max_error_rate: float,
) -> ArmResults:
    resolved_config = _config_for_tree(config, tree)
    if not resolved_config.exists():
        raise CapabilityABError(f"{label}: config does not exist: {resolved_config}")
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "run.json"
    signature = {
        "revision": revision,
        "config": str(config),
        "config_digest": _config_digest(resolved_config),
        "models": models,
        "runs": runs,
        "parallel": parallel,
        "limit": limit,
        "timeout": timeout,
        "test_filter": test_filter,
        "no_cache": no_cache,
        "trace_files": trace_files,
    }
    existing = newest_eval(output_dir)
    if reuse and existing and marker.exists():
        if json.loads(marker.read_text()) == signature:
            print(f"  {GREEN}✓{RESET} {label}: reusing {existing.name}")
            return parse_results(existing, label)

    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "eval_pipeline",
        "--config",
        str(resolved_config),
        "--models",
        ",".join(models),
        "--runs",
        str(runs),
        "--parallel",
        str(parallel),
        "--output-dir",
        str(output_dir),
        "-q",
    ]
    if limit:
        command += ["--limit", str(limit)]
    if timeout:
        command += ["--timeout", str(timeout)]
    if test_filter:
        command += ["--test", test_filter]
    if no_cache:
        command.append("--no-cache")
    if trace_files:
        command.append("--trace-files")

    print(f"\n{BOLD}▶ {label}: {len(models)} models × {runs} runs{RESET}")
    _run(command, cwd=tree, capture=False)
    produced = newest_eval(output_dir)
    if not produced:
        raise CapabilityABError(f"{label}: no .noo-eval.jsonl produced under {output_dir}")
    arm = parse_results(produced, label)
    if not arm.by_case:
        raise CapabilityABError(f"{label}: eval produced no usable result records")
    if arm.error_rate() > max_error_rate:
        raise CapabilityABError(
            f"{label}: {arm.error_rate():.0%} of samples errored; refusing to compare infrastructure failure"
        )
    marker.write_text(json.dumps(signature, indent=2, sort_keys=True))
    return arm


def run_ab(
    *,
    repo: Path,
    base_ref: str,
    head_ref: str,
    config: Path = DEFAULT_CONFIG,
    models: list[str] | None = None,
    runs: int = DEFAULT_RUNS,
    parallel: int = DEFAULT_PARALLEL,
    output_root: Path | None = None,
    experiment: str | None = None,
    env_file: Path | None = None,
    extra_packages: list[str] | None = None,
    limit: int | None = None,
    timeout: int | None = None,
    test_filter: str | None = None,
    no_cache: bool = False,
    trace_files: bool = False,
    reuse: bool = False,
    head_first: bool = True,
    policy: ComparisonPolicy | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> ABRunResult:
    """Run and compare two revisions using isolated, identically prepared arms."""
    repo = repo.resolve()
    base_sha = _git(repo, "rev-parse", base_ref)
    head_sha = _git(repo, "rev-parse", head_ref)
    config = config if config.is_absolute() else Path(config)
    model_config = config if config.is_absolute() else repo / config
    selected_models = models or configured_models(model_config)
    if not selected_models:
        raise CapabilityABError("no models selected")
    policy = policy or ComparisonPolicy()

    output_root = (output_root or repo / "tmp" / "capability-ab").resolve()
    if experiment is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        experiment = f"{timestamp}_{base_sha[:8]}_{head_sha[:8]}"
    experiment_dir = output_root / experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)

    packages = list(extra_packages) if extra_packages is not None else discover_env_extras(repo)
    env_file = env_file or repo / ".env"

    with ExitStack() as stack:
        base_tree = stack.enter_context(isolated_worktree(repo, base_sha, "base"))
        head_tree = stack.enter_context(isolated_worktree(repo, head_sha, "head"))
        print(f"{BOLD}Preparing isolated worktrees{RESET}", flush=True)
        _prepare_tree(base_tree, env_file=env_file, extra_packages=packages)
        _prepare_tree(head_tree, env_file=env_file, extra_packages=packages)

        arguments = {
            "config": config,
            "models": selected_models,
            "runs": runs,
            "parallel": parallel,
            "limit": limit,
            "timeout": timeout,
            "test_filter": test_filter,
            "no_cache": no_cache,
            "trace_files": trace_files,
            "reuse": reuse,
            "max_error_rate": policy.max_error_rate,
        }

        def run_base() -> ArmResults:
            return _run_arm(
                base_tree,
                label=f"BASE ({base_ref}@{base_sha[:8]})",
                revision=base_sha,
                output_dir=experiment_dir / "base",
                **arguments,
            )

        def run_head() -> ArmResults:
            return _run_arm(
                head_tree,
                label=f"HEAD ({head_ref}@{head_sha[:8]})",
                revision=head_sha,
                output_dir=experiment_dir / "head",
                **arguments,
            )

        if head_first:
            head, base = run_head(), run_base()
        else:
            base, head = run_base(), run_head()

    diff = compare(
        base,
        head,
        base_ref,
        head_sha,
        runs,
        policy=policy,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    report_path = experiment_dir / "capability-report.md"
    summary_path = experiment_dir / "summary.json"
    report_path.write_text(diff.markdown)
    summary = {
        "base_ref": base_ref,
        "base_sha": base_sha,
        "head_ref": head_ref,
        "head_sha": head_sha,
        "models": selected_models,
        "runs": runs,
        "parallel": parallel,
        "base_result": str(base.result_path),
        "head_result": str(head.result_path),
        "base_counts": base.counts(),
        "head_counts": head.counts(),
        "clean": diff.clean,
        "floor_breach": diff.floor_breach,
        "regressions": diff.regressions,
        "new_errors": diff.new_errors,
        "beyond_noise": diff.beyond_noise,
        "statistics": diff.statistics,
        "policy": asdict(policy),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print_comparison(base, head, diff)
    print(f"\n  markdown: {report_path}")
    print(f"  summary:  {summary_path}")
    return ABRunResult(
        base=base,
        head=head,
        diff=diff,
        base_sha=base_sha,
        head_sha=head_sha,
        report_path=report_path,
        summary_path=summary_path,
    )


def _split_models(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [model.strip() for model in value.split(",") if model.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_ref", help="baseline git revision")
    parser.add_argument("head_ref", nargs="?", default="HEAD", help="candidate revision")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--models",
        help="comma-separated models; defaults to agent_models in the selected config",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--extra-package", action="append", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--test", dest="test_filter")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--trace-files", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--base-first", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="exit 1 when the comparison needs review; infrastructure failures always exit 1",
    )
    arguments = parser.parse_args()

    try:
        result = run_ab(
            repo=REPO,
            base_ref=arguments.base_ref,
            head_ref=arguments.head_ref,
            config=arguments.config,
            models=_split_models(arguments.models),
            runs=arguments.runs,
            parallel=arguments.parallel,
            output_root=arguments.output_dir,
            experiment=arguments.experiment,
            env_file=arguments.env_file,
            extra_packages=arguments.extra_package,
            limit=arguments.limit,
            timeout=arguments.timeout,
            test_filter=arguments.test_filter,
            no_cache=arguments.no_cache,
            trace_files=arguments.trace_files,
            reuse=arguments.reuse,
            head_first=not arguments.base_first,
            bootstrap_samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        )
    except CapabilityABError as error:
        print(f"{RED}error: {error}{RESET}", file=sys.stderr)
        return 1
    return 1 if arguments.require_clean and not result.diff.clean else 0


if __name__ == "__main__":
    raise SystemExit(main())
