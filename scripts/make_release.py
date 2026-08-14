# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One codified release path: check, diff capabilities, review, publish.

Runs every pre-release check in a fixed order, prints a capability regression
report against the previous release, and stops at two human gates before doing
anything irreversible.

    uv run python scripts/make_release.py v0.0.9

The capability step has one hard gate and everything else advisory. The gate is
an absolute *floor* on the stable tier, not a delta threshold: LLM pass rates
vary run to run, so a delta threshold would either block good releases or get
routinely bypassed until it meant nothing, while a low floor only fires on
catastrophe. Regressions relative to the previous release are classified
(collapse / new errors / beyond-noise) and shown to a human to decide on.

Why both arms run fresh: comparing HEAD against a stored baseline cannot
distinguish "we regressed" from "the endpoint behind a model alias changed".
Checking out the previous tag and running it back to back with HEAD, against
the same endpoints in the same session, controls for provider drift so the
delta is attributable to our code.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parent.parent

# Four models: three small ones spanning providers (a regression in tool
# schemas or structured output is usually provider-specific), plus one large
# one to catch breakage that only shows up with stronger reasoning.
GATE_MODELS = [
    "claude-haiku",
    "gpt-5.4-mini",
    "nemotron3-nano-30b",
    "claude-opus-4-8",
]
GATE_RUNS = 3
GATE_PARALLEL = 40
CAPABILITY_CONFIG = Path("tests/capability/config.yaml")
PACKAGES = ["nooa", "nooa-cli", "nooa-acp", "nooa-memory", "nooa-bench"]
REPORT_PATH = REPO / "tmp" / "release-check" / "capability-report.md"

# An absolute floor on the stable tier, mirroring the MR pipeline's gate. A
# *floor* survives run-to-run LLM variance in a way a delta threshold cannot:
# it only fires on catastrophe, so it can be enforced without being bypassed.
STABLE_FLOOR = 0.60

# Classification thresholds. These shape the *report*, not a pass/fail verdict.
AGGREGATE_NOISE_PTS = 5.0  # overall/per-model drop worth calling out
# Above this share of errored samples an arm describes the network, not the code.
MAX_ERROR_RATE = 0.5
COLLAPSE_BEFORE = 0.80  # a test that used to pass at least this often...
COLLAPSE_AFTER = 0.20  # ...and now passes at most this often, has collapsed

# `git describe --tags --abbrev=0` alone resolves to `nooa-cybergym` in this
# repo. Every tag lookup must filter to version tags or the "previous release"
# silently becomes a random feature tag.
VERSION_TAG_GLOB = "v[0-9]*"

BOLD, DIM, RED, YELLOW, GREEN, RESET = (
    ("\033[1m", "\033[2m", "\033[31m", "\033[33m", "\033[32m", "\033[0m")
    if sys.stdout.isatty()
    else ("", "", "", "", "", "")
)


def die(msg: str) -> NoReturn:
    print(f"\n{RED}✗ {msg}{RESET}", file=sys.stderr)
    sys.exit(1)


def step(msg: str) -> None:
    print(f"\n{BOLD}▶ {msg}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def run(
    cmd: list[str], cwd: Path | None = None, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess:
    """Run a command, echoing it when output is not captured."""
    if not capture:
        print(f"  {DIM}$ {' '.join(cmd)}{RESET}")
    proc = subprocess.run(
        cmd,
        cwd=cwd or REPO,
        text=True,
        capture_output=capture,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() if capture else ""
        die(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}")
    return proc


def git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    return run(["git", *args], cwd=cwd, check=check).stdout.strip()


def confirm(prompt: str) -> bool:
    """Ask a yes/no question. Refuses to assume anything without a TTY."""
    if not sys.stdin.isatty():
        die(f"not a TTY — refusing to auto-confirm: {prompt}")
    return input(f"\n{BOLD}{prompt}{RESET} [y/N] ").strip().lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------


def preflight(tag: str, allow_dirty: bool) -> tuple[str, str]:
    """Validate repo state. Returns (head_sha, previous_version_tag)."""
    step(f"Preflight for {tag}")

    if not tag.startswith("v"):
        die(f"tag must look like v0.0.9, got {tag!r}")

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        die(f"on branch {branch!r}, releases are cut from main")
    ok("on main")

    dirty = git("status", "--porcelain")
    if dirty and not allow_dirty:
        die(f"working tree is not clean:\n{dirty}")
    if dirty:
        warn("working tree is dirty (--allow-dirty)")
    else:
        ok("working tree clean")

    git("fetch", "--tags", "origin", check=False)
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/main", check=False)
    if remote and head != remote:
        die("HEAD differs from origin/main — push or pull first")
    ok("in sync with origin/main")

    existing = git("tag", "-l", tag)
    if existing:
        die(f"tag {tag} already exists locally")
    remote_tag = run(["git", "ls-remote", "--tags", "origin", tag]).stdout.strip()
    if remote_tag:
        die(f"tag {tag} already exists on origin")
    ok(f"{tag} is unused")

    prev = git("describe", "--tags", "--abbrev=0", "--match", VERSION_TAG_GLOB, check=False)
    if not prev:
        die("no previous version tag found — cannot compute a capability diff")
    ok(f"previous release: {prev}")

    return head, prev


# ---------------------------------------------------------------------------
# 2. Fast checks
# ---------------------------------------------------------------------------


def fast_checks() -> None:
    """Everything cheap, before spending money on LLM calls."""
    step("Fast checks (lint, headers, unit tests)")
    for label, cmd in [
        ("ruff lint", ["uv", "run", "ruff", "check", "."]),
        ("ruff format", ["uv", "run", "ruff", "format", "--check", "."]),
        ("license headers", ["uv", "run", "python", "scripts/check_license_headers.py"]),
        ("unit tests", ["uv", "run", "pytest", "-q", "-m", "not integration and not stress"]),
    ]:
        proc = run(cmd, check=False)
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-2000:], file=sys.stderr)
            die(f"{label} failed")
        ok(label)


# ---------------------------------------------------------------------------
# 3. Build + smoke test
# ---------------------------------------------------------------------------


@contextmanager
def temporary_tag(tag: str, sha: str):
    """Create the tag locally just long enough to build under it.

    The version is derived from `git describe`, so building before the tag
    exists yields `X.Y.Z.devN` and proves nothing about what the release will
    publish. The tag is removed again on the way out — `gh release create`
    creates the real one, at this same commit.
    """
    git("tag", tag, sha)
    try:
        yield
    finally:
        git("tag", "-d", tag, check=False)


def build_and_smoke(tag: str, sha: str) -> None:
    step("Build wheels and smoke test")
    expected = tag.lstrip("v")
    dist = REPO / "dist"

    with temporary_tag(tag, sha):
        if dist.exists():
            shutil.rmtree(dist)
        for pkg in PACKAGES:
            run(
                ["uv", "build", "--no-sources", "--package", pkg, "--out-dir", "dist"],
                capture=True,
            )
        ok(f"built {len(PACKAGES)} packages")

        wheels = sorted(dist.glob("*.whl"))
        if len(wheels) != len(PACKAGES):
            die(f"expected {len(PACKAGES)} wheels, found {len(wheels)}")
        for wheel in wheels:
            version = wheel.name.split("-")[1]
            if version != expected:
                die(f"{wheel.name} has version {version}, expected {expected}")
            if "dev" in version:
                die(f"{wheel.name} is a dev version — the tag was not reachable")
        ok(f"all wheels at version {expected}")

        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / "smoke"
            python = venv / "bin" / "python"
            run(["uv", "venv", str(venv), "--python", "3.12"])
            # `--python` targets the throwaway venv explicitly. Without it uv
            # resolves VIRTUAL_ENV/.venv from cwd and the wheels land in the
            # project env — the smoke test would then be importing the working
            # tree rather than the built artifacts.
            run(
                ["uv", "pip", "install", "--python", str(python), *[str(w) for w in wheels]],
                capture=True,
            )
            proc = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import nooa, nooa_cli, nooa_memory, nooa_bench; print(nooa.__version__)",
                ],
                text=True,
                capture_output=True,
            )
            if proc.returncode != 0:
                die(f"smoke import failed:\n{proc.stderr}")
            ok(f"smoke import OK ({proc.stdout.strip()})")


# ---------------------------------------------------------------------------
# 4. Capability diff
# ---------------------------------------------------------------------------


@dataclass
class ArmResults:
    """Everything one checkout's eval run produced, indexed for comparison."""

    label: str
    # test_case ("sentiment_single_001") -> pass flags across every model and run
    by_case: dict[str, list[bool]] = field(default_factory=lambda: defaultdict(list))
    case_tier: dict[str, str] = field(default_factory=dict)
    # (model, test_name) -> pass flags; the granularity collapse detection needs
    by_test: dict[tuple[str, str], list[bool]] = field(default_factory=lambda: defaultdict(list))
    errors: dict[tuple[str, str], dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    output_tokens: int = 0
    total_tokens: int = 0
    errored: int = 0

    def error_rate(self) -> float:
        _, total = self.counts()
        return self.errored / total if total else 0.0

    def rate(self, key: tuple[str, str]) -> float | None:
        results = self.by_test.get(key)
        return sum(results) / len(results) if results else None

    def counts(self) -> tuple[int, int]:
        flags = [p for rs in self.by_case.values() for p in rs]
        return sum(flags), len(flags)

    def overall(self) -> float:
        passed, total = self.counts()
        return passed / total if total else 0.0

    def tier_counts(self) -> dict[str, tuple[int, int]]:
        acc: dict[str, list[bool]] = defaultdict(list)
        for case, flags in self.by_case.items():
            acc[self.case_tier.get(case, "stable")].extend(flags)
        return {t: (sum(f), len(f)) for t, f in acc.items()}

    def per_model(self) -> dict[str, float]:
        acc: dict[str, list[bool]] = defaultdict(list)
        for (model, _), results in self.by_test.items():
            acc[model].extend(results)
        return {m: sum(r) / len(r) for m, r in acc.items() if r}


def parse_results(path: Path, label: str) -> ArmResults:
    arm = ArmResults(label=label)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("_type") != "result":
            continue
        passed = bool(rec.get("passed"))
        test_name = rec.get("test_name") or rec.get("agent_class", "?")
        case = rec.get("test_case") or test_name
        arm.by_case[case].append(passed)
        arm.case_tier.setdefault(case, rec.get("tier") or "stable")
        arm.by_test[(rec.get("model", "?"), test_name)].append(passed)
        if rec.get("error_type"):
            arm.errors[(rec.get("model", "?"), test_name)][rec["error_type"]] += 1
            arm.errored += 1
        arm.output_tokens += rec.get("output_tokens") or 0
        arm.total_tokens += rec.get("total_tokens") or 0
    return arm


def env_extras() -> list[str]:
    """Install specs present in this venv but absent from `uv.lock`.

    The NVIDIA model aliases the gate resolves through (`claude-haiku`,
    `nemotron3-nano-30b`, …) ship in `nemo-oo-agents-nvidia`, which is installed
    from a local path and deliberately not in the lock. A freshly synced
    worktree would not have it, so those models would fail to resolve on the
    baseline arm and the two sides would not be comparable. Mirroring whatever
    the current env carries keeps both arms resolving the same aliases.
    """
    locked = {
        name.lower()
        for name in re.findall(r'^name = "([^"]+)"', (REPO / "uv.lock").read_text(), re.MULTILINE)
    }
    extras: list[str] = []
    for raw in run(["uv", "pip", "freeze"]).stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # freeze emits three shapes, and all three occur in this project:
        #   -e file:///path          editable install
        #   name @ file:///path      direct-URL (non-editable) install
        #   name==version            ordinary registry install
        if line.startswith("-e "):
            name, location = "", line[3:]
        elif " @ " in line:
            name, _, location = line.partition(" @ ")
        elif "==" in line:
            name, location = line.split("==")[0], ""
        else:
            continue

        # CRITICAL: skip anything inside this checkout. `uv pip freeze` lists
        # the workspace packages themselves as local installs pointing at REPO,
        # and mirroring those would put HEAD's nooa into the baseline worktree —
        # silently comparing HEAD against itself.
        if location.strip().removeprefix("file://").startswith(str(REPO)):
            continue
        if name.strip().lower() in locked:
            continue
        # The worktree is disposable, so a plain (non-editable) install is fine.
        extras.append(line.removeprefix("-e ").strip())
    return extras


def newest_eval(out_dir: Path) -> Path | None:
    """Most recent `.noo-eval.jsonl` under `out_dir`.

    Recursive on purpose: eval_pipeline writes into a timestamped subdirectory
    (`capability_<ts>_p40/capability_<ts>.noo-eval.jsonl`), not into the
    `--output-dir` root, so a plain glob finds nothing.
    """
    found = sorted(out_dir.rglob("*.noo-eval.jsonl"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def run_capability_arm(
    tree: Path,
    label: str,
    cache_key: str,
    models: list[str],
    runs: int,
    limit: int | None,
) -> ArmResults:
    """Run the capability suite in `tree`, reusing cached results when present.

    The script has two human gates and a run takes a while; without a cache,
    aborting at the review prompt costs another full paid run to get back. The
    signature covers models/runs/limit so a cheap smoke run never gets mistaken
    for a real gate run.
    """
    out_dir = REPO / "tmp" / "release-check" / cache_key
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "run.json"
    signature = {"models": sorted(models), "runs": runs, "limit": limit}

    existing = newest_eval(out_dir)
    if existing and marker.exists() and json.loads(marker.read_text()) == signature:
        ok(f"{label}: reusing cached results ({existing.name})")
        return parse_results(existing, label)

    scope = f"{len(models)} models × {runs} runs" + (f" × {limit} samples" if limit else "")
    print(f"  {DIM}{label}: full suite × {scope}{RESET}")

    if tree != REPO:
        # A worktree is a clean git checkout, so gitignored local config does
        # not come with it. Without .env the baseline arm has no credentials and
        # every sample fails with "Missing credentials" — which reads as a
        # spectacular improvement rather than a broken run.
        dotenv = REPO / ".env"
        if dotenv.exists():
            target = tree / ".env"
            shutil.copyfile(dotenv, target)
            target.chmod(0o600)
            print(f"  {DIM}copied .env into the worktree (removed with it){RESET}")

        print(f"  {DIM}syncing {tree}...{RESET}")
        # --inexact: this repo's dev env legitimately carries packages that are
        # not in the lock (see env_extras), and an exact sync would strip them.
        run(
            ["uv", "sync", "--all-extras", "--no-extra", "sandbox", "--inexact"],
            cwd=tree,
            capture=True,
        )
        extras = env_extras()
        if extras:
            print(f"  {DIM}mirroring {len(extras)} out-of-lock package(s) into the worktree{RESET}")
            # --python is load-bearing. `uv sync`/`uv run` are project commands
            # and use the worktree's .venv, but `uv pip install` is a pip
            # interface command that honours VIRTUAL_ENV — which this script
            # inherits from its own `uv run`, pointing at the MAIN repo venv.
            # Without --python the packages land back in the developer's env and
            # the worktree silently has no model aliases.
            venv_python = tree / ".venv" / "bin" / "python"
            run(
                ["uv", "pip", "install", "--python", str(venv_python), *extras],
                cwd=tree,
                capture=True,
            )
            # Fail here rather than 5 minutes into an eval that resolves no
            # models: this exact mistake produced "Config models: []".
            installed = run(
                ["uv", "pip", "list", "--python", str(venv_python)], cwd=tree
            ).stdout.lower()
            for spec in extras:
                name = Path(spec.removeprefix("file://")).name.lower()
                if name and name not in installed.replace("_", "-"):
                    die(f"{label}: {name} did not install into {venv_python}")

    cmd = [
        "uv",
        "run",
        # The env for each arm is prepared above; --no-sync keeps `uv run` from
        # touching it again mid-run (and from mutating the developer's own venv
        # on the HEAD arm).
        "--no-sync",
        "python",
        "-m",
        "eval_pipeline",
        "--config",
        str(CAPABILITY_CONFIG),
        "--models",
        ",".join(models),
        "--runs",
        str(runs),
        "--parallel",
        str(GATE_PARALLEL),
        "--output-dir",
        str(out_dir),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    run(cmd, cwd=tree, capture=False)

    produced = newest_eval(out_dir)
    if not produced:
        die(f"{label}: eval produced no .noo-eval.jsonl under {out_dir}")
    arm = parse_results(produced, label)
    # The eval CLI exits 0 even when every sample errors, so these two checks are
    # the only thing standing between an infrastructure outage and a meaningless
    # report. A broken BASELINE arm is the dangerous case: it reads as a huge
    # improvement and sails through the gate as "no regressions". Refuse to
    # compare rather than emit numbers that describe the network.
    if not arm.by_case:
        die(f"{label}: eval produced no usable results — check credentials and model aliases")
    if arm.error_rate() > MAX_ERROR_RATE:
        top = sorted(((n, e) for d in arm.errors.values() for e, n in d.items()), reverse=True)[:1]
        detail = f" (most common: {top[0][1]})" if top else ""
        die(
            f"{label}: {arm.error_rate():.0%} of samples errored{detail} — "
            f"this is an infrastructure failure, not a capability result"
        )
    marker.write_text(json.dumps(signature))
    return arm


@contextmanager
def worktree_at(tag: str):
    """A detached worktree at `tag`, cleaned up afterwards.

    Each arm runs entirely against its own checkout — its own nooa, its own
    capability config and agents. That keeps each side self-consistent (HEAD's
    test agents may use APIs the old nooa lacks), at the cost that a change to
    the harness itself shows up as a capability delta. Tests that exist on only
    one side are reported separately rather than compared.
    """
    tmp = Path(tempfile.mkdtemp(prefix="nooa-release-"))
    path = tmp / "tree"
    git("worktree", "add", "--detach", str(path), tag)
    try:
        yield path
    finally:
        git("worktree", "remove", "--force", str(path), check=False)
        shutil.rmtree(tmp, ignore_errors=True)


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

    @property
    def clean(self) -> bool:
        # `removed` counts, `added` does not: a test that vanished may be hiding
        # a regression, whereas a new test has no baseline to regress from.
        return not (
            self.regressions
            or self.new_errors
            or self.beyond_noise
            or self.removed
            or self.floor_breach
        )


def _mark(delta: float, *, inverse: bool = False) -> str:
    """Trend marker matching the MR report's vocabulary."""
    if delta == 0:
        return "➖"
    good = delta < 0 if inverse else delta > 0
    return "✅" if good else "❌"


def _bar(base_rate: float, head_rate: float, width: int = 20) -> str:
    """Blue up to the shared level, green for gain / red for loss, grey for the rest."""
    now, was = round(head_rate * width), round(base_rate * width)
    shared = min(now, was)
    gained, lost = max(0, now - was), max(0, was - now)
    return "🟦" * shared + "🟩" * gained + "🟥" * lost + "⬜" * (width - shared - gained - lost)


def compare(
    base: ArmResults, head: ArmResults, prev_tag: str, sha: str, runs: int = GATE_RUNS
) -> Diff:
    diff = Diff()
    bp, bt = base.counts()
    hp, ht = head.counts()
    b, h = base.overall(), head.overall()
    delta_pts = (h - b) * 100

    # A model present in only one arm makes every one of its tests look added or
    # removed. That is a change to GATE_MODELS, not to the test suite, so it is
    # tracked separately and never counted as a disappearing test.
    shared_models = {m for m, _ in base.by_test} & {m for m, _ in head.by_test}
    diff.models_changed = sorted(
        ({m for m, _ in base.by_test} | {m for m, _ in head.by_test}) - shared_models
    )

    # ---- collapse / new-error detection, at (model, test) granularity -------
    # Per-test deltas are reported only on collapse. With ~3 samples per test
    # per run a per-test wobble is mostly noise, and a report that lists every
    # wobble trains people to skim past the part that matters.
    for key in sorted(set(base.by_test) | set(head.by_test)):
        model, test = key
        if model not in shared_models:
            continue
        br, hr = base.rate(key), head.rate(key)
        if br is None:
            diff.added.append(f"{model}/{test}")
            continue
        if hr is None:
            # Counts against `clean`: deleting or renaming a failing test would
            # otherwise erase the regression it represents, and the run would
            # report "no regressions" while quietly testing less than before.
            diff.removed.append(f"{model}/{test}")
            continue
        if br >= COLLAPSE_BEFORE and hr <= COLLAPSE_AFTER:
            diff.regressions.append(
                f"| `{test}` | {model} | {br:.0%} | {hr:.0%} | {(hr - br) * 100:+.1f}% ❌ |"
            )
        before = set(base.errors.get(key, {}))
        for etype, count in head.errors.get(key, {}).items():
            if etype not in before:
                diff.new_errors.append(f"| `{test}` | {model} | {count}× `{etype}` | 0 before |")

    if delta_pts < -AGGREGATE_NOISE_PTS:
        diff.beyond_noise.append(f"overall {delta_pts:+.1f} pts (band ±{AGGREGATE_NOISE_PTS})")
    bm, hm = base.per_model(), head.per_model()
    for model in sorted(set(bm) & set(hm)):
        d = (hm[model] - bm[model]) * 100
        if d < -AGGREGATE_NOISE_PTS:
            diff.beyond_noise.append(f"{model} {d:+.1f} pts")

    # Per tier as well as overall: a stable-tier regression can be masked in the
    # aggregate by frontier tests improving at the same time, which is exactly
    # the trade nobody wants to make silently.
    tiers_head, tiers_base = head.tier_counts(), base.tier_counts()
    for tier in sorted(set(tiers_head) & set(tiers_base)):
        p0, t0 = tiers_base[tier]
        p1, t1 = tiers_head[tier]
        if not (t0 and t1):
            continue
        d = (p1 / t1 - p0 / t0) * 100
        if d < -AGGREGATE_NOISE_PTS:
            diff.beyond_noise.append(f"{tier} tier {d:+.1f} pts")

    # ---- floor gate --------------------------------------------------------
    # An absolute floor, not a delta threshold. A low floor survives run-to-run
    # LLM variance while still catching catastrophe; the delta stays advisory.
    stable_p, stable_t = tiers_head.get("stable", (0, 0))
    stable_rate = stable_p / stable_t if stable_t else 0.0
    if stable_t and stable_rate < STABLE_FLOOR:
        diff.floor_breach = f"stable tier at {stable_rate:.1%}, floor is {STABLE_FLOOR:.0%}"

    # ---- markdown ----------------------------------------------------------
    md: list[str] = ["## 🧪 Capability Test Results", ""]
    if diff.floor_breach:
        md.append(f"❌ **Release BLOCKED** — {diff.floor_breach}")
    elif diff.clean:
        md.append(
            f"✅ **Release OK** — Stable tier at {stable_rate:.1%} "
            f"(floor: {STABLE_FLOOR:.0%}), no regressions beyond noise"
        )
    else:
        md.append(
            f"⚠️ **Review required** — Stable tier at {stable_rate:.1%} "
            f"(floor: {STABLE_FLOOR:.0%}), {len(diff.regressions)} collapse(s), "
            f"{len(diff.new_errors)} new error type(s), "
            f"{len(diff.removed)} removed test(s)"
        )
    md += [
        "",
        "---",
        "",
        f"**{h:.1%}** {_bar(b, h)} **{delta_pts:+.1f}%**",
        "",
        f"{hp}/{ht} tests passing *({hp - bp:+d} from {prev_tag})*",
        "",
        "| Metric | " + prev_tag + " | This release | Change |",
        "|--------|----------|--------------|--------|",
        f"| Tests Passed | {bp}/{bt} | {hp}/{ht} | {hp - bp:+d} {_mark(hp - bp)} |",
        f"| Success Rate | {b:.1%} | {h:.1%} | {delta_pts:+.1f}% {_mark(delta_pts)} |",
        f"| Collapsed tests | — | {len(diff.regressions)} | "
        f"{len(diff.regressions)} {_mark(len(diff.regressions), inverse=True)} |",
        f"| New error types | — | {len(diff.new_errors)} | "
        f"{len(diff.new_errors)} {_mark(len(diff.new_errors), inverse=True)} |",
        f"| Output Tokens | {base.output_tokens:,} | {head.output_tokens:,} | "
        f"{head.output_tokens - base.output_tokens:+,} |",
        f"| Total Tokens | {base.total_tokens:,} | {head.total_tokens:,} | "
        f"{head.total_tokens - base.total_tokens:+,} |",
        "",
        "<details>",
        "<summary>📊 Per-tier breakdown</summary>",
        "",
        f"| Tier | {prev_tag} | This release | Change | Expected |",
        "|------|----------|--------------|--------|----------|",
    ]
    for tier in sorted(set(tiers_head) | set(tiers_base)):
        p0, t0 = tiers_base.get(tier, (0, 0))
        p1, t1 = tiers_head.get(tier, (0, 0))
        r0 = p0 / t0 if t0 else 0.0
        r1 = p1 / t1 if t1 else 0.0
        d = (r1 - r0) * 100
        expected = f"≥{STABLE_FLOOR:.0%}" if tier == "stable" else "—"
        md.append(
            f"| {tier.title()} | {p0}/{t0} ({r0:.1%}) | {p1}/{t1} ({r1:.1%}) | "
            f"{p1 - p0:+d} / {d:+.1f}% {_mark(d)} | {expected} |"
        )
    md += ["", "</details>", ""]

    if diff.regressions:
        md += [
            "<details open>",
            "<summary>❌ Collapsed tests</summary>",
            "",
            f"| Test | Model | {prev_tag} | This release | Change |",
            "|------|-------|----------|--------------|--------|",
            *diff.regressions,
            "",
            "</details>",
            "",
        ]
    if diff.new_errors:
        md += [
            "<details open>",
            "<summary>❌ New error types</summary>",
            "",
            "| Test | Model | This release | Baseline |",
            "|------|-------|--------------|----------|",
            *diff.new_errors,
            "",
            "</details>",
            "",
        ]

    md += [
        "<details>",
        "<summary>📋 Per-test breakdown</summary>",
        "",
        "| Test | Status |",
        "|------|--------|",
    ]
    for case in sorted(set(head.by_case) | set(base.by_case)):
        flags = head.by_case.get(case)
        if flags is None:
            md.append(f"| {case} | ⬜ removed |")
            continue
        icon = "✅" if all(flags) else "❌"
        new = " *(new)*" if case not in base.by_case else ""
        md.append(f"| {case} | {icon} {sum(flags)}/{len(flags)}{new} |")
    md += [
        "",
        "</details>",
        "",
        f"*{prev_tag} → `{sha[:8]}`* | *{len({m for m, _ in head.by_test})} models × "
        f"{runs} runs* | *both arms run fresh*",
    ]
    diff.markdown = "\n".join(md)

    # ---- terminal view -----------------------------------------------------
    print()
    print(f"{BOLD}{'═' * 72}{RESET}")
    print(f"{BOLD} CAPABILITY DIFF   {prev_tag} → HEAD ({sha[:8]}){RESET}")
    n_models = len({m for m, _ in head.by_test})
    print(f" {DIM}{n_models} models · {len(head.by_case)} cases · {runs} runs{RESET}")
    print(f"{BOLD}{'═' * 72}{RESET}\n")
    print(f" {_bar(b, h)}")
    colour = RED if delta_pts < -AGGREGATE_NOISE_PTS else (GREEN if delta_pts > 0 else "")
    print(
        f" {BOLD}{h:.1%}{RESET}  {colour}{delta_pts:+.1f} pts{RESET}   "
        f"{hp}/{ht} passing ({hp - bp:+d})\n"
    )

    print(f" {BOLD}PER TIER{RESET}")
    for tier in sorted(set(tiers_head) | set(tiers_base)):
        p0, t0 = tiers_base.get(tier, (0, 0))
        p1, t1 = tiers_head.get(tier, (0, 0))
        r0, r1 = (p0 / t0 if t0 else 0.0), (p1 / t1 if t1 else 0.0)
        d = (r1 - r0) * 100
        c = RED if d < -AGGREGATE_NOISE_PTS else ""
        print(f"   {tier:<12} {r0:>6.1%} → {r1:>6.1%}  {c}{d:+5.1f}{RESET}  ({p1}/{t1})")

    print(f"\n {BOLD}PER MODEL{RESET}")
    for model in sorted(set(bm) | set(hm)):
        if model not in bm or model not in hm:
            print(f"   {model:<26} {DIM}only in one arm{RESET}")
            continue
        d = (hm[model] - bm[model]) * 100
        c = RED if d < -AGGREGATE_NOISE_PTS else ""
        print(f"   {model:<26} {bm[model]:>6.1%} → {hm[model]:>6.1%}  {c}{d:+5.1f}{RESET}")

    if diff.regressions:
        print(
            f"\n {BOLD}{RED}COLLAPSED{RESET} {DIM}(≥{COLLAPSE_BEFORE:.0%} → "
            f"≤{COLLAPSE_AFTER:.0%}){RESET}"
        )
        for row in diff.regressions:
            print(f"   {RED}!{RESET} {row.strip('|').replace('|', ' ').replace('`', '')}")
    if diff.new_errors:
        print(f"\n {BOLD}{RED}NEW ERROR TYPES{RESET}")
        for row in diff.new_errors:
            print(f"   {RED}!{RESET} {row.strip('|').replace('|', ' ').replace('`', '')}")
    if diff.added or diff.removed or diff.models_changed:
        print(f"\n {BOLD}TEST SET CHANGES{RESET}")
        for line in diff.added[:10]:
            print(f"   {GREEN}+{RESET} {line} {DIM}(no baseline){RESET}")
        if len(diff.added) > 10:
            print(f"   {DIM}… and {len(diff.added) - 10} more added{RESET}")
        for line in diff.removed[:10]:
            print(f"   {RED}-{RESET} {line} {DIM}(gone from HEAD){RESET}")
        if len(diff.removed) > 10:
            print(f"   {DIM}… and {len(diff.removed) - 10} more removed{RESET}")
        for model in diff.models_changed:
            print(f"   {YELLOW}~{RESET} {model} {DIM}ran in only one arm — not compared{RESET}")

    print(f"\n{BOLD}{'─' * 72}{RESET}")
    if diff.floor_breach:
        print(f" {RED}VERDICT: BLOCKED — {diff.floor_breach}.{RESET}")
    elif diff.clean:
        print(
            f" {GREEN}VERDICT: OK — stable tier {stable_rate:.1%}, no regressions "
            f"beyond noise.{RESET}"
        )
    else:
        print(
            f" {YELLOW}VERDICT: {len(diff.regressions)} collapse(s), "
            f"{len(diff.new_errors)} new error type(s), "
            f"{len(diff.beyond_noise)} aggregate drop(s), "
            f"{len(diff.removed)} removed test(s) — REVIEW REQUIRED.{RESET}"
        )
    print(f"{BOLD}{'─' * 72}{RESET}")
    return diff


def capability_diff(
    prev_tag: str, sha: str, models: list[str], runs: int, limit: int | None
) -> Diff:
    step(f"Capability diff vs {prev_tag} (both arms run fresh)")
    # The cache key carries the scope, so a `--limit 1` smoke run and a real
    # gate run never share a directory.
    scope = f"m{len(models)}r{runs}" + (f"l{limit}" if limit else "")
    head_arm = run_capability_arm(
        REPO, f"HEAD ({sha[:8]})", f"{sha[:12]}-{scope}", models, runs, limit
    )
    with worktree_at(prev_tag) as tree:
        base_arm = run_capability_arm(
            tree, prev_tag, f"{prev_tag.replace('/', '_')}-{scope}", models, runs, limit
        )
    diff = compare(base_arm, head_arm, prev_tag, sha, runs)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(diff.markdown)
    print(f"\n  {DIM}markdown report: {REPORT_PATH}{RESET}")
    return diff


# ---------------------------------------------------------------------------
# 5. Release
# ---------------------------------------------------------------------------


def create_draft(tag: str, sha: str, report: str) -> None:
    step(f"Creating draft release {tag}")
    run(
        [
            "gh",
            "release",
            "create",
            tag,
            "--target",
            sha,
            "--title",
            f"NOOA {tag.lstrip('v')}",
            "--generate-notes",
            "--draft",
        ],
        capture=False,
    )
    if report:
        # Append the capability report to the generated notes, so each release
        # carries the evidence it was cut on. `--generate-notes` has already
        # written the changelog; read it back and extend rather than replace.
        notes = run(["gh", "release", "view", tag, "--json", "body", "-q", ".body"]).stdout
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
            fh.write(f"{notes.rstrip()}\n\n---\n\n{report}\n")
            notes_path = fh.name
        try:
            run(["gh", "release", "edit", tag, "--notes-file", notes_path])
            ok("capability report appended to the release notes")
        finally:
            Path(notes_path).unlink(missing_ok=True)
    url = run(["gh", "release", "view", tag, "--json", "url", "-q", ".url"]).stdout.strip()
    ok(f"draft created: {url}")


def publish(tag: str) -> None:
    step(f"Publishing {tag}")
    run(["gh", "release", "edit", tag, "--draft=false"], capture=False)
    ok(f"{tag} published — publish.yml will build and upload to PyPI")
    print(f"  {DIM}each package waits on its pypi-<name> environment approval{RESET}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full pre-release check, then create and publish a GitHub Release."
    )
    parser.add_argument("tag", help="release tag, e.g. v0.0.9")
    parser.add_argument(
        "--skip-capability",
        action="store_true",
        help="skip the capability diff (docs-only releases; the LLM eval is the slow, costly step)",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true", help="proceed with an unclean working tree"
    )
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="run every check and print the report, then stop without touching the release",
    )
    parser.add_argument(
        "--models",
        help=f"comma-separated model override for the capability diff "
        f"(default: {','.join(GATE_MODELS)})",
    )
    parser.add_argument(
        "--runs", type=int, default=GATE_RUNS, help=f"eval runs per test (default: {GATE_RUNS})"
    )
    parser.add_argument(
        "--limit", type=int, help="cap samples per test — for cheap rehearsals, not for a real gate"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the gh release commands instead of running them",
    )
    args = parser.parse_args()

    models = args.models.split(",") if args.models else GATE_MODELS
    rehearsal = args.limit or args.runs != GATE_RUNS or models != GATE_MODELS
    if rehearsal and not args.checks_only and not args.dry_run:
        die("--models/--runs/--limit reduce the gate's power; pair them with --checks-only")

    head_sha, prev_tag = preflight(args.tag, args.allow_dirty)
    fast_checks()
    build_and_smoke(args.tag, head_sha)

    report = ""
    if args.skip_capability:
        warn("capability diff SKIPPED — no evidence this release is free of regressions")
    else:
        diff = capability_diff(prev_tag, head_sha, models, args.runs, args.limit)
        report = diff.markdown
        if args.checks_only:
            return 0 if diff.clean else 1
        # The floor is the one place the script takes a position. Everything
        # else is advisory; a stable tier under the floor needs the override to
        # be typed out, not answered with a reflexive "y".
        if diff.floor_breach:
            print(f"\n{RED}{diff.floor_breach}{RESET}")
            if input(f"{BOLD}Type OVERRIDE to release anyway:{RESET} ").strip() != "OVERRIDE":
                print("Aborted. Cached eval results kept under tmp/release-check/.")
                return 1
        elif not confirm(f"Accept these capability results and draft {args.tag}?"):
            print("Aborted. Cached eval results kept under tmp/release-check/.")
            return 1

    if args.checks_only:
        return 0

    if args.dry_run:
        step("Dry run — the release steps that would follow")
        print(
            f"  {DIM}$ gh release create {args.tag} --target {head_sha[:12]} "
            f"--title 'NOOA {args.tag.lstrip('v')}' --generate-notes --draft{RESET}"
        )
        print(
            f"  {DIM}$ gh release edit {args.tag} --notes-file <notes + capability report>{RESET}"
        )
        print(f"  {DIM}$ gh release edit {args.tag} --draft=false{RESET}")
        ok("dry run complete — nothing was created")
        return 0

    create_draft(args.tag, head_sha, report)
    print(f"\n{DIM}Review the generated notes in the browser before continuing.{RESET}")
    if not confirm(f"Publish {args.tag} to GitHub and PyPI?"):
        # --cleanup-tag: plain `gh release delete` leaves the tag behind, and a
        # stray v0.0.9 tag makes the next attempt fail preflight ("tag already
        # exists on origin"). Harmless if the draft never created a tag.
        print(
            f"Aborted. The draft remains; delete with: gh release delete {args.tag} --cleanup-tag"
        )
        return 1

    publish(args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
