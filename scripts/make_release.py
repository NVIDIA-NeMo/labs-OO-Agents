# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One codified release gate: check, diff capabilities, and prepare a draft.

Runs every pre-release check in a fixed order, prints a capability regression
report against the previous release, and can create a GitHub draft. It contains
no publication operation: a reviewer publishes only in the GitHub UI.

    uv run python scripts/make_release.py v0.0.9

Strict ``--ci`` mode is noninteractive, freezes an explicit candidate SHA,
requires an explicit model-alias wheel, writes private evidence/manifests, and
creates a draft only after deterministic, infrastructure, and hard capability
gates pass. Advisory capability findings remain visible for human review.

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
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

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
PACKAGES = ["nooa", "nooa-cli", "nooa-memory", "nooa-bench"]
REPORT_PATH = REPO / "tmp" / "release-check" / "capability-report.md"
GITHUB_REPO = "NVIDIA-NeMo/labs-OO-Agents"
TAG_RE = re.compile(r"^v(?P<version>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PR_REF_RE = re.compile(r"^refs/pull/[1-9]\d*/head$")

# CI bounds are deliberately explicit. The eval runner checkpoints every
# completed result to JSONL, so an overall timeout still leaves useful evidence.
CI_SAMPLE_TIMEOUT = 20 * 60
CI_HANG_TIMEOUT = 25 * 60
CI_ARM_TIMEOUT = 2 * 60 * 60 + 30 * 60

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


class ReleaseError(RuntimeError):
    """Expected release-gate failure, safe to show without a traceback."""


def die(msg: str) -> NoReturn:
    raise ReleaseError(msg)


def step(msg: str) -> None:
    print(f"\n{BOLD}▶ {msg}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def run(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = True,
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, echoing it when output is not captured."""
    if not capture:
        print(f"  {DIM}$ {' '.join(cmd)}{RESET}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd or REPO,
            text=True,
            capture_output=capture,
            check=False,
            env=env,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        die(f"required executable is unavailable: {cmd[0]} ({exc})")
    except subprocess.TimeoutExpired:
        die(f"command timed out after {timeout}s: {' '.join(cmd)}")
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() if capture else ""
        die(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}")
    return proc


def git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    proc = run(["git", *args], cwd=cwd, check=check)
    # Some Git commands echo an unresolved argument to stdout on failure.
    # Callers using check=False treat an empty result as "not found", so never
    # let diagnostic stdout masquerade as a resolved ref or tag.
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def confirm(prompt: str) -> bool:
    """Ask a yes/no question. Refuses to assume anything without a TTY."""
    if not sys.stdin.isatty():
        die(f"not a TTY — refusing to auto-confirm: {prompt}")
    return input(f"\n{BOLD}{prompt}{RESET} [y/N] ").strip().lower() in ("y", "yes")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tag(tag: str) -> str:
    match = TAG_RE.fullmatch(tag)
    if not match:
        die(f"tag must be a canonical vX.Y.Z version, got {tag!r}")
    return match.group("version")


def validate_candidate_sha(sha: str) -> str:
    if not SHA_RE.fullmatch(sha):
        die("candidate SHA must be an exact 40-character lowercase hexadecimal commit SHA")
    return sha


def validate_candidate_ref(ref: str) -> str:
    if not PR_REF_RE.fullmatch(ref):
        die("candidate ref must be a canonical refs/pull/<number>/head ref")
    return ref


def validate_https_url(value: str, label: str) -> str:
    if not re.fullmatch(r"https://[A-Za-z0-9.-]+/[^\s]+", value) or "@" in value:
        die(f"{label} must be an HTTPS URL without embedded credentials")
    return value


def release_for_tag(tag: str) -> dict[str, Any] | None:
    """Return GitHub release metadata, including drafts, or None for 404."""
    proc = run(
        ["gh", "api", f"repos/{GITHUB_REPO}/releases/tags/{tag}"],
        check=False,
    )
    if proc.returncode == 0:
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            die(f"GitHub returned invalid release metadata for {tag}")
    # gh uses exit 1 for HTTP 404. Do not turn authentication/server failures
    # into "unused tag": its stderr contains the HTTP status, never a token.
    if "404" in proc.stderr or "release not found" in proc.stderr.lower():
        return None
    die(f"could not inspect GitHub release {tag}: {proc.stderr.strip()}")


def validate_existing_release(tag: str, sha: str) -> dict[str, Any] | None:
    """Allow only an idempotent draft already targeting this exact SHA."""
    release = release_for_tag(tag)
    remote_tag = run(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"], check=False
    ).stdout.strip()
    if release is None:
        if remote_tag:
            die(f"tag {tag} already exists on GitHub without a matching draft release")
        return None
    if not release.get("draft"):
        die(f"release {tag} is already published; refusing to modify it")
    target = str(release.get("target_commitish", "")).lower()
    if target != sha:
        die(f"draft {tag} targets {target or '<unknown>'}, not frozen candidate {sha}")
    if remote_tag:
        resolved = remote_tag.split()[0].removesuffix("^{}")
        # Annotated tags report the tag object first. Resolve through the local
        # fetched ref so exact-target checking remains correct in either case.
        local = git("rev-parse", f"refs/tags/{tag}^{{commit}}", check=False)
        if local:
            resolved = local
        if resolved.lower() != sha:
            die(f"tag {tag} resolves to {resolved}, not frozen candidate {sha}")
    return release


@dataclass
class ReleaseManifest:
    """Incrementally written private evidence index for CI diagnosis."""

    path: Path
    data: dict[str, Any]

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        tmp.replace(self.path)

    def update(self, **values: Any) -> None:
        self.data.update(values)
        self.write()

    def stage(self, name: str, outcome: str, detail: str | None = None) -> None:
        stages = self.data.setdefault("deterministic_checks", {})
        stages[name] = {"outcome": outcome}
        if detail:
            stages[name]["detail"] = detail
        self.write()


def tool_versions(image_digest: str | None) -> dict[str, str]:
    versions = {
        "runner_image": image_digest or "unknown",
        "python": platform.python_version(),
        "kernel": platform.release(),
    }
    for name, cmd in {
        "uv": ["uv", "--version"],
        "git": ["git", "--version"],
        "rg": ["rg", "--version"],
        "sha256sum": ["sha256sum", "--version"],
        "base64": ["base64", "--version"],
        "curl": ["curl", "--version"],
        "gcc": ["gcc", "--version"],
        "make": ["make", "--version"],
        "gh": ["gh", "--version"],
    }.items():
        if shutil.which(cmd[0]) is None:
            versions[name] = "unavailable"
            continue
        proc = run(cmd, check=False)
        versions[name] = (
            (proc.stdout or proc.stderr).splitlines()[0].strip()
            if (proc.stdout or proc.stderr).strip()
            else "unavailable"
        )
    return versions


# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------


def preflight(
    tag: str,
    allow_dirty: bool,
    *,
    candidate_sha: str | None = None,
    ci: bool = False,
    allow_unmerged_candidate: bool = False,
) -> tuple[str, str, str, dict[str, Any] | None]:
    """Validate repo state.

    Returns (head_sha, previous_version_tag, previous_version_sha,
    existing_draft_release_or_None).
    """
    step(f"Preflight for {tag}")

    validate_tag(tag)
    if ci:
        if candidate_sha is None:
            die("--candidate-sha is required in CI mode")
        candidate_sha = validate_candidate_sha(candidate_sha)

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if not ci and branch != "main":
        die(f"on branch {branch!r}, releases are cut from main")
    if not ci:
        ok("on main")

    dirty = git("status", "--porcelain")
    if dirty and not allow_dirty:
        die(f"working tree is not clean:\n{dirty}")
    if dirty:
        warn("working tree is dirty (--allow-dirty)")
    else:
        ok("working tree clean")

    fetch = run(["git", "fetch", "--force", "--tags", "origin", "main"], check=False)
    if fetch.returncode != 0:
        die(f"could not fetch public main and version tags: {fetch.stderr.strip()}")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/main")
    if ci:
        assert candidate_sha is not None
        if head != candidate_sha:
            die(f"checked-out HEAD is {head}, not frozen candidate {candidate_sha}")
        exists = run(["git", "cat-file", "-e", f"{candidate_sha}^{{commit}}"], check=False)
        if exists.returncode != 0:
            die(f"candidate {candidate_sha} does not exist in the fetched public repository")
        if allow_unmerged_candidate:
            existing_release = None
            ok(f"frozen unmerged rehearsal candidate: {candidate_sha[:12]}")
        else:
            reachable = run(
                ["git", "merge-base", "--is-ancestor", candidate_sha, remote], check=False
            )
            if reachable.returncode != 0:
                die(f"candidate {candidate_sha} is not reachable from GitHub main ({remote})")
            ok(f"frozen candidate is reachable from current GitHub main ({remote[:12]})")
            existing_release = validate_existing_release(tag, candidate_sha)
            ok(f"{tag} has a matching reusable draft" if existing_release else f"{tag} is unused")
    else:
        if head != remote:
            die("HEAD differs from origin/main — push or pull first")
        ok("in sync with origin/main")
        existing_release = validate_existing_release(tag, head)
        ok(f"{tag} has a matching reusable draft" if existing_release else f"{tag} is unused")

    prev = git("describe", "--tags", "--abbrev=0", "--match", VERSION_TAG_GLOB, check=False)
    if prev == tag:
        prev = git(
            "describe", "--tags", "--abbrev=0", "--match", VERSION_TAG_GLOB, f"{tag}^", check=False
        )
    if not prev:
        die("no previous version tag found — cannot compute a capability diff")
    prev_sha = git("rev-parse", f"{prev}^{{commit}}")
    ok(f"previous release: {prev} ({prev_sha[:12]})")

    return head, prev, prev_sha, existing_release


# ---------------------------------------------------------------------------
# 2. Fast checks
# ---------------------------------------------------------------------------


def fast_checks(
    manifest: ReleaseManifest | None = None, *, verify_containment: bool = False
) -> None:
    """Everything cheap, before spending money on LLM calls."""
    step("Fast checks (lint, headers, unit tests)")
    checks = [
        ("ruff lint", ["uv", "run", "ruff", "check", "."]),
        ("ruff format", ["uv", "run", "ruff", "format", "--check", "."]),
        ("license headers", ["uv", "run", "python", "scripts/check_license_headers.py"]),
        ("unit tests", ["uv", "run", "pytest", "-q", "-m", "not integration and not stress"]),
    ]
    if verify_containment:
        # Match public CI exactly. The release clone starts without a virtual
        # environment, and `uv run` alone does not install optional test extras.
        checks.insert(
            0,
            (
                "locked environment sync",
                ["uv", "sync", "--frozen", "--all-extras", "--no-extra", "sandbox"],
            ),
        )

    for label, cmd in checks:
        proc = run(cmd, check=False)
        if proc.returncode != 0:
            if manifest:
                manifest.stage(label, "failed")
            print(proc.stdout[-4000:])
            print(proc.stderr[-2000:], file=sys.stderr)
            die(f"{label} failed")
        if manifest:
            manifest.stage(label, "passed")
        ok(label)

    if verify_containment:
        # These prove the worker is a separate process and that both filesystem
        # and network containment are enforceable. A green run with skipped
        # kernel guard tests is not acceptable for the credential-bearing job.
        label = "sandbox containment"
        cmd = [
            "uv",
            "run",
            "pytest",
            "-q",
            "-m",
            "sandbox",
            "-rs",
            "tests/runtime/sandbox/test_codeact_sandbox.py::test_sandbox_cell_runs_in_a_separate_process",
            "tests/runtime/sandbox/test_guards.py::test_file_read_closed_with_sandbox",
            "tests/runtime/sandbox/test_guards.py::test_network_closed_with_sandbox",
        ]
        proc = run(cmd, check=False)
        output = f"{proc.stdout}\n{proc.stderr}"
        if proc.returncode != 0 or re.search(r"\bskipped\b", output, re.IGNORECASE):
            if manifest:
                manifest.stage(label, "failed", "required containment test failed or skipped")
            print(proc.stdout[-4000:])
            print(proc.stderr[-2000:], file=sys.stderr)
            die("sandbox containment is unavailable or a required containment test skipped")
        if manifest:
            manifest.stage(label, "passed")
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
    existing = git("rev-parse", f"refs/tags/{tag}^{{commit}}", check=False)
    created = not existing
    if existing and existing != sha:
        die(f"local tag {tag} resolves to {existing}, not candidate {sha}")
    if created:
        git("tag", tag, sha)
    try:
        yield
    finally:
        if created:
            git("tag", "-d", tag, check=False)


def build_and_smoke(
    tag: str, sha: str, manifest: ReleaseManifest | None = None
) -> list[dict[str, str]]:
    step("Build wheels and smoke test")
    expected = tag.lstrip("v")
    dist = REPO / "dist"

    with temporary_tag(tag, sha):
        if dist.exists():
            shutil.rmtree(dist)
        for pkg in PACKAGES:
            proc = run(
                ["uv", "build", "--no-sources", "--package", pkg, "--out-dir", "dist"],
                check=False,
            )
            if proc.returncode != 0:
                if manifest:
                    manifest.stage("build", "failed", pkg)
                die(f"build failed for {pkg}: {(proc.stderr or proc.stdout)[-2000:]}")
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
            cli = run([str(venv / "bin" / "nooa"), "--version"], check=False)
            if cli.returncode != 0 or expected not in cli.stdout:
                die(
                    f"CLI smoke test failed or reported the wrong version: {cli.stdout}{cli.stderr}"
                )
            ok(f"CLI version OK ({cli.stdout.strip()})")

    artifacts = [
        {"path": str(path.relative_to(REPO)), "sha256": sha256(path)}
        for path in sorted(dist.iterdir())
        if path.is_file()
    ]
    if manifest:
        manifest.stage("build", "passed")
        manifest.stage("wheel versions", "passed", expected)
        manifest.stage("clean wheel smoke", "passed")
        manifest.update(distributions=artifacts)
    return artifacts


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
    result_path: str | None = None
    lock_sha256: str | None = None

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
    *,
    strict_ci: bool = False,
    internal_wheel: Path | None = None,
    output_root: Path | None = None,
    sample_timeout: float | None = None,
    hang_timeout: float | None = None,
    overall_timeout: float | None = None,
) -> ArmResults:
    """Run the capability suite in `tree`, with cache only for local fallback.

    Local review can reuse a paid run after aborting at its prompt. Strict CI
    always runs fresh. The signature covers models/runs/limit so a cheap smoke
    run never gets mistaken for a real gate run.
    """
    out_dir = (output_root or REPO / "tmp" / "release-check") / cache_key
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "run.json"
    signature = {"models": sorted(models), "runs": runs, "limit": limit}

    existing = newest_eval(out_dir)
    if (
        not strict_ci
        and existing
        and marker.exists()
        and json.loads(marker.read_text()) == signature
    ):
        ok(f"{label}: reusing cached results ({existing.name})")
        return parse_results(existing, label)

    scope = f"{len(models)} models × {runs} runs" + (f" × {limit} samples" if limit else "")
    print(f"  {DIM}{label}: full suite × {scope}{RESET}")

    if strict_ci:
        if internal_wheel is None:
            die(f"{label}: strict CI requires an explicit internal model wheel")
        print(f"  {DIM}syncing a fresh locked environment for {label}...{RESET}")
        run(
            ["uv", "sync", "--frozen", "--all-extras", "--no-extra", "sandbox"],
            cwd=tree,
        )
        venv_python = tree / ".venv" / "bin" / "python"
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python),
                "--no-deps",
                str(internal_wheel),
            ],
            cwd=tree,
        )
        # Resolve every alias through the entry-point registry. This fails
        # before any paid call and does not print model endpoints or secrets.
        alias_check = (
            "from nooa.unifiedllm.registry import get_registry_config; "
            f"names={models!r}; missing=[n for n in names if not get_registry_config(n)]; "
            "assert not missing, 'missing model aliases: '+','.join(missing)"
        )
        run([str(venv_python), "-c", alias_check], cwd=tree)
        ok(f"{label}: explicit internal wheel installed and all model aliases resolved")
    elif tree != REPO:
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
    if strict_ci:
        cmd += ["--no-cache", "--trace-files"]
    if sample_timeout:
        cmd += ["--timeout", str(sample_timeout)]
    if hang_timeout:
        cmd += ["--hang-timeout", str(hang_timeout)]
    if limit:
        cmd += ["--limit", str(limit)]
    child_env = os.environ.copy()
    if strict_ci:
        # The evaluator always starts its own headless backend. Prevent an
        # inherited endpoint from duplicating private traces externally.
        child_env.pop("OTLP_ENDPOINT", None)
    run(
        cmd,
        cwd=tree,
        capture=False,
        env=child_env,
        timeout=overall_timeout,
    )

    produced = newest_eval(out_dir)
    if not produced:
        die(f"{label}: eval produced no .noo-eval.jsonl under {out_dir}")
    arm = parse_results(produced, label)
    arm.result_path = str(produced)
    lock = tree / "uv.lock"
    arm.lock_sha256 = sha256(lock) if lock.exists() else None
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
    def hard_gate_passed(self) -> bool:
        return self.floor_breach is None

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
    if not stable_t:
        diff.floor_breach = "candidate produced no stable-tier samples"
    elif stable_rate < STABLE_FLOOR:
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
    prev_tag: str,
    sha: str,
    models: list[str],
    runs: int,
    limit: int | None,
    *,
    strict_ci: bool = False,
    internal_wheel: Path | None = None,
    artifact_dir: Path | None = None,
    sample_timeout: float | None = None,
    hang_timeout: float | None = None,
    overall_timeout: float | None = None,
) -> tuple[Diff, ArmResults, ArmResults]:
    step(f"Capability diff vs {prev_tag} (both arms run fresh)")
    # The cache key carries the scope, so a `--limit 1` smoke run and a real
    # gate run never share a directory.
    scope = f"m{len(models)}r{runs}" + (f"l{limit}" if limit else "")
    output_root = (artifact_dir / "capability") if artifact_dir else None
    common = {
        "strict_ci": strict_ci,
        "internal_wheel": internal_wheel,
        "output_root": output_root,
        "sample_timeout": sample_timeout,
        "hang_timeout": hang_timeout,
        "overall_timeout": overall_timeout,
    }
    if strict_ci:
        # Run both sides from disposable worktrees so neither arm can inherit a
        # project environment. They receive the same explicit adapter wheel.
        with worktree_at(sha) as head_tree:
            head_arm = run_capability_arm(
                head_tree,
                f"candidate ({sha[:8]})",
                f"candidate-{sha[:12]}-{scope}",
                models,
                runs,
                limit,
                **common,
            )
        with worktree_at(prev_tag) as base_tree:
            base_arm = run_capability_arm(
                base_tree,
                prev_tag,
                f"baseline-{prev_tag.replace('/', '_')}-{scope}",
                models,
                runs,
                limit,
                **common,
            )
    else:
        head_arm = run_capability_arm(
            REPO, f"HEAD ({sha[:8]})", f"{sha[:12]}-{scope}", models, runs, limit
        )
        with worktree_at(prev_tag) as tree:
            base_arm = run_capability_arm(
                tree, prev_tag, f"{prev_tag.replace('/', '_')}-{scope}", models, runs, limit
            )
    diff = compare(base_arm, head_arm, prev_tag, sha, runs)
    report_path = artifact_dir / "capability-report.md" if artifact_dir else REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(diff.markdown)
    print(f"\n  {DIM}markdown report: {report_path}{RESET}")
    return diff, base_arm, head_arm


def arm_summary(arm: ArmResults) -> dict[str, Any]:
    passed, total = arm.counts()
    return {
        "label": arm.label,
        "passed": passed,
        "total": total,
        "sample_count": total,
        "success_rate": arm.overall(),
        "error_rate": arm.error_rate(),
        "tiers": {
            tier: {"passed": counts[0], "total": counts[1]}
            for tier, counts in sorted(arm.tier_counts().items())
        },
        "models": dict(sorted(arm.per_model().items())),
        "result_path": arm.result_path,
        "uv_lock_sha256": arm.lock_sha256,
    }


# ---------------------------------------------------------------------------
# 5. Draft release (publication is intentionally not implemented here)
# ---------------------------------------------------------------------------


def sanitize_public_text(text: str) -> str:
    """Conservative last-line defense before private evidence becomes public."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = re.sub(
        r"(?i)\b(api[_-]?key|authorization|token|secret)\b\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        text,
    )
    text = re.sub(r"https?://[^\s/@]+:[^\s/@]+@", "https://[REDACTED]@", text)
    text = re.sub(r"(?<![\w])/(?:home|builds|tmp)/[^\s`|)]+", "[private-path]", text)
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)


def generated_change_notes(tag: str, sha: str, prev_tag: str) -> str:
    proc = run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{GITHUB_REPO}/releases/generate-notes",
            "-f",
            f"tag_name={tag}",
            "-f",
            f"target_commitish={sha}",
            "-f",
            f"previous_tag_name={prev_tag}",
        ],
        check=False,
    )
    if proc.returncode != 0:
        die(f"could not generate GitHub change notes: {proc.stderr.strip()}")
    try:
        return str(json.loads(proc.stdout).get("body", "")).strip()
    except json.JSONDecodeError:
        die("GitHub returned invalid generated release notes")


def local_change_notes(sha: str, prev_tag: str) -> str:
    """Generate rehearsal-only notes without requiring GitHub credentials."""
    proc = run(
        ["git", "log", "--format=- %s (`%h`)", f"{prev_tag}..{sha}"],
        check=False,
    )
    if proc.returncode != 0:
        die(f"could not generate local rehearsal change notes: {proc.stderr.strip()}")
    return proc.stdout.strip()


def build_public_notes(
    *,
    tag: str,
    sha: str,
    prev_tag: str,
    diff: Diff,
    change_notes: str,
    pipeline_url: str,
    distributions: list[dict[str, str]],
    capability_gate_ran: bool,
) -> str:
    advisory_count = (
        len(diff.regressions) + len(diff.new_errors) + len(diff.beyond_noise) + len(diff.removed)
    )
    if not capability_gate_ran:
        verdict = "NOT RUN — the capability gate was skipped for this candidate"
    elif diff.hard_gate_passed:
        verdict = "PASS — all automated hard gates passed"
    else:
        verdict = f"FAIL — {diff.floor_breach}"
    checksum_lines = [
        f"- `{Path(item['path']).name}` — `{item['sha256']}`" for item in distributions
    ]
    sections = [
        "# Unpublished NOOA release candidate",
        "",
        "> **This draft is not published. Publishing it is the sole human release approval and immediately starts automatic PyPI publication.**",
        "",
        f"- Version: `{tag}`",
        f"- Immutable candidate: `{sha}`",
        f"- Previous release: `{prev_tag}`",
        f"- Automated hard-gate verdict: **{verdict}**",
        f"- Advisory findings: **{advisory_count}** (review the open sections below)",
        f"- Private pipeline evidence: {pipeline_url}",
        "",
        sanitize_public_text(diff.markdown),
        "",
        "## Changes",
        "",
        sanitize_public_text(change_notes) or "No generated change notes were returned.",
        "",
        "## Candidate build provenance",
        "",
        *checksum_lines,
        "",
        "The candidate pipeline built and smoke-tested these artifacts. The current publication workflow rebuilds from this exact tag; artifact promotion remains a follow-up.",
        "",
        "## Reviewer checklist",
        "",
        f"- Confirm the candidate SHA is exactly `{sha}`.",
        "- Review advisory findings and the private GitLab evidence/traces.",
        "- **Accept:** click GitHub’s **Publish release** button. This immediately triggers `publish.yml` and automatic PyPI publication.",
        "- **Reject:** delete this draft (including its tag) or leave it for the documented cleanup procedure.",
    ]
    return sanitize_public_text("\n".join(sections).rstrip() + "\n")


def create_or_update_draft(
    tag: str,
    sha: str,
    notes_path: Path,
) -> tuple[str, int]:
    """Create exactly one draft, or update the matching unpublished draft."""
    # Re-read immediately before mutation. A reviewer may have published a
    # pre-existing draft while this multi-hour gate was running; in that case
    # validate_existing_release fails before `gh release edit` can unpublish it.
    current = validate_existing_release(tag, sha)
    step(f"{'Updating' if current else 'Creating'} draft release {tag}")
    common = [
        "--repo",
        GITHUB_REPO,
        "--target",
        sha,
        "--title",
        f"NOOA {tag.lstrip('v')}",
        "--notes-file",
        str(notes_path),
        "--draft",
    ]
    if current:
        cmd = ["gh", "release", "edit", tag, *common]
    else:
        cmd = ["gh", "release", "create", tag, *common]
    run(cmd, capture=False)
    release = validate_existing_release(tag, sha)
    if release is None:
        die(f"GitHub did not return the draft after {'update' if current else 'creation'}")
    url = str(release.get("html_url", ""))
    release_id = int(release["id"])
    ok(f"draft ready: {url}")
    return url, release_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the release gate and optionally create/update a GitHub draft."
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
    parser.add_argument(
        "--ci",
        action="store_true",
        help="strict, noninteractive CI mode (never discovers ambient extras or copies .env)",
    )
    parser.add_argument("--candidate-sha", help="exact 40-character public candidate SHA")
    parser.add_argument(
        "--candidate-ref",
        default="",
        help="canonical refs/pull/<number>/head ref authenticated by the private controller",
    )
    parser.add_argument("--internal-wheel", type=Path, help="explicit internal model-alias wheel")
    parser.add_argument("--controller-sha", help="exact nooa-dev commit that built the wheel")
    parser.add_argument("--artifact-dir", type=Path, help="private CI evidence output directory")
    parser.add_argument(
        "--create-draft",
        action="store_true",
        help="after a production-strength CI gate passes, create/update the GitHub draft",
    )
    parser.add_argument("--pipeline-url", default="", help="private GitLab pipeline URL")
    parser.add_argument("--job-url", default="", help="private GitLab job URL")
    parser.add_argument("--image-digest", default="", help="immutable runner image reference")
    parser.add_argument(
        "--sample-timeout", type=float, default=CI_SAMPLE_TIMEOUT, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--hang-timeout", type=float, default=CI_HANG_TIMEOUT, help=argparse.SUPPRESS
    )
    parser.add_argument("--arm-timeout", type=float, default=CI_ARM_TIMEOUT, help=argparse.SUPPRESS)
    return parser


def _copy_distributions(artifact_dir: Path) -> list[dict[str, str]]:
    target = artifact_dir / "dist"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted((REPO / "dist").iterdir()):
        if not source.is_file():
            continue
        destination = target / source.name
        shutil.copy2(source, destination)
        records.append({"path": str(destination), "sha256": sha256(destination)})
    checksum_file = artifact_dir / "CHECKSUMS.sha256"
    checksum_file.write_text(
        "".join(f"{item['sha256']}  {Path(item['path']).name}\n" for item in records)
    )
    return records


def _write_job_summary(artifact_dir: Path, manifest: ReleaseManifest) -> None:
    data = manifest.data
    cap = data.get("capability", {})
    lines = [
        "# NOOA release gate summary",
        "",
        f"- Status: **{data.get('status', 'running')}**",
        f"- Version: `{data.get('requested_version', '')}`",
        f"- Candidate: `{data.get('candidate_sha', '')}`",
        f"- Previous: `{data.get('previous_release_tag', '')}`",
        f"- Capability hard gate: **{cap.get('hard_gate_outcome', 'not run')}**",
    ]
    if data.get("github_draft_url"):
        lines.append(f"- GitHub draft: {data['github_draft_url']}")
    if data.get("failure"):
        lines.append(f"- Failure: {data['failure']}")
    (artifact_dir / "job-summary.md").write_text("\n".join(lines) + "\n")


def ci_main(args: argparse.Namespace) -> int:
    if args.allow_dirty or args.skip_capability or args.dry_run:
        die("--ci forbids --allow-dirty, --skip-capability, and --dry-run")
    if not args.candidate_sha or not args.controller_sha:
        die("--ci requires --candidate-sha and --controller-sha")
    candidate_sha = validate_candidate_sha(args.candidate_sha)
    controller_sha = validate_candidate_sha(args.controller_sha)
    if args.internal_wheel is None or not args.internal_wheel.is_file():
        die("--ci requires an existing --internal-wheel")
    internal_wheel = args.internal_wheel.resolve()
    if internal_wheel.suffix != ".whl" or "nemo_oo_agents_nvidia" not in internal_wheel.name:
        die("--internal-wheel must be the nemo-oo-agents-nvidia wheel")
    if args.artifact_dir is None:
        die("--ci requires --artifact-dir")
    candidate_ref = validate_candidate_ref(args.candidate_ref) if args.candidate_ref else None
    unmerged_candidate = candidate_ref is not None
    models = args.models.split(",") if args.models else GATE_MODELS
    if any(not model.strip() for model in models):
        die("--models contains an empty alias")
    rehearsal = bool(args.limit) or args.runs != GATE_RUNS or models != GATE_MODELS
    if args.runs < 1 or (args.limit is not None and args.limit < 1):
        die("--runs and --limit must be positive")
    if rehearsal and args.create_draft:
        die("a reduced rehearsal can never create a draft")
    if unmerged_candidate and not rehearsal:
        die("--candidate-ref requires a reduced rehearsal")
    if unmerged_candidate and args.create_draft:
        die("an unmerged candidate can never create a draft")
    if args.checks_only and args.create_draft:
        die("--checks-only can never be combined with --create-draft")
    if not os.getenv("NVIDIA_INTERNAL_API_KEY"):
        die("NVIDIA_INTERNAL_API_KEY is required for the live capability gate")
    if not unmerged_candidate and not os.getenv("GH_TOKEN"):
        die("GH_TOKEN is required in CI to inspect and reconcile release state")
    validate_https_url(args.pipeline_url, "pipeline URL")
    validate_https_url(args.job_url, "job URL")
    if not re.search(r"@sha256:[0-9a-f]{64}$", args.image_digest):
        die("--image-digest must identify an image pinned by an immutable sha256 digest")
    for value, name in [
        (args.sample_timeout, "sample timeout"),
        (args.hang_timeout, "hang timeout"),
        (args.arm_timeout, "arm timeout"),
    ]:
        if value <= 0:
            die(f"{name} must be positive")

    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = ReleaseManifest(
        artifact_dir / "release-manifest.json",
        {
            "schema_version": 1,
            "status": "running",
            "requested_version": args.tag,
            "candidate_sha": candidate_sha,
            "candidate_ref": candidate_ref,
            "candidate_source": (
                "canonical_github_pull_ref" if unmerged_candidate else "github_main"
            ),
            "nooa_dev_sha": controller_sha,
            "internal_model_wheel": {
                "filename": internal_wheel.name,
                "sha256": sha256(internal_wheel),
            },
            "pipeline_url": args.pipeline_url,
            "job_url": args.job_url,
            "environment": {"runner_image": args.image_digest},
            "capability": {"hard_gate_outcome": "not_run"},
            "github_draft_url": None,
            "github_release_database_id": None,
        },
    )
    manifest.write()
    try:
        manifest.update(environment=tool_versions(args.image_digest))
        head_sha, prev_tag, prev_sha, existing = preflight(
            args.tag,
            False,
            candidate_sha=candidate_sha,
            ci=True,
            allow_unmerged_candidate=unmerged_candidate,
        )
        manifest.update(
            previous_release_tag=prev_tag,
            previous_release_sha=prev_sha,
            existing_matching_draft=bool(existing),
            capability_config_sha256=sha256(REPO / CAPABILITY_CONFIG),
            capability_scope={
                "models": models,
                "runs": args.runs,
                "sample_limit": args.limit,
                "parallelism": GATE_PARALLEL,
                "sample_timeout_seconds": args.sample_timeout,
                "hang_timeout_seconds": args.hang_timeout,
                "arm_timeout_seconds": args.arm_timeout,
                "fresh_arms": True,
                "response_cache": False,
            },
        )

        fast_checks(manifest, verify_containment=True)
        build_and_smoke(args.tag, head_sha, manifest)
        distributions = _copy_distributions(artifact_dir)
        manifest.update(distributions=distributions)

        manifest.update(capability={"hard_gate_outcome": "running"})
        try:
            diff, baseline, candidate = capability_diff(
                prev_tag,
                head_sha,
                models,
                args.runs,
                args.limit,
                strict_ci=True,
                internal_wheel=internal_wheel,
                artifact_dir=artifact_dir,
                sample_timeout=args.sample_timeout,
                hang_timeout=args.hang_timeout,
                overall_timeout=args.arm_timeout,
            )
        except ReleaseError as exc:
            manifest.update(
                capability={
                    "hard_gate_outcome": "infrastructure_failed",
                    "hard_gate_detail": sanitize_public_text(str(exc)),
                }
            )
            raise
        hard_gate = "passed" if diff.hard_gate_passed else "failed"
        capability = {
            "hard_gate_outcome": hard_gate,
            "hard_gate_detail": diff.floor_breach,
            "advisory": {
                "collapses": diff.regressions,
                "new_errors": diff.new_errors,
                "aggregate_drops": diff.beyond_noise,
                "removed_tests": diff.removed,
                "models_changed": diff.models_changed,
            },
            "baseline": arm_summary(baseline),
            "candidate": arm_summary(candidate),
        }
        manifest.update(
            capability=capability,
            public_uv_lock_sha256={
                "candidate": candidate.lock_sha256,
                "previous": baseline.lock_sha256,
            },
        )
        if diff.floor_breach:
            die(f"capability hard gate failed: {diff.floor_breach}")

        change_notes = (
            local_change_notes(head_sha, prev_tag)
            if unmerged_candidate
            else generated_change_notes(args.tag, head_sha, prev_tag)
        )
        notes = build_public_notes(
            tag=args.tag,
            sha=head_sha,
            prev_tag=prev_tag,
            diff=diff,
            change_notes=change_notes,
            pipeline_url=args.pipeline_url,
            distributions=distributions,
            capability_gate_ran=True,
        )
        notes_path = artifact_dir / "public-release-notes.md"
        notes_path.write_text(notes)
        manifest.update(public_release_notes_sha256=sha256(notes_path))

        if args.create_draft:
            url, release_id = create_or_update_draft(args.tag, head_sha, notes_path)
            manifest.update(
                github_draft_url=url,
                github_release_database_id=release_id,
            )
        else:
            warn("checks-only CI run complete; GitHub draft creation was disabled")
        manifest.update(
            status="passed",
            rehearsal=rehearsal,
            unmerged_candidate=unmerged_candidate,
        )
        _write_job_summary(artifact_dir, manifest)
        return 0
    except ReleaseError as exc:
        message = sanitize_public_text(str(exc))
        manifest.update(status="failed", failure=message)
        _write_job_summary(artifact_dir, manifest)
        raise


def local_main(args: argparse.Namespace) -> int:
    if (
        args.create_draft
        or args.candidate_sha
        or args.candidate_ref
        or args.internal_wheel
        or args.artifact_dir
    ):
        die("CI-only arguments require --ci")

    models = args.models.split(",") if args.models else GATE_MODELS
    rehearsal = args.limit or args.runs != GATE_RUNS or models != GATE_MODELS
    if rehearsal and not args.checks_only and not args.dry_run:
        die("--models/--runs/--limit reduce the gate's power; pair them with --checks-only")

    head_sha, prev_tag, _prev_sha, existing = preflight(args.tag, args.allow_dirty)
    fast_checks()
    build_and_smoke(args.tag, head_sha)

    report = ""
    if args.skip_capability:
        warn("capability diff SKIPPED — no evidence this release is free of regressions")
    else:
        diff, _baseline, _candidate = capability_diff(
            prev_tag, head_sha, models, args.runs, args.limit
        )
        report = diff.markdown
        if args.checks_only:
            return 0 if not diff.floor_breach else 1
        if diff.floor_breach:
            die(f"capability hard gate failed: {diff.floor_breach}")
        elif not confirm(f"Accept these capability results and draft {args.tag}?"):
            print("Aborted. Cached eval results kept under tmp/release-check/.")
            return 1

    if args.checks_only:
        return 0

    if args.dry_run:
        step("Dry run — the release steps that would follow")
        print(
            f"  {DIM}$ gh release create {args.tag} --repo {GITHUB_REPO} "
            f"--target {head_sha} --title 'NOOA {args.tag.lstrip('v')}' "
            f"--notes-file <review notes> --draft{RESET}"
        )
        ok("dry run complete — nothing was created")
        return 0

    change_notes = generated_change_notes(args.tag, head_sha, prev_tag)
    diff = diff if report else Diff(markdown="Capability comparison skipped.")
    notes = build_public_notes(
        tag=args.tag,
        sha=head_sha,
        prev_tag=prev_tag,
        diff=diff,
        change_notes=change_notes,
        pipeline_url="Local emergency fallback; no private CI evidence URL.",
        distributions=[
            {"path": str(path), "sha256": sha256(path)}
            for path in sorted((REPO / "dist").iterdir())
            if path.is_file()
        ],
        capability_gate_ran=not args.skip_capability,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(notes)
        notes_path = Path(fh.name)
    try:
        url, _release_id = create_or_update_draft(args.tag, head_sha, notes_path)
    finally:
        notes_path.unlink(missing_ok=True)
    print(f"\nReview {url}. Accept only by clicking GitHub's Publish release button.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return ci_main(args) if args.ci else local_main(args)
    except ReleaseError as exc:
        print(f"\n{RED}✗ {exc}{RESET}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
