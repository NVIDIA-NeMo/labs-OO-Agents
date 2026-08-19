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
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

try:
    from scripts import capability_ab
except ImportError:  # `python scripts/make_release.py`
    import capability_ab

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


# Re-export the pure comparison API for callers and existing release tests.
ArmResults = capability_ab.ArmResults
Diff = capability_ab.Diff
parse_results = capability_ab.parse_results
compare = capability_ab.compare
_bar = capability_ab._bar
_mark = capability_ab._mark


def capability_diff(
    prev_tag: str, sha: str, models: list[str], runs: int, limit: int | None
) -> Diff:
    """Run the shared commit-to-commit capability A/B workflow for a release."""
    step(f"Capability diff vs {prev_tag} (both arms run fresh)")
    scope = f"m{len(models)}r{runs}" + (f"l{limit}" if limit else "")
    experiment = f"release-{prev_tag.replace('/', '_')}-{sha[:12]}-{scope}"
    result = capability_ab.run_ab(
        repo=REPO,
        base_ref=prev_tag,
        head_ref=sha,
        config=CAPABILITY_CONFIG,
        models=models,
        runs=runs,
        parallel=GATE_PARALLEL,
        output_root=REPO / "tmp" / "release-check",
        experiment=experiment,
        env_file=REPO / ".env",
        limit=limit,
        reuse=True,
        head_first=True,
        policy=capability_ab.ComparisonPolicy(
            stable_floor=STABLE_FLOOR,
            aggregate_noise_points=AGGREGATE_NOISE_PTS,
            max_error_rate=MAX_ERROR_RATE,
            collapse_before=COLLAPSE_BEFORE,
            collapse_after=COLLAPSE_AFTER,
        ),
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(result.diff.markdown)
    print(f"\n  {DIM}markdown report: {REPORT_PATH}{RESET}")
    return result.diff


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
