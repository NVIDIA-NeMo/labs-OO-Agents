# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run and compare deterministic context captures at two Git revisions."""

from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
PREFILL_ID_RE = re.compile(r"\bprefill_[0-9a-fA-F]{8}\b")


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _revision(repo: Path, revision: str) -> str:
    return _run(["git", "rev-parse", f"{revision}^{{commit}}"], cwd=repo).strip()


def _extract_revision(repo: Path, revision: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(destination, filter="data")


def _capture(tree: Path, capture_script: Path, output: Path, log: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(tree / "src"), str(tree))),
            "PYTHON_DOTENV_DISABLED": "1",
        }
    )
    env.pop("OTLP_ENDPOINT", None)
    result = subprocess.run(
        [sys.executable, str(capture_script), "--output", str(output)],
        cwd=tree,
        env=env,
        text=True,
        capture_output=True,
    )
    log.write_text(f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}")
    if result.returncode:
        raise RuntimeError(f"capture failed in {tree}; see {log}")


def canonicalize(value: Any, uuid_map: dict[str, str] | None = None) -> Any:
    """Normalize incidental IDs and legacy XML-envelope byte formatting."""
    uuid_map = uuid_map if uuid_map is not None else {}
    if isinstance(value, str):
        text = "\n".join(line.rstrip() for line in value.replace("\r\n", "\n").split("\n"))

        def replace(match: re.Match[str]) -> str:
            raw = match.group(0).lower()
            return uuid_map.setdefault(raw, f"<UUID-{len(uuid_map) + 1}>")

        text = UUID_RE.sub(replace, text)
        text = PREFILL_ID_RE.sub(replace, text)
        if text.startswith("<context>\n") and text.endswith("\n</context>"):
            text = text[len("<context>\n") : -len("\n</context>")]
        return re.sub(r"(</[A-Za-z_][\w.-]*>)\n\n(?=<[A-Za-z_])", r"\1\n", text)
    if isinstance(value, list):
        return [canonicalize(item, uuid_map) for item in value]
    if isinstance(value, dict):
        return {key: canonicalize(item, uuid_map) for key, item in value.items()}
    return value


def _counts(capture: dict[str, Any]) -> tuple[int, int, int]:
    scenarios = capture["scenarios"]
    requests = [request for scenario in scenarios.values() for request in scenario["requests"]]
    messages = [message for request in requests for message in request["messages"]]
    chars = sum(len(json.dumps(message, sort_keys=True)) for message in messages)
    return len(requests), len(messages), chars


def compare_captures(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_normalized = canonicalize(baseline)
    candidate_normalized = canonicalize(candidate)
    baseline_text = json.dumps(baseline_normalized, indent=2, sort_keys=True).splitlines()
    candidate_text = json.dumps(candidate_normalized, indent=2, sort_keys=True).splitlines()
    diff = list(
        difflib.unified_diff(
            baseline_text,
            candidate_text,
            fromfile="baseline",
            tofile="candidate",
            lineterm="",
        )
    )
    return {
        "passed": not diff,
        "exact_bytes_equal": baseline == candidate,
        "baseline_counts": _counts(baseline),
        "candidate_counts": _counts(candidate),
        "diff": diff,
    }


def _report(
    result: dict[str, Any], baseline_sha: str, candidate_sha: str, scenarios: list[str]
) -> str:
    baseline_requests, baseline_messages, baseline_chars = result["baseline_counts"]
    candidate_requests, candidate_messages, candidate_chars = result["candidate_counts"]
    status = "PASS" if result["passed"] else "FAIL"
    lines = [
        "# Context parity report",
        "",
        f"- Result: **{status}**",
        f"- Baseline: `{baseline_sha}`",
        f"- Candidate: `{candidate_sha}`",
        f"- Scenarios: {len(scenarios)} ({', '.join(scenarios)})",
        f"- Baseline: {baseline_requests} requests, {baseline_messages} messages, {baseline_chars:,} serialized message characters",
        f"- Candidate: {candidate_requests} requests, {candidate_messages} messages, {candidate_chars:,} serialized message characters",
        f"- Raw captures byte-identical: {result['exact_bytes_equal']}",
        "- Normalization: generated IDs, CRLF, trailing whitespace, and the legacy outer `<context>` envelope/separators",
        "",
    ]
    if result["diff"]:
        lines.extend(["## Diff", "", "```diff", *result["diff"], "```", ""])
    else:
        lines.append("No semantic or structural differences were found.\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="main")
    parser.add_argument("--candidate", default="HEAD")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    capture_script = Path(__file__).with_name("capture.py")
    baseline_sha = _revision(repo, args.baseline)
    candidate_sha = _revision(repo, args.candidate)
    output = args.output_root or (
        repo
        / "experiments"
        / "context_parity"
        / "results"
        / f"{baseline_sha[:12]}_vs_{candidate_sha[:12]}"
    )
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nooa-context-parity-") as temporary:
        root = Path(temporary)
        baseline_tree = root / "baseline"
        candidate_tree = root / "candidate"
        _extract_revision(repo, baseline_sha, baseline_tree)
        _extract_revision(repo, candidate_sha, candidate_tree)
        _capture(
            baseline_tree,
            capture_script,
            output / "baseline.json",
            output / "baseline.log",
        )
        _capture(
            candidate_tree,
            capture_script,
            output / "candidate.json",
            output / "candidate.log",
        )

    baseline = json.loads((output / "baseline.json").read_text())
    candidate = json.loads((output / "candidate.json").read_text())
    result = compare_captures(baseline, candidate)
    scenarios = list(baseline["scenarios"])
    report = _report(result, baseline_sha, candidate_sha, scenarios)
    (output / "report.md").write_text(report)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": baseline_sha,
        "candidate": candidate_sha,
        "scenarios": scenarios,
        "passed": result["passed"],
        "exact_bytes_equal": result["exact_bytes_equal"],
        "normalization": [
            "generated UUID and prefill IDs",
            "CRLF",
            "trailing whitespace",
            "legacy outer <context> envelope and inter-block separators",
        ],
        "artifacts": [
            "baseline.json",
            "candidate.json",
            "baseline.log",
            "candidate.log",
            "report.md",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(report)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
