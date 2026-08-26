# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import NOOA OTLP or portable journal traces from Harbor into the viewer.

Walks a Harbor job directory (or any directory containing one), finds all
traces under ``artifacts/traces/*.jsonl``, enriches them with Harbor metadata
(trial name, task name, reward score, experiment grouping), and posts them to
the viewer.

Usage:
    nooa import-harbor ./jobs/my-job/
    nooa import-harbor ./workspaces/ --endpoint http://host:5001
    nooa import-harbor ./jobs/ --experiment my-eval --batch-id run-42
"""

import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import click

from ._otlp_helpers import (
    OtlpRequestError,
    _viewer_headers,
    check_endpoint_reachable,
    get_journal_record,
    inject_resource_attrs,
    post_journal_record,
    post_traces_batch,
    session_exists,
    validate_endpoint,
)

NAME = "import-harbor"


def _find_harbor_traces(root: Path) -> list[Path]:
    """Find all OTLP or portable journal trace files under Harbor artifacts.

    Harbor copies the container's ``/logs/artifacts/`` to ``trial_dir/artifacts/``
    on the host. The agent decides the layout within that directory — a common
    convention is ``artifacts/traces/*.jsonl``, but we search the full subtree
    to be robust to other layouts.
    """
    return sorted(root.rglob("artifacts/**/*.jsonl"))


def _read_json(path: Path) -> dict:
    """Read a JSON file, returning an empty dict on any failure."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _coerce_float(value: object) -> float | None:
    """Coerce a value to float, returning None if it cannot be coerced."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None


def _read_score(trial_dir: Path, trial_result: dict) -> float | None:
    """Read the trial reward score, trying current Harbor result shapes in order.

    Different Harbor/BinPool versions write the scalar reward in different places.
    Fallback order (first coercible float wins):

    1. ``verifier/reward.json["score"]``
    2. ``verifier/reward.json["reward"]``
    3. ``result.json["verifier_result"]["rewards"]["score"]``
    4. ``result.json["verifier_result"]["rewards"]["reward"]``
    5. ``verifier/reward.txt`` (plain float string)

    Explicit ``None`` checks (not truthiness) ensure a valid ``0.0`` is returned.
    """
    reward_json = _read_json(trial_dir / "verifier" / "reward.json")
    for key in ("score", "reward"):
        score = _coerce_float(reward_json.get(key))
        if score is not None:
            return score

    verifier_result = trial_result.get("verifier_result")
    rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
    if isinstance(rewards, dict):
        for key in ("score", "reward"):
            score = _coerce_float(rewards.get(key))
            if score is not None:
                return score

    reward_txt = trial_dir / "verifier" / "reward.txt"
    if reward_txt.exists():
        return _coerce_float(reward_txt.read_text().strip())

    return None


def _trial_meta(jsonl_path: Path) -> dict:
    """Extract Harbor metadata for a trace file from its surrounding directory structure.

    Expected layout::

        <job_dir>/
            result.json              ← job-level (stats.evals for experiment name)
            <trial_name>/
                result.json          ← trial_name, task_name, agent_info
                verifier/
                    reward.json      ← {"reward"|"score": <float>}  (or reward.txt)
                artifacts/           ← copy of /logs/artifacts/ from container
                    [traces/]        ← agent-defined layout; traces can be here
                        <file>.jsonl ← this file
    """
    # Walk up from the JSONL file to find the 'artifacts' directory;
    # trial_dir is its parent (works regardless of depth under artifacts/).
    trial_dir = jsonl_path.parent
    for parent in jsonl_path.parents:
        if parent.name == "artifacts":
            trial_dir = parent.parent
            break
    job_dir = trial_dir.parent

    trial_result = _read_json(trial_dir / "result.json")
    job_result = _read_json(job_dir / "result.json")

    trial_name = trial_result.get("trial_name") or trial_dir.name
    task_name = trial_result.get("task_name", "")
    config = trial_result.get("config") if isinstance(trial_result.get("config"), dict) else {}
    agent_config = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    task_config = config.get("task") if isinstance(config.get("task"), dict) else {}
    agent_name = (trial_result.get("agent_info") or {}).get("name", "") or agent_config.get(
        "name", ""
    )
    model_name = agent_config.get("model_name", "")
    agent_type = ""
    kwargs = agent_config.get("kwargs") if isinstance(agent_config.get("kwargs"), dict) else {}
    if kwargs:
        agent_type = str(kwargs.get("agent_type") or "")
    source = trial_result.get("source") or task_config.get("source") or ""

    # Reward scalar lives in different places across Harbor versions; see _read_score.
    score = _read_score(trial_dir, trial_result)

    # Keep the Harbor eval key as metadata, but group viewer Evaluations by
    # Harbor job by default. The eval key is usually broad (for example,
    # "nemo-oo-agents__swebench_all") and otherwise collapses separate model
    # jobs into one row.
    harbor_eval = ""
    evals = (job_result.get("stats") or {}).get("evals") or {}
    if evals:
        harbor_eval = next(iter(evals))

    return {
        "trial_name": trial_name,
        "task_name": task_name,
        "agent_name": agent_name,
        "agent_type": agent_type,
        "model_name": model_name,
        "source": source,
        "started_at": trial_result.get("started_at", ""),
        "finished_at": trial_result.get("finished_at", ""),
        "score": score,
        "harbor_eval": harbor_eval,
        "experiment": job_dir.name or harbor_eval or "harbor",
        "job_name": job_dir.name,
    }


def _harbor_resource_attrs(meta: dict, experiment: str, batch_id: str) -> dict[str, str | bool]:
    """Build viewer resource attrs that make a Harbor trial appear as an eval row."""
    attrs: dict[str, str | bool] = {
        "session.id": meta["trial_name"],
        "experiment": experiment,
        "batch_id": batch_id,
        "eval.test_id": meta["task_name"] or meta["trial_name"],
        "eval.test_name": meta["task_name"] or meta["trial_name"],
        "eval.display_name": meta["task_name"] or meta["trial_name"],
        "eval.method": "harbor",
        "eval.harbor_trial_name": meta["trial_name"],
    }
    if meta.get("model_name"):
        attrs["eval.model"] = str(meta["model_name"])
    if meta.get("agent_type"):
        attrs["eval.agent_class"] = str(meta["agent_type"])
    elif meta.get("agent_name"):
        attrs["eval.agent_class"] = str(meta["agent_name"])
    if meta.get("agent_name"):
        attrs["eval.agent_name"] = str(meta["agent_name"])
    if meta.get("source"):
        attrs["eval.suite_name"] = str(meta["source"])
    if meta.get("harbor_eval"):
        attrs["eval.harbor_eval"] = str(meta["harbor_eval"])
    if meta.get("score") is not None:
        score = meta["score"]
        attrs["eval.score"] = str(score)
        attrs["eval.weighted_score"] = str(score)
        attrs["eval.passed"] = score >= 1.0
    return attrs


def _find_matching_live_session(endpoint: str, meta: dict, experiment: str) -> str | None:
    """Ask the viewer for the live-streamed session matching this Harbor trial."""
    task_name = meta.get("task_name")
    if not task_name:
        return None

    def request_match(search_experiment: str) -> str | None:
        query = {
            "task_name": task_name,
            "model": meta.get("model_name") or "",
            "started_at": meta.get("started_at") or "",
            "finished_at": meta.get("finished_at") or "",
            "experiment": search_experiment,
        }
        url = f"{endpoint.rstrip('/')}/api/eval/match-session?{urllib.parse.urlencode(query)}"
        try:
            req = urllib.request.Request(url, headers=_viewer_headers({}), method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status >= 300:
                    return None
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
        match = data.get("match") if isinstance(data, dict) else None
        if not isinstance(match, dict):
            return None
        session_id = match.get("session_id")
        return str(session_id) if session_id else None

    return request_match("default") or request_match(experiment)


def _build_eval_only_body(resource_attrs: dict[str, str | bool]) -> dict:
    """Build a minimal OTLP payload that enriches/creates one viewer eval session."""
    now_ns = time.time_ns()
    passed = bool(resource_attrs.get("eval.passed", False))
    score = resource_attrs.get("eval.score")
    span_attrs: list[dict] = [{"key": "eval.passed", "value": {"boolValue": passed}}]
    if score is not None:
        try:
            span_attrs.append({"key": "eval.score", "value": {"doubleValue": float(score)}})
            span_attrs.append(
                {"key": "eval.weighted_score", "value": {"doubleValue": float(score)}}
            )
        except (TypeError, ValueError):
            span_attrs.append({"key": "eval.score", "value": {"stringValue": str(score)}})

    def value(v: str | bool) -> dict:
        if isinstance(v, bool):
            return {"boolValue": v}
        return {"stringValue": str(v)}

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": key, "value": value(val)} for key, val in resource_attrs.items()
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "harbor-import"},
                        "spans": [
                            {
                                "traceId": uuid.uuid4().hex,
                                "spanId": uuid.uuid4().hex[:16],
                                "name": "eval",
                                "kind": 1,
                                "startTimeUnixNano": str(now_ns),
                                "endTimeUnixNano": str(now_ns),
                                "attributes": span_attrs,
                                "status": {"code": 1 if passed else 2, "message": ""},
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _trial_dirs(root: Path) -> list[Path]:
    """Find Harbor trial directories under a job dir or parent dir."""
    roots = [root]
    roots.extend(p.parent for p in root.rglob("result.json") if p.parent != root)
    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in roots:
        if candidate in seen:
            continue
        seen.add(candidate)
        result = candidate / "result.json"
        if not result.exists():
            continue
        data = _read_json(result)
        if data.get("trial_name") or (candidate / "config.json").exists():
            out.append(candidate)
    return sorted(out)


def _trial_meta_from_dir(trial_dir: Path) -> dict:
    """Extract Harbor metadata when only a trial directory is available."""
    return _trial_meta(trial_dir / "artifacts" / "traces" / "_synthetic.jsonl")


def _import_trace_file(
    endpoint: str,
    jsonl_path: Path,
    resource_attrs: dict[str, str | bool | int],
    batch_lines: int,
    batch_bytes: int,
) -> tuple[bool, list[str]]:
    """Import one OTLP or portable journal JSONL file.

    Accumulates OTLP bodies and flushes them in batches: many ``resourceSpans``
    envelopes are merged into one POST, avoiding one HTTP request per line. A flush
    is triggered when the batch reaches ``batch_lines`` envelopes or ``batch_bytes``
    of raw input (an approximation of the eventual POST size). Returns
    ``(file_imported, errors)`` where ``file_imported`` is True if any flush
    succeeded (preserving the previous any-success semantics).
    """
    file_imported = False
    errors: list[str] = []
    batch: list[dict] = []
    batch_input_bytes = 0
    flush_count = 0

    def flush() -> None:
        nonlocal file_imported, batch, batch_input_bytes, flush_count
        if not batch:
            return
        flush_count += 1
        if post_traces_batch(endpoint, batch):
            file_imported = True
        else:
            errors.append(f"{jsonl_path.name}: batch #{flush_count} failed to post")
        batch = []
        batch_input_bytes = 0

    with open(jsonl_path) as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                body = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            journal_record = get_journal_record(body)
            if journal_record is not None:
                session_id = str(resource_attrs["session.id"])
                if not post_journal_record(endpoint, journal_record, session_id):
                    errors.append(f"{jsonl_path.name}: failed to post journal record")
                continue
            if "resourceSpans" not in body:
                continue

            inject_resource_attrs(body, resource_attrs)
            batch.append(body)
            # Approximation: raw line length before injection; the re-serialized
            # POST body (with injected resource attrs) is slightly larger.
            batch_input_bytes += len(raw_line)

            if len(batch) >= batch_lines or batch_input_bytes >= batch_bytes:
                flush()

        flush()

    return file_imported, errors


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--endpoint",
    default="http://localhost:5001",
    show_default=True,
    help="Viewer API endpoint.",
)
@click.option(
    "--experiment",
    default=None,
    help="Override experiment name (default: Harbor job directory name).",
)
@click.option(
    "--batch-id",
    default=None,
    help="Batch ID for this import (default: job directory name).",
)
@click.option(
    "--batch-lines",
    default=1000,
    show_default=True,
    help="Max OTLP lines combined into a single POST (per trace file).",
)
@click.option(
    "--batch-bytes",
    default=4_000_000,
    show_default=True,
    help="Max raw input bytes accumulated before flushing a POST (per trace file).",
)
@click.option(
    "--eval-only",
    is_flag=True,
    help="Post Harbor result metadata as eval spans without importing trace JSONL files.",
)
def command(
    path: str,
    endpoint: str,
    experiment: str | None,
    batch_id: str | None,
    batch_lines: int,
    batch_bytes: int,
    eval_only: bool,
):
    """Import NVIDIA OO Agents OTLP traces from a Harbor job directory.

    \b
    PATH can be:
      - A Harbor job directory (contains result.json + trial subdirs)
      - Any parent directory — traces are discovered recursively

    \b
    Examples:
        nooa import-harbor ./jobs/my-job/
        nooa import-harbor ./workspaces/ --endpoint http://host:5001
        nooa import-harbor ./jobs/ --experiment my-eval
        nooa import-harbor ./jobs/ --batch-lines 2000 --batch-bytes 8000000

    OTLP lines are posted in batches (combining many resourceSpans into one
    request) to keep large imports fast; tune with --batch-lines/--batch-bytes.
    """
    root = Path(path)
    files = _find_harbor_traces(root)

    validate_endpoint(endpoint)

    try:
        reachable = check_endpoint_reachable(endpoint)
    except OtlpRequestError as error:
        click.echo(f"Viewer at {endpoint} rejected the request: {error}")
        if error.status_code in (401, 403):
            click.echo("Check NOOA_VIEWER_AUTH_TOKEN and try again.")
        raise SystemExit(1) from None
    if not reachable:
        click.echo(f"Cannot reach viewer at {endpoint}. Is it running?")
        raise SystemExit(1)

    imported = 0
    skipped = 0
    already_exist = 0
    errors = []

    if eval_only:
        trial_dirs = _trial_dirs(root)
        if not trial_dirs:
            click.echo(f"No Harbor trial result directories found under {path}")
            raise SystemExit(1)
        click.echo(f"Found {len(trial_dirs)} Harbor trial result(s)...")
        for trial_dir in trial_dirs:
            meta = _trial_meta_from_dir(trial_dir)
            session_id = meta["trial_name"]
            exp = experiment or meta["experiment"]
            bid = batch_id or meta["job_name"]
            resource_attrs = _harbor_resource_attrs(meta, exp, bid)
            matched_session_id = _find_matching_live_session(endpoint, meta, exp)
            if matched_session_id:
                resource_attrs["session.id"] = matched_session_id
            body = _build_eval_only_body(resource_attrs)
            if post_traces_batch(endpoint, [body]):
                imported += 1
                score_str = f"{meta['score']:.3f}" if meta["score"] is not None else "n/a"
                match_str = f" -> {matched_session_id}" if matched_session_id else ""
                click.echo(
                    f"  + {session_id}{match_str}  score={score_str}  task={meta['task_name']}"
                )
            else:
                skipped += 1
                errors.append(f"{session_id}: failed to post eval metadata")
    else:
        if not files:
            click.echo(f"No Harbor trace files found under {path}")
            click.echo("Expected: <job>/<trial>/artifacts/traces/*.jsonl")
            click.echo("Tip: use --eval-only to group Harbor result metadata without trace files")
            raise SystemExit(1)

        click.echo(f"Found {len(files)} trace file(s)...")

        for jsonl_path in files:
            meta = _trial_meta(jsonl_path)
            session_id = meta["trial_name"]
            exp = experiment or meta["experiment"]
            bid = batch_id or meta["job_name"]

            if session_exists(endpoint, session_id):
                click.echo(f"  ! {session_id}: already exists, skipping")
                already_exist += 1
                continue

            resource_attrs = _harbor_resource_attrs(meta, exp, bid)

            file_imported, file_errors = _import_trace_file(
                endpoint, jsonl_path, resource_attrs, batch_lines, batch_bytes
            )
            errors.extend(file_errors)

            if file_imported:
                imported += 1
                score_str = f"{meta['score']:.3f}" if meta["score"] is not None else "n/a"
                click.echo(f"  + {session_id}  score={score_str}  task={meta['task_name']}")
            else:
                skipped += 1

    click.echo(f"\n{imported} imported, {skipped} skipped, {already_exist} already existed")
    if errors:
        for err in errors[:10]:
            click.echo(f"  ! {err}")
        if len(errors) > 10:
            click.echo(f"  ... and {len(errors) - 10} more errors")

    if imported:
        encoded_batch = urllib.parse.quote(bid or "", safe="")
        encoded_exp = urllib.parse.quote(exp or "", safe="")
        click.echo(f"\nView at: {endpoint}/traces?batch_id={encoded_batch}")
        click.echo(f"Evaluations: {endpoint}/evaluations/{encoded_exp}")
