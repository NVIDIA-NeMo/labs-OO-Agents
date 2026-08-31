# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import NOOA OTLP or portable journal traces from Harbor into the viewer.

Walks a Harbor job directory (or any directory containing one), finds persisted
trace JSONL files, enriches them with trial metadata and arbitrary verifier
results, and posts them to the viewer. A synthetic ``eval`` span makes the
verifier result visible in both experiment and trace detail views.

Usage:
    nooa import-harbor ./jobs/my-job/
    nooa import-harbor ./workspaces/ --endpoint http://host:5001
    nooa import-harbor ./jobs/ --experiment my-eval --batch-id run-42
"""

import hashlib
import json
import math
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

MAX_VERIFIER_FIELDS = 100
MAX_VERIFIER_DEPTH = 4
MAX_VERIFIER_SCALAR_CHARS = 1_000
MAX_VERIFIER_OUTPUT_CHARS = 50_000
MAX_TRACE_SNIFF_RECORDS = 20

OtlpScalar = str | bool | int | float


@dataclass(frozen=True)
class HarborVerifierOutcome:
    """Benchmark-neutral data recovered from Harbor verifier artifacts."""

    result: dict[str, Any] | None
    embedded_result: dict[str, Any] | None
    reward: dict[str, Any] | None
    reward_text: str | None
    score: float | None
    passed: bool | None
    error: str | None
    source: str | None

    def output(self) -> object | None:
        """Return the complete verifier documents with their artifact provenance."""
        if (
            self.result is not None
            and self.embedded_result is None
            and self.reward is None
            and self.reward_text is None
            and self.error is None
        ):
            return self.result
        if (
            self.result is None
            and self.embedded_result is None
            and self.reward is not None
            and self.reward_text is None
            and self.error is None
        ):
            return self.reward

        documents: dict[str, Any] = {}
        if self.result is not None:
            documents["result"] = self.result
        if self.embedded_result is not None:
            documents["embedded_result"] = self.embedded_result
        if self.reward is not None:
            documents["reward"] = self.reward
        if self.reward_text is not None:
            documents["reward_text"] = self.reward_text
        if self.error is not None:
            documents["harness_error"] = self.error
        return documents or None


def _find_harbor_traces(root: Path) -> list[Path]:
    """Find trace JSONL files within Harbor trials, independent of layout."""
    traces: set[Path] = set()
    for trial_dir in _trial_dirs(root):
        seen_contents: set[str] = set()
        candidates = sorted(
            trial_dir.rglob("*.jsonl"),
            key=lambda path: (len(path.relative_to(trial_dir).parts), str(path)),
        )
        for path in candidates:
            if not _is_trace_jsonl(path):
                continue
            content_hash = _trace_content_hash(path)
            if content_hash is None or content_hash in seen_contents:
                continue
            seen_contents.add(content_hash)
            traces.add(path)
    return sorted(traces)


def _is_trace_jsonl(path: Path) -> bool:
    """Return whether a JSONL file contains an OTLP envelope or Nooa journal record."""
    records_seen = 0
    try:
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    body = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records_seen += 1
                if isinstance(body, dict) and (
                    isinstance(body.get("resourceSpans"), list)
                    or get_journal_record(body) is not None
                ):
                    return True
                if records_seen >= MAX_TRACE_SNIFF_RECORDS:
                    break
    except (OSError, UnicodeError):
        return False
    return False


def _trace_content_hash(path: Path) -> str | None:
    """Return a streaming digest used to deduplicate copies within one trial."""
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning an empty dict on any failure."""
    try:
        value = json.loads(path.read_text())
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _coerce_float(value: object) -> float | None:
    """Coerce a value to float, returning None if it cannot be coerced."""
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _score_from_document(document: dict[str, Any] | None) -> float | None:
    """Select a conventional score from an arbitrary verifier document."""
    if not document:
        return None

    for key in ("score", "reward"):
        score = _coerce_float(document.get(key))
        if score is not None:
            return score

    rewards = document.get("rewards")
    if not isinstance(rewards, dict):
        return None
    for key in ("score", "reward"):
        score = _coerce_float(rewards.get(key))
        if score is not None:
            return score

    # Some validators give their single normalized metric a benchmark-specific
    # name (for example "progress"). Counts and multiple metrics remain
    # metadata rather than being guessed into the viewer's 0..1 score field.
    numeric_rewards = [
        score for value in rewards.values() if (score := _coerce_float(value)) is not None
    ]
    if len(numeric_rewards) == 1 and 0.0 <= numeric_rewards[0] <= 1.0:
        return numeric_rewards[0]
    return None


def _explicit_passed(*documents: dict[str, Any] | None) -> bool | None:
    """Return the first explicit boolean pass result from verifier documents."""
    for document in documents:
        if document and isinstance(document.get("passed"), bool):
            return document["passed"]
    return None


def _format_error(value: object) -> str | None:
    if value in (None, "", {}):
        return None
    if isinstance(value, str):
        return value[:MAX_VERIFIER_SCALAR_CHARS]
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)[
            :MAX_VERIFIER_SCALAR_CHARS
        ]
    except (TypeError, ValueError):
        return str(value)[:MAX_VERIFIER_SCALAR_CHARS]


def _read_verifier_outcome(trial_dir: Path, trial_result: dict[str, Any]) -> HarborVerifierOutcome:
    """Read Harbor verifier artifacts without assuming benchmark semantics."""
    result_path = trial_dir / "verifier" / "result.json"
    result = _read_json(result_path)
    source: str | None = "verifier/result.json" if result else None

    embedded = trial_result.get("verifier_result")
    embedded_result = embedded if isinstance(embedded, dict) else None
    if not result and embedded_result is not None:
        result = embedded_result
        embedded_result = None
        source = "result.json:verifier_result"

    reward = _read_json(trial_dir / "verifier" / "reward.json")
    if source is None and reward:
        source = "verifier/reward.json"
    reward_text: str | None = None
    reward_text_path = trial_dir / "verifier" / "reward.txt"
    if reward_text_path.exists():
        try:
            reward_text = reward_text_path.read_text().strip()
        except OSError:
            reward_text = None
    if source is None and reward_text is not None:
        source = "verifier/reward.txt"

    score = _score_from_document(reward)
    if score is None:
        score = _score_from_document(result)
    if score is None and reward_text is not None:
        score = _coerce_float(reward_text)

    passed = _explicit_passed(result, reward)
    if passed is None and score is not None and 0.0 <= score <= 1.0:
        # Preserve Harbor's established 0/1 reward convention. Scores outside
        # the normalized range are never assigned a guessed pass threshold.
        passed = score >= 1.0

    error = _format_error(
        trial_result.get("exception_info")
        or trial_result.get("error")
        or trial_result.get("exception")
    )

    return HarborVerifierOutcome(
        result=result or None,
        embedded_result=embedded_result,
        reward=reward or None,
        reward_text=reward_text,
        score=score,
        passed=passed,
        error=error,
        source=source,
    )


def _read_score(trial_dir: Path, trial_result: dict) -> float | None:
    """Compatibility wrapper returning the generic verifier outcome's score."""
    return _read_verifier_outcome(trial_dir, trial_result).score


def _attribute_segment(value: object) -> str:
    """Make one arbitrary JSON key safe and readable in an OTLP attribute path."""
    segment = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")[:80]
    return segment or "field"


def _flatten_verifier_fields(outcome: HarborVerifierOutcome) -> dict[str, OtlpScalar]:
    """Flatten bounded scalar verifier leaves for dynamic experiment columns."""
    flattened: dict[str, OtlpScalar] = {}

    def walk(value: object, prefix: str, depth: int) -> None:
        if len(flattened) >= MAX_VERIFIER_FIELDS:
            return
        if isinstance(value, bool):
            flattened[prefix] = value
        elif isinstance(value, int):
            flattened[prefix] = value
        elif isinstance(value, float) and math.isfinite(value):
            flattened[prefix] = value
        elif isinstance(value, str):
            flattened[prefix] = value[:MAX_VERIFIER_SCALAR_CHARS]
        elif isinstance(value, dict) and depth < MAX_VERIFIER_DEPTH:
            for key in sorted(value, key=str):
                walk(value[key], f"{prefix}.{_attribute_segment(key)}", depth + 1)

    if outcome.result is not None:
        walk(outcome.result, "eval.verifier", 0)
    if outcome.reward is not None:
        reward_prefix = "eval.verifier.reward_artifact" if outcome.result else "eval.verifier"
        walk(outcome.reward, reward_prefix, 0)
    if outcome.reward_text is not None:
        flattened.setdefault(
            "eval.verifier.reward_text", outcome.reward_text[:MAX_VERIFIER_SCALAR_CHARS]
        )
    if outcome.source:
        flattened["eval.verifier_source"] = outcome.source
    return flattened


def _serialize_verifier_output(outcome: HarborVerifierOutcome) -> str | None:
    """Serialize complete verifier artifacts into one bounded, valid JSON attribute."""
    output = outcome.output()
    if output is None:
        return None
    serialized = json.dumps(output, ensure_ascii=False, sort_keys=True, default=str)
    if len(serialized) <= MAX_VERIFIER_OUTPUT_CHARS:
        return serialized
    return json.dumps(
        {
            "_truncated": True,
            "original_chars": len(serialized),
            # Leave ample headroom for JSON escaping inside the preview string.
            "preview": serialized[: MAX_VERIFIER_OUTPUT_CHARS // 8],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _trial_dir_for_trace(jsonl_path: Path) -> Path:
    """Resolve a trace file to its nearest enclosing Harbor trial directory."""
    for parent in jsonl_path.parents:
        result = _read_json(parent / "result.json")
        if result.get("trial_name") or (parent / "config.json").is_file():
            return parent
    return jsonl_path.parent


def _datetime_ns(value: object, fallback: int) -> int:
    if not isinstance(value, str) or not value:
        return fallback
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0, int(parsed.timestamp() * 1_000_000_000))


def _trial_meta_from_dir(trial_dir: Path) -> dict:
    """Extract Harbor metadata from a discovered trial directory.

    Expected layout::

        <job_dir>/
            result.json              ← job-level (stats.evals for experiment name)
            <trial_name>/
                result.json          ← trial_name, task_name, agent_info
                verifier/
                    result.json      ← arbitrary benchmark verifier result
                    reward.json      ← optional scalar reward (or reward.txt)
                <any path>/*.jsonl   ← OTLP or portable journal traces
    """
    job_dir = trial_dir.parent

    trial_result = _read_json(trial_dir / "result.json")
    job_result = _read_json(job_dir / "result.json")

    trial_name = trial_result.get("trial_name") or trial_dir.name
    task_name = trial_result.get("task_name", "")
    config_value = trial_result.get("config")
    config: dict[str, Any] = config_value if isinstance(config_value, dict) else {}
    agent_value = config.get("agent")
    agent_config: dict[str, Any] = agent_value if isinstance(agent_value, dict) else {}
    task_value = config.get("task")
    task_config: dict[str, Any] = task_value if isinstance(task_value, dict) else {}
    agent_info_value = trial_result.get("agent_info")
    agent_info: dict[str, Any] = agent_info_value if isinstance(agent_info_value, dict) else {}
    agent_name = agent_info.get("name", "") or agent_config.get("name", "")
    model_name = agent_config.get("model_name", "")
    agent_type = ""
    kwargs = agent_config.get("kwargs") if isinstance(agent_config.get("kwargs"), dict) else {}
    if kwargs:
        agent_type = str(kwargs.get("agent_type") or "")
    source = trial_result.get("source") or task_config.get("source") or ""

    verifier = _read_verifier_outcome(trial_dir, trial_result)

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
        "score": verifier.score,
        "verifier": verifier,
        "harbor_eval": harbor_eval,
        "experiment": job_dir.name or harbor_eval or "harbor",
        "job_name": job_dir.name,
    }


def _trial_meta(jsonl_path: Path) -> dict:
    """Extract Harbor metadata for a trace file from its enclosing trial."""
    return _trial_meta_from_dir(_trial_dir_for_trace(jsonl_path))


def _harbor_resource_attrs(meta: dict, experiment: str, batch_id: str) -> dict[str, OtlpScalar]:
    """Build viewer resource attrs that make a Harbor trial appear as an eval row."""
    attrs: dict[str, OtlpScalar] = {
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

    verifier = meta.get("verifier")
    if isinstance(verifier, HarborVerifierOutcome):
        attrs.update(_flatten_verifier_fields(verifier))
        attrs["eval.has_exception"] = verifier.error is not None
        if verifier.error is not None:
            attrs["eval.error"] = verifier.error
        if verifier.score is not None:
            attrs["eval.score"] = verifier.score
            attrs["eval.weighted_score"] = verifier.score
        if verifier.passed is not None:
            attrs["eval.passed"] = verifier.passed
    elif meta.get("score") is not None:
        # Compatibility for callers constructing metadata directly.
        score = float(meta["score"])
        attrs["eval.score"] = score
        attrs["eval.weighted_score"] = score
        if 0.0 <= score <= 1.0:
            attrs["eval.passed"] = score >= 1.0
    return attrs


def _trace_resource_attrs(resource_attrs: dict[str, OtlpScalar]) -> dict[str, OtlpScalar]:
    """Keep large verifier detail on the single eval span, not every trace envelope."""
    return {
        key: value
        for key, value in resource_attrs.items()
        if not key.startswith("eval.verifier") and key not in {"eval.error", "eval.has_exception"}
    }


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


def _build_eval_only_body(
    resource_attrs: dict[str, OtlpScalar], meta: dict[str, Any] | None = None
) -> dict:
    """Build a deterministic OTLP eval span for regular or eval-only imports."""
    now_ns = time.time_ns()
    start_ns = _datetime_ns(meta.get("started_at") if meta else None, now_ns)
    end_ns = _datetime_ns(meta.get("finished_at") if meta else None, start_ns)
    end_ns = max(start_ns, end_ns)

    def value(v: OtlpScalar) -> dict:
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}
        if isinstance(v, float):
            return {"doubleValue": v}
        return {"stringValue": str(v)}

    span_attrs: list[dict[str, Any]] = []
    for key in (
        "eval.test_id",
        "eval.test_name",
        "eval.display_name",
        "eval.method",
        "eval.model",
        "eval.agent_class",
        "eval.score",
        "eval.weighted_score",
        "eval.passed",
        "eval.error",
        "eval.has_exception",
    ):
        if key in resource_attrs:
            span_attrs.append({"key": key, "value": value(resource_attrs[key])})

    verifier = meta.get("verifier") if meta else None
    serialized_output = (
        _serialize_verifier_output(verifier)
        if isinstance(verifier, HarborVerifierOutcome)
        else None
    )
    if serialized_output is not None:
        span_attrs.append({"key": "eval.output", "value": {"stringValue": serialized_output}})

    trial_name = str(resource_attrs.get("eval.harbor_trial_name") or resource_attrs["session.id"])
    digest = hashlib.sha256(
        "\0".join(
            (
                str(resource_attrs.get("experiment", "")),
                str(resource_attrs.get("batch_id", "")),
                trial_name,
                "harbor-eval-v1",
            )
        ).encode()
    ).hexdigest()
    error = verifier.error if isinstance(verifier, HarborVerifierOutcome) else None

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
                                "traceId": digest[:32],
                                "spanId": digest[32:48],
                                "name": "eval",
                                "kind": 1,
                                "startTimeUnixNano": str(start_ns),
                                "endTimeUnixNano": str(end_ns),
                                "attributes": span_attrs,
                                "status": {"code": 2 if error else 1, "message": error or ""},
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


def _import_trace_file(
    endpoint: str,
    jsonl_path: Path,
    resource_attrs: dict[str, OtlpScalar],
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

            inject_resource_attrs(
                body,
                resource_attrs,
                overwrite_keys={"session.id", "experiment", "batch_id"},
            )
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
    enriched = 0
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
            body = _build_eval_only_body(resource_attrs, meta)
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
            click.echo("Expected: OTLP or Nooa journal JSONL within a Harbor trial directory")
            click.echo("Tip: use --eval-only to group Harbor result metadata without trace files")
            raise SystemExit(1)

        files_by_trial: dict[Path, list[Path]] = {}
        for jsonl_path in files:
            files_by_trial.setdefault(_trial_dir_for_trace(jsonl_path), []).append(jsonl_path)
        click.echo(
            f"Found {len(files)} trace file(s) across {len(files_by_trial)} Harbor trial(s)..."
        )

        for _trial_dir, trial_files in sorted(files_by_trial.items()):
            meta = _trial_meta(trial_files[0])
            session_id = meta["trial_name"]
            exp = experiment or meta["experiment"]
            bid = batch_id or meta["job_name"]
            resource_attrs = _harbor_resource_attrs(meta, exp, bid)

            matched_session_id = _find_matching_live_session(endpoint, meta, exp)
            target_session_id = matched_session_id or session_id
            resource_attrs["session.id"] = target_session_id

            exists = session_exists(endpoint, target_session_id)
            trace_imported = False
            if exists:
                already_exist += 1
            else:
                trace_resource_attrs = _trace_resource_attrs(resource_attrs)
                for jsonl_path in trial_files:
                    file_imported, file_errors = _import_trace_file(
                        endpoint,
                        jsonl_path,
                        trace_resource_attrs,
                        batch_lines,
                        batch_bytes,
                    )
                    trace_imported = trace_imported or file_imported
                    errors.extend(file_errors)

            eval_imported = post_traces_batch(
                endpoint, [_build_eval_only_body(resource_attrs, meta)]
            )
            if not eval_imported:
                errors.append(f"{session_id}: failed to post eval metadata")

            if exists and eval_imported:
                enriched += 1
                match_str = f" -> {target_session_id}" if matched_session_id else ""
                click.echo(f"  ~ {session_id}{match_str}: evaluation metadata enriched")
            elif trace_imported:
                imported += 1
                score_str = f"{meta['score']:.3f}" if meta["score"] is not None else "n/a"
                match_str = f" -> {target_session_id}" if matched_session_id else ""
                click.echo(
                    f"  + {session_id}{match_str}  score={score_str}  task={meta['task_name']}"
                )
            else:
                skipped += 1

    click.echo(
        f"\n{imported} imported, {enriched} enriched, {skipped} skipped, "
        f"{already_exist} already existed"
    )
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
