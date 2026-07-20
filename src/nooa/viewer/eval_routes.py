# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation viewer API routes -- backed by otlp_store.

All evaluation data is stored as OTLP traces with eval.* attributes.
Experiments are groups of sessions sharing the same ``experiment`` resource attribute.

The eval metadata is stored as a single JSON blob in SQLite.  This module
unpacks it and returns all keys as top-level fields in each test dict,
plus a ``metadata_keys`` list so the frontend can dynamically build columns.
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from . import otlp_store


def _epoch_to_iso(epoch: str | float) -> str:
    """Convert a Unix epoch (seconds) to an ISO 8601 string."""
    try:
        return datetime.fromtimestamp(float(epoch), tz=UTC).isoformat()
    except (ValueError, TypeError, OSError):
        return ""


router = APIRouter(prefix="/api/eval")


# =============================================================================
# Response models
# =============================================================================


class ExperimentSummaryItem(BaseModel):
    id: str
    timestamp: str
    prompt_version: str = "v1"
    models: list[str]
    test_count: int
    passed_count: int
    status: str
    suite_name: str | None = None


class PaginatedExperimentsResponse(BaseModel):
    experiments: list[ExperimentSummaryItem]
    total: int
    page: int
    limit: int
    has_more: bool


class ExperimentStatus(BaseModel):
    running: bool
    last_modified: str
    test_count: int


# =============================================================================
# Helpers
# =============================================================================

# Keys that are used for the test name column and should not appear as
# separate metadata columns in the table.
_NAME_KEYS = {"display_name", "test_name", "test_id"}

# Keys that contain large/complex data and belong in the detail view only.
_DETAIL_ONLY_KEYS = {"input", "output", "expected", "scores", "trace_file", "duration_seconds"}

# Trace-derived table metrics. These are rendered as fixed frontend columns,
# not discovered as eval metadata.
_TRACE_METRIC_KEYS = {"duration_ms", "span_count"}


def _build_experiment_summary_item(experiment: str) -> ExperimentSummaryItem | None:
    """Build summary for a single experiment from indexed sessions."""
    sessions = otlp_store.list_sessions(experiment=experiment, eval_only=True)
    if not sessions:
        return None

    models = sorted(
        {s["eval"].get("model", "") for s in sessions if s.get("eval", {}).get("model")}
    )
    passed = sum(1 for s in sessions if s.get("eval", {}).get("passed"))
    modified = max((float(s["modified"]) for s in sessions), default=0.0)

    return ExperimentSummaryItem(
        id=experiment,
        timestamp=_epoch_to_iso(modified),
        models=models,
        test_count=len(sessions),
        passed_count=passed,
        status="completed",
    )


def _session_to_test_dict(
    session: dict[str, Any],
    include_detail: bool = False,
) -> dict[str, Any]:
    """Convert an otlp_store session dict to a test result dict for the frontend.

    All eval metadata keys are returned as top-level fields.  The frontend
    discovers available columns from the ``metadata_keys`` field returned
    at the experiment level.
    """
    ev = session.get("eval", {})

    d: dict[str, Any] = {
        "session_id": session["id"],
        "test_id": ev.get("test_id", session["id"]),
        "test_name": ev.get("test_name", ""),
        "test_type": ev.get("test_type", ""),
        "display_name": ev.get("display_name") or ev.get("test_name") or session["id"],
        "passed": ev.get("passed"),
    }

    for key, val in ev.items():
        if key == "passed":
            continue
        if not include_detail and key in _DETAIL_ONLY_KEYS:
            continue
        d.setdefault(key, val)

    # Trace-derived metrics (merged in by _sessions_to_test_dicts) override
    # any same-named eval-metadata field.
    d["duration_ms"] = session.get("duration_ms")
    d["span_count"] = session.get("span_count")

    return d


def _sessions_to_test_dicts(
    sessions: list[dict[str, Any]], *, include_detail: bool = False
) -> list[dict[str, Any]]:
    """Build test rows, enriching with trace-derived ``duration_ms``.

    ``span_count`` is already on the session dict from the sessions row;
    ``duration_ms`` is computed (latest span end − earliest start) via a
    single batched query.
    """
    durations = otlp_store.get_session_durations_ms([s["id"] for s in sessions])
    return [
        _session_to_test_dict(
            {**s, "duration_ms": durations.get(s["id"])}, include_detail=include_detail
        )
        for s in sessions
    ]


def _collect_metadata_keys(tests: list[dict[str, Any]]) -> list[str]:
    """Return the sorted union of metadata keys across all tests.

    Excludes keys used for test name and detail-only keys.
    """
    all_keys: set[str] = set()
    skip = {"session_id"} | _NAME_KEYS | _DETAIL_ONLY_KEYS | _TRACE_METRIC_KEYS
    for t in tests:
        all_keys.update(k for k in t if k not in skip)
    return sorted(all_keys)


# =============================================================================
# Routes
# =============================================================================


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "experiment_count": len(otlp_store.list_experiments()),
    }


def _parse_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


@router.get("/match-session")
def match_session(
    task_name: str = Query(..., description="Harbor task name, e.g. django__django-12345"),
    model: str | None = Query(None, description="Model name from Harbor result metadata"),
    started_at: str | None = Query(None, description="Harbor trial started_at timestamp or epoch"),
    finished_at: str | None = Query(
        None, description="Harbor trial finished_at timestamp or epoch"
    ),
    experiment: str = Query("default", description="Live-stream experiment to search"),
) -> dict[str, Any]:
    match = otlp_store.find_eval_session_for_task(
        task_name,
        model=model,
        started_at=_parse_epoch(started_at),
        finished_at=_parse_epoch(finished_at),
        experiment=experiment,
    )
    return {"match": match}


@router.get("/experiments")
def list_experiments(
    page: int = 1,
    limit: int = 50,
    search: str | None = None,
) -> PaginatedExperimentsResponse:
    limit = max(1, min(limit, 200))
    page = max(1, page)

    all_experiments = otlp_store.list_experiments()

    summaries: list[ExperimentSummaryItem] = []
    for name in all_experiments:
        item = _build_experiment_summary_item(name)
        if item:
            summaries.append(item)

    summaries.sort(key=lambda x: x.timestamp, reverse=True)

    if search:
        search_lower = search.lower()
        summaries = [s for s in summaries if search_lower in s.id.lower()]

    total = len(summaries)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated = summaries[start_idx:end_idx]

    return PaginatedExperimentsResponse(
        experiments=paginated,
        total=total,
        page=page,
        limit=limit,
        has_more=end_idx < total,
    )


@router.get("/experiments/all")
def list_all_experiments() -> list[ExperimentSummaryItem]:
    summaries: list[ExperimentSummaryItem] = []
    for name in otlp_store.list_experiments():
        item = _build_experiment_summary_item(name)
        if item:
            summaries.append(item)
    summaries.sort(key=lambda x: x.timestamp, reverse=True)
    return summaries


def _collect_column_info(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return column metadata with unique values for each metadata key."""
    skip = {"session_id"} | _NAME_KEYS | _DETAIL_ONLY_KEYS | _TRACE_METRIC_KEYS
    key_values: dict[str, set[str]] = {}
    for t in tests:
        for k, v in t.items():
            if k in skip:
                continue
            if k not in key_values:
                key_values[k] = set()
            if v is not None and v != "":
                key_values[k].add(str(v))
    return [{"key": k, "values": sorted(vals)} for k, vals in sorted(key_values.items()) if vals]


@router.get("/experiment/{experiment_id}")
def get_experiment(
    experiment_id: str,
    request: Request,
    page: int | None = None,
    limit: int = 50,
    sort_by: str | None = None,
    sort_dir: str = "desc",
    search: str | None = None,
):
    sessions = otlp_store.list_sessions(experiment=experiment_id, eval_only=True)
    if not sessions:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    tests = _sessions_to_test_dicts(sessions, include_detail=True)
    metadata_keys = _collect_metadata_keys(tests)
    columns = _collect_column_info(tests)

    models = sorted({t.get("model", "") for t in tests if t.get("model")})
    variants = sorted({t.get("variant", "") for t in tests if t.get("variant")})

    summary = otlp_store.get_experiment_summary(experiment_id)

    # --- Filtering ---
    known_params = {"page", "limit", "sort_by", "sort_dir", "search"}
    meta_filters = {k: v for k, v in request.query_params.items() if k not in known_params and v}

    if search:
        sl = search.lower()
        tests = [
            t
            for t in tests
            if sl in (t.get("display_name") or "").lower()
            or sl in (t.get("test_id") or "").lower()
            or sl in (t.get("test_name") or "").lower()
        ]

    for key, val in meta_filters.items():
        if val.startswith("~"):
            substr = val[1:].lower()
            tests = [t for t in tests if substr in str(t.get(key, "")).lower()]
        else:
            tests = [t for t in tests if str(t.get(key, "")) == val]

    # --- Sorting ---
    if sort_by:
        reverse = sort_dir != "asc"

        def _sort_key(t: dict[str, Any]):
            v = t.get(sort_by)
            if v is None:
                return (0, 0, "")
            if isinstance(v, bool):
                return (1, int(v), "")
            if isinstance(v, (int, float)):
                return (1, v, "")
            return (2, 0, str(v).lower())

        tests.sort(key=_sort_key, reverse=reverse)

    total = len(tests)

    # --- Pagination (only when page is explicitly requested) ---
    if page is not None:
        page = max(1, page)
        limit = max(1, min(limit, 500))
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated = tests[start_idx:end_idx]
        has_more = end_idx < total
    else:
        paginated = tests
        has_more = False

    return {
        "experiment_id": experiment_id,
        "metadata": {
            "version": 1,
            "metadata": {
                "timestamp": _epoch_to_iso(summary.get("modified", "0")),
                "models": models,
                "suite_name": experiment_id,
            },
        },
        "results": paginated,
        "total": total,
        "page": page or 1,
        "limit": limit,
        "has_more": has_more,
        "metadata_keys": metadata_keys,
        "columns": columns,
        "completion": None,
        "annotations": [],
        "timestamp": _epoch_to_iso(summary.get("modified", "0")),
        "prompt_version": "v1",
        "status": "completed",
        "variants": variants,
        "models": models,
    }


@router.get("/experiment/{experiment_id}/summary")
def get_experiment_summary_endpoint(
    experiment_id: str, request: Request, search: str | None = None
):
    sessions = otlp_store.list_sessions(experiment=experiment_id, eval_only=True)
    if not sessions:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    tests = _sessions_to_test_dicts(sessions)

    # Apply the same filters as the detail endpoint
    known_params = {"search"}
    meta_filters = {k: v for k, v in request.query_params.items() if k not in known_params and v}

    if search:
        sl = search.lower()
        tests = [
            t
            for t in tests
            if sl in (t.get("display_name") or "").lower()
            or sl in (t.get("test_id") or "").lower()
            or sl in (t.get("test_name") or "").lower()
        ]

    for key, val in meta_filters.items():
        if val.startswith("~"):
            substr = val[1:].lower()
            tests = [t for t in tests if substr in str(t.get(key, "")).lower()]
        else:
            tests = [t for t in tests if str(t.get(key, "")) == val]

    # Compute summary from filtered tests
    total = len(tests)
    passed = sum(1 for t in tests if t.get("passed"))
    total_score = 0.0
    scored = 0
    by_model: dict[str, dict] = {}
    by_tier: dict[str, dict] = {}
    by_test_type: dict[str, dict] = {}
    matrix: dict[str, dict[str, dict]] = {}

    for t in tests:
        model = t.get("model", "unknown")
        test_name = t.get("test_name", "unknown")
        t_passed = t.get("passed", False)
        tier = t.get("tier", "")

        # by_model
        if model not in by_model:
            by_model[model] = {"total": 0, "passed": 0}
        by_model[model]["total"] += 1
        if t_passed:
            by_model[model]["passed"] += 1

        # by_tier
        if tier:
            if tier not in by_tier:
                by_tier[tier] = {"total": 0, "passed": 0}
            by_tier[tier]["total"] += 1
            if t_passed:
                by_tier[tier]["passed"] += 1

        # by_test_type
        if test_name not in by_test_type:
            by_test_type[test_name] = {"total": 0, "passed": 0, "score_sum": 0.0}
        by_test_type[test_name]["total"] += 1
        if t_passed:
            by_test_type[test_name]["passed"] += 1
        score = t.get("score") if t.get("score") is not None else t.get("weighted_score")
        if score is not None:
            try:
                s = float(score)
                total_score += s
                scored += 1
                by_test_type[test_name]["score_sum"] += s
            except (ValueError, TypeError):
                pass

        # matrix
        if test_name not in matrix:
            matrix[test_name] = {}
        if model not in matrix[test_name]:
            matrix[test_name][model] = {"total": 0, "passed": 0}
        matrix[test_name][model]["total"] += 1
        if t_passed:
            matrix[test_name][model]["passed"] += 1

    formatted_matrix: dict[str, dict[str, dict]] = {}
    for test_type, models_data in matrix.items():
        formatted_matrix[test_type] = {}
        for model, stats in models_data.items():
            rate = round(stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            formatted_matrix[test_type][model] = {
                "passed": stats["passed"],
                "total": stats["total"],
                "rate": rate,
            }

    avg_score = (total_score / scored) if scored > 0 else 0.0
    success_rate = (passed / total) if total > 0 else 0.0

    return {
        "overall": {
            "total": total,
            "passed": passed,
            "avg_score": avg_score,
            "success_rate": success_rate,
            "run_count": 1,
        },
        "by_model": by_model,
        "by_test_type": by_test_type,
        "by_tier": by_tier,
        "matrix": {
            "models": sorted(by_model.keys()),
            "test_types": sorted(by_test_type.keys()),
            "cells": formatted_matrix,
        },
        "completion": None,
    }


@router.get("/experiment/{experiment_id}/tests")
def get_experiment_tests(experiment_id: str):
    sessions = otlp_store.list_sessions(experiment=experiment_id, eval_only=True)
    if not sessions:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    tests = _sessions_to_test_dicts(sessions, include_detail=True)
    metadata_keys = _collect_metadata_keys(tests)
    return {"tests": tests, "metadata_keys": metadata_keys}


@router.get("/experiment/{experiment_id}/status")
def get_experiment_status_endpoint(experiment_id: str) -> ExperimentStatus:
    sessions = otlp_store.list_sessions(experiment=experiment_id, eval_only=True)
    if not sessions:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    modified = max((float(s["modified"]) for s in sessions), default=0.0)
    return ExperimentStatus(
        running=False,
        last_modified=_epoch_to_iso(modified),
        test_count=len(sessions),
    )


@router.get("/experiment/{experiment_id}/trace/{test_id:path}")
def get_test_trace(experiment_id: str, test_id: str):
    """Get trace events for a specific test.

    Since traces are now stored as OTLP, returns the session's spans
    in the OTLP format. The frontend can link directly to
    /traces/view?session_id={session_id} instead.
    """
    sessions = otlp_store.list_sessions(experiment=experiment_id, eval_only=True)
    session_id = None
    for s in sessions:
        ev = s.get("eval", {})
        if ev.get("test_id") == test_id or s["id"] == test_id:
            session_id = s["id"]
            break

    if not session_id:
        return {"events": [], "trace_file": None, "session_id": None}

    try:
        spans = otlp_store.get_session_spans(session_id)
    except FileNotFoundError:
        spans = []

    return {
        "events": spans,
        "trace_file": None,
        "session_id": session_id,
        "format": "otlp",
    }


@router.get("/experiments/metrics")
def get_experiments_metrics(limit: int = -1):
    """Get aggregate metrics across experiments."""
    experiments = otlp_store.list_experiments()

    history = []
    for name in experiments:
        summary = otlp_store.get_experiment_summary(name)
        if summary["total"] == 0:
            continue

        tier_metrics = {
            "overall": {
                "success_rate": summary.get("success_rate", 0.0) * 100,
                "tests_passed": summary.get("passed", 0),
                "tests_total": summary.get("total", 0),
            }
        }
        for tier_name, tier_data in summary.get("by_tier", {}).items():
            t = tier_data["total"]
            p = tier_data["passed"]
            tier_metrics[tier_name] = {
                "success_rate": (p / t * 100) if t > 0 else 0.0,
                "tests_passed": p,
                "tests_total": t,
            }

        history.append(
            {
                "experiment_id": name,
                "created_at": _epoch_to_iso(summary.get("modified", "0")),
                "sha": name[:8],
                "tier_metrics": tier_metrics,
            }
        )

    history.sort(key=lambda x: x["created_at"])
    if limit > 0:
        history = history[-limit:]

    return {
        "generated_at": "",
        "history": history,
    }


@router.get("/debug/experiments")
def debug_experiments():
    """Debug endpoint to show experiment indexing."""
    experiments = otlp_store.list_experiments()
    result = []
    for name in experiments:
        sessions = otlp_store.list_sessions(experiment=name, eval_only=True)
        result.append(
            {
                "experiment": name,
                "session_count": len(sessions),
                "sessions": [s["id"] for s in sessions[:10]],
            }
        )
    return {"experiments": result, "db_path": str(otlp_store.DB_PATH)}
