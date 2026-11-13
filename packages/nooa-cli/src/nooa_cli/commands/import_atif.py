# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import ATIF trajectories into the viewer as OTLP traces.

Accepts ATIF ``.json`` files (one trajectory each, e.g. Harbor's
``trajectory.json``) and ``.jsonl`` files (one trajectory per line), or a
directory searched recursively for both. Each trajectory becomes one session.

Usage:
    nooa import-atif trajectory.json
    nooa import-atif ./jobs/my-job/ --experiment my-eval
    nooa import-atif traces.jsonl --endpoint http://host:5001 --batch-id run-42
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import click

from ._otlp_helpers import (
    OtlpRequestError,
    check_endpoint_reachable,
    post_traces_batch_with_retry,
    session_exists,
    sync_ingest,
    validate_endpoint,
)

NAME = "import-atif"

_BATCH_TRAJECTORIES = 50


def _is_trajectory(value: object) -> bool:
    return (
        isinstance(value, dict)
        and str(value.get("schema_version", "")).startswith("ATIF")
        and isinstance(value.get("steps"), list)
    )


def _trajectories(path: Path, errors: list[str]) -> Iterator[tuple[Path, int, dict]]:
    """Yield (file, line or 0, trajectory) for every ATIF trajectory under path."""
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix in (".json", ".jsonl"))
    else:
        files = [path]
    for file in files:
        if file.suffix == ".jsonl":
            with file.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as error:
                        errors.append(f"{file}:{line_number}: invalid JSON: {error.msg}")
                        continue
                    if _is_trajectory(value):
                        yield file, line_number, value
            continue
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            if not path.is_dir():  # In a directory, other JSON files are expected.
                errors.append(f"{file}: {error}")
            continue
        if _is_trajectory(value):
            yield file, 0, value
        elif not path.is_dir():
            errors.append(f"{file}: not an ATIF trajectory")


def _session_id(trajectory: dict, file: Path, line: int, root: Path, prefix: str) -> str:
    """Prefer the trajectory's own ID; fall back to its location in the import."""
    name = trajectory.get("trajectory_id") or trajectory.get("session_id")
    if not name:
        relative = file.relative_to(root) if root.is_dir() else Path(file.name)
        name = str(relative.with_suffix("")) + (f"-{line}" if line else "")
    return prefix + re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-")


@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--endpoint", default="http://localhost:5001", show_default=True, help="Viewer API endpoint."
)
@click.option("--batch-id", default=None, help="Batch ID for this import (default: generated).")
@click.option("--experiment", default=None, help="Experiment name to group the sessions under.")
@click.option("--session-prefix", default="", help="Prefix for generated session IDs.")
@click.option(
    "--code-tool",
    "code_tools",
    multiple=True,
    help="Tool name whose calls render as code execution (repeatable; "
    "default: execute_python, python_cell).",
)
@click.option(
    "--max-retries",
    default=5,
    show_default=True,
    type=click.IntRange(min=0),
    help="Retries for transient viewer errors such as HTTP 503.",
)
def command(
    path: Path,
    endpoint: str,
    batch_id: str | None,
    experiment: str | None,
    session_prefix: str,
    code_tools: tuple[str, ...],
    max_retries: int,
):
    """Import ATIF trajectory files into the viewer."""
    from nooa.atif.importer import DEFAULT_CODE_TOOLS, trajectory_to_otlp

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

    if batch_id is None:
        batch_id = f"import_atif_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    resource = {"batch_id": batch_id}
    tools = frozenset(code_tools) if code_tools else DEFAULT_CODE_TOOLS

    errors: list[str] = []
    seen: set[str] = set()
    pending: list[dict] = []
    imported = existing = failed = 0

    def flush() -> None:
        nonlocal imported, failed
        if not pending:
            return
        try:
            post_traces_batch_with_retry(endpoint, pending, max_retries=max_retries)
            imported += len(pending)
        except OtlpRequestError as error:
            errors.append(f"batch of {len(pending)} trajectories: {error}")
            failed += len(pending)
        pending.clear()

    for file, line, trajectory in _trajectories(path, errors):
        session_id = _session_id(trajectory, file, line, path, session_prefix)
        base, n = session_id, 2
        while session_id in seen:
            session_id, n = f"{base}-{n}", n + 1
        seen.add(session_id)
        if session_exists(endpoint, session_id):
            existing += 1
            continue
        try:
            body = trajectory_to_otlp(
                trajectory,
                session_id,
                experiment=experiment,
                resource=resource,
                code_tools=tools,
            )
        except (TypeError, ValueError, KeyError) as error:
            errors.append(f"{file}{f':{line}' if line else ''}: cannot convert: {error}")
            failed += 1
            continue
        pending.append(body)
        if len(pending) >= _BATCH_TRAJECTORIES:
            flush()
    flush()

    if imported:
        try:
            sync_ingest(endpoint)
        except OtlpRequestError as error:
            errors.append(f"waiting for ingestion: {error}")

    click.echo(
        f"Imported {imported} trajectory(ies) (batch_id={batch_id}); "
        f"{existing} already present, {failed} failed."
    )
    for error in errors:
        click.echo(f"  ! {error}")
    if not imported and not existing:
        raise SystemExit(1)
    if failed:
        raise SystemExit(1)
