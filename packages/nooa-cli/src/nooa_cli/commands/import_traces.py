# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import OTLP or portable NOOA journal .jsonl files into the viewer.

Usage:
    nooa import-traces ./traces/
    nooa import-traces my_trace.jsonl --endpoint http://host:5001
    nooa import-traces ./experiment/ --batch-id my-experiment-v2
"""

import json
import shlex
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import click

from ._otlp_helpers import (
    OtlpRequestError,
    check_endpoint_reachable,
    get_journal_record,
    get_session_span_count,
    inject_resource_attrs,
    post_annotations,
    post_journal_record,
    post_traces_batch_with_retry,
    session_exists,
    sync_ingest,
    validate_endpoint,
)

NAME = "import-traces"

TRACE_EXTENSIONS = (".nooa.jsonl", ".jsonl")


def _find_trace_files(path: Path) -> list[Path]:
    """Find all trace JSONL files in a file or directory."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    files = []
    for ext in TRACE_EXTENSIONS:
        files.extend(path.rglob(f"*{ext}"))
    seen = set()
    unique = []
    for f in sorted(files):
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique.append(f)
    return unique


def _detect_format(line: dict) -> str:
    """Detect whether a parsed JSON line is OTLP or legacy format."""
    if "resourceSpans" in line:
        return "otlp"
    if "span_id" in line or "trace_id" in line:
        return "legacy"
    return "unknown"


def _session_id_from_filename(path: Path) -> str:
    """Derive a session ID from the trace file's basename."""
    name = path.name
    for ext in sorted(TRACE_EXTENSIONS, key=len, reverse=True):
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def _count_otlp_spans(body: dict) -> int:
    """Count spans in a validated OTLP envelope."""
    count = 0
    for resource_spans in body.get("resourceSpans", []):
        if not isinstance(resource_spans, dict):
            continue
        scope_spans = resource_spans.get("scopeSpans", [])
        if not isinstance(scope_spans, list):
            continue
        for scope in scope_spans:
            if not isinstance(scope, dict):
                continue
            spans = scope.get("spans", [])
            if isinstance(spans, list):
                count += len(spans)
    return count


@dataclass
class _TraceBatch:
    """Mutable state for one bounded OTLP request."""

    bodies: list[dict] = field(default_factory=list)
    input_bytes: int = 0
    first_line: int = 0
    last_line: int = 0
    attempted: bool = False
    accepted: bool = False

    def clear(self) -> None:
        """Reset buffered request data while preserving aggregate status."""
        self.bodies = []
        self.input_bytes = 0
        self.first_line = 0
        self.last_line = 0


def _post_batch(
    endpoint: str,
    bodies: list[dict],
    *,
    max_retries: int,
    file_name: str,
    first_line: int,
    last_line: int,
) -> str | None:
    """Post one trace batch and return a user-facing error, if any."""
    line_range = str(first_line) if first_line == last_line else f"{first_line}-{last_line}"
    try:
        post_traces_batch_with_retry(
            endpoint,
            bodies,
            max_retries=max_retries,
        )
    except OtlpRequestError as error:
        return f"{file_name}:{line_range}: {error}"
    return None


def _flush_batch(
    endpoint: str,
    state: _TraceBatch,
    *,
    max_retries: int,
    file_name: str,
) -> str | None:
    """Post and clear buffered trace bodies, returning a user-facing error."""
    if not state.bodies:
        return None
    state.attempted = True
    error = _post_batch(
        endpoint,
        state.bodies,
        max_retries=max_retries,
        file_name=file_name,
        first_line=state.first_line,
        last_line=state.last_line,
    )
    state.clear()
    if error is None:
        state.accepted = True
    return error


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--endpoint",
    default="http://localhost:5001",
    show_default=True,
    help="Viewer API endpoint.",
)
@click.option(
    "--batch-id",
    default=None,
    help="Batch ID for this import (default: auto-generated).",
)
@click.option(
    "--batch-lines",
    default=1000,
    show_default=True,
    type=click.IntRange(min=1),
    help="Max OTLP lines combined into one request.",
)
@click.option(
    "--batch-bytes",
    default=4_000_000,
    show_default=True,
    type=click.IntRange(min=1),
    help="Max raw input bytes combined into one request.",
)
@click.option(
    "--max-retries",
    default=5,
    show_default=True,
    type=click.IntRange(min=0),
    help="Retries for transient viewer errors such as HTTP 503.",
)
def command(
    path: str,
    endpoint: str,
    batch_id: str | None,
    batch_lines: int,
    batch_bytes: int,
    max_retries: int,
):
    """Import OTLP and portable NOOA journal .jsonl files into the viewer."""
    target = Path(path)
    files = _find_trace_files(target)

    if not files:
        click.echo(f"No trace files found in {path}")
        raise SystemExit(1)

    validate_endpoint(endpoint)

    if batch_id is None:
        batch_id = f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

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

    click.echo(f"Importing {len(files)} trace file(s) (batch_id={batch_id})...")

    imported = 0
    skipped = 0
    failed = 0
    already_exist = 0
    annotations_imported = 0
    errors: list[str] = []

    for file in files:
        session_id = _session_id_from_filename(file)

        # Check for existing session before importing
        if session_exists(endpoint, session_id):
            click.echo(
                f"  ! {file.name}: session '{session_id}' already exists, skipping "
                f"(delete it first or rename the file to import as a new session)"
            )
            already_exist += 1
            continue

        inject_attrs = {"batch_id": batch_id, "session.id": session_id}

        file_errors: list[str] = []
        file_imported = False
        expected_span_count = 0
        is_legacy = False
        deferred_annotations: list[dict] = []
        batch = _TraceBatch()

        with open(file) as f:
            for line_num, raw_line in enumerate(f, 1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                try:
                    body = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    file_errors.append(f"{file.name}:{line_num}: invalid JSON: {error.msg}")
                    continue

                journal_record = get_journal_record(body)
                if journal_record is not None:
                    if not post_journal_record(endpoint, journal_record, session_id):
                        file_errors.append(f"{file.name}:{line_num}: failed to post journal record")
                    else:
                        file_imported = True
                    continue

                # Handle annotation lines from exported traces
                if "annotations" in body and "resourceSpans" not in body:
                    anns = body["annotations"]
                    if isinstance(anns, list):
                        deferred_annotations.extend(anns)
                    continue

                fmt = _detect_format(body)

                if fmt == "legacy":
                    if not is_legacy:
                        file_errors.append(f"{file.name}: legacy format not supported, skipping")
                        is_legacy = True
                    continue

                if fmt != "otlp":
                    continue

                resource_spans = body.get("resourceSpans")
                if not isinstance(resource_spans, list):
                    file_errors.append(
                        f"{file.name}:{line_num}: resourceSpans must be a JSON array"
                    )
                    continue
                if not resource_spans:
                    continue

                raw_line_bytes = len(raw_line.encode("utf-8"))
                if batch.bodies and batch.input_bytes + raw_line_bytes > batch_bytes:
                    batch_error = _flush_batch(
                        endpoint,
                        batch,
                        max_retries=max_retries,
                        file_name=file.name,
                    )
                    if batch_error:
                        file_errors.append(batch_error)
                        break

                inject_resource_attrs(body, inject_attrs, overwrite=True)
                if not batch.bodies:
                    batch.first_line = line_num
                batch.last_line = line_num
                batch.bodies.append(body)
                batch.input_bytes += raw_line_bytes
                expected_span_count += _count_otlp_spans(body)

                if len(batch.bodies) >= batch_lines or batch.input_bytes >= batch_bytes:
                    batch_error = _flush_batch(
                        endpoint,
                        batch,
                        max_retries=max_retries,
                        file_name=file.name,
                    )
                    if batch_error:
                        file_errors.append(batch_error)
                        break

            else:
                batch_error = _flush_batch(
                    endpoint,
                    batch,
                    max_retries=max_retries,
                    file_name=file.name,
                )
                if batch_error:
                    file_errors.append(batch_error)

        # A 200 from /v1/traces only means queued. Wait for durable processing
        # before importing annotations or reporting success.
        if batch.attempted:
            try:
                sync_ingest(endpoint)
            except OtlpRequestError as error:
                file_errors.append(f"{file.name}: failed to sync viewer ingest: {error}")

        if not file_errors and batch.accepted:
            try:
                stored_span_count = get_session_span_count(endpoint, session_id)
            except OtlpRequestError as error:
                file_errors.append(f"{file.name}: failed to verify viewer ingest: {error}")
            else:
                if stored_span_count != expected_span_count:
                    file_errors.append(
                        f"{file.name}: viewer stored {stored_span_count}/{expected_span_count} spans"
                    )
                else:
                    file_imported = True

        if not file_errors and deferred_annotations:
            count = post_annotations(endpoint, deferred_annotations)
            annotations_imported += count
            if count != len(deferred_annotations):
                file_errors.append(
                    f"{file.name}: imported {count}/{len(deferred_annotations)} annotations"
                )
            else:
                file_imported = True

        if file_errors:
            failed += 1
            errors.extend(file_errors)
        elif file_imported:
            imported += 1
        elif not is_legacy:
            skipped += 1

    click.echo(f"  {imported} imported, {skipped} skipped")
    if failed:
        click.echo(f"  {failed} failed")
    if already_exist:
        click.echo(f"  {already_exist} skipped (already exist)")
    if annotations_imported:
        click.echo(f"  {annotations_imported} annotation(s) imported")
    if errors:
        for err in errors[:10]:
            click.echo(f"  ! {err}")
        if len(errors) > 10:
            click.echo(f"  ... and {len(errors) - 10} more errors")

    encoded_batch = urllib.parse.quote(batch_id, safe="")
    if errors:
        click.echo(
            f"\nImport incomplete. Partial data may exist in batch '{batch_id}'.\n"
            f"Delete it before retrying:\n"
            f"  nooa delete-traces --batch-id {shlex.quote(batch_id)} "
            f"--endpoint {shlex.quote(endpoint)}"
        )
        raise SystemExit(1)

    click.echo(f"\nView at: {endpoint}/traces?batch_id={encoded_batch}")
