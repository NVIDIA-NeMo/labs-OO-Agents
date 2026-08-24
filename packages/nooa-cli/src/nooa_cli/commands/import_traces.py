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
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import click

from ._otlp_helpers import (
    OtlpRequestError,
    check_endpoint_reachable,
    get_journal_record,
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

    # Verify endpoint is reachable
    if not check_endpoint_reachable(endpoint):
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
        trace_post_attempted = False
        trace_batch_accepted = False
        is_legacy = False
        deferred_annotations: list[dict] = []
        batch: list[dict] = []
        batch_input_bytes = 0
        batch_first_line = 0
        batch_last_line = 0

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

                inject_resource_attrs(body, inject_attrs)
                if not batch:
                    batch_first_line = line_num
                batch_last_line = line_num
                batch.append(body)
                batch_input_bytes += len(raw_line.encode("utf-8"))

                if len(batch) >= batch_lines or batch_input_bytes >= batch_bytes:
                    trace_post_attempted = True
                    batch_error = _post_batch(
                        endpoint,
                        batch,
                        max_retries=max_retries,
                        file_name=file.name,
                        first_line=batch_first_line,
                        last_line=batch_last_line,
                    )
                    batch = []
                    batch_input_bytes = 0
                    if batch_error:
                        file_errors.append(batch_error)
                        break
                    trace_batch_accepted = True

            else:
                if batch:
                    trace_post_attempted = True
                    batch_error = _post_batch(
                        endpoint,
                        batch,
                        max_retries=max_retries,
                        file_name=file.name,
                        first_line=batch_first_line,
                        last_line=batch_last_line,
                    )
                    if batch_error:
                        file_errors.append(batch_error)
                    else:
                        trace_batch_accepted = True

        # A 200 from /v1/traces only means queued. Wait for durable processing
        # before importing annotations or reporting success.
        if trace_post_attempted:
            try:
                sync_ingest(endpoint)
            except OtlpRequestError as error:
                file_errors.append(f"{file.name}: failed to sync viewer ingest: {error}")

        if not file_errors and trace_batch_accepted:
            if deferred_annotations:
                count = post_annotations(endpoint, deferred_annotations)
                annotations_imported += count
                if count != len(deferred_annotations):
                    file_errors.append(
                        f"{file.name}: imported {count}/{len(deferred_annotations)} annotations"
                    )

        if file_errors:
            failed += 1
            errors.extend(file_errors)
        elif trace_batch_accepted:
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
