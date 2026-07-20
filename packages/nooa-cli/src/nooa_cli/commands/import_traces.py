# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import OTLP trace .jsonl files into the viewer.

Usage:
    nooa import-traces ./traces/
    nooa import-traces my_trace.jsonl --endpoint http://host:5001
    nooa import-traces ./experiment/ --batch-id my-experiment-v2
"""

import json
import urllib.parse
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import click

from ._otlp_helpers import (
    check_endpoint_reachable,
    inject_resource_attrs,
    post_annotations,
    post_trace,
    session_exists,
    validate_endpoint,
)

NAME = "import-traces"

TRACE_EXTENSIONS = {".jsonl"}


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
def command(path: str, endpoint: str, batch_id: str | None):
    """Import OTLP trace .jsonl files into the viewer."""
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
    already_exist = 0
    annotations_imported = 0
    errors = []

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

        inject_attrs = {"batch_id": batch_id}

        file_has_session = False
        file_imported = False
        is_legacy = False
        deferred_annotations: list[dict] = []

        with open(file) as f:
            for line_num, raw_line in enumerate(f, 1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                try:
                    body = json.loads(raw_line)
                except json.JSONDecodeError:
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
                        errors.append(f"{file.name}: legacy format not supported, skipping")
                        is_legacy = True
                    continue

                if fmt != "otlp":
                    continue

                if not file_has_session:
                    inject_attrs["session.id"] = session_id
                    file_has_session = True

                inject_resource_attrs(body, inject_attrs)

                if post_trace(endpoint, body):
                    file_imported = True
                else:
                    errors.append(f"{file.name}:{line_num}: failed to post")

        if file_imported:
            imported += 1
            # Import annotations after spans so the session exists
            if deferred_annotations:
                count = post_annotations(endpoint, deferred_annotations)
                annotations_imported += count
        elif not is_legacy:
            skipped += 1

    click.echo(f"  {imported} imported, {skipped} skipped")
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
    click.echo(f"\nView at: {endpoint}/traces?batch_id={encoded_batch}")
