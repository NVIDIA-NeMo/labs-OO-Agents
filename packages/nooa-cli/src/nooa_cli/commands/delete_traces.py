# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delete imported trace batches from the viewer.

Usage:
    nooa delete-traces --batch-id my-experiment-v2
    nooa delete-traces --batch-id my-experiment-v2 --endpoint http://host:5001
"""

import json
import urllib.parse
import urllib.request

import click

from nooa.tracing._viewer_auth import apply_viewer_auth

NAME = "delete-traces"


def _validate_endpoint(endpoint: str) -> None:
    """Validate that the endpoint uses an HTTP(S) scheme."""
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in ("http", "https"):
        raise click.BadParameter(
            f"Endpoint must use http:// or https:// scheme, got: {endpoint}",
            param_hint="'--endpoint'",
        )


@click.command()
@click.option(
    "--batch-id",
    required=True,
    help="Batch ID to delete.",
)
@click.option(
    "--endpoint",
    default="http://localhost:5001",
    show_default=True,
    help="Viewer API endpoint.",
)
def command(batch_id: str, endpoint: str):
    """Delete all traces belonging to a batch."""
    _validate_endpoint(endpoint)

    # Verify endpoint is reachable
    try:
        request = urllib.request.Request(
            f"{endpoint.rstrip('/')}/api/version",
            headers=apply_viewer_auth({}),
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=5):
            pass
    except Exception:
        click.echo(f"Cannot reach viewer at {endpoint}. Is it running?")
        raise SystemExit(1) from None

    url = f"{endpoint.rstrip('/')}/api/traces?batch_id={urllib.parse.quote(batch_id, safe='')}"
    req = urllib.request.Request(url, headers=apply_viewer_auth({}), method="DELETE")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            count = result.get("deleted", 0)
            click.echo(f"Deleted {count} trace(s) from batch '{batch_id}'")
    except Exception as e:
        click.echo(f"Failed to delete batch '{batch_id}': {e}")
        raise SystemExit(1) from None
