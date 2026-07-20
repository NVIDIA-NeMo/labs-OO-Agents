# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for OTLP trace import commands."""

import json
import urllib.parse
import urllib.request

import click


def validate_endpoint(endpoint: str) -> None:
    """Validate that the endpoint uses an HTTP(S) scheme."""
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in ("http", "https"):
        raise click.BadParameter(
            f"Endpoint must use http:// or https:// scheme, got: {endpoint}",
            param_hint="'--endpoint'",
        )


def inject_resource_attrs(body: dict, attrs: dict[str, str | bool | int]) -> dict:
    """Inject additional resource attributes into an OTLP body (skips existing keys).

    Values are typed: str → stringValue, bool → boolValue, int → intValue.
    """
    for rs in body.get("resourceSpans", []):
        resource = rs.setdefault("resource", {})
        existing = resource.setdefault("attributes", [])
        existing_keys = {a["key"] for a in existing}
        for key, val in attrs.items():
            if key not in existing_keys:
                if isinstance(val, bool):
                    otlp_val = {"boolValue": val}
                elif isinstance(val, int):
                    otlp_val = {"intValue": val}
                else:
                    otlp_val = {"stringValue": val}
                existing.append({"key": key, "value": otlp_val})
    return body


def post_trace(endpoint: str, body: dict, timeout: float = 30) -> bool:
    """POST a single OTLP trace body to the viewer endpoint.

    ``timeout`` defaults to 30s so large batched bodies (see ``post_traces_batch``)
    don't spuriously time out; per-line callers are unaffected.
    """
    url = f"{endpoint.rstrip('/')}/v1/traces"
    data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 300
    except Exception:
        return False


def post_traces_batch(endpoint: str, bodies: list[dict]) -> bool:
    """POST multiple OTLP bodies as one request by merging their ``resourceSpans``.

    Combines the ``resourceSpans`` arrays of every body into a single OTLP envelope
    and posts it once, drastically reducing HTTP round-trips for large imports.
    Returns True (a no-op success) when there are no spans to send.
    """
    merged: dict = {"resourceSpans": []}
    for body in bodies:
        # Guard against malformed bodies whose ``resourceSpans`` is missing or
        # not a list (e.g. None or a dict); skip rather than raise on extend.
        spans = body.get("resourceSpans")
        if isinstance(spans, list):
            merged["resourceSpans"].extend(spans)
    if not merged["resourceSpans"]:
        return True
    return post_trace(endpoint, merged)


def post_annotations(endpoint: str, annotations: list[dict]) -> int:
    """POST annotations to the viewer endpoint. Returns count of successfully imported."""
    url = f"{endpoint.rstrip('/')}/api/annotations"
    imported = 0
    for ann in annotations:
        ann = {k: v for k, v in ann.items() if k != "id"}
        data = json.dumps(ann, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status < 300:
                    imported += 1
        except Exception:
            pass
    return imported


def session_exists(endpoint: str, session_id: str) -> bool:
    """Check whether a session already exists in the viewer."""
    url = f"{endpoint.rstrip('/')}/api/trace-count?session_id={urllib.parse.quote(session_id)}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_endpoint_reachable(endpoint: str) -> bool:
    """Return True if the viewer API is reachable."""
    try:
        with urllib.request.urlopen(f"{endpoint.rstrip('/')}/api/version", timeout=5):
            return True
    except Exception:
        return False
