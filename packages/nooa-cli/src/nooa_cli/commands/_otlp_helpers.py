# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for OTLP trace import commands."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import click

JOURNAL_ENVELOPE_KEY = "nooaJournal"
JOURNAL_FORMAT = "nooa.message_journal"
JOURNAL_VERSION = 1
RETRYABLE_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class OtlpRequestError(RuntimeError):
    """An OTLP viewer request failed with details suitable for CLI output."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after


def validate_endpoint(endpoint: str) -> None:
    """Validate that the endpoint uses an HTTP(S) scheme."""
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in ("http", "https"):
        raise click.BadParameter(
            f"Endpoint must use http:// or https:// scheme, got: {endpoint}",
            param_hint="'--endpoint'",
        )


def _viewer_headers(headers: dict[str, str]) -> dict[str, str]:
    """Apply viewer authentication without slowing unrelated CLI commands."""
    from nooa.tracing._viewer_auth import apply_viewer_auth

    return apply_viewer_auth(headers)


def inject_resource_attrs(
    body: dict,
    attrs: dict[str, str | bool | int | float],
    *,
    overwrite: bool = False,
    overwrite_keys: set[str] | None = None,
) -> dict:
    """Inject additional resource attributes into an OTLP body.

    Existing keys are preserved unless ``overwrite`` is true or the key is in
    ``overwrite_keys``. Values are typed: str → stringValue, bool →
    boolValue, int → intValue, float → doubleValue.
    """
    overwrite_keys = overwrite_keys or set()

    def otlp_value(val: str | bool | int | float) -> dict:
        if isinstance(val, bool):
            return {"boolValue": val}
        if isinstance(val, int):
            # OTLP JSON encodes int64 values as decimal strings so values above
            # JavaScript's safe integer limit retain their exact value.
            return {"intValue": str(val)}
        if isinstance(val, float):
            return {"doubleValue": val}
        return {"stringValue": val}

    resource_spans = body.get("resourceSpans")
    if not isinstance(resource_spans, list):
        return body

    for rs in resource_spans:
        if not isinstance(rs, dict):
            continue
        resource = rs.setdefault("resource", {})
        if not isinstance(resource, dict):
            continue
        existing = resource.setdefault("attributes", [])
        if not isinstance(existing, list):
            continue
        existing_by_key = {
            attribute["key"]: attribute
            for attribute in existing
            if isinstance(attribute, dict) and isinstance(attribute.get("key"), str)
        }
        for key, val in attrs.items():
            if key in existing_by_key:
                if overwrite or key in overwrite_keys:
                    existing_by_key[key]["value"] = otlp_value(val)
            else:
                existing.append({"key": key, "value": otlp_value(val)})
    return body


def _http_error_body(error: urllib.error.HTTPError) -> str:
    """Read a bounded, printable HTTP error response body."""
    try:
        body = error.read(2048).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    return body


def _retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    """Return a numeric Retry-After value when the server supplied one."""
    value = error.headers.get("Retry-After") if error.headers else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _post_trace_checked(endpoint: str, body: dict, timeout: float = 30) -> None:
    """POST one OTLP body, raising a detailed error on failure."""

    url = f"{endpoint.rstrip('/')}/v1/traces"
    data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=_viewer_headers({"Content-Type": "application/json"}),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 300:
                raise OtlpRequestError(
                    f"HTTP {resp.status} from {url}",
                    status_code=resp.status,
                    retryable=resp.status in RETRYABLE_HTTP_STATUS_CODES,
                )
    except urllib.error.HTTPError as error:
        response_body = _http_error_body(error)
        detail = f": {response_body}" if response_body else ""
        raise OtlpRequestError(
            f"HTTP {error.code} {error.reason} from {url}{detail}",
            status_code=error.code,
            retryable=error.code in RETRYABLE_HTTP_STATUS_CODES,
            retry_after=_retry_after_seconds(error),
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise OtlpRequestError(
            f"request to {url} failed: {reason}",
        ) from error


def post_trace(endpoint: str, body: dict, timeout: float = 30) -> bool:
    """POST a single OTLP body, preserving the legacy boolean API."""
    try:
        _post_trace_checked(endpoint, body, timeout=timeout)
    except OtlpRequestError:
        return False
    return True


def post_trace_with_retry(
    endpoint: str,
    body: dict,
    *,
    timeout: float = 30,
    max_retries: int = 5,
    initial_backoff: float = 0.25,
    max_backoff: float = 5.0,
) -> None:
    """POST an OTLP body, retrying transient HTTP failures.

    Raises :class:`OtlpRequestError` with the HTTP status and response body when
    all attempts fail. ``max_retries`` counts retries after the initial request.
    Transport failures are not replayed because the server may already have
    accepted the request, and OTLP ingest is not idempotent.
    """
    for retry_index in range(max_retries + 1):
        try:
            _post_trace_checked(endpoint, body, timeout=timeout)
            return
        except OtlpRequestError as error:
            if not error.retryable or retry_index >= max_retries:
                attempts = retry_index + 1
                raise OtlpRequestError(
                    f"{error} (after {attempts} attempt{'s' if attempts != 1 else ''})",
                    status_code=error.status_code,
                    retryable=error.retryable,
                    retry_after=error.retry_after,
                ) from error
            delay = (
                min(error.retry_after, max_backoff)
                if error.retry_after is not None
                else min(initial_backoff * (2**retry_index), max_backoff)
            )
            time.sleep(delay)


def _merge_resource_spans(bodies: list[dict]) -> dict:
    """Merge valid ``resourceSpans`` arrays into one OTLP envelope."""
    merged: dict = {"resourceSpans": []}
    for body in bodies:
        spans = body.get("resourceSpans")
        if isinstance(spans, list):
            merged["resourceSpans"].extend(spans)
    return merged


def post_traces_batch(endpoint: str, bodies: list[dict]) -> bool:
    """POST multiple OTLP bodies as one request by merging their ``resourceSpans``.

    Combines the ``resourceSpans`` arrays of every body into a single OTLP envelope
    and posts it once, drastically reducing HTTP round-trips for large imports.
    Returns True (a no-op success) when there are no spans to send.
    """
    merged = _merge_resource_spans(bodies)
    if not merged["resourceSpans"]:
        return True
    return post_trace(endpoint, merged)


def post_traces_batch_with_retry(
    endpoint: str,
    bodies: list[dict],
    *,
    max_retries: int = 5,
) -> None:
    """Merge and reliably POST a bounded batch of OTLP envelopes."""
    merged = _merge_resource_spans(bodies)
    if not merged["resourceSpans"]:
        return
    post_trace_with_retry(endpoint, merged, max_retries=max_retries)


def sync_ingest(endpoint: str, timeout: float = 35) -> None:
    """Wait until the viewer has processed every accepted OTLP request."""
    url = f"{endpoint.rstrip('/')}/v1/sync"
    request = urllib.request.Request(url, headers=_viewer_headers({}), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status >= 300:
                raise OtlpRequestError(
                    f"HTTP {response.status} from {url}",
                    status_code=response.status,
                    retryable=response.status in RETRYABLE_HTTP_STATUS_CODES,
                )
    except urllib.error.HTTPError as error:
        response_body = _http_error_body(error)
        detail = f": {response_body}" if response_body else ""
        raise OtlpRequestError(
            f"HTTP {error.code} {error.reason} from {url}{detail}",
            status_code=error.code,
            retryable=error.code in RETRYABLE_HTTP_STATUS_CODES,
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise OtlpRequestError(f"request to {url} failed: {reason}") from error


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
            headers=_viewer_headers({"Content-Type": "application/json"}),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status < 300:
                    imported += 1
        except Exception:
            pass
    return imported


def get_journal_record(body: dict) -> dict | None:
    """Return a validated NOOA journal envelope, or ``None`` for other lines."""
    record = body.get(JOURNAL_ENVELOPE_KEY)
    if not isinstance(record, dict):
        return None
    if record.get("format") != JOURNAL_FORMAT or record.get("version") != JOURNAL_VERSION:
        return None
    return record


def post_journal_record(endpoint: str, record: dict, session_id: str) -> bool:
    """POST one portable-file journal record to the viewer.

    ``session_id`` is authoritative so renaming a trace file and Harbor's
    trial-name remapping affect OTLP spans and their journal sideband equally.
    Manifest records are accepted as no-ops.
    """
    record_type = record.get("type")
    if record_type == "manifest":
        return True
    if record_type == "blocks":
        payload = record.get("blocks")
        if not isinstance(payload, list):
            return False
        path = "/v1/journal/blocks"
        headers = {"Content-Type": "application/json", "X-Session-Id": session_id}
    elif record_type == "call":
        source = record.get("call")
        if not isinstance(source, dict):
            return False
        payload = dict(source)
        payload["session_id"] = session_id
        path = "/v1/journal/calls"
        headers = {"Content-Type": "application/json"}
    else:
        return False

    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}{path}",
        data=data,
        headers=_viewer_headers(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status < 300
    except Exception:
        return False


def session_exists(endpoint: str, session_id: str) -> bool:
    """Check whether a session already exists in the viewer."""
    url = f"{endpoint.rstrip('/')}/api/trace-count?session_id={urllib.parse.quote(session_id)}"
    req = urllib.request.Request(url, headers=_viewer_headers({}), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_endpoint_reachable(endpoint: str) -> bool:
    """Return true when reachable, raising detailed HTTP response failures."""
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/version",
        headers=_viewer_headers({}),
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5):
            return True
    except urllib.error.HTTPError as error:
        response_body = _http_error_body(error)
        detail = f": {response_body}" if response_body else ""
        raise OtlpRequestError(
            f"HTTP {error.code} {error.reason} from {request.full_url}{detail}",
            status_code=error.code,
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def get_session_span_count(endpoint: str, session_id: str) -> int:
    """Return the viewer's stored span count for one session."""
    url = f"{endpoint.rstrip('/')}/api/trace-count?session_id={urllib.parse.quote(session_id)}"
    request = urllib.request.Request(url, headers=_viewer_headers({}), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        response_body = _http_error_body(error)
        detail = f": {response_body}" if response_body else ""
        raise OtlpRequestError(
            f"HTTP {error.code} {error.reason} from {url}{detail}",
            status_code=error.code,
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise OtlpRequestError(f"request to {url} failed: {reason}") from error
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise OtlpRequestError(f"invalid trace-count response from {url}: {error}") from error

    count = payload.get("event_count") if isinstance(payload, dict) else None
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise OtlpRequestError(f"invalid trace-count response from {url}: {payload!r}")
    return count
