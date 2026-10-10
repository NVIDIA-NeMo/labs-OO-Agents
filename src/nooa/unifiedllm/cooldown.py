# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read structured overload feedback without inspecting provider error messages."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Any

_OVERLOAD_STATUSES = frozenset({429, 503, 529})
_MAX_ERROR_CHAIN = 8
_DELTA_SECONDS = re.compile(r"[0-9]+\Z")
_DAY = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_CLOCK = r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
_HTTP_DATE = re.compile(
    rf"(?:{_DAY}, [0-9]{{2}} {_MONTH} [0-9]{{4}} {_CLOCK} GMT"
    rf"|(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), "
    rf"[0-9]{{2}}-{_MONTH}-[0-9]{{2}} {_CLOCK} GMT"
    rf"|{_DAY} {_MONTH} (?: [0-9]|[0-9]{{2}}) {_CLOCK} [0-9]{{4}})\Z"
)


def _retry_after_header(headers: Any) -> tuple[bool, str | None]:
    """Read a case-insensitive Retry-After field from a header mapping."""
    if not isinstance(headers, Mapping):
        return False, None
    for name, value in headers.items():
        if isinstance(name, str) and name.lower() == "retry-after":
            return True, value if isinstance(value, str) else None
    return False, None


def _structured_status(error: BaseException) -> int | None:
    """Read an HTTP status without inferring it from exception text."""
    status = getattr(error, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool):
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _error_retry_after(error: Exception) -> str | None:
    """Follow bounded SDK wrapper chains to the nearest overload Retry-After."""
    if _structured_status(error) not in _OVERLOAD_STATUSES:
        return None
    seen: set[int] = set()
    candidate: BaseException | None = error
    for _ in range(_MAX_ERROR_CHAIN):
        if candidate is None or id(candidate) in seen:
            return None
        seen.add(id(candidate))
        status = _structured_status(candidate)
        if status is not None:
            if status not in _OVERLOAD_STATUSES:
                return None
            present, value = _retry_after_header(
                getattr(getattr(candidate, "response", None), "headers", None)
            )
            if not present:
                present, value = _retry_after_header(getattr(candidate, "headers", None))
            if present:
                return value
        if candidate.__cause__ is not None:
            candidate = candidate.__cause__
        elif not candidate.__suppress_context__:
            candidate = candidate.__context__
        else:
            return None
    return None


def retry_after_delay(error: Exception) -> float | None:
    """Return a positive Retry-After delay from a structured overload error.

    Only HTTP 429, 503, and 529 are recognized. Headers may be attached to the
    error's response or directly to the error. SDK causes and unsuppressed
    contexts are inspected up to eight exceptions, with cycle protection.
    Explicit causes take precedence, and the nearest Retry-After is
    authoritative even when malformed. An outer non-overload error or a
    nested explicit non-overload status cannot inherit stale overload headers.
    RFC delta-seconds are unsigned ASCII integers; HTTP dates are converted
    relative to the current wall clock. Malformed, nonpositive, expired, and
    nonfinite values are ignored. This helper never reads the exception message
    or other header values. The controller, rather than the parser, caps the
    returned delay.
    """
    value = _error_retry_after(error)
    if value is None:
        return None
    value = value.strip()

    if _DELTA_SECONDS.fullmatch(value):
        try:
            delay = float(value)
        except (OverflowError, ValueError):
            return None
    else:
        if not _HTTP_DATE.fullmatch(value):
            return None
        try:
            date = parsedate_to_datetime(value)
            # The obsolete asctime HTTP-date format has no timezone; HTTP
            # dates nevertheless always describe a UTC time.
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            delay = date.timestamp() - time.time()
        except (OverflowError, OSError, TypeError, ValueError):
            return None

    return delay if delay > 0 and math.isfinite(delay) else None


__all__ = ["retry_after_delay"]
