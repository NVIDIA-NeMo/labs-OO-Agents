# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured Retry-After parsing for opt-in shared admission cooldowns."""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import litellm
import pytest
from litellm.llms.openai.common_utils import OpenAIError

from nooa.unifiedllm import cooldown
from nooa.unifiedllm.cooldown import retry_after_delay


def _error(status: int = 429, headers: object | None = None) -> Exception:
    """Construct the structured status/header shape used by provider adapters."""
    error = Exception("provider error")
    error.status_code = status  # type: ignore[attr-defined]
    error.headers = headers  # type: ignore[attr-defined]
    return error


@pytest.mark.parametrize("status", [429, 503, 529])
def test_real_httpx_error_reads_response_status_and_headers(status: int) -> None:
    """Support response-only status codes and case-insensitive HTTPX headers."""
    response = httpx.Response(
        status,
        request=httpx.Request("GET", "https://example.test"),
        headers={"Retry-After": "2", "Authorization": "private"},
    )
    with pytest.raises(httpx.HTTPStatusError) as caught:
        response.raise_for_status()
    assert retry_after_delay(caught.value) == 2.0


def test_real_litellm_rate_limit_error_preserves_response_guidance() -> None:
    """Read Retry-After through the SDK's actual structured overload wrapper."""
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
        headers={"Retry-After": "3"},
    )
    error = litellm.RateLimitError(
        message="overloaded", llm_provider="openai", model="test-model", response=response
    )
    assert retry_after_delay(error) == 3.0


@pytest.mark.parametrize("name", ["Retry-After", "retry-after", "RETRY-AFTER", "ReTrY-aFtEr"])
def test_adapter_error_reads_case_insensitive_direct_headers(name: str) -> None:
    """Accept LiteLLM-style structured attributes without requiring its SDK."""
    assert retry_after_delay(_error(headers={name: " 3 "})) == 3.0


@pytest.mark.parametrize("value", ["0", "-1", "+1", "1.5", "1e2", "NaN", "inf", "", " ", "١"])
def test_invalid_delta_seconds_are_ignored(value: str) -> None:
    """Reject values that are not positive RFC unsigned ASCII delta-seconds."""
    assert retry_after_delay(_error(headers={"Retry-After": value})) is None


@pytest.mark.parametrize("value", ["9" * 400, "9" * 5000])
def test_huge_delta_seconds_are_ignored(value: str) -> None:
    """An unrepresentable server delay cannot overflow into the call path."""
    assert retry_after_delay(_error(headers={"Retry-After": value})) is None


@pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 408, 500, 502, 504])
def test_other_statuses_do_not_start_cooldown(status: int) -> None:
    """A Retry-After field on another status is outside this overload policy."""
    assert retry_after_delay(_error(status, {"Retry-After": "2"})) is None


@pytest.mark.parametrize("status", [None, True, False, "429", 429.0])
def test_missing_or_noninteger_status_is_ignored(status: object) -> None:
    """Do not infer an overload code from strings, booleans, or error text."""
    error = _error(headers={"Retry-After": "2"})
    error.status_code = status  # type: ignore[attr-defined]
    assert retry_after_delay(error) is None


def test_http_date_uses_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate the server's absolute UTC date into a finite relative delay."""
    now = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC).timestamp()
    monkeypatch.setattr(cooldown.time, "time", lambda: now)
    value = format_datetime(datetime.fromtimestamp(now + 17, UTC), usegmt=True)
    assert retry_after_delay(_error(headers={"Retry-After": value})) == 17.0


@pytest.mark.parametrize("offset", [-30, 0])
def test_expired_http_date_is_ignored(offset: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """An expired or current-date Retry-After cannot delay future attempts."""
    now = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC).timestamp()
    monkeypatch.setattr(cooldown.time, "time", lambda: now)
    value = format_datetime(datetime.fromtimestamp(now + offset, UTC), usegmt=True)
    assert retry_after_delay(_error(headers={"Retry-After": value})) is None


@pytest.mark.parametrize(
    "value",
    [
        "Sunday, 04-Oct-26 12:00:17 GMT",
        "Sun Oct  4 12:00:17 2026",
    ],
)
def test_obsolete_http_dates_are_interpreted_as_utc(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recognize the obsolete HTTP-date formats still accepted by HTTP clients."""
    now = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC).timestamp()
    monkeypatch.setattr(cooldown.time, "time", lambda: now)
    assert retry_after_delay(_error(headers={"Retry-After": value})) == 17.0


@pytest.mark.parametrize(
    "value",
    [
        "tomorrow",
        "Sun, 99 Oct 2026 12:00:00 GMT",
        "Sun, 04 Oct 99999 12:00:00 GMT",
        "Sun, 04 Oct 2026 12:00:00",
        "Sun, 04 Oct 2026 12:00:00 GMT trailing",
        "04 Oct 2026 12:00:00 GMT",
        "Sun, 04 Oct 2026 12:00:00 +0000",
    ],
)
def test_malformed_http_date_is_ignored(value: str) -> None:
    """Reject unparseable or out-of-range server dates without propagating errors."""
    assert retry_after_delay(_error(headers={"Retry-After": value})) is None


@pytest.mark.parametrize("headers", [None, [], "Retry-After: 2", {"Retry-After": 2}])
def test_missing_or_malformed_headers_are_ignored(headers: object) -> None:
    """Only a string field in a header mapping supplies cooldown information."""
    assert retry_after_delay(_error(headers=headers)) is None


def test_response_header_precedes_direct_header() -> None:
    """The provider response is authoritative when both header shapes exist."""
    error = _error(headers={"Retry-After": "9"})
    error.response = SimpleNamespace(headers={"Retry-After": "2"})  # type: ignore[attr-defined]
    assert retry_after_delay(error) == 2.0


def test_direct_header_fallback_when_response_has_no_retry_after() -> None:
    """Support adapter-level Retry-After metadata when the response omits it."""
    error = _error(headers={"Retry-After": "2"})
    error.response = SimpleNamespace(headers={"other": "field"})  # type: ignore[attr-defined]
    assert retry_after_delay(error) == 2.0


def test_structured_error_status_precedes_response_status() -> None:
    """A contradictory response must not override a structured non-overload code."""
    error = _error(401, {"Retry-After": "2"})
    error.response = SimpleNamespace(status_code=429, headers={})  # type: ignore[attr-defined]
    assert retry_after_delay(error) is None


def test_error_message_and_unrelated_header_values_are_not_inspected() -> None:
    """Keep secrets out of the parser and avoid error-message heuristics."""

    class SecretError(Exception):
        """Fail if parsing attempts to stringify private exception text."""

        status_code = 429
        headers = {"Authorization": object(), "Retry-After": "2"}

        def __str__(self) -> str:
            """Make any accidental read of the exception message observable."""
            raise AssertionError("Do not inspect the exception message")

    assert retry_after_delay(SecretError()) == 2.0


def test_large_finite_delay_is_left_for_controller_to_cap() -> None:
    """Parsing returns server guidance; policy applies the configured maximum."""
    assert retry_after_delay(_error(headers={"Retry-After": "1000000"})) == 1_000_000.0


def test_litellm_wrapper_recovers_structured_adapter_context_headers() -> None:
    """LiteLLM can discard headers on its outer error while retaining its context."""
    adapter = OpenAIError(status_code=429, message="overloaded", headers={"Retry-After": "3"})
    outer = litellm.RateLimitError(message="overloaded", llm_provider="openai", model="test-model")
    outer.__context__ = adapter
    assert not outer.response.headers.get("Retry-After")
    assert retry_after_delay(outer) == 3.0


def test_explicit_cause_precedes_implicit_context() -> None:
    """Honor Python exception precedence instead of mixing two error histories."""
    outer = _error()
    outer.__context__ = _error(headers={"Retry-After": "2"})
    outer.__cause__ = _error(503, {"Retry-After": "5"})
    assert retry_after_delay(outer) == 5.0


def test_suppressed_context_is_not_inspected() -> None:
    """An explicit suppression cannot expose an unrelated hidden error's guidance."""
    outer = _error()
    outer.__context__ = _error(headers={"Retry-After": "2"})
    outer.__suppress_context__ = True
    assert retry_after_delay(outer) is None


def test_explicit_cause_without_headers_does_not_fall_back_to_outer_context() -> None:
    """A preferred cause selects one chain even when its nodes contain no header."""
    outer = _error()
    outer.__context__ = _error(headers={"Retry-After": "2"})
    outer.__cause__ = _error()
    assert retry_after_delay(outer) is None


@pytest.mark.parametrize("headers", [{"Retry-After": "bad"}, {"Retry-After": 2}])
def test_nearest_present_header_remains_authoritative(headers: object) -> None:
    """Malformed wrapper guidance must not inherit a different underlying delay."""
    outer = _error(headers=headers)
    outer.__context__ = _error(headers={"Retry-After": "2"})
    assert retry_after_delay(outer) is None


def test_malformed_response_header_does_not_fall_back_to_direct_header() -> None:
    """A malformed field is present, unlike an absent response field."""
    error = _error(headers={"Retry-After": "2"})
    error.response = SimpleNamespace(headers={"Retry-After": object()})  # type: ignore[attr-defined]
    assert retry_after_delay(error) is None


@pytest.mark.parametrize("status", [400, 401, 404, 500])
def test_outer_nonoverload_error_cannot_inherit_context_overload(status: int) -> None:
    """Reject stale rate-limit context when the actual failure has another status."""
    outer = _error(status)
    outer.__context__ = _error(headers={"Retry-After": "2"})
    assert retry_after_delay(outer) is None


def test_nested_explicit_nonoverload_status_stops_chain() -> None:
    """A contradictory inner status separates this failure from older overloads."""
    outer = _error()
    middle = _error(401)
    middle.__context__ = _error(headers={"Retry-After": "2"})
    outer.__context__ = middle
    assert retry_after_delay(outer) is None


def test_headerless_wrapper_without_status_can_be_traversed() -> None:
    """Neutral wrappers may connect two structured overload exceptions."""
    outer = _error()
    middle = Exception("neutral wrapper")
    middle.__context__ = _error(headers={"Retry-After": "2"})
    outer.__context__ = middle
    assert retry_after_delay(outer) == 2.0


def test_nested_header_without_status_is_not_used() -> None:
    """Every exception supplying guidance must itself have an overload status."""
    outer = _error()
    middle = Exception("neutral wrapper")
    middle.headers = {"Retry-After": "2"}  # type: ignore[attr-defined]
    outer.__context__ = middle
    assert retry_after_delay(outer) is None


@pytest.mark.parametrize("attribute", ["__cause__", "__context__"])
def test_exception_chain_cycles_terminate(attribute: str) -> None:
    """Malformed adapter chains cannot make feedback parsing loop forever."""
    first, second = _error(), _error()
    setattr(first, attribute, second)
    setattr(second, attribute, first)
    assert retry_after_delay(first) is None


@pytest.mark.parametrize("length, expected", [(8, 2.0), (9, None)])
def test_exception_chain_depth_is_bounded(length: int, expected: float | None) -> None:
    """Inspect at most eight exceptions, including the original provider failure."""
    chain = [_error() for _ in range(length)]
    for outer, inner in zip(chain, chain[1:], strict=False):
        outer.__cause__ = inner
    chain[-1].headers = {"Retry-After": "2"}  # type: ignore[attr-defined]
    assert retry_after_delay(chain[0]) == expected
