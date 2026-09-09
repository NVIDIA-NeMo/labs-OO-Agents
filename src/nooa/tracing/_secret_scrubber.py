# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secret scrubbing for telemetry spans.

Intercepts OpenTelemetry spans before export and redacts any detected
secrets (API keys, tokens, private keys) from string attributes.

Uses high-precision regex patterns (not entropy-based) to minimize
false positives on code and log output.

Usage::

    from nooa.tracing._secret_scrubber import SecretScrubSpanProcessor

    provider.add_span_processor(SecretScrubSpanProcessor(inner_processor))
"""

import json
import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "api_key",
        "x_api_key",
        "access_token",
        "refresh_token",
        "auth_token",
        "session_token",
        "id_token",
        "client_secret",
        "client_assertion",
        "password",
        "passwd",
        "private_key",
        "secret_key",
        "code_verifier",
        "cookie",
        "set_cookie",
    }
)

# Provider-owned replay state is not a user credential, but it has the same
# telemetry rule: it may go back to its issuer and nowhere else. Provider
# adapters add their exact wire keys here as support is introduced.
_OPAQUE_PROVIDER_STATE_KEYS = frozenset(
    {
        "encrypted_content",
        "nooa_llm_state",
        "thought_signature",
        "thought_signatures",
        "thoughtsignature",
        "thoughtsignatures",
    }
)


def _is_sensitive_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
    return normalized in _SENSITIVE_KEYS or any(
        normalized.endswith(f"_{sensitive}") for sensitive in _SENSITIVE_KEYS
    )


def _is_opaque_provider_state_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
    return normalized in _OPAQUE_PROVIDER_STATE_KEYS


def _redact_key(key: Any) -> str | None:
    if _is_sensitive_key(key):
        return "sensitive_key"
    if _is_opaque_provider_state_key(key):
        return "opaque_provider_state"
    return None


# ---------------------------------------------------------------------------
# Regex-based secret patterns — high precision, low false positives
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # LiteLLM may append Gemini thought signatures to tool-call IDs.
    (
        "gemini_inline_thought_signature",
        re.compile(r"__thought__(?P<secret>[A-Za-z0-9+/=_-]+)"),
    ),
    # AWS Access Key IDs (AKIA, ASIA, AIDA, AROA + 16 alphanumeric)
    ("aws_access_key", re.compile(r"(?P<secret>(?:AKIA|ASIA|AIDA|AROA)[A-Z0-9]{16})")),
    # AWS Secret Access Keys (40-char base64 after known prefix)
    (
        "aws_secret_key",
        re.compile(
            r"(?:aws_secret_access_key|secret.?key)[\"\']?\s*[=:]\s*[\"\'\s]*(?P<secret>[A-Za-z0-9/+=]{40})",
            re.IGNORECASE,
        ),
    ),
    # GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_)
    ("github_token", re.compile(r"(?P<secret>gh[pousr]_[A-Za-z0-9_]{36,})")),
    # GitLab tokens (glpat-)
    ("gitlab_token", re.compile(r"(?P<secret>glpat-[A-Za-z0-9\-_]{20,})")),
    # Slack tokens (xox[bporas]-)
    ("slack_token", re.compile(r"(?P<secret>xox[bporas]-[A-Za-z0-9\-]{10,})")),
    # Stripe keys (sk_live_, pk_live_, etc.)
    ("stripe_key", re.compile(r"(?P<secret>(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{20,})")),
    # NVIDIA API keys (nvapi-)
    ("nvidia_api_key", re.compile(r"(?P<secret>nvapi-[A-Za-z0-9_\-]{20,})")),
    # Anthropic API keys (sk-ant-) — must precede the broader openai_key pattern
    ("anthropic_key", re.compile(r"(?P<secret>sk-ant-[A-Za-z0-9_\-]{20,})")),
    # OpenAI API keys (sk-...)
    ("openai_key", re.compile(r"(?P<secret>sk-[A-Za-z0-9]{20,})")),
    # Google API keys (AIza...)
    ("google_key", re.compile(r"(?P<secret>AIza[A-Za-z0-9_\-]{35})")),
    # Bearer tokens in Authorization headers
    ("bearer_token", re.compile(r"[Bb]earer\s+(?P<secret>[A-Za-z0-9_.\-/+=]{20,})")),
    # Generic API key/token in key=value or key: value patterns
    (
        "generic_api_key",
        re.compile(
            r"(?:api[_-]?key|api[_-]?token|auth[_-]?token|access[_-]?token|"
            r"client[_-]?secret|refresh[_-]?token|code[_-]?verifier|"
            r"secret[_-]?key|private[_-]?key|password|passwd)"
            r"[\"\']?\s*[=:]+\s*[\"\'\s]*(?P<secret>[A-Za-z0-9_.\-/+=]{1,})",
            re.IGNORECASE,
        ),
    ),
    # Private keys (PEM format)
    (
        "private_key",
        re.compile(
            r"(?P<secret>-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
            r"[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----)",
        ),
    ),
    # Hex tokens after known prefix (64+ char hex strings)
    (
        "hex_token",
        re.compile(
            r"(?:token|key|secret|hash)[\"\']?\s*[=:]\s*[\"\'\s]*(?P<secret>[0-9a-f]{64,})",
            re.IGNORECASE,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Counters for telemetry
# ---------------------------------------------------------------------------


class ScrubStats:
    """Thread-safe counters for secret scrubbing telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_scrubbed = 0
        self._by_pattern: dict[str, int] = {}
        self._spans_scrubbed = 0

    def record(self, pattern_name: str, count: int = 1) -> None:
        with self._lock:
            self._total_scrubbed += count
            self._by_pattern[pattern_name] = self._by_pattern.get(pattern_name, 0) + count

    def record_span(self) -> None:
        with self._lock:
            self._spans_scrubbed += 1

    @property
    def total_scrubbed(self) -> int:
        with self._lock:
            return self._total_scrubbed

    @property
    def spans_scrubbed(self) -> int:
        with self._lock:
            return self._spans_scrubbed

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of all counters."""
        with self._lock:
            return {
                "total_secrets_redacted": self._total_scrubbed,
                "spans_with_secrets": self._spans_scrubbed,
                "by_pattern": dict(self._by_pattern),
            }

    def reset(self) -> None:
        with self._lock:
            self._total_scrubbed = 0
            self._by_pattern.clear()
            self._spans_scrubbed = 0


# Module-level singleton
stats = ScrubStats()


# ---------------------------------------------------------------------------
# Core scrubbing functions
# ---------------------------------------------------------------------------


def scrub_string(text: str) -> tuple[str, int]:
    """Scrub secrets from a string, replacing them with [REDACTED].

    Returns ``(scrubbed_text, redaction_count)``. The count is the number of
    secrets redacted in *this* string, tracked locally so callers do not have
    to diff the global ``stats`` singleton (which is not safe under concurrent
    span processing). Returns the original string unchanged with a count of 0
    if no secrets are found (fast path).
    """
    if not text:
        return text, 0

    count = 0
    result = text
    for name, pattern in _SECRET_PATTERNS:

        def _replace(m: re.Match, _name: str = name) -> str:
            nonlocal count
            full = m.group(0)
            secret = m.group("secret")
            stats.record(_name)
            count += 1
            return full.replace(secret, REDACTED)

        result = pattern.sub(_replace, result)

    return result, count


def scrub_value(value: Any) -> tuple[Any, int]:
    """Scrub secrets from a span attribute value.

    Handles strings, mappings, and sequences. Values under credential-bearing
    keys are always replaced, including short or provider-specific secrets.
    Returns ``(scrubbed_value, redaction_count)`` where the count is the
    number of secrets redacted within this value.
    """
    if isinstance(value, str):
        # OpenInference records LLM inputs as JSON string span attributes.
        # Parse only valid JSON and serialize it again only if the recursive
        # walk found additional keyed state to redact.
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return scrub_string(value)
        decoded, json_count = scrub_value(decoded)
        if json_count:
            return json.dumps(decoded, separators=(",", ":")), json_count
        return value, 0
    if isinstance(value, dict):
        scrubbed_mapping: dict[Any, Any] = {}
        count = 0
        is_thinking = value.get("type") == "thinking"
        is_redacted_thinking = value.get("type") == "redacted_thinking"
        for key, item in value.items():
            reason = _redact_key(key)
            if reason is None and is_thinking and key == "signature":
                reason = "opaque_provider_state"
            if reason is None and is_redacted_thinking and key == "data":
                reason = "opaque_provider_state"
            if reason is not None:
                scrubbed_mapping[key] = REDACTED
                stats.record(reason)
                count += 1
            else:
                scrubbed_mapping[key], n = scrub_value(item)
                count += n
        return scrubbed_mapping, count
    if isinstance(value, (list, tuple)):
        new_items = []
        count = 0
        for v in value:
            new_v, n = scrub_value(v)
            new_items.append(new_v)
            count += n
        return type(value)(new_items), count
    return value, 0


# ---------------------------------------------------------------------------
# OpenTelemetry SpanProcessor
# ---------------------------------------------------------------------------
try:
    from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

    class SecretScrubSpanProcessor(SpanProcessor):
        """Wraps another SpanProcessor, scrubbing secrets from span attributes.

        Also adds ``secrets.redacted_count`` to each span that had secrets,
        for observability into how often scrubbing fires.
        """

        def __init__(self, inner: SpanProcessor) -> None:
            self._inner = inner

        @property
        def inner(self) -> SpanProcessor:
            """The wrapped span processor (e.g. Batch/Simple)."""
            return self._inner

        def on_start(self, span: Span, parent_context: Any = None) -> None:
            self._inner.on_start(span, parent_context)

        def on_end(self, span: ReadableSpan) -> None:
            if span.attributes:
                scrubbed: dict[str, Any] = {}
                redacted_count = 0

                for key, value in span.attributes.items():
                    if reason := _redact_key(key):
                        new_value, n = REDACTED, 1
                        stats.record(reason)
                    else:
                        new_value, n = scrub_value(value)
                    scrubbed[key] = new_value
                    redacted_count += n

                if redacted_count > 0:
                    stats.record_span()
                    scrubbed["secrets.redacted_count"] = redacted_count
                    if hasattr(span, "_attributes"):
                        span._attributes = scrubbed  # type: ignore[attr-defined]
                        logger.debug(
                            "Scrubbed %d secrets from span %s",
                            redacted_count,
                            span.name,
                        )
                    else:
                        logger.warning(
                            "SecretScrubSpanProcessor: span._attributes not found; "
                            "%d secret(s) may be exported in span %s",
                            redacted_count,
                            span.name,
                        )

            self._inner.on_end(span)

        def shutdown(self) -> None:
            self._inner.shutdown()

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return self._inner.force_flush(timeout_millis)

except ImportError as exc:  # pragma: no cover - opentelemetry always installed in tests
    logger.warning("opentelemetry.sdk not available; SecretScrubSpanProcessor disabled: %s", exc)
