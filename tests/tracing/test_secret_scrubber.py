# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for secret scrubbing in telemetry."""

import json
from unittest.mock import patch

import pytest

from nooa.tracing._secret_scrubber import (
    REDACTED,
    ScrubStats,
    scrub_string,
    scrub_value,
    stats,
)


@pytest.fixture(autouse=True)
def _reset_stats():
    """Reset global stats before each test."""
    stats.reset()
    yield
    stats.reset()


class TestScrubString:
    def test_aws_access_key(self):
        """AWS access key IDs are redacted."""
        text = "key is AKIAIOSFODNN7EXAMPLE"
        result, count = scrub_string(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert REDACTED in result
        assert count == 1

    def test_aws_secret_key(self):
        """AWS secret access keys following a known prefix are redacted."""
        text = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        result, count = scrub_string(text)
        assert "wJalrXUtnFEMI" not in result
        assert REDACTED in result
        assert count >= 1

    def test_github_token(self):
        """GitHub personal access tokens (ghp_) are redacted."""
        text = "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        result, _ = scrub_string(text)
        assert "ghp_" not in result
        assert REDACTED in result

    def test_gitlab_token(self):
        """GitLab personal access tokens (glpat-) are redacted."""
        text = "GITLAB_TOKEN=glpat-abcdef1234567890abcd"
        result, _ = scrub_string(text)
        assert "glpat-" not in result
        assert REDACTED in result

    def test_slack_token(self):
        """Slack tokens (xoxb-) are redacted."""
        text = "slack: xoxb-1234567890-abcdefghij"
        result, _ = scrub_string(text)
        assert "xoxb-" not in result
        assert REDACTED in result

    def test_stripe_key(self):
        """Stripe live/test keys are redacted."""
        text = "STRIPE_KEY=sk_live_abcdefghijklmnopqrstuv"
        result, _ = scrub_string(text)
        assert "sk_live_" not in result
        assert REDACTED in result

    def test_nvidia_api_key(self):
        """NVIDIA API keys (nvapi-) are redacted."""
        text = "export NVIDIA_KEY=nvapi-abcdefghijklmnopqrstuv"
        result, _ = scrub_string(text)
        assert "nvapi-" not in result
        assert REDACTED in result

    def test_openai_key(self):
        """OpenAI API keys (sk-) are redacted."""
        text = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuv"
        result, _ = scrub_string(text)
        assert "sk-abcdefghijklmnopqrstuv" not in result
        assert REDACTED in result

    def test_anthropic_key(self):
        """Anthropic API keys (sk-ant-) are redacted."""
        text = "key=sk-ant-abcdefghijklmnopqrstuvwxyz"
        result, _ = scrub_string(text)
        assert "sk-ant-" not in result
        assert REDACTED in result

    def test_google_key(self):
        """Google API keys (AIza...) are redacted."""
        text = "GOOGLE_KEY=AIzaSyA1234567890abcdefghijklmnopqrstuv"
        result, _ = scrub_string(text)
        assert "AIzaSy" not in result
        assert REDACTED in result

    def test_bearer_token(self):
        """Bearer tokens in Authorization headers are redacted."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdef"
        result, _ = scrub_string(text)
        assert "eyJhbGci" not in result
        assert REDACTED in result

    def test_private_key(self):
        """The full PEM private key block (header, body, footer) is redacted."""
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEAsecretkeymaterial1234567890\n"
            "-----END RSA PRIVATE KEY-----"
        )
        result, _ = scrub_string(text)
        assert "BEGIN RSA PRIVATE KEY" not in result
        assert "MIIEowIBAAK" not in result
        assert REDACTED in result

    def test_generic_api_key(self):
        """Generic key=value secrets are redacted."""
        text = "api_key=abcdefghijklmnopqrstuvwxyz1234"
        result, _ = scrub_string(text)
        assert "abcdefghijklmnopqrstuvwxyz1234" not in result
        assert REDACTED in result

    def test_hex_token(self):
        """Long hex tokens following a known prefix are redacted."""
        text = "token=" + "a" * 64
        result, _ = scrub_string(text)
        assert "a" * 64 not in result
        assert REDACTED in result

    def test_no_secrets(self):
        """Clean text passes through unchanged with a zero count."""
        text = "This is just normal code output with no secrets at all."
        result, count = scrub_string(text)
        assert result == text
        assert count == 0

    def test_empty_string(self):
        """Empty strings return unchanged with a zero count."""
        assert scrub_string("") == ("", 0)

    def test_short_string(self):
        """A clean short string passes through unchanged."""
        assert scrub_string("hello") == ("hello", 0)

    def test_short_generic_secret(self):
        result, count = scrub_string("client_secret=s3cr3t")
        assert "s3cr3t" not in result
        assert count == 1

    def test_preserves_surrounding_text(self):
        """Surrounding non-secret text is preserved around a redaction."""
        text = "before AKIAIOSFODNN7EXAMPLE after"
        result, _ = scrub_string(text)
        assert "before" in result
        assert "after" in result
        assert REDACTED in result

    def test_multiple_secrets(self):
        """Multiple distinct secrets are each redacted and counted."""
        text = "aws: AKIAIOSFODNN7EXAMPLE and github: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        result, count = scrub_string(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "ghp_" not in result
        assert result.count(REDACTED) >= 2
        assert count >= 2

    def test_quoted_json_key(self):
        """Secrets under a JSON-quoted key (e.g. \'"api_key": "..."\') are redacted."""
        text = '{"api_key": "abcdefghijklmnopqrstuvwxyz1234"}'
        result, count = scrub_string(text)
        assert "abcdefghijklmnopqrstuvwxyz1234" not in result
        assert REDACTED in result
        assert count == 1


class TestScrubValue:
    @pytest.mark.parametrize("as_array", [False, True])
    def test_json_redaction_preserves_literal_unicode(self, as_array):
        payload = {"api_key": "synthetic-secret", "text": "héllo 世界"}
        attribute = " \n" + json.dumps([payload] if as_array else payload, ensure_ascii=False)

        result, count = scrub_value(attribute)

        assert "héllo 世界" in result
        assert "synthetic-secret" not in result
        expected = {"api_key": REDACTED, "text": "héllo 世界"}
        assert json.loads(result) == ([expected] if as_array else expected)
        assert count == 1

    @pytest.mark.parametrize("text", ["ordinary code output", " \n héllo 世界", "", "42"])
    def test_plain_text_never_attempts_json_parsing(self, text):
        with patch("nooa.tracing._secret_scrubber.json.loads") as parse:
            assert scrub_value(text) == (text, 0)
        parse.assert_not_called()

    def test_plain_text_still_redacts_without_json_parsing(self):
        with patch("nooa.tracing._secret_scrubber.json.loads") as parse:
            result, count = scrub_value("api_key=synthetic-secret")
        parse.assert_not_called()
        assert result == f"api_key={REDACTED}"
        assert count == 1

    def test_invalid_json_falls_back_to_text_scrubbing(self):
        result, count = scrub_value("[progress] api_key=synthetic-secret")
        assert result == f"[progress] api_key={REDACTED}"
        assert count == 1

    def test_clean_json_keeps_original_formatting(self):
        attribute = ' \n{ "text": "héllo 世界" }\n'
        assert scrub_value(attribute) == (attribute, 0)

    def test_string(self):
        """A string value is scrubbed and its redaction count returned."""
        result, count = scrub_value("key=AKIAIOSFODNN7EXAMPLE")
        assert REDACTED in result
        assert count == 1

    def test_list_of_strings(self):
        """List items are scrubbed individually and counts aggregated."""
        result, count = scrub_value(["AKIAIOSFODNN7EXAMPLE", "normal"])
        assert REDACTED in result[0]
        assert result[1] == "normal"
        assert count == 1

    def test_tuple(self):
        """Tuple values keep their type and are scrubbed element-wise."""
        result, count = scrub_value(("AKIAIOSFODNN7EXAMPLE",))
        assert isinstance(result, tuple)
        assert REDACTED in result[0]
        assert count == 1

    def test_int_passthrough(self):
        """Non-string scalars pass through unchanged with a zero count."""
        assert scrub_value(42) == (42, 0)

    def test_bool_passthrough(self):
        """Booleans pass through unchanged with a zero count."""
        result, count = scrub_value(True)
        assert result is True
        assert count == 0

    def test_none_passthrough(self):
        """None passes through unchanged with a zero count."""
        assert scrub_value(None) == (None, 0)

    def test_nested_sensitive_keys(self):
        result, count = scrub_value(
            {"safe": {"client_secret": "short", "refresh_token": "provider-specific"}}
        )
        assert result == {"safe": {"client_secret": REDACTED, "refresh_token": REDACTED}}
        assert count == 2

    def test_openai_encrypted_reasoning_is_redacted_from_mapping(self):
        result, count = scrub_value(
            {"input": [{"type": "reasoning", "encrypted_content": "opaque-openai-state"}]}
        )

        assert result["input"][0]["encrypted_content"] == REDACTED
        assert count == 1

    def test_openai_encrypted_reasoning_is_redacted_from_json_attribute(self):
        attribute = json.dumps(
            {"input": [{"type": "reasoning", "encrypted_content": "opaque-openai-state"}]}
        )

        result, count = scrub_value(attribute)

        assert json.loads(result)["input"][0]["encrypted_content"] == REDACTED
        assert "opaque-openai-state" not in result
        assert count == 1

    @pytest.mark.parametrize(
        ("provider_state", "secret"),
        [
            (
                {"thinking_blocks": [{"type": "thinking", "signature": "anthropic-sig"}]},
                "anthropic-sig",
            ),
            (
                {"thinking_blocks": [{"type": "redacted_thinking", "data": "opaque-data"}]},
                "opaque-data",
            ),
            (
                {"provider_specific_fields": {"thought_signature": "gemini-sig"}},
                "gemini-sig",
            ),
            (
                {"provider_specific_fields": {"thought_signatures": ["gemini-sig"]}},
                "gemini-sig",
            ),
            (
                {"providerSpecificFields": {"thoughtSignature": "gemini-camel-sig"}},
                "gemini-camel-sig",
            ),
            (
                {"providerSpecificFields": {"thoughtSignatures": ["gemini-camel-sig"]}},
                "gemini-camel-sig",
            ),
        ],
    )
    def test_cross_provider_opaque_state_is_redacted(self, provider_state, secret):
        result, count = scrub_value(provider_state)

        assert secret not in repr(result)
        assert REDACTED in repr(result)
        assert count == 1

    def test_gemini_inline_tool_call_signature_is_redacted_from_flat_string(self):
        result, count = scrub_value("call_1__thought__Z2VtaW5pLXNpZw==")

        assert result == f"call_1__thought__{REDACTED}"
        assert count == 1

    def test_gemini_inline_tool_call_signature_is_redacted_from_json_attribute(self):
        result, count = scrub_value('{"tool_calls":[{"id":"call_1__thought__Z2VtaW5pLXNpZw=="}]}')

        assert "Z2VtaW5pLXNpZw==" not in result
        assert REDACTED in result
        assert count == 1


class TestScrubStats:
    def test_record_and_snapshot(self):
        """record() and record_span() accumulate into snapshot() correctly."""
        s = ScrubStats()
        s.record("aws_access_key", 2)
        s.record("github_token", 1)
        s.record_span()
        snap = s.snapshot()
        assert snap["total_secrets_redacted"] == 3
        assert snap["spans_with_secrets"] == 1
        assert snap["by_pattern"]["aws_access_key"] == 2
        assert snap["by_pattern"]["github_token"] == 1

    def test_reset(self):
        """reset() zeroes all counters and clears the per-pattern map."""
        s = ScrubStats()
        s.record("test", 5)
        s.reset()
        assert s.total_scrubbed == 0
        assert s.snapshot()["by_pattern"] == {}

    def test_global_stats_increment(self):
        """Global stats should increment when scrub_string finds secrets."""
        scrub_string("key: AKIAIOSFODNN7EXAMPLE")
        assert stats.total_scrubbed > 0

    def test_global_stats_no_increment_on_clean(self):
        """Global stats should not increment for clean strings."""
        before = stats.total_scrubbed
        scrub_string("just normal text nothing secret here")
        assert stats.total_scrubbed == before
