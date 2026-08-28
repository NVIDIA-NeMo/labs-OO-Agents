# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TruncationConfig model_validator — catches bad values on construction."""

import pytest

from nooa.config.truncation_config import (
    CaptureConfig,
    FormatConfig,
    MediaCaptureConfig,
    TruncationConfig,
)


class TestTruncationConfigValidation:
    def test_default_config_is_valid(self) -> None:
        TruncationConfig()  # must not raise

    def test_rejects_negative_max_stdout(self) -> None:
        with pytest.raises(ValueError, match="max_stdout"):
            CaptureConfig(max_stdout=-1)

    def test_rejects_zero_max_stderr(self) -> None:
        with pytest.raises(ValueError, match="max_stderr"):
            CaptureConfig(max_stderr=0)

    def test_rejects_zero_max_error(self) -> None:
        with pytest.raises(ValueError, match="max_error"):
            CaptureConfig(max_error=0)

    def test_rejects_zero_max_media_attachments(self) -> None:
        with pytest.raises(ValueError, match="max_attachments_per_execution"):
            MediaCaptureConfig(max_attachments_per_execution=0)

    def test_rejects_zero_value_max_length(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            FormatConfig(max_length=0)

    def test_rejects_zero_value_max_string(self) -> None:
        with pytest.raises(ValueError, match="max_string"):
            FormatConfig(max_string=0)

    def test_rejects_zero_value_max_depth(self) -> None:
        with pytest.raises(ValueError, match="max_depth"):
            FormatConfig(max_depth=0)

    def test_allows_none_value_limits(self) -> None:
        FormatConfig(max_length=None, max_string=None, max_depth=None)

    def test_rejects_negative_capture_tail(self) -> None:
        with pytest.raises(ValueError, match="capture.tail"):
            CaptureConfig(tail=-1)

    def test_rejects_capture_tail_gte_max_stdout(self) -> None:
        with pytest.raises(ValueError, match="capture.tail"):
            CaptureConfig(max_stdout=1000, tail=1000)

    def test_rejects_capture_tail_gte_max_stderr(self) -> None:
        with pytest.raises(ValueError, match="capture.tail"):
            CaptureConfig(max_stderr=500, tail=500)

    def test_rejects_capture_tail_gte_max_error(self) -> None:
        with pytest.raises(ValueError, match="capture.tail"):
            CaptureConfig(max_error=500, tail=500)

    def test_allows_valid_capture_tail(self) -> None:
        cfg = CaptureConfig(max_stdout=1000, max_stderr=500, max_error=600, tail=200)
        assert cfg.tail == 200

    def test_rejects_nonpositive_context_error_batch(self) -> None:
        with pytest.raises(ValueError, match="context_error_event_batch"):
            TruncationConfig(context_error_event_batch=0)

    def test_rejects_removed_token_budget_keys(self) -> None:
        with pytest.raises(ValueError, match="max_context_tokens"):
            TruncationConfig(max_context_tokens=100)
