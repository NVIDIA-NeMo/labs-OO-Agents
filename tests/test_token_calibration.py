# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TokenCalibration — EMA-based token count correction (#204)."""

import pytest

from nooa.unifiedllm.unifiedllm import (
    TokenCalibration,
    _token_calibration,
    _update_token_calibration,
)


class TestTokenCalibration:
    """Unit tests for the TokenCalibration EMA tracker."""

    def test_default_ratio_is_one(self):
        cal = TokenCalibration()
        assert cal.ratio("any-model") == 1.0

    def test_custom_default_ratio(self):
        cal = TokenCalibration(default_ratio=1.5)
        assert cal.ratio("unseen-model") == 1.5

    def test_first_observation_sets_ratio(self):
        cal = TokenCalibration()
        cal.update("model-a", estimated=100, actual=150)
        assert cal.ratio("model-a") == pytest.approx(1.5)

    def test_ema_converges(self):
        """After several observations at the same ratio, EMA converges."""
        cal = TokenCalibration(alpha=0.3)
        for _ in range(20):
            cal.update("model-a", estimated=100, actual=200)
        assert cal.ratio("model-a") == pytest.approx(2.0, abs=0.01)

    def test_ema_smooths_outliers(self):
        """A single outlier doesn't dominate."""
        cal = TokenCalibration(alpha=0.3)
        # Establish baseline at 1.5x
        for _ in range(10):
            cal.update("model-a", estimated=100, actual=150)
        baseline = cal.ratio("model-a")
        # One outlier at 3.0x
        cal.update("model-a", estimated=100, actual=300)
        after_outlier = cal.ratio("model-a")
        # Should shift but not jump to 3.0
        assert after_outlier > baseline
        assert after_outlier < 2.5

    def test_per_model_isolation(self):
        cal = TokenCalibration()
        cal.update("model-a", estimated=100, actual=200)
        cal.update("model-b", estimated=100, actual=120)
        assert cal.ratio("model-a") == pytest.approx(2.0)
        assert cal.ratio("model-b") == pytest.approx(1.2)

    def test_calibrate_applies_ratio(self):
        cal = TokenCalibration()
        cal.update("model-a", estimated=100, actual=170)
        result = cal.calibrate("model-a", 500)
        assert result == int(500 * 1.7)

    def test_calibrate_unseen_uses_default(self):
        cal = TokenCalibration(default_ratio=1.0)
        assert cal.calibrate("unseen", 500) == 500

    def test_zero_estimate_ignored(self):
        cal = TokenCalibration()
        cal.update("model-a", estimated=0, actual=100)
        assert cal.ratio("model-a") == 1.0  # unchanged

    def test_zero_actual_ignored(self):
        cal = TokenCalibration()
        cal.update("model-a", estimated=100, actual=0)
        assert cal.ratio("model-a") == 1.0  # unchanged

    def test_negative_values_ignored(self):
        cal = TokenCalibration()
        cal.update("model-a", estimated=-10, actual=100)
        cal.update("model-a", estimated=100, actual=-10)
        assert cal.ratio("model-a") == 1.0

    def test_repr(self):
        cal = TokenCalibration()
        cal.update("m", estimated=100, actual=200)
        assert "m: 2.000" in repr(cal)


class TestUpdateTokenCalibration:
    """Tests for the _update_token_calibration helper."""

    def setup_method(self):
        """Reset the module-level calibration before each test."""
        _token_calibration._ratios.clear()

    def teardown_method(self):
        """Clean up module-level singleton after each test."""
        _token_calibration._ratios.clear()

    def test_updates_from_messages(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        _update_token_calibration(
            "gpt-4o", messages, {"prompt_tokens": 50, "completion_tokens": 10}
        )
        ratio = _token_calibration.ratio("gpt-4o")
        # API reports 50 prompt tokens; litellm raw estimate is ~9 tokens for this text
        # Ratio should be significantly > 1.0 (50/9 ≈ 5.6)
        assert ratio > 1.0
        assert ratio < 100.0  # sanity upper bound

    def test_skips_empty_usage(self):
        messages = [{"role": "user", "content": "hello"}]
        _update_token_calibration("gpt-4o", messages, {"prompt_tokens": 0})
        assert _token_calibration.ratio("gpt-4o") == 1.0  # unchanged

    def test_handles_multipart_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        _update_token_calibration("gpt-4o", messages, {"prompt_tokens": 30, "completion_tokens": 5})
        ratio = _token_calibration.ratio("gpt-4o")
        assert ratio > 0  # didn't crash on multipart
