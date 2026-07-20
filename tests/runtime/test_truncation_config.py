# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TruncationConfig."""

from nooa.config.truncation_config import (
    DEFAULT_TRUNCATION_CONFIG,
    CaptureConfig,
    FormatConfig,
    MediaCaptureConfig,
    TruncationConfig,
)


class TestTruncationConfig:
    """Tests for TruncationConfig dataclass."""

    def test_default_values(self):
        """Default config should have expected values."""
        config = TruncationConfig()

        assert config.capture.max_stdout == 50_000
        assert config.capture.max_stderr == 2_000
        assert config.capture.max_error == 10_000
        assert config.media_capture.max_attachments_per_execution == 5
        # events: generous (rendered every turn for the rest of the trajectory)
        assert config.event_format.max_length == 200
        assert config.event_format.max_string == 10_000
        assert config.event_format.max_depth == 5
        # prefill: one-shot per call, agent author controls inputs
        assert config.prefill_format.max_length == 25
        assert config.prefill_format.max_string == 2_000
        assert config.prefill_format.max_depth == 4
        # context_blocks: unlimited (agent-author-curated, L4 handles overflow)
        assert config.context_block_format.max_length is None
        assert config.context_block_format.max_string is None
        assert config.context_block_format.max_depth is None

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = TruncationConfig(
            capture=CaptureConfig(max_stdout=100_000, max_stderr=30_000),
            media_capture=MediaCaptureConfig(max_attachments_per_execution=20),
            event_format=FormatConfig(max_length=100, max_string=1000, max_depth=5),
        )

        assert config.capture.max_stdout == 100_000
        assert config.capture.max_stderr == 30_000
        assert config.media_capture.max_attachments_per_execution == 20
        assert config.event_format.max_length == 100
        assert config.event_format.max_string == 1000
        assert config.event_format.max_depth == 5

    def test_none_value_limits(self):
        """Value limits can be None for unlimited."""
        config = TruncationConfig(
            event_format=FormatConfig(max_length=None, max_string=None, max_depth=None),
        )

        assert config.event_format.max_length is None
        assert config.event_format.max_string is None
        assert config.event_format.max_depth is None

    def test_merge_with_none(self):
        """Merging with None should return self."""
        config = TruncationConfig(capture=CaptureConfig(max_stdout=100_000))
        merged = config.merge_with(None)

        assert merged.capture.max_stdout == 100_000
        assert merged is config  # Returns self when merging with None

    def test_merge_with_overrides(self):
        """Merging should override with explicitly-set values."""
        base = TruncationConfig(
            capture=CaptureConfig(max_stdout=50_000),
            event_format=FormatConfig(max_length=50),
        )
        override = TruncationConfig(
            capture=CaptureConfig(max_stdout=100_000),
            event_format=FormatConfig(max_length=100),
        )

        merged = base.merge_with(override)

        assert merged.capture.max_stdout == 100_000
        assert merged.event_format.max_length == 100

    def test_merge_with_empty_fields_set_raises(self):
        """merge_with(TruncationConfig()) raises since no fields are explicitly set."""
        import pytest

        base = TruncationConfig(
            capture=CaptureConfig(max_stdout=100_000, max_stderr=40_000),
        )
        override = TruncationConfig()  # All defaults — model_fields_set is empty

        with pytest.raises(ValueError, match="merge_with"):
            base.merge_with(override)

    def test_merge_creates_new_instance(self):
        """Merge should create a new config instance."""
        config1 = TruncationConfig(capture=CaptureConfig(max_stdout=100_000))
        config2 = TruncationConfig(capture=CaptureConfig(max_stderr=30_000))

        merged = config1.merge_with(config2)

        # Should be different object
        assert merged is not config1
        assert merged is not config2

    def test_default_truncation_config_instance(self):
        """DEFAULT_TRUNCATION_CONFIG should exist and have defaults."""
        assert DEFAULT_TRUNCATION_CONFIG.capture.max_stdout == 50_000
        assert DEFAULT_TRUNCATION_CONFIG.capture.max_stderr == 2_000
        assert DEFAULT_TRUNCATION_CONFIG.capture.max_error == 10_000
        assert DEFAULT_TRUNCATION_CONFIG.media_capture.max_attachments_per_execution == 5
        assert DEFAULT_TRUNCATION_CONFIG.event_format.max_length == 200
        assert DEFAULT_TRUNCATION_CONFIG.prefill_format.max_length == 25
        assert DEFAULT_TRUNCATION_CONFIG.context_block_format.max_length is None

    def test_all_limits_positive(self):
        """All limits should be positive integers."""
        config = TruncationConfig()

        assert config.capture.max_stdout > 0
        assert config.capture.max_stderr > 0
        assert config.capture.max_error > 0
        assert config.media_capture.max_attachments_per_execution > 0

    def test_equality(self):
        """Two configs with same values should be equal."""
        config1 = TruncationConfig(capture=CaptureConfig(max_stdout=100_000))
        config2 = TruncationConfig(capture=CaptureConfig(max_stdout=100_000))

        assert config1 == config2

    def test_inequality(self):
        """Two configs with different values should not be equal."""
        config1 = TruncationConfig(capture=CaptureConfig(max_stdout=100_000))
        config2 = TruncationConfig(capture=CaptureConfig(max_stdout=50_000))

        assert config1 != config2


class TestMergeSemantics:
    """Tests for merge_with() merge semantics using model_fields_set."""

    def test_three_way_merge(self):
        """Three-way merge with non-default values."""
        class_level = TruncationConfig(capture=CaptureConfig(max_stdout=100_000))
        instance_level = TruncationConfig(
            capture=CaptureConfig(max_stdout=100_000),
            event_format=FormatConfig(max_length=100),
        )
        method_level = TruncationConfig(
            capture=CaptureConfig(max_stdout=100_000),
            event_format=FormatConfig(max_length=100, max_depth=10),
        )

        merged = class_level.merge_with(instance_level).merge_with(method_level)

        assert merged.capture.max_stdout == 100_000
        assert merged.event_format.max_length == 100
        assert merged.event_format.max_depth == 10

    def test_later_overrides_earlier(self):
        """Later configs with explicit values override earlier ones."""
        config1 = TruncationConfig(
            capture=CaptureConfig(max_stdout=100_000),
            event_format=FormatConfig(max_length=100),
        )
        config2 = TruncationConfig(
            capture=CaptureConfig(max_stdout=200_000),
            event_format=FormatConfig(max_length=100),
        )

        merged = config1.merge_with(config2)

        assert merged.capture.max_stdout == 200_000
        assert merged.event_format.max_length == 100

    def test_partial_override(self):
        """Partial override should only change specified fields, including
        across nested sub-configs (capture/value)."""
        base = TruncationConfig(
            capture=CaptureConfig(max_stdout=100_000, max_stderr=30_000),
            event_format=FormatConfig(max_length=100),
        )
        override = TruncationConfig(event_format=FormatConfig(max_length=200))

        merged = base.merge_with(override)

        assert merged.capture.max_stdout == 100_000
        assert merged.capture.max_stderr == 30_000
        assert merged.event_format.max_length == 200

    def test_partial_override_within_subconfig(self):
        """Sub-config field-level merge: overriding only one field of capture
        preserves the other capture fields from the base."""
        base = TruncationConfig(
            capture=CaptureConfig(max_stdout=100_000, max_stderr=40_000, max_error=20_000),
        )
        override = TruncationConfig(capture=CaptureConfig(max_stdout=200_000))

        merged = base.merge_with(override)

        assert merged.capture.max_stdout == 200_000
        assert merged.capture.max_stderr == 40_000  # preserved from base
        assert merged.capture.max_error == 20_000  # preserved from base

    def test_context_blocks_merge_independent_of_events(self):
        """Overriding context_blocks does not affect events, and vice versa.
        These are separate render moments with separate defaults."""
        base = TruncationConfig()
        override = TruncationConfig(
            context_block_format=FormatConfig(max_string=50_000),
        )
        merged = base.merge_with(override)

        # context_blocks overridden
        assert merged.context_block_format.max_string == 50_000
        # context_blocks' other fields preserved (still None from base)
        assert merged.context_block_format.max_length is None
        assert merged.context_block_format.max_depth is None
        # events untouched
        assert merged.event_format.max_string == 10_000
        # prefill untouched
        assert merged.prefill_format.max_string == 2_000

    def test_media_capture_merge_independent_of_text_capture(self):
        """Overriding media_capture does not affect stdout/stderr capture."""
        base = TruncationConfig(
            capture=CaptureConfig(max_stdout=100_000, max_stderr=30_000),
            media_capture=MediaCaptureConfig(max_attachments_per_execution=5),
        )
        override = TruncationConfig(
            media_capture=MediaCaptureConfig(max_attachments_per_execution=20),
        )

        merged = base.merge_with(override)

        assert merged.media_capture.max_attachments_per_execution == 20
        assert merged.capture.max_stdout == 100_000
        assert merged.capture.max_stderr == 30_000

    def test_explicit_none_overrides(self):
        """Explicitly passing None overrides the base value."""
        base = TruncationConfig(event_format=FormatConfig(max_length=100))
        override = TruncationConfig(event_format=FormatConfig(max_length=None))

        merged = base.merge_with(override)

        assert merged.event_format.max_length is None

    def test_unset_fields_dont_override(self):
        """Fields not passed to __init__ don't override base values."""
        base = TruncationConfig(
            event_format=FormatConfig(max_length=100),
            capture=CaptureConfig(max_stdout=80_000),
        )
        override = TruncationConfig(capture=CaptureConfig(max_stdout=90_000))

        merged = base.merge_with(override)

        assert merged.event_format.max_length == 100
        assert merged.capture.max_stdout == 90_000

    def test_explicit_default_value_overrides(self):
        """Explicitly setting a field to the default value still overrides."""
        base = TruncationConfig(max_context_tokens=10_000)
        override = TruncationConfig(max_context_tokens=20_000)

        merged = base.merge_with(override)

        assert merged.max_context_tokens == 20_000
