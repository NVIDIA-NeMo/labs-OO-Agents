# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An event channel's QueueOutput string preview is cut through nooa.snapshot (sweep CORE036)."""

from nooa.runtime.channels import _default_event_preview


def test_a_short_string_passes_through():
    assert _default_event_preview("x" * 200) == "x" * 200


def test_a_long_string_keeps_200_chars_in_snapshot_notation():
    value = "a" * 150 + "b" * 150
    assert _default_event_preview(value) == "str(len=300, [:200]='" + "a" * 150 + "b" * 50 + "')"


def test_other_values_are_pformat_previews():
    assert _default_event_preview([1, 2]) == "[1, 2]"
