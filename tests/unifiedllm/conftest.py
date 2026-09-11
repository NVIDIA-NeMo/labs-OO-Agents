# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An unexpected replay downgrade must fail tests, not merely log a warning."""

import pytest


@pytest.fixture(autouse=True)
def assert_expected_replay_edits(request, caplog):
    yield
    downgrades = [
        record
        for when in ("setup", "call")
        for record in caplog.get_records(when)
        if record.name == "nooa.unifiedllm.unifiedllm"
        and record.getMessage()
        == "Assistant message was edited; replaying it without native state."
    ]
    expected = request.node.get_closest_marker("expected_replay_edit") is not None
    assert bool(downgrades) == expected, (
        "Unexpected replay downgrade: rendering and the stored public message diverged. "
        "Mark only intentional edit/truncation cases with expected_replay_edit."
        if downgrades
        else "This test expected an edit downgrade, but none was observed."
    )
