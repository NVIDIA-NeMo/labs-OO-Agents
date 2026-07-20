# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TDD: pformat API change — max_total_chars and _truncated_out removed.

Change 1 of truncation-2.0: pformat becomes purely structural (M1).
Memory safety is M2's responsibility (truncating_pformat / TruncatingStringIO).
"""

import pytest

from nooa.agentdoc import pformat


class TestPformatRemovesBudgetParams:
    """pformat must reject max_total_chars and _truncated_out (TDD: fails until Change 1)."""

    def test_pformat_rejects_max_total_chars(self):
        # TDD: will fail until Change 1 is implemented
        with pytest.raises(TypeError):
            pformat([1, 2, 3], max_total_chars=1000)

    def test_pformat_rejects_truncated_out(self):
        # TDD: will fail until Change 1 is implemented
        with pytest.raises(TypeError):
            pformat([1, 2, 3], _truncated_out=[False])

    def test_pformat_still_accepts_structural_params(self):
        # TDD: will fail until Change 1 is implemented (silently passes today — verify after)
        result = pformat([1, 2, 3], max_length=10, max_string=100, max_depth=5)
        assert isinstance(result, str)
        assert "1" in result

    def test_pformat_without_budget_formats_large_list(self):
        # TDD: will fail until Change 1 is implemented
        # Without _budget abort, pformat with max_length=None renders everything.
        big = list(range(200))
        result = pformat(big, max_length=None)
        assert "199" in result  # tail visible — no abort-early cut

    def test_pformat_max_length_still_applies_element_head_tail(self):
        # max_length is purely structural — shows head+tail with truncation
        # 3.0's slice-keys marker. No char budget.
        items = list(range(100))
        result = pformat(items, max_length=10)
        assert "0" in result  # head
        assert "99" in result  # tail
        assert "list(len=100," in result  # truncation 3.0 marker, not "... +N"
