# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delegated context must respect even very small caller-provided budgets."""

import pytest
from nooa_cli.coding.context_rendering import render_delegated_context


@pytest.mark.parametrize("max_chars", [-10, 0, 1, 12, 13, 14, 50, 500])
def test_context_never_exceeds_character_budget(max_chars):
    """Truncation markers count toward the output budget, including at zero."""
    rendered = render_delegated_context({"payload": "x" * 200}, max_chars=max_chars)

    assert len(rendered) <= max(0, max_chars)
    if max_chars <= 0:
        assert rendered == ""
    if max_chars == 500:
        assert rendered == '{"payload": "' + "x" * 200 + '"}'
