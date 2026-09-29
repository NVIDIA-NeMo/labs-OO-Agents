# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reasoning gets its own allowance without increasing a caller's approved total."""

import pytest

from nooa.unifiedllm import connect


def plan(**kwargs):
    return connect.plan(
        "test",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"high": {"reasoning_effort": "high"}},
        **kwargs,
    )


def test_separate_caps_are_included_in_token_and_price_estimates():
    proposal = plan(
        catalogue={
            "id": "vendor/model",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        }
    )
    assert proposal.budget_tokens == connect.DEFAULT_CHECK_BUDGET
    assert proposal.entry["max_tokens"] == 32768
    assert [p.body["max_tokens"] for p in proposal.probes] == [32768] * 3
    assert [p.token_estimate for p in proposal.probes] == [33280] * 3
    assert proposal.token_estimate == 99840
    assert proposal.price_estimate == pytest.approx(3 * 512e-6 + 3 * 32768 * 2e-6)


@pytest.mark.asyncio
async def test_smaller_explicit_budget_skips_before_any_call(monkeypatch):
    from tests.unifiedllm.connect.connect_http import mock_http

    mock_http(monkeypatch, lambda request: pytest.fail("No budget for this request"))
    proposal = plan(budget_tokens=1024)
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    assert proposal.budget_tokens == 1024
    assert result.entry["provenance"]["probes"]["level:high"] == {
        "outcome": "not_probed",
        "reason": "budget exhausted",
    }


@pytest.mark.parametrize("value", [0, 32769])
def test_invalid_reasoning_allowance_rejected_during_plan(value):
    with pytest.raises(ValueError, match="reasoning_output_tokens"):
        plan(reasoning_output_tokens=value)
