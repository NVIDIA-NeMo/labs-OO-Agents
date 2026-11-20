# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task charges have a terminal source independent of metrics/tracing."""

import asyncio
from types import SimpleNamespace

import pytest

from nooa.runtime.actor import _harness_metrics_lifecycle, _make_llm_metrics_bridge
from nooa.runtime.token_usage import get_task_tokens, start_task_tokens
from nooa.unifiedllm._accounting import _logical_account_scope
from nooa.unifiedllm._observations import _NumericUsage


@pytest.mark.parametrize("error", [None, ValueError("paid parse"), asyncio.CancelledError()])
def test_terminal_failed_cancelled_success_no_metrics(error):
    start_task_tokens()
    try:
        with _harness_metrics_lifecycle(False) as metrics:
            assert metrics is None
            with _logical_account_scope() as account:
                for usage in [
                    _NumericUsage(input_tokens=7, output_tokens=2),
                    _NumericUsage(input_tokens=3, output_tokens=None),
                ]:
                    index, owner = account.attempt()
                    owner.writer.entering_provider()
                    owner.writer.delivered(usage)
                    account.fold(index, owner)
                if error is not None:
                    raise error
    except BaseException as caught:
        assert caught is error
    assert get_task_tokens() == {"n_input_tokens": 10, "n_output_tokens": 2}
    assert not account.snapshot.subtotal("output_tokens").complete


def test_owned_success_metrics_not_charge_but_repairs_and_unowned_fallback():
    repairs = []
    bridge = _make_llm_metrics_bridge(
        SimpleNamespace(record_json_fence_removed=lambda: repairs.append("fence"))
    )
    start_task_tokens()
    with _harness_metrics_lifecycle(False):
        with _logical_account_scope() as account:
            index, owner = account.attempt()
            owner.writer.entering_provider()
            owner.writer.delivered(_NumericUsage(input_tokens=7, output_tokens=2))
            account.fold(index, owner)
            bridge("token_usage", {"input_tokens": 100, "output_tokens": 100})
            bridge("json_fence_removed")
        assert get_task_tokens() == {"n_input_tokens": 7, "n_output_tokens": 2}
        bridge("token_usage", {"input_tokens": 3, "output_tokens": 1})
    assert repairs == ["fence"]
    assert get_task_tokens() == {"n_input_tokens": 10, "n_output_tokens": 3}


def test_receiver_captures_task_not_eventual_context():
    start_task_tokens()
    with _harness_metrics_lifecycle(False):
        with _logical_account_scope() as account:
            index, owner = account.attempt()
            owner.writer.entering_provider()
            owner.writer.delivered(_NumericUsage(input_tokens=7, output_tokens=2))
            account.fold(index, owner)
            # A different task accumulator is installed after account creation.
            start_task_tokens()
    assert get_task_tokens() == {"n_input_tokens": 0, "n_output_tokens": 0}
    assert account.snapshot.subtotal("input_tokens").known == 7
