# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Result wording for failures seen in live NVIDIA Build and Inference Hub tests."""

import httpx
import pytest
from nooa_cli.commands import _connect_view as view

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http


def test_account_404_says_the_model_is_not_available_for_the_account():
    record = {"error": "NotFoundError", "status_code": 404, "unavailable_for_account": True}
    assert view.check_failure(record) == "This model isn't available for your account (HTTP 404)."
    # A plain 404 still means the route or interface is missing.
    assert view.check_failure({"error": "NotFoundError", "status_code": 404}) == (
        "No route found for this interface or model."
    )


def test_server_error_at_test_time_is_a_failure_in_the_summary_too():
    """A 503 is stored as not_probed; the summary must say Failed like the row does."""
    record = {
        "outcome": "not_probed",
        "error": "ServiceUnavailableError",
        "status_code": 503,
        "baseline_kind": "predict_int_v1",
        "typed_result_validated": False,
        "answer_correct": False,
    }
    assert view.connection_detail(record) == "Failed · Server error. Try again later."
    # A call that was never sent is still "Not tested".
    assert view.connection_detail({"outcome": "not_probed"}) == "Not tested"


@pytest.mark.asyncio
async def test_server_error_over_http_reads_failed_in_row_and_summary(monkeypatch):
    def handle(request):
        return httpx.Response(503, text="private upstream detail")

    mock_http(monkeypatch, handle)
    proposal = connect.plan("local", "model", "chat", "https://api.test/v1", "")
    result = await connect.check_stage(proposal, "routing", api_key="test-key")
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["status_code"] == 503
    assert view.connection_detail(record) == "Failed · Server error. Try again later."
    assert "private upstream detail" not in repr(result)


@pytest.mark.asyncio
async def test_off_level_that_still_reasons_is_never_verified(monkeypatch):
    """Nemotron and MiniMax still reason with Off; Off must not be saved as verified."""
    from nooa.unifiedllm.connect._verification import save_verification

    def handle(request):
        return httpx.Response(
            200,
            json={
                "id": "r",
                "object": "chat.completion",
                "created": 1,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"value":323}',
                            "reasoning_content": "Thinking anyway.",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29},
            },
        )

    mock_http(monkeypatch, handle)
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    proposal = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"off": off},
        reasoning_level="off",
    )
    result = await connect.check_stage(proposal, "all", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:off"]
    assert record["outcome"] == "accepted" and record["reasoning_observed"] is True
    assert connect.ignored_off_reasoning_levels(result.entry) == ["off"]
    assert "level:off" in connect.verdict(result.entry).needs_attention
    summary, _ = save_verification(result.entry)
    assert summary["state"] != "verified"
    assert "level:off" in summary["unverified_scopes"]
