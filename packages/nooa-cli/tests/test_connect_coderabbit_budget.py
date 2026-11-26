# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local skipped media must not create wizard insufficient-budget warnings."""

from dataclasses import replace

import httpx
import pytest
from nooa_cli.commands import _connect_wizard as wizard

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.mark.parametrize("style", ["responses", "anthropic"])
def test_local_media_skips_do_not_warn_or_consume_budget(monkeypatch, capsys, style):
    """Exercise wizard display/execution without saving config or live calls."""
    sent = []
    mock_http(
        monkeypatch,
        lambda request: sent.append(request) or httpx.Response(200, json=response_body(style)),
    )
    proposal = connect.plan(
        "local",
        "model",
        style,
        "https://api.test/v1",
        "",
        reply_tokens=1000,
        test_media_inputs=["audio", "video"],
        budget_tokens=3024,
    )
    state = wizard.WizardState(
        api_key="fake",
        proposal=connect.refresh_plan(proposal),
        approval="all",
        media_inputs=[],
        test_media_inputs=["audio", "video"],
        interface_spent=0,
        yes=True,
    )
    state.remaining_estimate = state.proposal.token_estimate
    assert wizard.run_checks(state)
    output = capsys.readouterr()
    assert "Some tests will be skipped" not in output.err + output.out
    assert len(sent) == 2
    assert state.result.entry["provenance"]["tokens_charged_to_budget"] == 3024
    for kind in ("audio", "video"):
        assert (
            state.result.entry["provenance"]["probes"]["media:" + kind]["outcome"] == "not_probed"
        )
    assert "Audio input: Skipped" in output.out and "Video input: Skipped" in output.out
    # A genuinely insufficient runnable budget still warns.
    state.proposal = replace(proposal, budget_tokens=3023)
    state.remaining_estimate = state.proposal.token_estimate
    assert not wizard.run_checks(state)  # the earlier attempts exhausted this wizard's allowance
    output = capsys.readouterr()
    assert "Some tests will be skipped" in output.err + output.out
    assert len(sent) == 2  # another Test now never resets the allowance
    assert state.tokens_spent == 3024
