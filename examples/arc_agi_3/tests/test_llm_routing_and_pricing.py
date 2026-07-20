# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for two headless-fleet failures:

1. gpt-5.x must route through the Responses API. A gpt-5.6-sol competition fleet
   403'd on EVERY LLM call because ``build_llm`` only special-cased the literal
   ``"gpt-5.5"``, so gpt-5.6-sol fell back to chat-completions, which this gateway
   reroutes with a mangled model id → ``403 key_model_access_denied``. The fix
   widened the heuristic to the whole gpt-5 family.

2. gpt-5.6(-sol) must be priced in the viewers. The dashboard showed ``-`` in
   every ``$`` column and no cumulative budget because ``LLM_PRICING`` only knew
   ``gpt5.5``; an unknown model resolves to zero pricing by design.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.5", "gpt-5.6-sol"])
def test_gpt5_family_routes_through_responses_api(monkeypatch, model):
    monkeypatch.setenv("ARC_LLM_MODEL", f"openai/openai/{model}")
    monkeypatch.setenv("ARC_LLM_BASE_URL", "http://gateway.invalid")
    monkeypatch.setenv("ARC_LLM_API_KEY", "test-key")
    monkeypatch.delenv("ARC_LLM_USE_RESPONSES", raising=False)

    import arc_llm

    from nooa.unifiedllm.unifiedllm import ResponsesClient

    client = arc_llm.build_llm()  # constructs the client; does NOT call the gateway
    assert isinstance(client, ResponsesClient), (
        f"{model} must use the Responses API (chat-completions 403s on this gateway)"
    )


def test_non_gpt5_model_uses_completions(monkeypatch):
    monkeypatch.setenv("ARC_LLM_MODEL", "openai/openai/gpt-4o")
    monkeypatch.setenv("ARC_LLM_BASE_URL", "http://gateway.invalid")
    monkeypatch.setenv("ARC_LLM_API_KEY", "test-key")
    monkeypatch.delenv("ARC_LLM_USE_RESPONSES", raising=False)

    import arc_llm

    from nooa.unifiedllm import CompletionClient

    assert type(arc_llm.build_llm()) is CompletionClient


def test_use_responses_env_override(monkeypatch):
    monkeypatch.setenv("ARC_LLM_MODEL", "openai/openai/gpt-4o")  # would default to completions
    monkeypatch.setenv("ARC_LLM_BASE_URL", "http://gateway.invalid")
    monkeypatch.setenv("ARC_LLM_API_KEY", "test-key")
    monkeypatch.setenv("ARC_LLM_USE_RESPONSES", "1")

    import arc_llm

    from nooa.unifiedllm.unifiedllm import ResponsesClient

    assert isinstance(arc_llm.build_llm(), ResponsesClient)


def test_gpt56_is_priced_in_the_viewer():
    import viewer

    # gpt-5.6 must resolve to the same non-zero figures as gpt-5.5 (not $0).
    assert viewer.get_pricing("gpt5.6")["input"] > 0
    assert viewer.get_pricing("gpt5.6") == viewer.get_pricing("gpt5.5")
    # the run's model URI must map to the gpt5.6 pricing key.
    assert viewer._pricing_key("openai/openai/gpt-5.6-sol") == "gpt5.6"
    assert viewer._pricing_key("openai/openai/gpt-5.5") == "gpt5.5"


def test_unknown_model_still_zero_priced():
    import viewer

    # sanity: the zero-pricing fallback for a genuinely unknown model is intact
    # (so the gpt5.6 add is a real fix, not a blanket non-zero default).
    assert viewer.get_pricing("some-unknown-model")["input"] == 0.0
