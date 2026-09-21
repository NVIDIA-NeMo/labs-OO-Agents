# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate attribution is explicit, independent of the alias transport or environment."""

from types import SimpleNamespace

import pytest

from tests.integration import _release_gate as gate


@pytest.mark.parametrize("value", ["direct", "litellm", ""])
def test_process_override_is_rejected_before_alias_lookup(monkeypatch, value):
    monkeypatch.setenv("NOOA_LLM_TRANSPORT", value)
    monkeypatch.setattr(gate, "_entry", lambda _: pytest.fail("must stop before alias lookup"))
    with pytest.raises(pytest.fail.Exception, match="Unset NOOA_LLM_TRANSPORT"):
        gate.gate_client("openai", transport="litellm")


@pytest.mark.parametrize("transport", ["litellm", "direct"])
def test_selected_transport_is_not_owned_by_alias(monkeypatch, transport):
    monkeypatch.delenv("NOOA_LLM_TRANSPORT", raising=False)
    monkeypatch.setattr(gate, "_entry", lambda _: {"transport": "direct"})
    calls = []

    def construct(alias, **kwargs):
        calls.append((alias, kwargs))
        return SimpleNamespace(transport=kwargs["transport"])

    monkeypatch.setattr(gate, "get_llm_client", construct)
    gate.gate_client("openai", transport=transport)
    assert calls == [("release-gate-openai", {"transport": transport})]


def test_alias_cannot_override_shipped_cache_default(monkeypatch):
    monkeypatch.delenv("NOOA_LLM_TRANSPORT", raising=False)
    monkeypatch.setattr(gate, "_entry", lambda _: {"cache_breakpoint": "openai"})
    with pytest.raises(AssertionError, match="shipped cache defaults"):
        gate.gate_client("openai")
