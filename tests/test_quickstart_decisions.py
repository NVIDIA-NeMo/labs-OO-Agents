# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline configuration checks for the decisions quickstart."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def example(monkeypatch):
    """Load the example without running its entry point or probing tracing."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DECISION_ENDPOINT", raising=False)
    monkeypatch.delenv("DECISION_MODEL", raising=False)
    monkeypatch.setattr("nooa.agent._try_auto_enable_tracing", lambda: None)
    path = Path(__file__).resolve().parents[1] / "examples/quickstart/16_decisions.py"
    spec = importlib.util.spec_from_file_location("quickstart_decisions_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("endpoint", [None, "", "https://gateway.example/decisions", " "])
@pytest.mark.parametrize("api_key", [None, "test-key"])
async def test_endpoint_configuration_and_setup_guidance(example, monkeypatch, endpoint, api_key):
    """Default empty endpoints, preserve overrides, and reject missing credentials early."""
    if endpoint is not None:
        monkeypatch.setenv("DECISION_ENDPOINT", endpoint)
    if api_key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
    construct = Mock(wraps=example.DecisionClient)
    monkeypatch.setattr(example, "DecisionClient", construct)
    router = SimpleNamespace(
        department=AsyncMock(return_value=example.Department.BILLING),
        confident_department=AsyncMock(
            return_value=SimpleNamespace(
                selected=example.Department.BILLING,
                probabilities={example.Department.BILLING: 0.9},
                confidence=0.8,
                value=example.Department.BILLING,
            )
        ),
        triage=AsyncMock(return_value=example.Triage(department="billing", urgent=False)),
    )
    router_class = Mock(return_value=router)
    monkeypatch.setattr(example, "SupportRouter", router_class)

    if not (endpoint or api_key):
        with pytest.raises(SystemExit) as raised:
            await example.main()
        message = str(raised.value)
        assert "OPENROUTER_API_KEY=... uv run python examples/quickstart/16_decisions.py" in message
        assert "DECISION_ENDPOINT" in message
        assert "DECISION_MODEL" in message
        assert "System One-compatible decisions API" in message
        construct.assert_not_called()
        router_class.assert_not_called()
        return

    await example.main()

    construct.assert_called_once_with(
        "typesafe/jev-1.13",
        endpoint=endpoint or "https://openrouter.ai/api/alpha/decisions",
        api_key=api_key,
    )
    client = router_class.call_args.kwargs["decision_model"]
    assert client.api_style == "systemone"
    assert client._client.is_closed
    for method in (router.department, router.confident_department, router.triage):
        assert method.await_count == len(example.MESSAGES)


async def test_custom_model_is_preserved(example, monkeypatch):
    """Pass DECISION_MODEL through unchanged and close the client after routing."""
    monkeypatch.setenv("DECISION_ENDPOINT", "https://gateway.example/decisions")
    monkeypatch.setenv("DECISION_MODEL", "custom/model")
    construct = Mock(wraps=example.DecisionClient)
    monkeypatch.setattr(example, "DecisionClient", construct)
    monkeypatch.setattr(example, "MESSAGES", [])
    monkeypatch.setattr(example, "SupportRouter", Mock())

    await example.main()

    construct.assert_called_once_with(
        "custom/model", endpoint="https://gateway.example/decisions", api_key=None
    )
    assert example.SupportRouter.call_args.kwargs["decision_model"]._client.is_closed
