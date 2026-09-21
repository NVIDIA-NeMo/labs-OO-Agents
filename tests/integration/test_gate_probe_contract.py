# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for live-gate diagnostics and exporter isolation."""

from itertools import permutations
from types import SimpleNamespace

import pytest

from tests.integration.test_open_model_tool_reasoning_live import (
    readable_seed_reasoning,
    seed_messages,
)


def test_glm_task_requires_a_derived_key_with_unique_optimum():
    prompt = seed_messages("glm")[1]["content"]
    durations = {"A": 3, "B": 2, "C": 4, "D": 1}
    weights = {"A": 3, "B": 6, "C": 2, "D": 4}
    for values in (durations, weights):
        assert ", ".join(f"{key}={value}" for key, value in values.items()) in prompt
    assert "A must precede C and B must precede D" in prompt
    assert "key must be the four job letters in that order" in prompt
    scores = []
    for order in permutations(durations):
        if order.index("A") > order.index("C") or order.index("B") > order.index("D"):
            continue
        elapsed = cost = 0
        for job in order:
            elapsed += durations[job]
            cost += weights[job] * elapsed
        scores.append((cost, "".join(order)))
    scores.sort()
    assert scores[0] == (62, "BDAC")
    assert scores[1][0] > scores[0][0]
    assert "BDAC" not in prompt


@pytest.mark.parametrize("family", ["deepseek", "kimi", "qwen"])
def test_other_gate_prompts_are_unchanged(family):
    assert seed_messages(family)[1]["content"] == (
        "Is 17 times 19 less than 18 squared plus offset? First call lookup with "
        "key=offset; do not answer until the tool returns. Then give the difference."
    )


@pytest.mark.parametrize(
    "fields",
    [{}, {"reasoning_content": None}, {"reasoning_content": ""}, {"reasoning_content": " "}],
)
def test_absent_reasoning_is_an_informative_assertion(fields):
    seed = SimpleNamespace(
        raw_response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(**fields))]),
        finish_reason="tool_calls",
    )
    with pytest.raises(
        AssertionError, match="no nonempty reasoning_content.*finish_reason=tool_calls"
    ):
        readable_seed_reasoning(seed)


def test_reasoning_is_kept_byte_for_byte():
    raw = "  synthetic reasoning\n"
    seed = SimpleNamespace(
        raw_response=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(reasoning_content=raw))]
        ),
        finish_reason="tool_calls",
    )
    assert readable_seed_reasoning(seed) == raw


@pytest.mark.parametrize("endpoint", [None, "http://viewer.example/v1/traces"])
def test_gate_never_discovers_or_registers_exporters(isolated_gate_tracing, monkeypatch, endpoint):
    import nooa.tracing as tracing
    from nooa.tracing import _llm_hooks
    from nooa.tracing._session_processor import SessionSpanProcessor

    if endpoint is None:
        monkeypatch.delenv("OTLP_ENDPOINT", raising=False)
    else:
        monkeypatch.setenv("OTLP_ENDPOINT", endpoint)
    monkeypatch.setattr(tracing, "probe_otlp_endpoint", lambda *_: pytest.fail("viewer probe"))
    # Exercise startup after isolation, not the fixture's stubbed return value.
    monkeypatch.setattr(tracing, "_default_exporters", lambda: pytest.fail("exporter discovery"))
    tracing.enable_tracing()  # Same no-argument path used by Agent startup.
    with tracing._provider.get_tracer("gate-test").start_as_current_span("gate-test"):
        pass
    tracing._provider.force_flush()
    assert not _llm_hooks.callbacks
    processors = tracing._provider._active_span_processor._span_processors
    assert all(isinstance(processor, SessionSpanProcessor) for processor in processors)
