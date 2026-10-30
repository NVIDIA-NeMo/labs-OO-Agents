# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""return_result fix-ups in a real AtomAgent turn: keyword aliases and invalid values.

``return_result(value=x)`` reads as ``return_result(x)`` when the return type
has no ``value`` field. An invalid value goes back to the model as a cell
error naming the call form; the turn fails only after ``_MAX_RESULT_RETRIES``.
Both are counted as harness metrics on the turn's span.
"""

import asyncio

import pytest
from atom_test_agents import AtomModels, cell, done
from nooa_atom.agent.agent import _MAX_RESULT_RETRIES
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done
from nooa.runtime import harness_metrics

UNIFIED = "nooa_atom.agent:AtomAgent"
TIMEOUT = 60


@pytest.fixture
def models():
    return AtomModels()


@pytest.fixture
async def registry(sessions_dir, models):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    yield registry
    await registry.close_all()


@pytest.fixture
def options(tmp_path, sessions_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SessionOptions(workspace=workspace, agent_spec=UNIFIED, sessions_dir=sessions_dir)


@pytest.fixture
def span_attributes(monkeypatch):
    """The attributes each harness-metrics session flushes to its span."""
    flushed: list[dict] = []
    flush = harness_metrics.HarnessMetrics.flush_to_span

    def spy(self):
        flushed.append(self.to_span_attributes())
        return flush(self)

    monkeypatch.setattr(harness_metrics.HarnessMetrics, "flush_to_span", spy)
    return flushed


def _merged(flushed):
    merged: dict = {}
    for attrs in flushed:
        for key, value in attrs.items():
            if key.startswith("harness.return_result"):
                merged[key] = value
    return merged


def _cell_output(models, call):
    return str(models.llms[None].calls[call].messages[-2])


async def test_value_keyword_is_read_as_the_result(registry, options, models, span_attributes):
    models.scripts[None] = [cell("return_result(value=Done(explanation='aliased'))")]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("hi"), TIMEOUT) == Done(explanation="aliased")
    assert len(models.llms[None].calls) == 1
    attrs = _merged(span_attributes)
    assert attrs["harness.return_result_keyword_alias.count"] == 1
    assert attrs["harness.return_result_keyword_alias.keywords"] == ["value"]
    assert "harness.return_result_invalid.count" not in attrs


async def test_result_keyword_keeps_its_meaning(registry, options, models, span_attributes):
    # Done has a ``result`` field, so result= stays the wrapper key (as before).
    models.scripts[None] = [cell("return_result(result=Done(explanation='as before'))")]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("hi"), TIMEOUT) == Done(explanation="as before")
    assert "harness.return_result_keyword_alias.count" not in _merged(span_attributes)


async def test_invalid_values_are_retryable_cell_errors(registry, options, models, span_attributes):
    # Before: the third invalid value in a turn failed it (GenerationError,
    # "Did not finish" in ACP). Now each is a cell error naming the call form.
    models.scripts[None] = [
        cell("return_result('a')"),
        cell("return_result('b')"),
        cell("return_result('c')"),
        done("fixed"),
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("hi"), TIMEOUT) == Done(explanation="fixed")
    assert len(models.llms[None].calls) == 4
    error = _cell_output(models, 1)
    assert "ResultStatus.ERROR" in error
    assert (
        "Call it as one of: return_result(Done(explanation=...)), "
        "return_result(NeedInput(question=...)), return_result(Waiting(explanation=..., on=...))"
    ) in error
    assert f"(attempt 1 of {_MAX_RESULT_RETRIES})" in error
    assert f"(attempt 3 of {_MAX_RESULT_RETRIES})" in _cell_output(models, 3)
    attrs = _merged(span_attributes)
    assert attrs["harness.return_result_invalid.count"] == 3
    assert attrs["harness.return_result_invalid.types"] == ["ValidationError"] * 3


async def test_invalid_values_stay_bounded(registry, options, models):
    from nooa_atom.session.session import TurnFailedError

    models.scripts[None] = [cell(f"return_result({i})") for i in range(_MAX_RESULT_RETRIES)]
    root = await registry.create(options)
    with pytest.raises(TurnFailedError, match=f"failed after {_MAX_RESULT_RETRIES} attempts"):
        await asyncio.wait_for(root.prompt("hi"), TIMEOUT)
    assert len(models.llms[None].calls) == _MAX_RESULT_RETRIES
