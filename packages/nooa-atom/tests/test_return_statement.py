# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An Atom cell ends the turn with a top-level ``return``, not ``return_result``.

A ``return x`` at the cell's top level (not in a nested def/lambda/class)
ends the turn with ``x``, validated like core's ``return_result(x)``: an
invalid value goes back to the model as a cell error naming the form to
return; the turn fails only after ``_MAX_RESULT_RETRIES``. A bare
``return`` is refused before the cell runs, a return that does not execute
does not end the turn, and ``return_result`` is not defined (a call says
what ends a turn and is counted as ``harness.return_result_called.count``).
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


def _merged(flushed, prefix="harness.return"):
    merged: dict = {}
    for attrs in flushed:
        for key, value in attrs.items():
            if key.startswith(prefix):
                merged[key] = value
    return merged


def _cell_output(models, call):
    return str(models.llms[None].calls[call].messages[-2])


async def _prompt(registry, options, text="hi"):
    root = await registry.create(options)
    return await asyncio.wait_for(root.prompt(text), TIMEOUT)


async def test_top_level_return_ends_the_turn_with_its_value(
    registry, options, models, span_attributes
):
    models.scripts[None] = [cell("x = 'answered'\nreturn Done(explanation=x)")]
    assert await _prompt(registry, options) == Done(explanation="answered")
    assert len(models.llms[None].calls) == 1
    attrs = _merged(span_attributes, "harness.")
    assert attrs["harness.explicit_return_auto_completed"] == 1


async def test_return_in_a_compound_statement_ends_the_turn(registry, options, models):
    models.scripts[None] = [
        cell("for i in range(3):\n    if i == 1:\n        return Done(explanation=str(i))")
    ]
    assert await _prompt(registry, options) == Done(explanation="1")


async def test_a_return_that_does_not_run_does_not_end_the_turn(registry, options, models):
    # Falling off the end is not ``return None``: no result, no retry spent.
    models.scripts[None] = [
        cell("if False:\n    return Done(explanation='never')\nprint('fell through')"),
        done("next cell"),
    ]
    assert await _prompt(registry, options) == Done(explanation="next cell")
    output = _cell_output(models, 1)
    assert "fell through" in output
    assert "ResultStatus.COMPLETE" in output
    assert "attempt" not in output


async def test_nested_def_return_does_not_end_the_turn(registry, options, models):
    models.scripts[None] = [
        cell(
            "def make():\n    return Done(explanation='inner')\n"
            "inner = make()\nfirst = (lambda: 1)()\n"
            "class K:\n    def m(self):\n        return 2\n"
            "print(inner.explanation, first, K().m())"
        ),
        cell("return Done(explanation=inner.explanation + ' returned later')"),
    ]
    assert await _prompt(registry, options) == Done(explanation="inner returned later")
    assert "inner 1 2" in _cell_output(models, 1)


async def test_trailing_expression_does_not_end_the_turn(registry, options, models):
    models.scripts[None] = [cell("Done(explanation='not returned')"), done("returned")]
    assert await _prompt(registry, options) == Done(explanation="returned")


async def test_bare_return_is_refused_before_the_cell_runs(registry, options, models):
    models.scripts[None] = [
        cell("self.v.ran = True\nreturn"),
        cell("return Done(explanation=str(getattr(self.v, 'ran', False)))"),
    ]
    assert await _prompt(registry, options) == Done(explanation="False")
    error = _cell_output(models, 1)
    assert "ResultStatus.ERROR" in error
    assert "a bare `return` (line 2) gives the turn no result" in error
    assert "`return Done(message=..., explanation=...)`" in error


async def test_invalid_values_are_retryable_cell_errors(registry, options, models, span_attributes):
    models.scripts[None] = [
        cell("return 'a'"),
        cell("return 'b'"),
        cell("return None"),
        done("fixed"),
    ]
    assert await _prompt(registry, options) == Done(explanation="fixed")
    assert len(models.llms[None].calls) == 4
    error = _cell_output(models, 1)
    assert "ResultStatus.ERROR" in error
    assert "return validation error" in error
    assert (
        "Write it as one of: return Done(explanation=...), "
        "return NeedInput(question=...), return Waiting(explanation=..., on=...)"
    ) in error
    assert f"fix the value and return it again (attempt 1 of {_MAX_RESULT_RETRIES})" in error
    assert f"(attempt 3 of {_MAX_RESULT_RETRIES})" in _cell_output(models, 3)
    assert "return_result" not in error
    attrs = _merged(span_attributes)
    assert attrs["harness.return_result_invalid.count"] == 3
    assert attrs["harness.return_result_invalid.types"] == ["ValidationError"] * 3


async def test_invalid_values_stay_bounded(registry, options, models):
    from nooa_atom.session.session import TurnFailedError

    models.scripts[None] = [cell(f"return {i}") for i in range(_MAX_RESULT_RETRIES)]
    with pytest.raises(TurnFailedError, match=f"failed after {_MAX_RESULT_RETRIES} attempts"):
        await _prompt(registry, options)
    assert len(models.llms[None].calls) == _MAX_RESULT_RETRIES


async def test_role_postconditions_still_apply_to_a_returned_value(registry, options, models):
    # A Waiting on nothing that runs is rejected as before; the turn goes on.
    models.scripts[None] = [
        cell("return Waiting(explanation='w', on=['nothing'])"),
        done("ok"),
    ]
    assert await _prompt(registry, options) == Done(explanation="ok")
    error = _cell_output(models, 1)
    assert "no running job has that name" in error
    assert "(attempt 1 of" in error


async def test_return_result_is_not_defined_and_is_counted(
    registry, options, models, span_attributes
):
    models.scripts[None] = [
        cell("return_result(Done(explanation='old habit'))"),
        done("returned"),
    ]
    assert await _prompt(registry, options) == Done(explanation="returned")
    error = _cell_output(models, 1)
    assert "NameError" in error
    assert "end the turn with `return Done(...)`" in error
    attrs = _merged(span_attributes)
    assert attrs["harness.return_result_called.count"] == 1
    assert "harness.return_result_invalid.count" not in attrs
    assert not _merged(span_attributes, "harness.name_error_hint.count")


async def test_prompt_never_names_return_result(registry, options, models):
    models.scripts[None] = [done()]
    await _prompt(registry, options)
    call = models.llms[None].calls[0]
    rendered = "\n".join(str(message.get("content", "")) for message in call.messages)
    tools = "\n".join(tool.description for tool in models.llms[None].last_tools)
    assert "return_result" not in rendered
    assert "return_result" not in tools
    assert "To finish, `return value` at the cell's top level (not inside a def)." in tools
