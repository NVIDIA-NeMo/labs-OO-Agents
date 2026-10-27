# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Developer warnings report actual omissions without inspecting rendered text."""

import asyncio
import io
import sys
import warnings
from dataclasses import dataclass

import pytest

from nooa.agentdoc import pformat
from nooa.agentdoc._format_observation import observe_truncation
from nooa.runtime.actor import ContextVarStream, _stderr_buffer_var
from nooa.strategies.current_call import CurrentCall
from nooa.strategies.prefill import PrefillTruncationWarning, warn_on_prefill_truncation


@pytest.mark.parametrize(
    ("value", "limits", "reason"),
    [
        ("x" * 30, {"max_string": 10}, "max_string"),
        (["x" * 30], {"max_string": 10}, "max_string"),
        (list(range(30)), {"max_length": 10}, "max_length"),
        ({str(i): i for i in range(30)}, {"max_length": 10}, "max_length"),
        ([[1]], {"max_depth": 1}, "max_depth"),
    ],
)
def test_warns_for_actual_formatting_omissions(value, limits, reason):
    """Each structural limit produces one warning with actionable advice."""
    call = CurrentCall(id="test", method_name="analyze", decorator="strategy")
    with pytest.warns(PrefillTruncationWarning, match=reason) as caught:
        with warn_on_prefill_truncation(call):
            pformat(value, **limits)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "analyze()" in message
    assert "prefill_format" in message
    assert "spec(" in message
    assert "REPL variables" in message
    assert "x" * 30 not in message


@pytest.mark.parametrize("value", ["str(len=999, ...)", "...", [1, 2], [[]], 42])
def test_complete_values_do_not_warn(value):
    """Marker-like input, empty containers, and scalar depth checks aren't omissions."""
    call = CurrentCall(id="test", method_name="analyze", decorator="strategy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with warn_on_prefill_truncation(call):
            pformat(value, max_depth=1)
    assert not caught


def test_structured_fields_report_omissions():
    """Structured objects participate in the same observation as containers."""

    @dataclass
    class Data:
        first: str = "long value"
        second: int = 2

    with observe_truncation() as reasons:
        pformat(Data(), max_string=3, max_length=1)
    assert reasons == {"max_string", "max_length"}


def test_dictionary_key_warning_explains_fixed_limit():
    """Dictionary key previews aren't controlled by prefill_format.max_string."""
    call = CurrentCall(id="test", method_name="analyze", decorator="strategy")
    with pytest.warns(PrefillTruncationWarning) as caught:
        with warn_on_prefill_truncation(call):
            pformat({"k" * 100: 1})
    message = str(caught[0].message)
    assert "50-character preview limit" in message
    assert "prefill_format" not in message


@pytest.mark.asyncio
async def test_worker_omissions_survive_transport():
    """A worker reports omissions to its parent rather than warning inside its stderr."""
    from nooa.runtime.pprint import pprint
    from nooa.runtime.sandbox.cell_core import run_cell_source
    from nooa.runtime.sandbox.serialization import (
        dto_from_wire,
        dto_to_result,
        dto_to_wire,
        result_to_dto,
    )
    from nooa.runtime.sandbox.wire import Codec

    result = await run_cell_source(
        "pprint(data, max_string=3)", {"pprint": pprint, "data": "long input"}
    )
    assert result.success
    assert result.truncation_reasons == ["max_string"]
    assert not result.stderr
    codec = Codec()
    transported = dto_from_wire(codec.loads(codec.dumps(dto_to_wire(result_to_dto(result)))))
    assert dto_to_result(transported).truncation_reasons == ["max_string"]


@pytest.mark.parametrize("reason", [123, "untrusted\nwarning", ["max_string"]])
def test_worker_omission_names_are_validated(reason):
    """Untrusted worker metadata cannot inject arbitrary text into developer warnings."""
    from nooa.runtime.sandbox.errors import CellSerializationError
    from nooa.runtime.sandbox.serialization import dto_from_wire

    with pytest.raises(CellSerializationError):
        dto_from_wire({"truncation_reasons": [reason]})


def test_warning_bypasses_model_stderr(monkeypatch):
    """A nested call's warning goes to the developer, not an enclosing REPL cell."""

    def showwarning(message, category, filename, lineno, file=None, line=None):
        """Exercise Python's stderr delivery instead of pytest's warning recorder."""
        sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))

    developer_stderr = io.StringIO()
    model_stderr = io.StringIO()
    monkeypatch.setattr(
        "sys.stderr", ContextVarStream(developer_stderr, _stderr_buffer_var, "stderr")
    )
    token = _stderr_buffer_var.set(model_stderr)
    try:
        call = CurrentCall(id="test", method_name="analyze", decorator="strategy")
        with warnings.catch_warnings():
            # pytest records warnings instead of using Python's stderr writer.
            # Exercise the actual default warning delivery for this test.
            monkeypatch.setattr(warnings, "showwarning", showwarning)
            warnings.simplefilter("always", PrefillTruncationWarning)
            with warn_on_prefill_truncation(call):
                pformat("long input", max_string=3)
        assert "PrefillTruncationWarning" in developer_stderr.getvalue()
        assert not model_stderr.getvalue()
        assert _stderr_buffer_var.get() is model_stderr
    finally:
        _stderr_buffer_var.reset(token)


@pytest.mark.asyncio
async def test_observations_are_task_local_and_reset_after_errors():
    """Concurrent and failed inspections cannot leak state into other calls."""

    async def render(value):
        """Observe one task's formatting across a scheduling boundary."""
        with observe_truncation() as reasons:
            await asyncio.sleep(0)
            pformat(value, max_string=3)
        return reasons

    assert await asyncio.gather(render("long input"), render("ok")) == [{"max_string"}, set()]
    with pytest.raises(ValueError):
        with observe_truncation():
            raise ValueError("failed")
    assert await render("ok") == set()
