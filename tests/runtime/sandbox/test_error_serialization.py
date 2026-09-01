# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for errors crossing the sandbox process boundary."""

from __future__ import annotations

import asyncio
import pickle

import pytest

from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
from nooa.errors.formatting import format_error_for_llm
from nooa.events import _NO_RETURN, ExecutionResult, ExecutionSignal
from nooa.runtime.sandbox.cell_core import run_cell_source
from nooa.runtime.sandbox.errors import (
    CellMemoryError,
    CellSerializationError,
    CellTimeoutError,
    SandboxError,
    SandboxExecutionError,
    WorkerDiedError,
)
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.runtime.sandbox.readonly import SandboxStateError
from nooa.runtime.sandbox.serialization import (
    ErrorDTO,
    ResultDTO,
    dto_to_result,
    result_to_dto,
)


def _result_with_error(error: Exception, *, line_offset: int = 0) -> ExecutionResult:
    return ExecutionResult(error=error, wrapper_line_offset=line_offset)


class _PicklesOnlyOnce:
    """Value whose reducer fails if transport tries to serialize it twice."""

    calls = 0

    def __reduce__(self):
        type(self).calls += 1
        if type(self).calls > 1:
            raise RuntimeError("second pickle explodes")
        return str, ("serialized once",)


class SignalWithResult(ExecutionSignal):
    def __init__(self, result: object) -> None:
        self.result = result


def test_builtin_exception_reconstruction_keeps_worker_diagnostic() -> None:
    diagnostic = "Cell In[8], line 2\n    int('nope')\n    ^^^^^^^^^^^\nValueError: bad value"

    result = dto_to_result(
        ResultDTO(
            # The worker traceback is separate from the concise, source-aware rendering.
            error=ErrorDTO(
                "ValueError",
                "bad value",
                diagnostic,
            )
        )
    )

    assert isinstance(result.error, SandboxExecutionError)
    assert not isinstance(result.error, SandboxError)
    assert isinstance(result.error.original_error, ValueError)
    assert result.error.original_type == "ValueError"
    assert format_error_for_llm(result.error) == diagnostic


def test_custom_exception_uses_surrogate_with_original_type_and_diagnostic() -> None:
    class DomainFailure(Exception):
        pass

    try:
        raise DomainFailure("widget rejected")
    except DomainFailure as error:
        dto = result_to_dto(_result_with_error(error))

    result = dto_to_result(dto)

    assert isinstance(result.error, SandboxExecutionError)
    assert result.error.original_type == "DomainFailure"
    assert str(result.error) == "DomainFailure: widget rejected"
    assert format_error_for_llm(result.error).endswith("DomainFailure: widget rejected")


def test_syntax_error_keeps_formatted_source_and_caret() -> None:
    source = "answer = (1 + )"
    try:
        compile(source, "Cell In[12]", "exec")
    except SyntaxError as error:
        dto = result_to_dto(_result_with_error(error))

    result = dto_to_result(dto)
    diagnostic = format_error_for_llm(result.error)

    assert isinstance(result.error, SandboxExecutionError)
    assert isinstance(result.error.original_error, SyntaxError)
    assert "Cell In[12], line 1" in diagnostic
    assert source in diagnostic
    assert "^" in diagnostic
    assert diagnostic.endswith("SyntaxError: invalid syntax")


@pytest.mark.asyncio
async def test_worker_system_exit_keeps_adjusted_source_context() -> None:
    source = "marker = 1\nraise SystemExit('bye')"
    result = await run_cell_source(source, {}, execution_count=70)

    assert result.error is not None
    formatted = format_error_for_llm(
        result.error,
        source,
        line_offset=result.wrapper_line_offset,
    )
    assert "Cell In[70], line 2" in formatted
    assert "raise SystemExit('bye')" in formatted
    assert "SystemExit: bye" in formatted
    assert "direct cause" in formatted
    assert formatted.endswith(
        "RuntimeError: SystemExit raised inside generated code. Use return_result() "
        "or a normal return to finish a cell, not sys.exit()/exit()/quit()."
    )


@pytest.mark.asyncio
async def test_persisted_worker_helper_keeps_original_source_location() -> None:
    namespace: dict[str, object] = {}
    helper_source = """def boom():
    marker = "helper"
    raise ValueError("x")
"""
    await run_cell_source(helper_source, namespace, execution_count=71)
    namespace["persisted"] = 1

    failed = await run_cell_source("boom()", namespace, execution_count=73)

    assert failed.error is not None
    formatted = format_error_for_llm(
        failed.error,
        "boom()",
        line_offset=failed.wrapper_line_offset,
    )
    assert "Cell In[73], line 1" in formatted
    assert "Cell In[71], line 3, in boom" in formatted
    assert 'raise ValueError("x")' in formatted


def test_worker_formatter_receives_complete_context() -> None:
    calls = []

    def formatter(
        error: Exception,
        code: str | None = None,
        *,
        line_offset: int = 0,
        max_error: int | None = None,
        tail_chars: int | None = None,
    ) -> str:
        calls.append((error, code, line_offset, max_error, tail_chars))
        return f"worker diagnostic with tail {tail_chars}"

    dto = result_to_dto(
        _result_with_error(ValueError("failure"), line_offset=3),
        error_formatter=formatter,
        max_error=100,
        tail_chars=17,
    )

    assert dto.error is not None
    assert dto.error.diagnostic == "worker diagnostic with tail 17"
    assert len(calls) == 1
    error, code, line_offset, max_error, tail_chars = calls[0]
    assert isinstance(error, ValueError)
    assert str(error) == "failure"
    assert code is None
    assert line_offset == 3
    assert max_error == 100
    assert tail_chars == 17


def test_formatter_failure_does_not_destroy_worker_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import nooa.errors.formatting as formatting

    def fail_to_format(*args: object, **kwargs: object) -> str:
        raise RuntimeError("formatter failed")

    monkeypatch.setattr(formatting, "format_error_for_llm", fail_to_format)

    dto = result_to_dto(_result_with_error(ValueError("original failure")))

    assert dto.error == ErrorDTO(
        type_name="ValueError",
        message="original failure",
        diagnostic="ValueError: original failure",
    )


class _HostileString(str):
    def rstrip(self, chars: str | None = None) -> str:
        raise RuntimeError("hostile rstrip")


@pytest.mark.parametrize("invalid_result", [None, 123, b"bytes", _HostileString("text")])
def test_non_string_formatter_result_uses_safe_fallback(invalid_result: object) -> None:
    def invalid_formatter(
        error: Exception,
        code: str | None = None,
        *,
        line_offset: int = 0,
        max_error: int | None = None,
        tail_chars: int | None = None,
    ) -> str:
        return invalid_result  # type: ignore[return-value]

    dto = result_to_dto(
        _result_with_error(ValueError("original failure")),
        error_formatter=invalid_formatter,
    )

    assert dto.error is not None
    assert dto.error.diagnostic == "ValueError: original failure"


def test_worker_uses_formatter_captured_before_cell_module_mutation() -> None:
    import nooa.errors.formatting as formatting
    from nooa.runtime.sandbox.worker import _run_one

    original = formatting.format_error_for_llm
    loop = asyncio.new_event_loop()
    try:
        dto = _run_one(
            loop,
            {},
            {
                "code": (
                    "import nooa.errors.formatting as formatting\n"
                    "formatting.format_error_for_llm = "
                    "lambda *args, **kwargs: 'ValueError: forged'\n"
                    "raise RuntimeError('real failure')"
                ),
                "execution_count": 1,
            },
        )
    finally:
        formatting.format_error_for_llm = original
        loop.close()

    assert dto.error is not None
    assert dto.error.type_name == "RuntimeError"
    assert dto.error.message == "real failure"
    assert "RuntimeError: real failure" in dto.error.diagnostic
    assert "forged" not in dto.error.diagnostic


def test_broken_exception_string_still_crosses_worker_boundary() -> None:
    class BrokenStringError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("broken __str__")

    dto = result_to_dto(_result_with_error(BrokenStringError()))

    assert dto.error is not None
    assert dto.error.type_name == "BrokenStringError"
    assert dto.error.message == "BrokenStringError"
    assert dto.error.diagnostic == "BrokenStringError: BrokenStringError"


@pytest.mark.parametrize("invalid_limit", [0, -1, True])
def test_invalid_error_limit_falls_back_to_default(invalid_limit: object) -> None:
    dto = result_to_dto(
        _result_with_error(RuntimeError("x" * 20_000)),
        max_error=invalid_limit,  # type: ignore[arg-type]
    )

    assert dto.error is not None
    assert "Showing first 5,000 and last 5,000 chars" in dto.error.message


@pytest.mark.parametrize("invalid_tail", [-1, True])
def test_invalid_error_tail_falls_back_to_valid_split(invalid_tail: object) -> None:
    dto = result_to_dto(
        _result_with_error(RuntimeError("x" * 1_000)),
        max_error=100,
        tail_chars=invalid_tail,  # type: ignore[arg-type]
    )

    assert dto.error is not None
    assert "Showing first 50 and last 50 chars" in dto.error.message
    assert dto.error.message.endswith("x" * 50 + "\n</truncated-output>")


def test_error_dto_text_respects_configured_capture_limit_before_transport_limit() -> None:
    """Text above a custom content budget is truncated even if the DTO could carry it."""
    max_error = 100
    dto = result_to_dto(
        _result_with_error(RuntimeError("x" * (max_error + 1))),
        max_error=max_error,
    )

    assert dto.error is not None
    assert "Showing first 50 and last 50 chars" in dto.error.message
    assert len(dto.error.message) <= max_error + 1_024


def test_worker_formatter_output_is_only_subject_to_transport_ceiling() -> None:
    dto = result_to_dto(
        _result_with_error(RuntimeError("failure")),
        error_formatter=lambda error, code=None, *, line_offset=0, max_error=None, tail_chars=None: (
            "L" * 1_000
        ),
        max_error=100,
    )

    assert dto.error is not None
    assert dto.error.diagnostic == "L" * 1_000


def test_formatter_failure_fallback_is_transport_bounded() -> None:
    def broken_formatter(
        error: Exception,
        code: str | None = None,
        *,
        line_offset: int = 0,
        max_error: int | None = None,
        tail_chars: int | None = None,
    ) -> str:
        raise RuntimeError("formatter failed")

    dto = result_to_dto(
        _result_with_error(RuntimeError("F" * 1_000)),
        error_formatter=broken_formatter,
        max_error=100,
    )

    assert dto.error is not None
    assert dto.error.diagnostic == "RuntimeError: " + "F" * 1_000


def test_worker_formatted_envelope_is_not_nested() -> None:
    dto = result_to_dto(
        _result_with_error(RuntimeError("X" * 1_000)),
        max_error=100,
    )

    assert dto.error is not None
    assert dto.error.diagnostic.count("<truncated-output>") == 1
    assert dto.error.diagnostic.count("</truncated-output>") == 1


def test_formatter_owned_envelope_is_not_rebounded_by_transport() -> None:
    from nooa.errors.formatting import format_error_for_llm

    larger = format_error_for_llm(RuntimeError("X" * 2_000), max_error=400)
    dto = result_to_dto(
        _result_with_error(RuntimeError("surrogate")),
        error_formatter=lambda error, code=None, *, line_offset=0, max_error=None, tail_chars=None: (
            larger
        ),
        max_error=100,
    )

    assert dto.error is not None
    assert dto.error.diagnostic == larger


def test_error_dto_text_never_exceeds_transport_cap() -> None:
    max_transport = DEFAULT_TRUNCATION_CONFIG.capture.max_error + 1_024
    dto = result_to_dto(
        _result_with_error(RuntimeError("x" * 100_000)),
        max_error=max_transport * 2,
    )

    assert dto.error is not None
    assert len(dto.error.message) <= max_transport
    assert len(dto.error.diagnostic) <= max_transport


def test_error_dto_text_is_bounded_before_ipc() -> None:
    max_error = DEFAULT_TRUNCATION_CONFIG.capture.max_error
    configured_tail = DEFAULT_TRUNCATION_CONFIG.capture.tail
    tail = max_error // 2 if configured_tail is None else configured_tail
    dto = result_to_dto(_result_with_error(RuntimeError("x" * 100_000)))

    assert dto.error is not None
    max_transport = max_error + 1_024
    assert len(dto.error.message) <= max_transport
    assert len(dto.error.diagnostic) <= max_transport
    assert "<truncated-output>" in dto.error.message
    assert dto.error.message.endswith("x" * tail + "\n</truncated-output>")


@pytest.mark.parametrize("raised", [KeyboardInterrupt("interrupt"), SystemExit("exit")])
def test_base_exception_from_exception_string_does_not_escape_serialization(raised) -> None:
    class BrokenStringError(Exception):
        def __str__(self) -> str:
            raise raised

    dto = result_to_dto(_result_with_error(BrokenStringError()))

    assert dto.error is not None
    assert dto.error.message == "BrokenStringError"
    assert dto.error.diagnostic == "BrokenStringError: BrokenStringError"


def test_ordinary_return_is_pickled_only_once_before_transport() -> None:
    _PicklesOnlyOnce.calls = 0
    dto = result_to_dto(ExecutionResult(returned_value=_PicklesOnlyOnce()))

    transported_dto = pickle.loads(pickle.dumps(dto))
    result = dto_to_result(transported_dto)

    assert _PicklesOnlyOnce.calls == 1
    assert result.returned_value == "serialized once"


def test_return_result_payload_is_pickled_only_once_before_transport() -> None:
    class Signal(ExecutionSignal):
        def __init__(self) -> None:
            self.result = _PicklesOnlyOnce()

    _PicklesOnlyOnce.calls = 0
    dto = result_to_dto(ExecutionResult(signal=Signal()))

    transported_dto = pickle.loads(pickle.dumps(dto))
    result = dto_to_result(
        transported_dto,
        signal_factory=lambda value: SignalWithResult(value),
    )

    assert _PicklesOnlyOnce.calls == 1
    assert result.signal is not None
    assert result.signal.result == "serialized once"


def test_unpicklable_ordinary_return_becomes_serialization_error() -> None:
    def returned_value() -> None:
        pass

    dto = result_to_dto(ExecutionResult(returned_value=returned_value))
    result = dto_to_result(dto)

    assert isinstance(result.error, CellSerializationError)
    assert "Return value of type 'function' is not picklable" in str(result.error)
    assert "sandbox boundary" in str(result.error)
    assert result.returned_value is _NO_RETURN


def test_unpicklable_return_result_payload_becomes_serialization_error() -> None:
    class Signal(ExecutionSignal):
        def __init__(self) -> None:
            self.result = lambda: None

    dto = result_to_dto(ExecutionResult(signal=Signal()))
    result = dto_to_result(dto)

    assert isinstance(result.error, CellSerializationError)
    assert "return_result(...)" in str(result.error)
    assert "JSON/pickle-safe value" in str(result.error)
    assert result.signal is None


@pytest.mark.asyncio
async def test_broker_error_with_hostile_string_is_safe_and_bounded() -> None:
    class HostileError(Exception):
        def __str__(self) -> str:
            raise KeyboardInterrupt("hostile string")

    class Target:
        def explode(self) -> None:
            raise HostileError()

    executor = object.__new__(SandboxedExecutor)
    executor._agent = Target()
    executor._max_error = DEFAULT_TRUNCATION_CONFIG.capture.max_error

    response = await executor._dispatch_tool_call(
        {"kind": "call", "path": ["explode"], "args": (), "kwargs": {}}
    )

    assert response == {"ok": False, "error_type": "HostileError", "error": "HostileError"}


@pytest.mark.asyncio
async def test_broker_error_text_is_bounded_before_ipc() -> None:
    class Target:
        def explode(self) -> None:
            raise RuntimeError("x" * 100_000)

    executor = object.__new__(SandboxedExecutor)
    executor._agent = Target()
    executor._max_error = DEFAULT_TRUNCATION_CONFIG.capture.max_error

    response = await executor._dispatch_tool_call(
        {"kind": "call", "path": ["explode"], "args": (), "kwargs": {}}
    )

    assert response["ok"] is False
    assert len(response["error"]) <= DEFAULT_TRUNCATION_CONFIG.capture.max_error
    assert response["error"].endswith("...<truncated>")


def test_broker_error_with_multi_argument_builtin_uses_surrogate() -> None:
    from nooa.runtime.sandbox.worker import ParentToolError, _raise_broker_error

    with pytest.raises(ParentToolError) as caught:
        _raise_broker_error(
            {
                "error_type": "UnicodeDecodeError",
                "error": "codec failed",
                "call_hint": "decode(data: bytes)",
            }
        )

    assert str(caught.value) == "codec failed"
    assert caught.value.original_type == "UnicodeDecodeError"  # type: ignore[attr-defined]
    assert caught.value._nooa_call_hint == "decode(data: bytes)"  # type: ignore[attr-defined]


def test_sandbox_state_error_reconstructs_concrete_type_without_duplicate_prefix() -> None:
    result = dto_to_result(
        ResultDTO(
            error=ErrorDTO(
                "SandboxStateError",
                "SandboxStateError: cannot mutate module-level state 'CACHE'",
            )
        )
    )

    assert isinstance(result.error, SandboxStateError)
    assert str(result.error) == "cannot mutate module-level state 'CACHE'"


@pytest.mark.parametrize(
    ("error", "actionable"),
    [
        (CellTimeoutError("cell exceeded its 2s deadline and was killed"), "2s deadline"),
        (CellMemoryError("worker was killed; reduce memory use"), "reduce memory"),
        (WorkerDiedError("sandbox worker exited unexpectedly"), "exited unexpectedly"),
    ],
)
def test_synthetic_boundary_errors_are_concise_and_do_not_fabricate_source(
    error: Exception, actionable: str
) -> None:
    executor = object.__new__(SandboxedExecutor)
    executor._disabled = True

    result = executor._synth_error(error)
    diagnostic = format_error_for_llm(result.error)

    assert actionable in diagnostic
    assert diagnostic == f"{type(error).__name__}: {error}"
    assert "Cell In[" not in diagnostic
    assert "Traceback" not in diagnostic
    assert "^" not in diagnostic
