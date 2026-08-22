# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for errors crossing the sandbox process boundary."""

from __future__ import annotations

import asyncio

import pytest

from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
from nooa.errors.formatting import format_error_for_llm
from nooa.events import _NO_RETURN, ExecutionResult, ExecutionSignal
from nooa.runtime.sandbox.errors import (
    CellMemoryError,
    CellSerializationError,
    CellTimeoutError,
    WorkerDiedError,
)
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.runtime.sandbox.readonly import SandboxStateError
from nooa.runtime.sandbox.serialization import (
    ErrorDTO,
    ResultDTO,
    _SurrogateCellError,
    dto_to_result,
    result_to_dto,
)


def _result_with_error(error: Exception, *, line_offset: int = 0) -> ExecutionResult:
    return ExecutionResult(error=error, wrapper_line_offset=line_offset)


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

    assert isinstance(result.error, ValueError)
    assert result.formatted_error == diagnostic
    assert format_error_for_llm(result.error, formatted_error=result.formatted_error) == diagnostic


def test_custom_exception_uses_surrogate_with_original_type_and_diagnostic() -> None:
    class DomainFailure(Exception):
        pass

    try:
        raise DomainFailure("widget rejected")
    except DomainFailure as error:
        dto = result_to_dto(_result_with_error(error))

    result = dto_to_result(dto)

    assert isinstance(result.error, _SurrogateCellError)
    assert result.error.original_type == "DomainFailure"  # type: ignore[union-attr]
    assert str(result.error) == "widget rejected"
    assert format_error_for_llm(result.error, formatted_error=result.formatted_error).endswith(
        "DomainFailure: widget rejected"
    )


def test_syntax_error_keeps_formatted_source_and_caret() -> None:
    source = "answer = (1 + )"
    try:
        compile(source, "Cell In[12]", "exec")
    except SyntaxError as error:
        dto = result_to_dto(_result_with_error(error))

    result = dto_to_result(dto)
    diagnostic = format_error_for_llm(result.error, formatted_error=result.formatted_error)

    assert isinstance(result.error, SyntaxError)
    assert "Cell In[12], line 1" in diagnostic
    assert source in diagnostic
    assert "^" in diagnostic
    assert diagnostic.endswith("SyntaxError: invalid syntax")


def test_formatter_failure_does_not_destroy_worker_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import nooa.errors.formatting as formatting

    def fail_to_format(*args: object, **kwargs: object) -> str:
        raise RuntimeError("formatter failed")

    monkeypatch.setattr(formatting, "format_error_for_llm", fail_to_format)

    dto = result_to_dto(_result_with_error(ValueError("original failure")))

    assert dto.error == ErrorDTO(
        type_name="ValueError",
        message="original failure",
        formatted_error="ValueError: original failure",
    )


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
    assert "RuntimeError: real failure" in dto.error.formatted_error
    assert "forged" not in dto.error.formatted_error


def test_broken_exception_string_still_crosses_worker_boundary() -> None:
    class BrokenStringError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("broken __str__")

    dto = result_to_dto(_result_with_error(BrokenStringError()))

    assert dto.error is not None
    assert dto.error.type_name == "BrokenStringError"
    assert dto.error.message == "BrokenStringError"
    assert dto.error.formatted_error == "BrokenStringError: BrokenStringError"


def test_error_dto_text_is_bounded_before_ipc() -> None:
    max_error = DEFAULT_TRUNCATION_CONFIG.capture.max_error
    configured_tail = DEFAULT_TRUNCATION_CONFIG.capture.tail
    tail = max_error // 2 if configured_tail is None else configured_tail
    dto = result_to_dto(_result_with_error(RuntimeError("x" * 100_000)))

    assert dto.error is not None
    max_transport = max_error + 1_024
    assert len(dto.error.message) <= max_transport
    assert len(dto.error.formatted_error) <= max_transport
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
    assert dto.error.formatted_error == "BrokenStringError: BrokenStringError"


@pytest.mark.parametrize(
    ("returned_value", "expected"),
    [
        (lambda: None, "Return value of type 'function' is not picklable"),
    ],
)
def test_unpicklable_ordinary_return_becomes_serialization_error(
    returned_value: object, expected: str
) -> None:
    dto = result_to_dto(ExecutionResult(returned_value=returned_value))
    result = dto_to_result(dto)

    assert isinstance(result.error, CellSerializationError)
    assert expected in str(result.error)
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
