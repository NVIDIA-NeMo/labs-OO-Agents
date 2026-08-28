# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IPC-safe marshaling of execution results across the process boundary.

``ExecutionResult`` carries fields that cannot be pickled (live callables,
arbitrary return values, exceptions). This module converts a worker-side result
into a picklable :class:`ResultDTO` and reconstructs a faithful
``ExecutionResult`` on the parent, following the contract in the design doc:

* ``defined_methods`` / ``captured_locals`` stay in the worker (empty on parent).
* ``returned_value`` crosses only if picklable, else becomes a serialization error.
* ``error`` is reduced to type/message plus a worker-formatted diagnostic and
  reconstructed as a lightweight parent-side exception.
* ``signal`` (``return_result``) is marshaled as a picklable record.
* ``images`` are already dicts.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Protocol

from nooa.agentdoc import TruncatingStringIO
from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
from nooa.errors.formatting import _diagnostic_budget, _hard_bound_text
from nooa.runtime.sandbox.errors import CellSerializationError, SandboxExecutionError

# ``TruncatingStringIO`` adds a human-readable envelope around retained
# head/tail content. Sandbox IPC has a fixed safety ceiling independent of a
# caller's configured capture budget.
_MAX_ERROR_CONTENT = DEFAULT_TRUNCATION_CONFIG.capture.max_error
_MAX_ERROR_TRANSPORT = _MAX_ERROR_CONTENT + 1_024


class _ErrorFormatter(Protocol):
    """Complete formatter callable contract used inside the sandbox worker."""

    def __call__(
        self,
        error: Exception,
        code: str | None = None,
        *,
        line_offset: int = 0,
        max_error: int | None = None,
        tail_chars: int | None = None,
    ) -> str: ...


def effective_error_limit(max_error: int | None) -> int:
    """Return a valid diagnostic budget clamped to the sandbox IPC ceiling."""
    requested = (
        max_error
        if isinstance(max_error, int) and not isinstance(max_error, bool) and max_error > 0
        else _MAX_ERROR_CONTENT
    )
    return min(requested, _MAX_ERROR_CONTENT)


def _bounded_error_message(
    value: str,
    *,
    max_error: int | None = None,
    tail_chars: int | None = None,
) -> str:
    """Apply the effective capture policy once to a raw message before IPC."""
    content_limit = effective_error_limit(max_error)
    _, effective_tail = _diagnostic_budget(content_limit, tail_chars)
    if len(value) <= content_limit:
        return value
    stream = TruncatingStringIO(limit=content_limit, tail_chars=effective_tail)
    stream.write(value)
    return _hard_bound_text(stream.getvalue(), _MAX_ERROR_TRANSPORT)


def _bounded_diagnostic(value: str, *, max_error: int | None = None) -> str:
    """Enforce the IPC ceiling after the worker formatter applies capture policy."""
    content_limit = effective_error_limit(max_error)
    transport_limit = min(_MAX_ERROR_TRANSPORT, content_limit + 1_024)
    return _hard_bound_text(
        value.rstrip(),
        transport_limit,
        closing="\n</truncated-output>",
    )


def is_picklable(value: Any) -> bool:
    try:
        pickle.dumps(value)
        return True
    except BaseException:
        return False


@dataclass
class ErrorDTO:
    """Picklable surrogate for an exception raised inside a cell."""

    type_name: str
    message: str
    diagnostic: str = ""


@dataclass
class SignalDTO:
    """Picklable surrogate for a ``return_result()`` control-flow signal."""

    result: bytes


@dataclass
class ResultDTO:
    """Everything from a worker cell run that can cross the pipe."""

    stdout: str = ""
    stderr: str = ""
    error: ErrorDTO | None = None
    signal: SignalDTO | None = None
    returned_value: bytes = b""
    has_return: bool = False
    explicit_return: bool = False
    images: list[dict[str, Any]] = field(default_factory=list)
    wrapper_line_offset: int = 0
    defined_method_names: list[str] = field(default_factory=list)


def result_to_dto(
    result: Any,
    *,
    error_formatter: _ErrorFormatter | None = None,
    max_error: int | None = None,
    tail_chars: int | None = None,
) -> ResultDTO:
    """Convert a worker-side ``ExecutionResult`` into a picklable DTO.

    The presence of a control-flow signal is keyed off ``result.signal`` (not a
    sentinel payload value), and the signal payload is picklability-checked just
    like ``returned_value`` so a non-picklable ``return_result(...)`` yields a
    clean error instead of crashing the worker on ``conn.send``.
    """
    from nooa.events import _NO_RETURN

    dto = ResultDTO(
        stdout=result.stdout or "",
        stderr=result.stderr or "",
        images=list(result.images or []),
        wrapper_line_offset=getattr(result, "wrapper_line_offset", 0),
        defined_method_names=sorted(getattr(result, "defined_methods", {}) or {}),
    )

    if result.error is not None:
        err = result.error
        # User-defined exception formatting must not break the worker send path.
        error_type = type(err).__name__
        try:
            message = str(err) or error_type
        except BaseException:
            message = error_type

        if error_formatter is None:
            from nooa.errors.formatting import format_error_for_llm

            error_formatter = format_error_for_llm

        effective_max_error = effective_error_limit(max_error)
        try:
            diagnostic = error_formatter(
                err,
                None,
                line_offset=getattr(result, "wrapper_line_offset", 0),
                max_error=effective_max_error,
                tail_chars=tail_chars,
            )
            if type(diagnostic) is not str:
                raise TypeError("error formatter must return str")
        except BaseException:
            diagnostic = f"{error_type}: {message}"

        dto.error = ErrorDTO(
            type_name=error_type,
            message=_bounded_error_message(
                message,
                max_error=effective_max_error,
                tail_chars=tail_chars,
            ),
            diagnostic=_bounded_diagnostic(
                diagnostic,
                max_error=effective_max_error,
            ),
        )
        return dto

    if result.signal is not None:
        payload = getattr(result.signal, "result", None)
        try:
            pickled_payload = pickle.dumps(payload)
        except BaseException:
            dto.error = ErrorDTO(
                type_name="CellSerializationError",
                message=(
                    "return_result(...) was called with a value that is not picklable and "
                    "cannot cross the sandbox boundary. Return a JSON/pickle-safe value "
                    "(numbers, str, list, dict, ndarray) instead."
                ),
            )
        else:
            dto.signal = SignalDTO(result=pickled_payload)
        return dto

    rv = result.returned_value
    if rv is not _NO_RETURN:
        try:
            pickled_return = pickle.dumps(rv)
        except BaseException:
            dto.error = ErrorDTO(
                type_name="CellSerializationError",
                message=(
                    f"Return value of type {type(rv).__name__!r} is not picklable and "
                    "cannot cross the sandbox boundary. Keep it in the namespace and "
                    "return a JSON/pickle-safe summary instead."
                ),
            )
        else:
            dto.returned_value = pickled_return
            dto.has_return = True
            dto.explicit_return = bool(result.explicit_return)
    return dto


def _reconstruct_error(err: ErrorDTO) -> Exception:
    """Rebuild an exception and wrap worker diagnostics in a typed boundary."""
    import builtins as _bi

    if err.type_name == "CellSerializationError":
        original: Exception = CellSerializationError(err.message)
    elif err.type_name == "SandboxStateError":
        from nooa.runtime.sandbox.readonly import SandboxStateError

        prefix = "SandboxStateError: "
        message = err.message[len(prefix) :] if err.message.startswith(prefix) else err.message
        original = SandboxStateError(message)
    else:
        cls = getattr(_bi, err.type_name, None)
        if isinstance(cls, type) and issubclass(cls, Exception):
            try:
                message = err.message
                prefix = f"{err.type_name}: "
                if message.startswith(prefix):
                    message = message[len(prefix) :]
                original = cls(message)
            except Exception:
                original = Exception(err.message)
        else:
            original = Exception(err.message)

    if not err.diagnostic:
        return original
    return SandboxExecutionError(
        original_type=err.type_name,
        message=err.message,
        diagnostic=err.diagnostic,
        original_error=original,
    )


def dto_to_result(dto: ResultDTO, *, signal_factory: Any = None) -> Any:
    """Reconstruct a parent-side ``ExecutionResult`` from a :class:`ResultDTO`.

    ``signal_factory(payload) -> ExecutionSignal`` rebuilds the ``return_result``
    signal from its marshaled payload (supplied by the caller that owns the
    concrete signal type).
    """
    from nooa.events import _NO_RETURN, ExecutionResult

    error: Exception | None = None
    if dto.error is not None:
        error = _reconstruct_error(dto.error)

    signal = None
    returned_value: Any = _NO_RETURN
    try:
        if dto.signal is not None and signal_factory is not None:
            signal = signal_factory(pickle.loads(dto.signal.result))
        if dto.has_return:
            returned_value = pickle.loads(dto.returned_value)
    except BaseException:
        error = CellSerializationError(
            "A sandbox result could not be deserialized across the process boundary."
        )
        signal = None
        returned_value = _NO_RETURN

    return ExecutionResult(
        stdout=dto.stdout,
        stderr=dto.stderr,
        error=error,
        signal=signal,
        defined_methods={},
        returned_value=returned_value,
        explicit_return=dto.explicit_return,
        captured_locals={},
        images=dto.images,
        wrapper_line_offset=dto.wrapper_line_offset,
    )
