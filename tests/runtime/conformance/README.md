# CodeAct backend conformance matrix

Which observable CodeAct contracts are expected to match across
`execution_backend="inprocess"` and `execution_backend="sandbox"`, and which
are deliberately backend-specific. Tracks NVIDIA-NeMo/labs-OO-Agents#187.

## Shared contracts

Same assertions run against both backends. Failures name the backend in the
pytest node ID, e.g. `test_stdout_is_captured[sandbox]`.

| Contract | Observable via | Notes |
| --- | --- | --- |
| Implicit and explicit returns | `PythonOutput.value`, `.explicit_return` | Explicit return also auto-completes the task on both backends |
| stdout capture | `PythonOutput.stdout` | |
| stderr capture | `PythonOutput.stderr` | |
| Namespace persistence across cells | cell 2 reads cell 1's binding | |
| Helper definitions across cells | `def` in cell 1, called in cell 2 | |
| Restriction enforcement | `execution_status`, `.error` | `RestrictionsConfig` defaults apply to both backends |
| Validation retry | session completes after a rejected cell | **Deferred** — not yet covered |
| Nested agent/tool calls | event ordering | **Deferred** — not yet covered |
| `PythonOutput` status and ordering | `event_type` sequence, `tool_call_id` | |
| Runtime exception surfacing | `execution_status is ResultStatus.ERROR` | Status and exception type |
| Runtime exception source context | `PythonOutput.error` | Frames, source lines and carets; regression guard for #191 |
| SyntaxError source and caret fidelity | `PythonOutput.error` | **Deferred** — not yet covered |
| Source line and wrapper-offset fidelity | `PythonOutput.error` | Line numbers pinned per frame |
| `return_result()` success and failure | method return value | Failure transport **deferred** — not yet covered |
| Cancellation and timeout | TBD | Parity boundary undefined; see open question |

## Backend-specific contracts

Not parameterised. Remain in their existing suites.

| Contract | Suite |
| --- | --- |
| OS capability probes and security restrictions | `tests/runtime/sandbox/test_guards.py` |
| IPC serialization failures | `tests/runtime/sandbox/test_executor.py` |
| Broker and proxy behaviour | `tests/runtime/sandbox/test_broker_deadline.py`, `test_nested_async_proxy.py` |
| Worker death and restart | `tests/runtime/sandbox/test_executor.py` |
| Filesystem and network confinement | `tests/runtime/sandbox/test_guards.py` |
| CPU, memory and hard-timeout enforcement | `tests/runtime/sandbox/test_guards.py`, `test_executor.py` |
| In-process implementation details and mocks | `tests/strategies/test_codeact_strategy.py` |

## Capability-dependent skips

Sandbox params skip with an explicit reason when the host cannot enforce the
guards the backend relies on — non-Linux, or missing landlock, seccomp or
rlimit. `SandboxConfig(require=False)` would otherwise let a cell run with
enforcement absent, so a pass on such a host would say nothing about the
behaviour being asserted.

## Open question

Which cancellation and timeout semantics are intended to match? `cell_timeout`
is a *hard* bound under sandbox and a cooperative one in-process, so full parity
is not obviously the goal. Raised on #187; unanswered at time of writing.