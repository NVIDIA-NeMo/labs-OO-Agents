# CodeAct OS sandbox removal

The legacy `CodeActConfig(execution_backend="sandbox")` backend and the entire
`nooa.runtime.sandbox` package are removed by owner decision. CodeAct runs in the
agent process. Explicit `execution_backend="inprocess"` remains accepted.

Requests for the removed backend, or any `CodeActConfig(sandbox=...)` input,
raise a configuration error, including JSON/YAML mappings and disabled/null
sandbox settings. Imports from the removed package fail. Runtime
`execute_code(sandbox_executor=...)` is removed from the implementation and
protocol; callers receive a signature error rather than an unconfined fallback.

Migrate isolation to the whole agent before removing old settings: use a
container, VM or external OpenShell environment. In-process `cell_timeout` is
cooperative and cannot terminate blocking synchronous code. It replaces none of
the removed hard cell deadlines, Landlock filesystem confinement, seccomp
network filtering or CPU/memory rlimits. Static code restrictions remain defense
in depth, without a containment guarantee.

ARC-AGI-3 rejects `--sandbox-cells`, `--sandbox-workspace`, `--sandbox-mem-mb`,
`--sandbox-cpu-s`, `--sandbox-require`, all `ARC_SANDBOX_*` environment inputs,
and the run_multi `sandbox:` YAML key. Rejection happens before process launch,
even for disabled/null inputs. Its whole-agent `--sandbox` modes (including the
default `drop`), uid guards, and namespace/container isolation remain available.

The `nooa[sandbox]` extra installs external OpenShell; it does not implement this
backend and remains available. Atom/SBX, benchmark containers and other external
services are independent and retained. The only orphaned direct dependency was
`msgpack`, used for the worker wire format. Older changelog entries describe
earlier changes; the removal entry supersedes their backend-specific claims.

## Updating custom integrations

After configuring external isolation, remove `sandbox` from saved CodeAct
configuration mappings and remove imports of `SandboxConfig`, `FileRule`,
`SandboxedExecutor` and the worker exception types. Configure ordinary CodeAct
execution with `CodeActConfig()`; an explicit `execution_backend="inprocess"`
is also accepted. Keep application-specific code restrictions and result
validators as needed, but do not treat them as OS isolation.

Custom runtimes and middleware should stop forwarding `sandbox_executor` to
`execute_code()`. Execution errors are ordinary in-process exceptions; remove
worker-specific handling of `SandboxExecutionError.original_type`,
`original_error` and `diagnostic`. Source-aware error formatting, output capture,
namespace persistence, result validation and cancellation remain available.

Remove test invocations of `tests/runtime/sandbox/` and `pytest -m sandbox`:
the worker tests and marker are removed. Continue testing the in-process
contracts your integration uses, and verify your external isolation separately.
Passing NOOA's unit tests does not establish a containment boundary for your
deployment.
