# Design: Sandboxed (process-backed) cell execution for nemo-oo

- **Status:** Proposed → implemented (this MR)
- **Area:** `src/nooa/runtime/sandbox/` (new), `src/nooa/runtime/actor.py` (`execute_code` delegation seam), `src/nooa/strategies/codeact.py` (session lifecycle + context block), `src/nooa/config/strategy_config.py` (`SandboxConfig`)
- **Reference implementation (pattern only):** `progressive-learning/beam/agent/async_repl.py` (`AsyncREPL`) — a battle-tested process-backed REPL. Copies live in `tmp/beam_reference/`. We port the *pattern* (worker + hard-timeout + tool broker + serialization contract), not the code.

---

## 1. Problem

CodeAct runs LLM-generated cells **in the agent's own process, on the agent's own event loop**. That has two independent failure modes:

1. **No enforceable resource bound.** `CodeActConfig.cell_timeout` is applied with `asyncio.wait_for`, which can only cancel a coroutine that yields back to the loop. A CPU-bound cell (`while True`, a BFS, an `itertools` sweep) never yields, so the timeout callback never fires and the cell runs unbounded. Measured: a single cell ran **2345 s under a 60 s cap** and returned `success=True`. There is no memory or CPU bound at all.
2. **No isolation.** A cell can read any file the process can (`open("/etc/shadow")`, the agent's own API keys), write anywhere, and open arbitrary network sockets. Nothing stops generated code from exfiltrating secrets or corrupting the workspace.

Both stem from the same root: **untrusted code shares the trusted process.** The only real fix is to run the cell in a **separate process** the parent can (a) hard-kill and (b) lock down with OS-enforced guards *before the cell runs*.

## 2. Goals

1. **Four configurable guardrails**, each independently toggleable, all enforced by the kernel in the child:
   1. **Timeout** — a hard wall-clock kill of a CPU-bound cell. **Reuses the existing `cell_timeout`** as the deadline; the sandbox is what makes it enforceable.
   2. **Memory & CPU caps** — `RLIMIT_AS` (address space) and `RLIMIT_CPU` (CPU-seconds), set with `soft == hard` so the cell cannot raise them.
   3. **Filesystem access control** — confine to a workspace subdir, or an explicit allow-list of read / read-write paths. Default-deny via **Landlock**.
   4. **Network access on/off** — block all internet sockets via **seccomp-BPF** (deny `socket(AF_INET/AF_INET6)`), leaving `AF_UNIX` intact for the parent broker.
2. **A context block** telling the agent exactly which constraints are active, so it adapts (writes only under the workspace, expects no network, keeps cells short).
3. **100 % reliable by design.** Guards are installed in the child *after fork, before any cell bytecode executes*, using **irrevocable** mechanisms (`no_new_privs` + seccomp, Landlock `restrict_self`, `setrlimit` with `soft==hard`). Once installed, arbitrary Python in the cell cannot lift them. The parent hard-kill is the backstop for hangs. **Fail-closed:** if a requested guard cannot be enforced on the host, the sandbox refuses to start rather than run unguarded.
4. **Opt-in, backward compatible.** Default `execution_backend="inprocess"` → zero behavior change. `execution_backend="sandbox"` selects the process backend.
5. **Preserve cell semantics** for the common case: persistent REPL namespace across cells, `self.method()` tool calls, `return_result()`, `show()` images, stdout streaming, error tracebacks.

### Non-goals
- Replacing the in-process executor (it stays the default and fast path).
- Data-confidentiality isolation of the parent's memory (with `fork`, the child inherits a COW copy of parent state; the threat model is *action* containment — runaway/exfiltration/damage — not hiding the parent's RAM from itself).
- Windows / macOS. Linux-only (the ARC + CI target). The sandbox probes capabilities and fails closed elsewhere.
- Defending against a *fully adversarial* cell that reaches framework internals to attack the parent. The parent↔worker channel is `pickle`-based and the parent trusts what the worker sends; the OS guardrails contain **actions**, and defense against adversarial parent-RCE is the separate whole-process (uid/namespace) sandbox's job. We do close the trivial direct escape (the broker/pipe is not a proxy attribute — see `worker.py`).

### Known semantic differences from in-process (documented in the `<sandbox>` block)
Because `self` in a cell is a proxy over the parent's live agent, and the worker owns the REPL namespace, a few in-process behaviors change under the sandbox:
- **`self.<attr>` returns a fresh copy**, so in-place mutation (`self.items.append(x)`) mutates a throwaway and is not persisted. Cells must call a method (`self.record(x)`) or reassign (`self.items = self.items + [x]`, which brokers a `__setattr__` to the parent). Attribute *assignment* and container ops on a non-picklable live attribute (`self.context["k"] = v`, `self.memory.remember(x)`) are brokered; in-place mutation of a *fetched copy* is not.
- **`return_result` must pass the value**, not a variable-name string — the parent has no view of the worker's cell locals to resolve a bare identifier.
- **`Out[n]` history and caller-seeded `session_locals` are not present** in the worker namespace (they live on the parent). A cell must recompute rather than reference `Out[-1]`.
- **A hard timeout / OOM kill resets the worker's namespace** (variables/functions from earlier cells are gone). The synthesized error says so, so the model rebuilds state; `recovery="restart_empty"` (default) restarts fresh, `recovery="disabled"` stops the backend.

The middleware pipeline (`execute_python`) and `before/after_code_execution` hooks **do** fire for sandboxed cells: `execute_code` runs the executor internally, keeping hooks, events and tracing on the parent unchanged. `deny` sub-trees and the `spawn` start method are **not** supported (allowlist-only filesystem; `fork` only, since the worker inherits the live agent).

## 3. Enforcement mechanisms (verified on the target: kernel 6.8, Landlock ABI 4)

| Guardrail | Mechanism | Irrevocable? | Verified behaviour |
|---|---|---|---|
| Timeout | parent `asyncio.wait_for(recv, cell_timeout+grace)` → `SIGTERM`→`SIGKILL` → restart worker | n/a (parent-side) | `while True` killed at deadline; parent survives; next cell runs |
| CPU cap | `RLIMIT_CPU` (`soft==hard`) | yes (`soft==hard`) | spin loop → `SIGXCPU`/`SIGKILL` near the cap |
| Memory cap | `RLIMIT_AS` (`soft==hard`) + parent RSS poll | yes | `bytearray(1 GiB)` under 256 MiB cap → clean `MemoryError` |
| Filesystem | Landlock path-beneath rules, default-deny | yes (`landlock_restrict_self`) | read/write outside the allow-list → `PermissionError`; inside → OK |
| Network | seccomp-BPF: `socket(AF_INET/AF_INET6)` → `EACCES` | yes (`no_new_privs`) | inet socket/connect → `PermissionError`; `AF_UNIX` still works |

Why these and not namespaces: unprivileged network/mount namespaces (`unshare`) are blocked by many container seccomp policies (including this CI/dev image), so they can't be the reliable primitive. seccomp + Landlock + rlimits are **unprivileged, self-imposed, and irrevocable**, and work inside containers. All three are installed by the child on itself; nothing in the cell can undo them.

Each mechanism is implemented with raw `ctypes` syscalls (`landlock_create_ruleset`/`add_rule`/`restrict_self`, `seccomp`, `prctl`, `setrlimit`) — **no new dependencies**.

## 4. Architecture

```
Parent (agent event loop)                     Child worker (one long-lived process)
--------------------------                    -------------------------------------
CodeActStrategy.execute()
  creates SandboxedExecutor (lazy, 1st cell)
  fork worker ─────────────────────────────►  install_guards(spec):   [runs ONCE, before any cell]
                                                 setrlimit(AS, CPU)
                                                 landlock_restrict_self(file rules)
                                                 seccomp(no AF_INET)     ← irrevocable from here on
                                                 build persistent namespace; self→ParentAgentProxy
  runtime.execute_code(code, sandbox=exec)
    → executor.run_cell(code, timeout) ──────►  run_cell_source(code, ns)  [nooa wrapper semantics]
       {op:run, code, cell_timeout}                 exec cell; capture stdout/stderr/return/signal
       await wait_for(recv, timeout+grace)          self.method(x) ──► RPC {tool_call} to parent
         parent runs real bound method on live agent ◄──────────────  (await if coroutine)
         result ──► {tool_result} ─────────────────────────────────►  continues cell
       ◄── {ok, ExecutionResultDTO, stdout,...}  ◄── IPC-safe DTO (picklable subset)
    on timeout: terminate→kill→restart(recovery); synth ExecutionResult(error=CellTimeoutError)
```

**Key inversion:** in the child, `self` is a `ParentAgentProxy`. Every `self.tool(...)` RPCs to the parent's **live** agent, so agent state stays authoritative and the child never needs network or the LLM (which seccomp blocks anyway). Cell-local state (functions defined in cell 1, used in cell 3) lives in the child and never crosses the boundary — only explicitly returned / snapshotted picklable values do.

### Files

| File | Role |
|---|---|
| `runtime/sandbox/config.py` | `SandboxConfig`, `FileRule`, enums (pydantic, frozen) |
| `runtime/sandbox/capabilities.py` | probe what the host can enforce; power fail-closed + test skips |
| `runtime/sandbox/guards.py` | `ctypes` primitives: `apply_rlimits`, `apply_landlock`, `apply_seccomp_no_inet`, `install_guards` |
| `runtime/sandbox/cell_core.py` | `run_cell_source(...)` — the wrapper/compile/exec/capture core, mirroring `actor.execute_code` |
| `runtime/sandbox/worker.py` | child entrypoint: `install_guards`, op loop, `ParentAgentProxy` |
| `runtime/sandbox/executor.py` | parent `SandboxedExecutor`: fork/restart, hard-timeout, tool broker, RSS watchdog, IPC |
| `runtime/sandbox/serialization.py` | IPC-safe `ExecutionResult` DTO, error surrogate, picklability checks |
| `runtime/sandbox/context_block.py` | render active constraints into an agent-facing block |
| `runtime/sandbox/errors.py` | `CellTimeoutError`, `CellMemoryError`, `SandboxUnavailable`, `CellSerializationError` |
| `config/strategy_config.py` | `+ execution_backend`, `+ sandbox: SandboxConfig` on `CodeActConfig` |
| `runtime/actor.py` | `execute_code(..., sandbox_executor=None)`: when set, delegate exec to it (hooks/events unchanged) |
| `strategies/codeact.py` | create/teardown executor per session; inject sandbox context block via `get_block_overrides` |

## 5. Config surface

```python
class FileRule(BaseModel):        # frozen
    path: str
    access: Literal["read", "read_write"] = "read"

class SandboxConfig(BaseModel):   # frozen
    # filesystem (guardrail 3)
    filesystem: bool = True                 # enforce Landlock at all
    workspace: str | None = None            # a dir the cell gets read+write to
    allow: list[FileRule] = []              # extra explicit allow rules
    system_paths: bool = True               # auto-allow read of interpreter/libs so Python works
    # network (guardrail 4)
    network: bool = False                   # False → internet blocked; True → allowed
    # memory + cpu (guardrail 2)
    max_memory_mb: int = 0                  # 0 = disabled
    max_cpu_seconds: int = 0                # 0 = disabled
    rss_poll_s: float = 0.25                # parent RSS watchdog interval
    # timeout (guardrail 1) — reuses CodeActConfig.cell_timeout; this is only the extra grace
    timeout_grace_s: float = 2.0
    # process / recovery
    start_method: Literal["fork"] = "fork"   # fork only (worker inherits the live agent)
    recovery: Literal["restart_empty", "disabled"] = "restart_empty"
    require: bool = True                     # fail-closed if a requested guard is unenforceable
    context_block: bool = True               # inject the constraints context block

class CodeActConfig(BaseModel):
    ...
    cell_timeout: float | None = None
    execution_backend: Literal["inprocess", "sandbox"] = "inprocess"
    sandbox: SandboxConfig = SandboxConfig()
```

The **timeout guardrail deliberately has no timeout field of its own** — it uses `cell_timeout`. The sandbox turns that (previously advisory) value into a hard bound: `hard_deadline = cell_timeout + sandbox.timeout_grace_s`.

Opt-in example (the ARC agent):

```python
@strategy(CodeActStrategy(config=CodeActConfig(
    cell_timeout=60.0,
    execution_backend="sandbox",
    sandbox=SandboxConfig(
        workspace="/work/arc_runs",
        network=False,
        max_memory_mb=4096,
        max_cpu_seconds=120,
    ),
)))
async def solve(self): ...
```

## 6. The context block

When `sandbox.context_block` is on, the strategy injects one block each turn (via `get_block_overrides()` → a `DynamicContext` pointing at a strategy method) describing the *active* constraints, e.g.:

```
<sandbox>
Your code runs in an isolated process with enforced limits:
- Wall-clock: each cell is hard-killed after 60s. Keep cells short; return partial results.
- CPU: 120 CPU-seconds per cell. Memory: 4096 MB per cell (allocation past this raises MemoryError).
- Filesystem: read+write only under /work/arc_runs. Reads/writes elsewhere raise PermissionError.
- Network: disabled. Sockets to the internet raise PermissionError; do not attempt downloads.
Values returned from a cell must be picklable (JSON/number/str/list/dict/ndarray); keep live objects
in the namespace and return a summary instead.
</sandbox>
```

Only the guards actually in force are listed; a disabled guard is omitted.

## 7. Serialization contract (IPC boundary)

`ExecutionResult` has non-picklable fields. Rules (mirroring beam's `_safe_result_for_ipc`):
- `defined_methods`, `captured_locals` — **stay in the child** (the persistent namespace). Parent gets empty dicts; the values remain reachable by name in later cells.
- `returned_value` — if picklable, cross it; else replace with a `CellSerializationError` carrying a clear "return a summary instead" message.
- `error` — reduced to a picklable surrogate (type name, message, formatted traceback) and re-raised as a lightweight exception on the parent so `_format_error` still works, with `wrapper_line_offset` preserved.
- `signal` (`return_result`) — marshaled as a picklable record and re-raised as the signal on the parent.
- `images` (`show()`) — already dicts, picklable.
- Tool-call args/return values must be picklable; a clear `CellSerializationError` is raised otherwise.

## 8. Reliability argument (why "no leakage")

1. **Ordering.** `install_guards()` runs as the first thing in the worker, before the op loop, before any cell. There is no window in which cell code runs unguarded.
2. **Irrevocability.** seccomp (with `no_new_privs`) and Landlock (`restrict_self`) cannot be relaxed by the restricted process; `setrlimit` with `soft==hard` cannot be raised. So arbitrary cell code — even `import ctypes` — cannot remove a guard.
3. **Fail-closed.** `capabilities.probe()` runs at executor start; if `require=True` and a requested guard is unenforceable, the executor raises `SandboxUnavailable` instead of running the cell unguarded.
4. **Backstop.** If the child ignores an in-child interrupt or wedges in a C call, the parent's `wait_for` + `SIGKILL` terminates it regardless.

The test suite proves each link: a leak test shows the exploit succeeding *without* the guard, and a paired test shows the *same* exploit raising under the guard.

## 9. Test plan

`tests/runtime/sandbox/` (marker `integration` for the ones that fork):

**Guardrail leak ↔ closed pairs** (the headline):
- `timeout`: `while True` — in-process exceeds the cap (documents the bug); sandbox kills at `cell_timeout+grace`, parent survives, next cell runs.
- `memory`: `bytearray(2 GiB)` — unbounded child OOMs; sandbox → `CellMemoryError`.
- `cpu`: spin — unbounded child runs; sandbox → killed near `max_cpu_seconds`.
- `file-read`: `open(<secret outside workspace>)` — succeeds unguarded; sandbox → `PermissionError`.
- `file-write`: `open(<outside>, "w")` — succeeds unguarded; sandbox → `PermissionError`; write inside workspace OK.
- `network`: `socket.create_connection(...)` / `urllib` — succeeds unguarded; sandbox → `PermissionError`; `AF_UNIX` still works.

**Semantic parity:** persistent namespace across cells; `self.method()` broker (sync + async generation method, with the child's soft timer suspended during the RPC); `return_result()` signal; `show()` images; serialization boundary (`return lambda` → clear error, parent healthy); traceback line fidelity; restrictions parity (blocked import rejected in worker too).

**Capability / fail-closed:** `require=True` + unavailable guard → `SandboxUnavailable`; `probe()` reports each mechanism.

**No regression:** the whole existing CodeAct/actor suite is green with the default `inprocess` backend (the delegation branch is only taken when the sandbox is enabled).

## 10. Rollout

- Default `inprocess` → zero change; existing suites stay green.
- ARC example flips `execution_backend="sandbox"` on; its `cell_timeout=60` becomes a real bound and its cells lose filesystem/network reach they never needed.
- Composes with the existing whole-process OS sandbox (`examples/arc_agi_3/sandbox.py`): the worker is just another process inside that namespace.
