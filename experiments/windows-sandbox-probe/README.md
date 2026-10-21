# Windows native sandbox feasibility (M0)

## Research question

Can a native Windows **AppContainer** host the CodeAct worker process, with
filesystem, network, memory and CPU confinement strong enough to back a
`sandbox` execution backend — without WSL, Docker, Hyper-V, or administrator
rights?

This is the gating experiment for adding a Windows backend to
`src/nooa/runtime/sandbox/`, whose current implementation requires Linux
(`fork`, Landlock, seccomp, `RLIMIT_*`).

## Experiment design

`runner.py` drives eight scenarios against a real AppContainer process. Every
child runs a **private copy** of the interpreter (`.rt/`, built by
`_mkroot.py`), because the base interpreter directory denies DACL writes to a
non-admin user.

| # | Scenario | Question |
|---|---|---|
| 1 | reachable | Does the container start Python and report itself as a container? |
| 2 | reversible | Can a grant be undone byte-exactly, and does `revoke` alone suffice? |
| 3 | denied | Are reads/writes outside the grants refused, while granted paths work? |
| 4 | no-network | Is TCP blocked without a capability, and allowed with one? Is UDP egress observable? |
| 5 | job-limits | Do memory/CPU caps and hard kill work? |
| 6 | broker pipe | Can an inherited anonymous pipe carry a bidirectional round trip? |
| 7 | nooa-import | Does `import nooa` resolve to the runtime-root copy, with the checkout still denied? |
| 8 | mp-pipe | Does the broker's actual `multiprocessing` connection cross the boundary? |

Process creation uses `CreateProcessW` with
`PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES` (the only way into an
AppContainer: it is a creation-time attribute) plus
`PROC_THREAD_ATTRIBUTE_HANDLE_LIST`, created **suspended**, assigned to a Job
Object, then resumed — closing the escape window between creation and job
membership.

All results come from one machine: Windows 11 build 26300, Python 3.13.12,
non-admin user.

## Key metrics

- Container startup latency (create → resume → exit)
- Denial verdicts per path, with `errno`/`WinError`
- TCP/UDP egress under capability combinations
- Memory-cap behaviour and CPU-cap termination timing
- Pipe round-trip success and latency
- Runtime-root copy size and build time

## How to run

```bash
# Build the private interpreter root (≈4.5 min, ~420 MB)
python _mkroot.py

# Refresh the nooa package in an existing root (<2 s); run after source edits
python _mkroot.py --sync-nooa

# All eight scenarios
python runner.py

# One scenario
python runner.py --only 4
```

`_mkroot.py` is a thin wrapper over `nooa.runtime.sandbox.win_runtime`; the
implementation lives in the package so this experiment and the shipped backend
cannot drift apart. It targets this directory's `.rt`, deliberately separate
from the backend's per-user cache. Use the package entry point for a real
session:

```bash
python -m nooa.runtime.sandbox.win_runtime --build
python -m nooa.runtime.sandbox.win_runtime --status
```

Requires Windows 10 1809+ (AppContainer) and Windows 10 1803+ for named pipes.
No admin rights. Results are written to `results.json`.

## Results summary

One full run on 2026-10-17 (`results.json`).

### 1. Reachable — yes

| Metric | Value |
|---|---|
| `is_app_container` | `true` |
| Container SID | `S-1-15-2-606662641-…-4016704499` |
| `sys.prefix` inside | private root (`…/.rt`), not the system install |
| Startup (create→exit) | 0.19–0.56 s |
| Admin required | no |

The container cannot read its own script from the repository until that single
file is granted; granting the directory would expose the whole checkout. This
was observed directly: the first run failed with `Errno 13` on
`child_probe.py`.

### 2. Reversible — yes, with two preconditions

| Metric | Value |
|---|---|
| SDDL byte-equal after restore | **yes** |
| ACE set equal after restore | **yes** |
| ACEs added by grant | 1 |
| ACEs added by a second grant for the same SID | 1 (merged, not appended) |
| `revoke` alone restores | **yes** (byte-equal SDDL) |

Preconditions, both measured:

1. **The target must be user-owned.** DACL writes to the base interpreter
   (`D:\dev\uv\python\…`) and to `C:\Windows\System32` fail with `Errno 5`
   without elevation. This is why the interpreter is copied.
2. **Inheritance must be broken (`PROTECTED_DACL`).** On an inheriting
   directory the `AI` control bit is recomputed from the parent on every write,
   so no restore can be byte-equal by construction. The test therefore runs on a
   directory with `D:P…`.

`GRANT_ACCESS` merges into an existing ACE for the same SID
(`READ|EXECUTE|WRITE` → `0x12019f`), so a rollback journal that just replays
ACE additions will drift. **`REVOKE_ACCESS` is the reliable teardown path**;
store SDDL for audit, not as the undo mechanism.

### 3. Denied — all expected denials hold

| Probe | Result |
|---|---|
| read `C:\Windows\System32\config\SAM` | DENIED (EACCES) |
| read `C:\Windows\System32\config\SYSTEM` | DENIED (EACCES) |
| read `%USERPROFILE%\.ssh` | DENIED (EACCES) |
| read `%USERPROFILE%\.claude\CLAUDE.md` | DENIED (EACCES) |
| read `<repo>\AGENTS.md` | DENIED (EACCES) |
| write granted workspace | **ALLOWED** |
| write `System32`, `Windows\Temp`, `%USERPROFILE%` | DENIED (EACCES) |
| write next to the granted dir | DENIED (EACCES) |

Probe paths must exist: a missing file reports `ENOENT`, which is not evidence
of a denial. An earlier run used a nonexistent `.ssh/id_rsa` and was discarded.

### 4. Network — TCP solid, UDP unresolved

| Capability | TCP `1.1.1.1:443` | UDP to host socket | Loopback listen |
|---|---|---|---|
| none | **BLOCKED** (WSAEACCES 10013) | no datagram observed | ALLOWED |
| `internetClient` | **ALLOWED** | no datagram observed | — |
| `internetClient` + `privateNetworkClientServer` | — | no datagram observed | — |

TCP has a clean two-way control: blocked without the capability, allowed with
it. UDP is **not attributable**: `sendto` returns success in every case, but a
host-side socket observed no datagram even with the capability granted. The
only non-loopback address on this machine is `198.18.0.1`, a TUN adapter, so
container traffic to it may be dropped by the adapter rather than by the
AppContainer policy. Host-to-host UDP on that address does arrive, which rules
out a blanket host firewall but not the TUN path.

**Conclusion:** do not claim UDP isolation from this experiment. `network=False`
should be backed by an explicit WFP filter (or verified on a machine with a
real LAN adapter), not by capability omission alone.

**Resolved later** by [UDP egress](#udp-egress): with a working positive
control and a wire capture, omitting capability SIDs blocks UDP as well.

### 5. Job limits — both work

| Metric | Value |
|---|---|
| Memory cap 64 MB → allocation probe | `MemoryError` at 48 MB allocated |
| CPU cap 2 s → process terminated | exit code `3221225540` (`0xC0000044`, `STATUS_QUOTA_EXCEEDED`) |
| CPU-cap wall clock | 2.6–5.9 s (container overhead varies) |
| `TerminateJobObject` reap | 0.003–0.006 s |
| Hard-kill exit code | 1 |

Memory is enforced as a raised `MemoryError` inside Python, not a silent kill;
the worker can report it. CPU and wall-clock termination are hard kills.

### 6. Broker pipe — works

| Metric | Value |
|---|---|
| Child → parent payload | `HELLO` (length-prefixed frame) |
| Parent → child payload | `WORLD` |
| `bidirectional_ok` | `true` |
| Round trip latency | 0.19–0.28 s (includes interpreter start) |

`PROC_THREAD_ATTRIBUTE_HANDLE_LIST` accepts only **inheritable** handles; a
non-inheritable handle in the list makes `CreateProcessW` fail with
`ERROR_INVALID_PARAMETER` (87). `CreatePipe` + `SetHandleInformation` is
sufficient — Windows CPython has **no `AF_UNIX`** (`socket.AF_UNIX` does not
exist), so a socketpair design is not portable; anonymous pipes are the right
primitive.

### 7. nooa-import — works, and the checkout stays denied

Lets the container import the framework from the runtime root alone. Grants:
runtime root, the probe script, the work directory. Nothing on the checkout,
the venv, or `src`.

| Metric | Value |
|---|---|
| `_editable_impl_*.pth` remaining in root | 0 (all six deleted) |
| `import nooa` | OK, `__file__` inside the runtime root |
| `nooa.runtime.sandbox{,.config,.worker,.executor}` | all import |
| `nooa.strategies` | imports |
| `importlib.metadata.version("nooa")` | `0.0.11.dev415` (dist-info found, no fallback) |
| read `<repo>\src\nooa\__init__.py` | **DENIED** |
| Wall clock (create → exit) | 19.7 s (first run: cold bytecode cache) |
| Package sync | 307 files / 4.4 MB / 0.65–1.25 s |

Both halves are required for the verdict. An import that *works* while the
checkout stays readable would have proven nothing — it would mean the editable
hook survived and the "isolation" was a path alias. The probe asserts
`__file__` starts with `sys.prefix` and separately attempts a read of the
source package; one success without the other is a failure.

The 19.7 s figure is first-run only: it includes compiling 307 modules with no
`__pycache__` (the sync strips them deliberately, so stale bytecode cannot
shadow new source). A warm run should be materially faster; that is not yet
measured.

### 8. mp-pipe — the broker's real transport crosses the boundary

Scenario 6 used a hand-rolled anonymous pipe. The backend does not: its broker
speaks `multiprocessing.connection.Connection`, which on Windows is a **named
pipe** created with a default security descriptor. If the container had to open
that name, the DACL would refuse it.

It does not have to. `Popen.duplicate_for_child` hands the child a duplicated
handle, and `DuplicateHandle` does not consult the pipe's DACL.

| Metric | Value |
|---|---|
| `mp.Pipe(duplex=True)` round trip | `{'reply': 'HELLO', 'ok': True}` |
| Child exit code | 0 |
| Wall clock (create → reply → exit) | 0.30 s |
| Handle requirement | `SetHandleInformation(HANDLE_FLAG_INHERIT)` first |

`PROC_THREAD_ATTRIBUTE_HANDLE_LIST` rejects non-inheritable handles (87), so
the pipe end must be flagged inheritable before it appears in the list.

**Consequence:** the broker transport needs no replacement. `wire.py` and
`worker.py` can be reused verbatim; the Windows work is a process launcher, an
ACL layer, and a job object, not a new IPC protocol.

## Backend verification (M3)

The probe above established feasibility. The backend built on it
(`nooa/runtime/sandbox/win_api.py`, `win_runtime.py`, `win_launcher.py`,
`win_worker.py`) is verified by `m3_smoke.py`, which drives the shipped
`SandboxedExecutor` end to end — config → capability probe → ACL grants →
suspended `CreateProcessW` → job assignment → resume → ready handshake →
blueprint rebuild → cell execution → teardown.

| Stage | Result |
|---|---|
| construct | ok |
| cell (`value = self.answer(); value`) | returned `42` |
| broker cell (`self.echo('hi')`) | returned `'probe:hi'` |
| read `<repo>\src\nooa\__init__.py` | **DENIED** (`PermissionError`, errno 13) |
| close | ok |

```powershell
$env:NOOA_SANDBOX_RUNTIME = "$PWD\experiments\windows-sandbox-probe\.rt"
python experiments\windows-sandbox-probe\m3_smoke.py
```

Four defects were found and fixed by this run; each one would have been
invisible in code review:

1. **`psapi.GetProcessMemoryInfo` crashed on the 64-bit handle.** No
   `argtypes`, so ctypes marshalled the handle as a C `int` and raised
   `OverflowError`. The worker died before its ready message, with no output:
   the process is created without inherited std handles.
2. **A startup crash had no channel.** `CREATE_NO_WINDOW` plus no inherited
   std handles means stderr goes nowhere. The worker now appends a traceback to
   a file in the granted workspace, which is what surfaced defect 1.
3. **The agent's import root was granted but not on `sys.path`.** `-I` strips
   cwd and `PYTHONPATH`; the launcher now descends the blueprint's
   `module:qualname` to the module's own directory, grants read on it (never its
   ancestors), and passes it in `init["import_roots"]`.
4. **The ready handshake preceded the namespace build.** The first cell was
   charged for the child's cold import (≈20 s measured in scenario 7) and hit
   its 30 s deadline. `worker_main` now takes an `on_ready` callback fired after
   the namespace exists. The memory baseline is taken there too, so
   `max_memory_mb` is headroom above a worker that is ready to run cells —
   matching the `fork` baseline, which inherits the parent's mapped image.

### A semantic difference from Linux

Linux `spawn` imports the agent module *before* installing guards. An
AppContainer is confined from process creation, so the import happens **inside**
the sandbox and the module's directory must be granted. The consequence is that
the container can read the directory holding the agent module — for the probe,
`experiments/windows-sandbox-probe/` — while the checkout stays denied. A user
who keeps an agent module inside a sensitive tree should move it or accept that
one directory of it is readable by cell code.

## Environment inheritance

**Question.** Does a variable that exists only in the parent reach cell code,
and does narrowing the environment block break anything that worked before?

**Design.** `env_probe.py` sets a sentinel in the parent, starts the shipped
`SandboxedExecutor`, and asks the cell for its variable names, whether it sees
the sentinel, and whether Winsock, asyncio, `tempfile` and `Path.home()` work.
Only names, counts, booleans and exception types leave the cell; the workspace
is a temp directory removed afterwards. Run once on the launcher that copied
`os.environ`, once on the allowlist (`CONTAINER_ENV` + `env_passthrough`).

| Metric | Copy of `os.environ` | Allowlist |
|---|---|---|
| Parent variables | 119 | 119 |
| Visible in the container | 120 | 33 |
| Sentinel reached the cell | **yes** | **no** |
| Crossed but not allowlisted | — | none |
| `tempfile.gettempdir()` | `…\Packages\nooa-sandbox\AC\Temp` | same |
| `Path.home()`, `socket.socket()`, asyncio loop | ok | ok |

The one variable on top of the parent's is `CUSTOM_TIKTOKEN_CACHE_DIR`, which
the worker sets from argv; it and `TIKTOKEN_CACHE_DIR` are the 2 of 33 not on
the allowlist. Three of the parent's variables were credentials for unrelated
services, all readable by cell code before the change.

What fixed the allowlist's contents, each measured on a bare interpreter:

- **`SYSTEMROOT`** — without it Winsock does not initialise: `socket.socket()`
  raises `WinError 10106`.
- **`USERPROFILE`** — without it `Path.home()` raises `RuntimeError`; nooa's
  `get_user_dir` depends on it.
- **`USERNAME`** — without it `getpass.getuser()` raises `OSError`.
- **`TEMP` / `TMP` / `LOCALAPPDATA`** point at the container's own `AC` folder
  in both runs, not at the parent's values. Redirecting `TEMP` to the
  workspace was considered and dropped: `tempfile` already has a private,
  writable directory.

Found along the way, unrelated to the environment and not fixed by this change:
importing `nooa.util.quickstart` in the container raises `PermissionError`.
`find_project_root()` walks up from the package and calls `Path.exists()` on
each ancestor's `pyproject.toml`; for a file that exists but is unreadable that
raises instead of returning `False`. A runtime root inside a checkout (this
probe's `.rt`, CI's `.sandbox-rt`) has the checkout's `pyproject.toml` as an
ancestor. The default `%LOCALAPPDATA%` root was not measured.

**Fixed later** in the launcher, not in `nooa.paths`: it sets
`NEMO_OO_PROJECT_DIR` and `NEMO_OO_USER_DIR` for the container to a location
under the read-only root that does not exist, so discovery never walks up.
Measured first: with both set, the import succeeds and writes nothing.
`test_agent_built_on_the_quickstart_helpers_runs` fails without the change.

```powershell
uv run python experiments\windows-sandbox-probe\env_probe.py
```

## UDP egress

**Question.** Does omitting capability SIDs block UDP egress, or only TCP?
Scenario 4 could not tell: its check failed even with `internetClient`.

**Design.** `udp_dns_probe.py` runs one child three ways (host unconfined,
container with `internetClient`, container with no capability) and completes
DNS round trips to `1.1.1.1` and `8.8.8.8`. A reply proves the datagram left;
the capability case is the positive control. A missing reply cannot rule out
a one-way send, which is all exfiltration needs, so the CI step also captures
the wire with `pktmon`, filtered to the two resolvers.

| Check | Host | `internetClient` | No capability |
|---|---|---|---|
| TCP `1.1.1.1:443` | connected | connected | `WinError 10013` |
| UDP DNS `1.1.1.1` / `8.8.8.8` | reply | reply | timeout (4 s) |
| `getaddrinfo` | resolved | resolved | `gaierror 11001` |
| Wire (`pktmon`, CI only) | — | Ethernet frames to both | `Drop:` at the IP layer, no frame |

The table holds on the reference machine and on the CI runner (a real NIC,
`10.1.0.x`). The wire capture settles the one-way question: without the
capability the stack drops the datagram before any Ethernet frame is built.

Both ran with the Windows firewall service (`mpssvc`) running, whose WFP
filters implement AppContainer network isolation. Behaviour with it stopped
was not measured.

```powershell
uv run python experiments\windows-sandbox-probe\udp_dns_probe.py
```

## Concurrent sessions

**Question.** Can two sandbox sessions run at the same time without one
affecting the other?

**Design.** `concurrency_probe.py` starts session A, runs a cell, starts B while
A is live, then measures three things: whether B's startup revoked A's
workspace grant (A's DACL, then a write from A), whether closing A destroyed B's
journal, and whether the two sessions ran under the same container SID. All
three are measured against real processes.

**Result (before the fix).** Both sessions ran as `nooa-sandbox` and appended to
one `acl-journal.jsonl`. B's `prepare_root` recovered that journal — revoking
A's live workspace grant, so A's next write raised `PermissionError` — and
`A.release()` unlinked the journal B would need after a crash. Verdict:
`SHARED IDENTITY: concurrent sessions interfere. Not production-safe.`

**Result (after the fix).** Each session mints `nooa-sandbox-<12 hex>`, so the
SIDs differ; each owns `acl-journal-<name>.jsonl`; recovery reclaims only
journals whose lock no live process holds, and the owning process holds that
lock for its whole life (the kernel drops it on exit, however the process
dies). The same probe now reports `distinct_identity True`,
`a_grant_survived_b_start True`, `a_lost_write_access False`,
`journal_survived_a_close True`, `b_grant_survived_a_close True` — verdict
`INDEPENDENT`.

The profile is deleted at teardown: its name is unique per start, so leaving it
behind would accumulate entries in the user's package registry that nothing can
ever reuse.

```powershell
$env:NOOA_SANDBOX_RUNTIME = "$PWD\experiments\windows-sandbox-probe\.rt"
uv run python experiments\windows-sandbox-probe\concurrency_probe.py
```

## Consequences for the backend

1. **A private runtime root is mandatory, not an optimisation.** 22,598 files /
   419 MB / 274 s to build. Reuse it across processes; rebuilding per worker is
   not viable.
2. **The package is copied into the root; the checkout is never granted.** The
   editable installs in `.venv` are `.pth` files pointing at
   `E:\labs-OO-Agents\src`. Deleting those six hooks and copying `src/nooa`
   into the root (307 files, 4.4 MB, <2 s) gives the isolation of a non-editable
   install without an installer, a wheel build, or dependency resolution. The
   alternative — granting read on `src` — was rejected on two measured grounds:
   it hands over the whole checkout, and `src` is an inheriting directory where
   byte-exact DACL rollback is impossible (scenario 2), so a killed parent
   leaves a permanent ACE on the user's working tree. Every ACL write must stay
   inside the runtime root, which the root needs a grant on anyway.
3. **Teardown uses `revoke`, not SDDL replay.** Plus a per-session journal for
   crash recovery: a killed parent leaves ACEs behind, and they are persistent.
   The journal is keyed by the session's container name and locked by its owner
   for as long as it runs, so a start only reclaims journals whose lock no live
   process holds — see [Concurrent sessions](#concurrent-sessions).
4. **`network=False` is backed by capability omission alone**, for TCP and UDP
   ([UDP egress](#udp-egress)). No WFP filter of our own, so no admin rights.
5. **The launcher must be a separate, unconfined process**: it needs to write
   ACLs and create the job, and `SECURITY_CAPABILITIES` cannot be applied to an
   existing process.
6. **`PROC_THREAD_ATTRIBUTE_HANDLE_LIST` + `SECURITY_CAPABILITIES` compose
   fine**, which is what makes the broker design work unchanged.

## Not measured

- Warm-start cost after the bytecode cache is populated (19.7 s is cold).
- Long-running stability and startup cost under repeated worker restarts.
- Network isolation with the Windows firewall service stopped.
- Low-integrity / LPAC variants.
- Whether a worker can import an agent module that lives outside the granted
  workspace (the blueprint path needs `module:qualname` to be importable).
- Environment visibility on the Linux backend, which does not filter it.
